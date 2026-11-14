"""
train.py - Training pipeline for Libaas Lens.

Trains:
  1. Baseline: Frozen MobileNetV2 with new classification head (Adam 1e-3).
  2. Fine-Tuned: Top layers unfrozen + data augmentation (Adam 1e-5, ReduceLROnPlateau).

Embedded Preprocessing:
  tf.keras.layers.Rescaling(1./127.5, offset=-1.0) maps [0, 255] -> [-1, 1] inside the model.

Outputs:
  - models/baseline.keras
  - models/finetuned.keras
  - models/final_model.keras
  - reports/baseline_training.csv
  - reports/finetuned_training.csv
  - reports/experiments.csv
"""

import os
import csv
import random
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.applications.mobilenet_v2 import MobileNetV2

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROCESSED_DIR = "libaas/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

CLASSES = [
    "Dress",
    "Jacket - Blazer",
    "Jeans - Trousers",
    "Kurta",
    "Lehenga",
    "Nehru Jacket",
    "Saree",
    "Sherwani",
    "Shirt",
    "T-shirt",
]

BATCH_SIZE = 32
IMG_SIZE = (224, 224)


def load_datasets():
    print("\n--- Loading Datasets ---")
    train_ds = keras.utils.image_dataset_from_directory(
        os.path.join(PROCESSED_DIR, "train"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
        label_mode="int"
    )
    val_ds = keras.utils.image_dataset_from_directory(
        os.path.join(PROCESSED_DIR, "val"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int"
    )
    test_ds = keras.utils.image_dataset_from_directory(
        os.path.join(PROCESSED_DIR, "test"),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int"
    )

    assert train_ds.class_names == CLASSES, f"Mismatch in class names: {train_ds.class_names} != {CLASSES}"
    print(f"Verified {len(train_ds.class_names)} classes in alphabetical order:")
    for idx, c in enumerate(train_ds.class_names):
        print(f"  [{idx}] {c}")

    # Optimize data loading pipeline
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


def build_baseline_model():
    print("\n--- Building Baseline MobileNetV2 Model ---")
    inputs = layers.Input(shape=(224, 224, 3), name="input_image")
    # Rescaling layer directly reproduces MobileNetV2 preprocess_input: [0, 255] -> [-1, 1]
    rescaled = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="preprocess_input")(inputs)

    backbone = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False

    x = backbone(rescaled, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.2, name="top_dropout")(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LibaasLens_Baseline")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model, backbone


def train_baseline(model, train_ds, val_ds, epochs=12):
    print("\n=== Training Baseline Model ===")
    baseline_path = os.path.join(MODELS_DIR, "baseline.keras")
    csv_log_path = os.path.join(REPORTS_DIR, "baseline_training.csv")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(baseline_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.CSVLogger(csv_log_path)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    return history


def build_and_train_finetuned(train_ds, val_ds, baseline_model, epochs=15):
    print("\n=== Building & Training Fine-Tuned Model ===")
    
    # 1. Inputs & Augmentation
    inputs = layers.Input(shape=(224, 224, 3), name="input_image")
    augmented = layers.RandomFlip("horizontal", seed=SEED)(inputs)
    augmented = layers.RandomRotation(0.08, seed=SEED)(augmented)
    augmented = layers.RandomZoom(0.08, seed=SEED)(augmented)
    augmented = layers.RandomContrast(0.1, seed=SEED)(augmented)
    
    rescaled = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="preprocess_input")(augmented)

    # Re-use backbone from baseline
    backbone = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    
    # Unfreeze top 35 layers of MobileNetV2
    backbone.trainable = True
    for layer in backbone.layers[:-35]:
        layer.trainable = False
        
    # Keep BatchNormalization in inference mode to avoid corrupting statistics
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    x = backbone(rescaled, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="top_dropout")(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="predictions")(x)

    finetuned_model = keras.Model(inputs=inputs, outputs=outputs, name="LibaasLens_Finetuned")
    
    # Copy classification head weights from best baseline
    finetuned_model.get_layer("predictions").set_weights(baseline_model.get_layer("predictions").get_weights())

    finetuned_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    finetuned_path = os.path.join(MODELS_DIR, "finetuned.keras")
    csv_log_path = os.path.join(REPORTS_DIR, "finetuned_training.csv")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint(finetuned_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        keras.callbacks.CSVLogger(csv_log_path)
    ]

    history = finetuned_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    return finetuned_model, history


def log_experiments(baseline_model, finetuned_model, val_ds, base_hist, ft_hist):
    print("\n--- Logging Experiments ---")
    base_loss, base_acc = baseline_model.evaluate(val_ds, verbose=0)
    ft_loss, ft_acc = finetuned_model.evaluate(val_ds, verbose=0)

    print(f"Baseline Validation: Accuracy = {base_acc:.4f}, Loss = {base_loss:.4f}")
    print(f"Fine-Tuned Validation: Accuracy = {ft_acc:.4f}, Loss = {ft_loss:.4f}")

    exp_file = os.path.join(REPORTS_DIR, "experiments.csv")
    with open(exp_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Val_Accuracy", "Val_Loss", "Epochs_Trained", "Status"])
        writer.writerow(["Baseline (Frozen MobileNetV2)", f"{base_acc:.4f}", f"{base_loss:.4f}", len(base_hist.history["loss"]), "Trained"])
        writer.writerow(["Fine-Tuned (Top 35 Layers + Augmentation)", f"{ft_acc:.4f}", f"{ft_loss:.4f}", len(ft_hist.history["loss"]), "Trained"])

    # Select final model by validation accuracy
    if ft_acc >= base_acc:
        best_name = "Fine-Tuned"
        best_model = finetuned_model
    else:
        best_name = "Baseline"
        best_model = baseline_model

    final_path = os.path.join(MODELS_DIR, "final_model.keras")
    best_model.save(final_path)
    print(f"\nFinal model chosen based on validation score: {best_name} (saved to {final_path})")


def main():
    train_ds, val_ds, test_ds = load_datasets()
    
    # 1. Baseline
    base_model, backbone = build_baseline_model()
    base_hist = train_baseline(base_model, train_ds, val_ds, epochs=10)
    
    # Reload best baseline weights
    best_base = keras.models.load_model(os.path.join(MODELS_DIR, "baseline.keras"))
    
    # 2. Fine-Tuned
    ft_model, ft_hist = build_and_train_finetuned(train_ds, val_ds, best_base, epochs=12)
    
    # Reload best fine-tuned weights
    best_ft = keras.models.load_model(os.path.join(MODELS_DIR, "finetuned.keras"))
    
    # 3. Log and save final model
    log_experiments(best_base, best_ft, val_ds, base_hist, ft_hist)
    print("\n[SUCCESS] Model training complete!")


if __name__ == "__main__":
    main()
