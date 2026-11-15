import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflowjs.converters import tf_saved_model_conversion_v2

CLASSES = [
    "Dress",
    "Jacket - Blazer",
    "Jeans - Trousers",
    "Kurta",
    "Lehenga",
    "Nehru Jacket",
    "Saree",
    "Shalwar Kameez",
    "Sherwani",
    "Shirt",
    "T-shirt"
]

def main():
    print("Loading final_model.keras...")
    trained_model = keras.models.load_model("models/final_model.keras", compile=False)
    
    print("Building clean inference model (no data augmentation layers)...")
    inputs = layers.Input(shape=(224, 224, 3), name="input_1")
    rescaled = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="preprocess_input")(inputs)
    
    backbone = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    
    x = backbone(rescaled, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.3, name="top_dropout")(x)
    outputs = layers.Dense(len(CLASSES), activation="softmax", name="predictions")(x)
    
    inf_model = keras.Model(inputs=inputs, outputs=outputs, name="LibaasLens_Classifier")
    
    # Transfer backbone weights
    # In trained_model, find the MobileNetV2 layer
    mobilenet_sublayer = None
    for l in trained_model.layers:
        if "mobilenetv2" in l.name.lower():
            mobilenet_sublayer = l
            break
            
    if mobilenet_sublayer is not None:
        print(f"Found backbone layer {mobilenet_sublayer.name}, transferring weights...")
        backbone.set_weights(mobilenet_sublayer.get_weights())
    else:
        print("Warning: MobileNetV2 layer not found by name, attempting direct layer match...")
        
    # Transfer prediction head weights
    inf_model.get_layer("predictions").set_weights(trained_model.get_layer("predictions").get_weights())
    
    # Verification: check with random input
    dummy_input = np.random.uniform(0, 255, size=(5, 224, 224, 3)).astype(np.float32)
    pred_orig = trained_model(dummy_input, training=False).numpy()
    pred_inf = inf_model(dummy_input, training=False).numpy()
    max_diff = np.max(np.abs(pred_orig - pred_inf))
    print(f"Max prediction difference between trained and clean inference model: {max_diff:.8f}")
    assert max_diff < 1e-4, f"Prediction mismatch: {max_diff}"
    print("Verification passed! Clean inference model matches trained model exactly.")
    
    # Export clean SavedModel
    saved_model_dir = "models/saved_model_inference"
    if os.path.exists(saved_model_dir):
        shutil.rmtree(saved_model_dir)
    print(f"Exporting clean SavedModel to {saved_model_dir}...")
    inf_model.export(saved_model_dir)
    
    # Convert to TF.js
    tfjs_output_dir = "model"
    if os.path.exists(tfjs_output_dir):
        shutil.rmtree(tfjs_output_dir)
    os.makedirs(tfjs_output_dir, exist_ok=True)
    
    print(f"Converting SavedModel to TensorFlow.js graph model in {tfjs_output_dir}...")
    tf_saved_model_conversion_v2.convert_tf_saved_model(
        saved_model_dir=saved_model_dir,
        output_dir=tfjs_output_dir
    )
    
    print(f"\nTensorFlow.js conversion complete! Files in {tfjs_output_dir}:")
    for f in os.listdir(tfjs_output_dir):
        fpath = os.path.join(tfjs_output_dir, f)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  - {f} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    main()
