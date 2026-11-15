"""
evaluate.py - Full evaluation script for Libaas Lens.

Evaluates baseline and fine-tuned models on the fixed public test set:
  - Accuracy, macro precision, macro recall, macro F1
  - Per-class precision, recall, F1
  - Ethnic accuracy vs Western accuracy
  - Row-normalized confusion matrices (PNG)
  - Training curves (PNG)
  - Visual grids of correct and incorrect predictions (PNG)
  - Complete metrics saved to reports/metrics.json
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

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
    "T-shirt",
]

ETHNIC_CLASSES = {"Kurta", "Lehenga", "Nehru Jacket", "Saree", "Shalwar Kameez", "Sherwani"}
WESTERN_CLASSES = {"Dress", "Jacket - Blazer", "Jeans - Trousers", "Shirt", "T-shirt"}

PROCESSED_DIR = "libaas/processed"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_test_data():
    test_dir = os.path.join(PROCESSED_DIR, "test")
    test_ds = keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(224, 224),
        batch_size=32,
        shuffle=False,
        label_mode="int"
    )
    assert test_ds.class_names == CLASSES, f"Mismatch: {test_ds.class_names} != {CLASSES}"

    # Extract all images and true labels
    images_list, labels_list = [], []
    for imgs, lbls in test_ds:
        images_list.append(imgs.numpy())
        labels_list.append(lbls.numpy())

    X_test = np.concatenate(images_list, axis=0)
    y_test = np.concatenate(labels_list, axis=0)
    return X_test, y_test


def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n--- Evaluating {model_name} on Public Test Set ({len(y_test)} images) ---")
    y_probs = model.predict(X_test, batch_size=32, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    acc = float(accuracy_score(y_test, y_pred))
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)

    # Ethnic vs Western breakdown
    ethnic_indices = [i for i, c in enumerate(CLASSES) if c in ETHNIC_CLASSES]
    western_indices = [i for i, c in enumerate(CLASSES) if c in WESTERN_CLASSES]

    ethnic_mask = np.isin(y_test, ethnic_indices)
    western_mask = np.isin(y_test, western_indices)

    ethnic_acc = float(accuracy_score(y_test[ethnic_mask], y_pred[ethnic_mask])) if np.any(ethnic_mask) else 0.0
    western_acc = float(accuracy_score(y_test[western_mask], y_pred[western_mask])) if np.any(western_mask) else 0.0

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CLASSES))))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    per_class_metrics = {}
    for i, c in enumerate(CLASSES):
        per_class_metrics[c] = {
            "precision": round(float(p_per[i]), 4),
            "recall": round(float(r_per[i]), 4),
            "f1": round(float(f1_per[i]), 4),
            "support": int(sup_per[i]),
            "category": "Ethnic" if c in ETHNIC_CLASSES else "Western"
        }

    results = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "macro_precision": round(float(p_macro), 4),
        "macro_recall": round(float(r_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "ethnic_accuracy": round(ethnic_acc, 4),
        "western_accuracy": round(western_acc, 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": np.round(cm_norm, 4).tolist()
    }

    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro F1-Score:    {f1_macro:.4f}")
    print(f"Ethnic Accuracy:   {ethnic_acc * 100:.2f}%")
    print(f"Western Accuracy:  {western_acc * 100:.2f}%")

    return results, y_pred, y_probs, cm_norm


def plot_confusion_matrix(cm_norm, model_name, filename):
    plt.figure(figsize=(9, 8))
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name} (Row-Normalized)", fontsize=13, pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45, ha="right", fontsize=10)
    plt.yticks(tick_marks, CLASSES, fontsize=10)

    # Numbers inside cells
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            color = "white" if val > thresh else "black"
            text = f"{val:.2f}" if val > 0.005 else "0"
            plt.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.ylabel("True Class", fontsize=11)
    plt.xlabel("Predicted Class", fontsize=11)
    out_path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix plot: {out_path}")


def plot_training_curves():
    base_csv = os.path.join(REPORTS_DIR, "baseline_training.csv")
    ft_csv = os.path.join(REPORTS_DIR, "finetuned_training.csv")

    if not os.path.exists(base_csv) or not os.path.exists(ft_csv):
        print("Training logs not found, skipping curve plot.")
        return

    import pandas as pd
    df_base = pd.read_csv(base_csv)
    df_ft = pd.read_csv(ft_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy curves
    axes[0].plot(df_base["epoch"] + 1, df_base["accuracy"], label="Baseline Train", linestyle="--", color="#4A90E2")
    axes[0].plot(df_base["epoch"] + 1, df_base["val_accuracy"], label="Baseline Val", color="#1B365D", linewidth=2)
    axes[0].plot(df_ft["epoch"] + 1, df_ft["accuracy"], label="Fine-Tuned Train", linestyle="--", color="#E67E22")
    axes[0].plot(df_ft["epoch"] + 1, df_ft["val_accuracy"], label="Fine-Tuned Val", color="#A04000", linewidth=2)
    axes[0].set_title("Training & Validation Accuracy", fontsize=12)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss curves
    axes[1].plot(df_base["epoch"] + 1, df_base["loss"], label="Baseline Train", linestyle="--", color="#4A90E2")
    axes[1].plot(df_base["epoch"] + 1, df_base["val_loss"], label="Baseline Val", color="#1B365D", linewidth=2)
    axes[1].plot(df_ft["epoch"] + 1, df_ft["loss"], label="Fine-Tuned Train", linestyle="--", color="#E67E22")
    axes[1].plot(df_ft["epoch"] + 1, df_ft["val_loss"], label="Fine-Tuned Val", color="#A04000", linewidth=2)
    axes[1].set_title("Training & Validation Loss", fontsize=12)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    out_path = os.path.join(REPORTS_DIR, "training_curves.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves: {out_path}")


def plot_error_analysis(X_test, y_test, y_pred, y_probs):
    correct_idx = np.where(y_test == y_pred)[0]
    wrong_idx = np.where(y_test != y_pred)[0]

    np.random.seed(42)
    sample_correct = np.random.choice(correct_idx, min(6, len(correct_idx)), replace=False)
    sample_wrong = np.random.choice(wrong_idx, min(6, len(wrong_idx)), replace=False) if len(wrong_idx) > 0 else []

    # 1. Correct Predictions Grid
    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for ax, idx in zip(axes.flat, sample_correct):
        img = X_test[idx].astype("uint8")
        true_c = CLASSES[y_test[idx]]
        pred_c = CLASSES[y_pred[idx]]
        conf = y_probs[idx][y_pred[idx]] * 100
        ax.imshow(img)
        ax.set_title(f"True & Pred: {pred_c}\nConf: {conf:.1f}%", fontsize=10, color="green")
        ax.axis("off")
    plt.suptitle("Sample Correct Predictions (Public Test Set)", fontsize=13, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "sample_correct.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Misclassified Examples Grid
    if len(sample_wrong) > 0:
        n_show = len(sample_wrong)
        fig, axes = plt.subplots(1, n_show, figsize=(3.5 * n_show, 4))
        if n_show == 1:
            axes = [axes]
        for ax, idx in zip(axes, sample_wrong):
            img = X_test[idx].astype("uint8")
            true_c = CLASSES[y_test[idx]]
            pred_c = CLASSES[y_pred[idx]]
            conf = y_probs[idx][y_pred[idx]] * 100
            ax.imshow(img)
            ax.set_title(f"True: {true_c}\nPred: {pred_c} ({conf:.1f}%)", fontsize=9, color="crimson")
            ax.axis("off")
        plt.suptitle("Common Misclassifications (Public Test Set)", fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(REPORTS_DIR, "sample_misclassified.png"), dpi=200, bbox_inches="tight")
        plt.close()
        print("Saved sample correct and misclassified visual grids.")


def main():
    X_test, y_test = load_test_data()

    baseline_path = os.path.join(MODELS_DIR, "baseline.keras")
    finetuned_path = os.path.join(MODELS_DIR, "finetuned.keras")

    assert os.path.exists(baseline_path), f"Missing {baseline_path}"
    assert os.path.exists(finetuned_path), f"Missing {finetuned_path}"

    base_model = keras.models.load_model(baseline_path)
    ft_model = keras.models.load_model(finetuned_path)

    # Evaluate both
    base_res, base_pred, base_probs, base_cm_norm = evaluate_model(base_model, X_test, y_test, "Baseline")
    ft_res, ft_pred, ft_probs, ft_cm_norm = evaluate_model(ft_model, X_test, y_test, "Fine-Tuned")

    # Plots
    plot_confusion_matrix(base_cm_norm, "Baseline MobileNetV2", "confusion_matrix_baseline.png")
    plot_confusion_matrix(ft_cm_norm, "Fine-Tuned MobileNetV2", "confusion_matrix_finetuned.png")
    plot_training_curves()
    plot_error_analysis(X_test, y_test, ft_pred, ft_probs)

    # Save comprehensive metrics.json
    final_metrics = {
        "classes": CLASSES,
        "ethnic_classes": list(sorted(ETHNIC_CLASSES)),
        "western_classes": list(sorted(WESTERN_CLASSES)),
        "baseline": base_res,
        "finetuned": ft_res,
        "summary": {
            "test_images": len(y_test),
            "baseline_accuracy": base_res["accuracy"],
            "finetuned_accuracy": ft_res["accuracy"],
            "accuracy_gain": round(ft_res["accuracy"] - base_res["accuracy"], 4),
            "baseline_f1": base_res["macro_f1"],
            "finetuned_f1": ft_res["macro_f1"],
            "ethnic_accuracy": ft_res["ethnic_accuracy"],
            "western_accuracy": ft_res["western_accuracy"],
        }
    }

    metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"\nSaved comprehensive metrics to {metrics_path}")

    # Also copy to root for the website to read
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print("Saved metrics.json to workspace root for static webapp.")

    print("\n[SUCCESS] Evaluation complete!")


if __name__ == "__main__":
    main()
