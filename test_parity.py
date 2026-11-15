import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

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

TEST_DIR = "libaas/processed/test"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def main():
    print("=== Libaas Lens: 22-Image Parity Verification ===")
    
    # 1. Load models
    keras_model = keras.models.load_model("models/final_model.keras", compile=False)
    saved_model = tf.saved_model.load("models/saved_model_inference")
    infer_fn = saved_model.signatures["serving_default"]
    
    # 2. Select 2 test images per class (total 20 images)
    selected_images = []
    for cls in CLASSES:
        cls_dir = os.path.join(TEST_DIR, cls)
        files = sorted(os.listdir(cls_dir))
        if len(files) >= 2:
            selected_images.append((cls, os.path.join(cls_dir, files[0])))
            selected_images.append((cls, os.path.join(cls_dir, files[1])))
            
    print(f"Selected {len(selected_images)} test images across {len(CLASSES)} classes.")
    
    results = []
    max_discrepancy = 0.0
    all_match = True
    
    print("\nEvaluating images:")
    for idx, (gt_class, img_path) in enumerate(selected_images):
        # Load and resize image to 224x224 (raw [0, 255] RGB)
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_arr = np.array(img, dtype=np.float32)[np.newaxis, ...] # (1, 224, 224, 3)
        
        # Inference 1: Keras model
        keras_preds = keras_model(img_arr, training=False).numpy()[0]
        keras_top1_idx = int(np.argmax(keras_preds))
        keras_top1_class = CLASSES[keras_top1_idx]
        keras_top1_prob = float(keras_preds[keras_top1_idx])
        
        # Inference 2: SavedModel (direct source of TF.js graph)
        # Find input key for signature
        input_key = list(infer_fn.structured_input_signature[1].keys())[0]
        sm_out = infer_fn(**{input_key: tf.constant(img_arr)})
        output_key = list(sm_out.keys())[0]
        sm_preds = sm_out[output_key].numpy()[0]
        sm_top1_idx = int(np.argmax(sm_preds))
        sm_top1_class = CLASSES[sm_top1_idx]
        sm_top1_prob = float(sm_preds[sm_top1_idx])
        
        # Calculate max absolute difference across all 10 probabilities
        diff = np.max(np.abs(keras_preds - sm_preds))
        if diff > max_discrepancy:
            max_discrepancy = diff
            
        is_match = (keras_top1_class == sm_top1_class) and (diff <= 0.02)
        if not is_match:
            all_match = False
            
        print(f"[{idx+1:02d}/20] GT: {gt_class:<15} | Keras: {keras_top1_class} ({keras_top1_prob*100:.1f}%) | SM: {sm_top1_class} ({sm_top1_prob*100:.1f}%) | MaxDiff: {diff:.6f}")
        
        results.append({
            "index": idx + 1,
            "ground_truth": gt_class,
            "image_path": img_path.replace("\\", "/"),
            "keras_prediction": keras_top1_class,
            "keras_confidence": round(keras_top1_prob, 4),
            "keras_probabilities": [round(float(p), 4) for p in keras_preds],
            "saved_model_prediction": sm_top1_class,
            "saved_model_confidence": round(sm_top1_prob, 4),
            "saved_model_probabilities": [round(float(p), 4) for p in sm_preds],
            "max_probability_diff": round(float(diff), 6),
            "parity_passed": bool(diff <= 0.02 and keras_top1_class == sm_top1_class)
        })
        
    parity_report = {
        "num_test_images": len(selected_images),
        "classes_tested": len(CLASSES),
        "max_probability_discrepancy": round(float(max_discrepancy), 6),
        "discrepancy_threshold": 0.02, # 2% max allowed
        "parity_test_passed": bool(all_match and max_discrepancy <= 0.02),
        "results": results
    }
    
    out_path = os.path.join(REPORTS_DIR, "parity_report.json")
    with open(out_path, "w") as f:
        json.dump(parity_report, f, indent=2)
        
    print(f"\nParity summary: Max discrepancy = {max_discrepancy:.6f} (Threshold <= 0.02)")
    print(f"Overall parity status: {'PASSED' if parity_report['parity_test_passed'] else 'FAILED'}")
    print(f"Report saved to {out_path}")

if __name__ == "__main__":
    main()
