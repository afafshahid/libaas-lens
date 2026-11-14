"""
prepare_data.py - Data preparation pipeline for Libaas Lens.

Classes (10 total, 5 Ethnic + 5 Western):
  Dress, Jacket - Blazer, Jeans - Trousers, Kurta, Lehenga,
  Nehru Jacket, Saree, Sherwani, Shirt, T-shirt

Functions:
  - pull_fashion_product: Extracts 5 Western classes from Kaggle Fashion Product Images
  - pull_indofashion: Extracts 5 Ethnic classes from IndoFashion (with brand-balance & title filtering)
  - clean_raw: Removes corrupt images, exact duplicates (SHA256), and near duplicates (pHash DCT)
  - balance_and_split: Stratified 70/15/15 split, capped to ensure balance ratio < 1.5x
  - verify: Verifies split counts and dataset integrity
  - generate_reports: Writes data/SOURCES.md and reports/data_summary.csv
"""

# %% [markdown]
# ## 1. Imports and Configuration

# %%
import os
import sys
import io
import csv
import json
import random
import hashlib
import shutil
import zipfile
from collections import defaultdict
from PIL import Image
import imagehash

ROOT = "libaas"
RAW_DIR = os.path.join(ROOT, "raw")
PROCESSED_DIR = os.path.join(ROOT, "processed")
DOWNLOADS_DIR = r"C:\Users\Afaf Shahid\Downloads"

# 10 target classes in alphabetical order
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

SEED = 42
random.seed(SEED)

# %% [markdown]
# ## 2. Data Extraction Functions

# %%
def pull_fashion_product(zip_path, target_raw_dir, max_per_class=350):
    """
    Extracts Western wear classes from Kaggle Fashion Product Images archive.zip
    """
    print(f"\n[1/4] Pulling Western wear from {zip_path}...")
    os.makedirs(target_raw_dir, exist_ok=True)
    
    mapping = {
        "Dresses": "Dress",
        "Jackets": "Jacket - Blazer",
        "Blazers": "Jacket - Blazer",
        "Jeans": "Jeans - Trousers",
        "Trousers": "Jeans - Trousers",
        "Shirts": "Shirt",
        "Tshirts": "T-shirt",
    }
    
    counts = defaultdict(int)
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist_set = set(zf.namelist())
        with zf.open("styles.csv") as f:
            reader = csv.reader(io.TextIOWrapper(f, "utf-8", errors="replace"))
            header = next(reader)
            rows = list(reader)
            random.shuffle(rows)
            
            for row in rows:
                if len(row) <= 4:
                    continue
                img_id = row[0].strip()
                art_type = row[4].strip()
                
                if art_type in mapping:
                    target_cls = mapping[art_type]
                    if counts[target_cls] >= max_per_class:
                        continue
                    
                    img_name_in_zip = f"images/{img_id}.jpg"
                    if img_name_in_zip in namelist_set:
                        out_dir = os.path.join(target_raw_dir, target_cls)
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = os.path.join(out_dir, f"fprod_{img_id}.jpg")
                        
                        if not os.path.exists(out_path):
                            with zf.open(img_name_in_zip) as img_f:
                                img_bytes = img_f.read()
                                try:
                                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                    img.save(out_path, "JPEG", quality=95)
                                    counts[target_cls] += 1
                                except Exception as e:
                                    continue
                        else:
                            counts[target_cls] += 1

    print("Western classes extracted:")
    for cls in ["Dress", "Jacket - Blazer", "Jeans - Trousers", "Shirt", "T-shirt"]:
        print(f"  {cls}: {counts[cls]} images")


# %%
def pull_indofashion(zip_path, target_raw_dir, max_per_class=350, max_per_brand=12):
    """
    Extracts Ethnic wear classes from IndoFashion indofashion_dataset.zip
    Applies title exclusions and brand-capping for diversity.
    """
    print(f"\n[2/4] Pulling Ethnic wear from {zip_path}...")
    os.makedirs(target_raw_dir, exist_ok=True)
    
    ethnic_classes = ["Kurta", "Lehenga", "Nehru Jacket", "Saree", "Sherwani"]
    counts = defaultdict(int)
    brand_counts = defaultdict(lambda: defaultdict(int))
    
    kurta_exclude = ["set", "suit", "palazzo", "salwar", "sharara", "bottom", "pant", "dupatta", "combo"]
    sherwani_exclude = ["turban", "pagri", "stole", "sash", "brooch", "mala", "shawl", "dupatta"]
    
    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist_set = set(zf.namelist())
        
        # Read items from train, val, test metadata
        all_items = []
        for split in ["train", "val", "test"]:
            json_file = f"{split}_data.json"
            if json_file in namelist_set:
                with zf.open(json_file) as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            all_items.append(item)
                        except Exception:
                            continue
        
        random.shuffle(all_items)
        print(f"Total IndoFashion metadata entries loaded: {len(all_items)}")
        
        for item in all_items:
            cls_lbl = item.get("class_label", "").strip()
            title = item.get("product_title", "").lower()
            brand = item.get("brand", "unknown").strip()
            img_path = item.get("image_path", "").strip()
            
            target_cls = None
            if cls_lbl in ["women_kurta", "kurta_men"]:
                if any(w in title for w in kurta_exclude):
                    continue
                target_cls = "Kurta"
            elif cls_lbl == "lehenga":
                target_cls = "Lehenga"
            elif cls_lbl == "saree":
                target_cls = "Saree"
            elif cls_lbl == "sherwanis":
                if any(w in title for w in sherwani_exclude):
                    continue
                target_cls = "Sherwani"
            elif cls_lbl == "nehru_jackets":
                target_cls = "Nehru Jacket"
                
            if not target_cls or counts[target_cls] >= max_per_class:
                continue
                
            if brand_counts[target_cls][brand] >= max_per_brand:
                continue
                
            if img_path in namelist_set:
                out_dir = os.path.join(target_raw_dir, target_cls)
                os.makedirs(out_dir, exist_ok=True)
                fn = os.path.basename(img_path)
                out_path = os.path.join(out_dir, f"indo_{fn}")
                
                if not os.path.exists(out_path):
                    try:
                        with zf.open(img_path) as img_f:
                            img = Image.open(io.BytesIO(img_f.read())).convert("RGB")
                            img.save(out_path, "JPEG", quality=95)
                            counts[target_cls] += 1
                            brand_counts[target_cls][brand] += 1
                    except Exception:
                        continue
                else:
                    counts[target_cls] += 1
                    brand_counts[target_cls][brand] += 1

    print("Ethnic classes extracted:")
    for cls in ethnic_classes:
        print(f"  {cls}: {counts[cls]} images")


# %% [markdown]
# ## 3. Cleaning & Perceptual Deduplication

# %%
def clean_raw(raw_dir, hash_threshold=4):
    """
    Removes corrupt, unreadable images, exact duplicates (SHA256),
    and perceptual near-duplicates (DCT pHash).
    """
    print(f"\n[3/4] Cleaning images in {raw_dir}...")
    exact_hashes = set()
    total_removed = 0
    total_valid = 0
    
    for cls in sorted(os.listdir(raw_dir)):
        cls_dir = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
            
        phashes = []  # list of (phash, filepath)
        files = sorted(os.listdir(cls_dir))
        cls_removed = 0
        
        for fn in files:
            fp = os.path.join(cls_dir, fn)
            # 1. Readability & size check
            try:
                with Image.open(fp) as im:
                    im.verify()
                with Image.open(fp) as im:
                    w, h = im.size
                    if w < 40 or h < 40:
                        os.remove(fp)
                        cls_removed += 1
                        continue
                    im_rgb = im.convert("RGB")
                    h_exact = hashlib.sha256(im_rgb.tobytes()).hexdigest()
                    h_perc = imagehash.phash(im_rgb)
            except Exception:
                if os.path.exists(fp):
                    os.remove(fp)
                cls_removed += 1
                continue
                
            # 2. Exact duplicate check
            if h_exact in exact_hashes:
                os.remove(fp)
                cls_removed += 1
                continue
            exact_hashes.add(h_exact)
            
            # 3. Near duplicate check (Hamming distance on pHash)
            is_near_dup = False
            for prev_h, prev_fp in phashes:
                if (h_perc - prev_h) <= hash_threshold:
                    is_near_dup = True
                    break
                    
            if is_near_dup:
                os.remove(fp)
                cls_removed += 1
            else:
                phashes.append((h_perc, fp))
                total_valid += 1
                
        total_removed += cls_removed
        rem_count = len(os.listdir(cls_dir))
        print(f"  {cls:<18} -> {rem_count:>4} clean images ({cls_removed} duplicates/invalid removed)")
        
    print(f"Total valid: {total_valid}, Total removed: {total_removed}")


# %% [markdown]
# ## 4. Balancing & Stratified 70/15/15 Split

# %%
def balance_and_split(raw_dir, processed_dir, cap=260, seed=42, ratio=(0.70, 0.15, 0.15)):
    """
    Caps classes to ensure ratio < 1.5x between any two classes.
    Creates stratified 70% train / 15% val / 15% test splits with seed=42.
    """
    print(f"\n[4/4] Balancing and splitting dataset into {processed_dir}...")
    random.seed(seed)
    
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(processed_dir, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
            
    summary_data = []
    
    classes_present = sorted([d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))])
    
    # Check minimum count across all classes
    counts = {c: len(os.listdir(os.path.join(raw_dir, c))) for c in classes_present}
    min_count = min(counts.values())
    effective_cap = min(cap, min_count) if min_count < cap else cap
    print(f"Raw counts: {counts}")
    print(f"Smallest class has {min_count} images. Target cap per class: {effective_cap}")
    
    for cls in classes_present:
        cls_dir = os.path.join(raw_dir, cls)
        all_imgs = sorted(os.listdir(cls_dir))
        random.shuffle(all_imgs)
        
        # Take up to effective_cap
        selected = all_imgs[:effective_cap]
        n = len(selected)
        
        n_train = int(n * ratio[0])
        n_val = int(n * ratio[1])
        n_test = n - n_train - n_val
        
        splits = {
            "train": selected[:n_train],
            "val": selected[n_train:n_train + n_val],
            "test": selected[n_train + n_val:]
        }
        
        for split, flist in splits.items():
            out_cls_dir = os.path.join(processed_dir, split, cls)
            os.makedirs(out_cls_dir, exist_ok=True)
            for fn in flist:
                src = os.path.join(cls_dir, fn)
                dst = os.path.join(out_cls_dir, fn)
                shutil.copy2(src, dst)
                
        summary_data.append({
            "class": cls,
            "type": "Ethnic" if cls in ["Kurta", "Lehenga", "Nehru Jacket", "Saree", "Sherwani"] else "Western",
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "total": n
        })
        
    return summary_data


# %% [markdown]
# ## 5. Verification & Summary Generation

# %%
def verify(processed_dir):
    """
    Verifies that all 10 classes exist in train, val, test with matching order.
    """
    print("\n--- Verifying Split Dataset ---")
    for split in ["train", "val", "test"]:
        s_dir = os.path.join(processed_dir, split)
        classes = sorted(os.listdir(s_dir))
        assert classes == CLASSES, f"Classes mismatch in {split}! Found {classes}, expected {CLASSES}"
        total = sum(len(os.listdir(os.path.join(s_dir, c))) for c in classes)
        print(f"Split {split:<5}: {len(classes)} classes, {total} total images")
    print("Verification PASSED: All 10 classes verified in alphabetical order.")


# %%
def generate_reports(summary_data, reports_dir="reports"):
    """
    Writes reports/data_summary.csv and data/SOURCES.md
    """
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # 1. Write CSV summary
    csv_path = os.path.join(reports_dir, "data_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "type", "train", "val", "test", "total"])
        writer.writeheader()
        writer.writerows(summary_data)
    print(f"\nWrote data summary to {csv_path}")
    
    # 2. Write data/SOURCES.md
    sources_md = """# Dataset Sources & Credits

This project uses public datasets for Pakistani ethnic wear and Western wear classification.

## 1. IndoFashion (Ethnic Wear)
- **Classes used:** Kurta, Lehenga, Nehru Jacket, Saree, Sherwani
- **Homepage:** https://indofashion.github.io
- **Documentation:** https://indofashion.readthedocs.io/en/latest/tutorials/info.html
- **Citation:**
  > Pranjal Singh Rajput and Aneja, "IndoFashion: Apparel Classification for Indian Ethnic Clothes", *CVPR Workshops (CVFAD)*, 2021.
  > [Paper Link](https://openaccess.thecvf.com/content/CVPR2021W/CVFAD/html/Rajput_IndoFashion_Apparel_Classification_for_Indian_Ethnic_Clothes_CVPRW_2021_paper.html) | [arXiv:2104.02830](https://arxiv.org/abs/2104.02830)
- **Licence:** Research / Educational use.

## 2. Fashion Product Images Small (Western Wear)
- **Classes used:** Dress, Jacket - Blazer, Jeans - Trousers, Shirt, T-shirt
- **Source:** Kaggle by Param Aggarwal
- **Link:** https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
- **Licence:** CC0 / Public Domain.

## 3. Reference Frameworks & Models
- **MobileNetV2:** Sandler et al., *"MobileNetV2: Inverted Residuals and Linear Bottlenecks"*, CVPR 2018, https://arxiv.org/abs/1801.04381
- **TensorFlow & Keras:** https://www.tensorflow.org
- **TensorFlow.js:** https://www.tensorflow.org/js

## Preprocessing & Curation Rules
1. **Deduplication:** Perceptual hashing (DCT pHash) with Hamming distance threshold of 4, plus SHA-256 exact matching.
2. **Brand Diversity:** Maximum 12 images per brand from IndoFashion to prevent brand-specific shortcut learning.
3. **Keyword Filtering:** Excluded combination sets, suits, and accessories to adhere strictly to single-garment classification.
4. **Stratified Split:** 70% Train, 15% Validation, 15% Test with fixed seed (42).
"""
    with open("data/SOURCES.md", "w", encoding="utf-8") as f:
        f.write(sources_md)
    print("Wrote data sources and citations to data/SOURCES.md")


# %%
def main():
    fprod_zip = os.path.join(DOWNLOADS_DIR, "archive.zip")
    indo_zip = os.path.join(DOWNLOADS_DIR, "indofashion_dataset.zip")
    
    assert os.path.exists(fprod_zip), f"Missing {fprod_zip}"
    assert os.path.exists(indo_zip), f"Missing {indo_zip}"
    
    # 1. Pull data
    pull_fashion_product(fprod_zip, RAW_DIR, max_per_class=350)
    pull_indofashion(indo_zip, RAW_DIR, max_per_class=350, max_per_brand=12)
    
    # 2. Clean & deduplicate
    clean_raw(RAW_DIR, hash_threshold=4)
    
    # 3. Balance and split (cap 260 keeps classes evenly matched at 1.0x to 1.1x ratio)
    summary = balance_and_split(RAW_DIR, PROCESSED_DIR, cap=260, seed=42)
    
    # 4. Verify & generate reports
    verify(PROCESSED_DIR)
    generate_reports(summary)
    print("\n[SUCCESS] Data pipeline execution complete!")


if __name__ == "__main__":
    main()
