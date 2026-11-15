import os
import io
import shutil
import urllib.request
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

RAW_DIR = "libaas/raw"
CSV_URL = "https://raw.githubusercontent.com/alexeygrigorev/clothing-dataset/master/images.csv"
RAW_IMG_URL = "https://raw.githubusercontent.com/alexeygrigorev/clothing-dataset/master/images/{}.jpg"

TARGET_CLASSES = {
    "Dress": ["Dress"],
    "Jacket - Blazer": ["Blazer", "Outwear"],
    "Jeans - Trousers": ["Pants"],
    "Shirt": ["Shirt"],
    "T-shirt": ["T-Shirt"]
}

MAX_PER_CLASS = 260

def download_single(item):
    cls_name, img_id = item
    url = RAW_IMG_URL.format(img_id)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()
        img = Image.open(io.BytesIO(content))
        img = img.convert("RGB")
        if img.width < 100 or img.height < 100:
            return None
        return cls_name, img_id, img
    except Exception as e:
        return None

def main():
    print("=== Downloading High-Resolution Western Apparel Dataset ===")
    print(f"Reading CSV from {CSV_URL}...")
    csv_bytes = urllib.request.urlopen(CSV_URL).read()
    df = pd.read_csv(io.BytesIO(csv_bytes))
    print(f"Loaded {len(df)} metadata rows.")
    
    # Check tree to ensure images exist
    tree_url = "https://api.github.com/repos/alexeygrigorev/clothing-dataset/git/trees/master?recursive=1"
    tree_req = urllib.request.urlopen(tree_url)
    import json
    tree_data = json.loads(tree_req.read().decode("utf-8"))
    valid_ids = set(t["path"].replace("images/", "").replace(".jpg", "") for t in tree_data["tree"] if t["path"].startswith("images/"))
    df = df[df["image"].isin(valid_ids)]
    print(f"Verified {len(df)} images available on remote git.")
    
    download_tasks = []
    for cls_name, labels in TARGET_CLASSES.items():
        sub_df = df[df["label"].isin(labels)]
        selected_ids = sub_df["image"].tolist()[:MAX_PER_CLASS + 30] # buffer for corrupts
        for img_id in selected_ids:
            download_tasks.append((cls_name, img_id))
            
    print(f"Prepared {len(download_tasks)} download tasks across 5 Western categories.")
    
    # Clean previous raw directories for Western classes
    for cls_name in TARGET_CLASSES.keys():
        folder = os.path.join(RAW_DIR, cls_name)
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        
    counts = {cls: 0 for cls in TARGET_CLASSES.keys()}
    
    print("Downloading in parallel with 20 workers...")
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(download_single, item): item for item in download_tasks}
        for f in as_completed(futures):
            res = f.result()
            if res is not None:
                cls_name, img_id, img = res
                if counts[cls_name] < MAX_PER_CLASS:
                    out_path = os.path.join(RAW_DIR, cls_name, f"hd_{img_id}.jpg")
                    img.save(out_path, quality=92)
                    counts[cls_name] += 1
                    if counts[cls_name] % 50 == 0 or counts[cls_name] == MAX_PER_CLASS:
                        print(f"  [{cls_name}] saved {counts[cls_name]}/{MAX_PER_CLASS}")
                        
    print("\nDownload Summary:")
    for cls_name, count in counts.items():
        print(f"  {cls_name:<20}: {count} images (target: {MAX_PER_CLASS})")
        assert count == MAX_PER_CLASS, f"Insufficient images for {cls_name}: {count} < {MAX_PER_CLASS}"

if __name__ == "__main__":
    main()
