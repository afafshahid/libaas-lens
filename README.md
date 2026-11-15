# Libaas Lens (لباس لینز)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![TensorFlow.js 4.22](https://img.shields.io/badge/TensorFlow.js-4.22-FF6F00?logo=javascript&logoColor=white)](https://www.tensorflow.org/js)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status: Submittable](https://img.shields.io/badge/Status-Complete_v2.0-2E9D6A.svg)]()

> **Dual-Domain Clothing Type Classification Across Pakistani Ethnic and Western Apparel using Deep Transfer Learning on MobileNetV2.**  
> Built for **CS4045 Deep Learning for Perception**, FAST National University of Computer and Emerging Sciences (FAST-NUCES), Karachi Campus, Fall 2026.

- **GitHub Repository:** [https://github.com/afafshahid/libaas-lens](https://github.com/afafshahid/libaas-lens) (Branch: `master`)
- **Live Interactive Web App:** [https://afafshahid.github.io/libaas-lens/](https://afafshahid.github.io/libaas-lens/)
- **Comprehensive Academic Report:** [`report/REPORT.md`](file:///report/REPORT.md)
- **Slide Presentation Outline:** [`report/SLIDES_OUTLINE.md`](file:///report/SLIDES_OUTLINE.md)
- **Oral Defence Q&A:** [`report/DEFENSE_QA.md`](file:///report/DEFENSE_QA.md)

---

## 1. Project Overview & Motivation

Standard fashion computer vision models (trained on Western benchmarks like DeepFashion and Fashion-MNIST) consistently fail on traditional South Asian apparel. In Pakistan, individuals frequently alternate between traditional ethnic wear and contemporary Western fashion. 

**Libaas Lens** bridges this perception gap with a unified deep transfer-learning model that classifies **11 balanced apparel categories** (6 Pakistani ethnic wear, 5 Western wear) in a lightweight architecture. The final trained model is converted to a chunked **TensorFlow.js Graph Model** that executes **100% in the user's web browser**, guaranteeing complete privacy (no wardrobe photographs are ever uploaded to a server) and sub-50ms inference latency.

A critical design milestone in Libaas Lens was resolving the **open-set female apparel failure mode**: initial 10-class models lacked a general female suit category, forcing any real-world Pakistani woman's 2-pc or 3-pc lawn suit into *Lehenga*. By sourcing authentic Pakistani designer collections (Khaadi, J., Sapphire, Gul Ahmed, Alkaram) alongside IndoFashion suits, we established a dedicated **Shalwar Kameez** category, solving this failure mode with 99%+ test confidence on authentic Pakistani retail images.

---

## 2. 11-Class Taxonomy

Every image is annotated by its primary visible garment. Classes are sorted in strict alphabetical order matching the model's categorical softmax output:

| Index | Garment Class | Category Domain | Primary Source Dataset |
| :---: | :--- | :---: | :--- |
| **0** | `Dress` | Western | Real-World Clothing Dataset (Alexey Grigorev) |
| **1** | `Jacket - Blazer` | Western | Real-World Clothing Dataset (`blazer` + `outwear`) |
| **2** | `Jeans - Trousers` | Western | Real-World Clothing Dataset (`pants`) |
| **3** | `Kurta` | Ethnic | IndoFashion (`women_kurta`, `kurta_men`) |
| **4** | `Lehenga` | Ethnic | IndoFashion (`lehenga`) |
| **5** | `Waistcoat` (*Nehru Jacket*) | Ethnic | IndoFashion (`nehru_jackets`) |
| **6** | `Saree` | Ethnic | IndoFashion (`saree`) |
| **7** | `Shalwar Kameez` | Ethnic | Pakistani Fashion Retail (Khaadi, J., Sapphire, Gul Ahmed, Alkaram) + IndoFashion Suit Sets |
| **8** | `Sherwani` | Ethnic | IndoFashion (`sherwanis`) |
| **9** | `Shirt` | Western | Real-World Clothing Dataset (`shirt`) |
| **10** | `T-shirt` | Western | Real-World Clothing Dataset (`t-shirt`) |

*Scope & Nomenclature Note:* Index 5 is cataloged as `Nehru Jacket` in the IndoFashion benchmark and is represented throughout the UI and Pakistani commercial context as **Waistcoat** (*واسکت*).

---

## 3. Empirical Performance Summary

Evaluated on the fixed, held-out **429-image test set** (39 images per class, strictly unseen during training and validation):

| Metric | Baseline Model (Frozen) | Fine-Tuned Model (Top 35 Layers) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 81.12% (348/429) | **86.25%** (370/429) | **$+5.13\%$** |
| **Macro Precision** | 81.29% | **86.86%** | **$+5.57\%$** |
| **Macro Recall** | 81.12% | **86.25%** | **$+5.13\%$** |
| **Macro F1-Score** | 81.09% | **86.38%** | **$+5.29\%$** |
| **Western-Class Accuracy** | 83.59% | **87.18%** | **$+3.59\%$** |
| **Ethnic-Class Accuracy** | 79.06% | **85.47%** | **$+6.41\%$** |
| **TF.js Parity Discrepancy** | — | **$\le 0.000001$ (PASSED)** | — |

### Key Perception Findings
1. **Resolution Asymmetry Resolution:** Early models trained on blurry $60 \times 80$ Kaggle thumbnails developed inductive shortcut learning (associating sharp studio lighting and fine edges exclusively with ethnic categories). Upgrading Western classes to high-resolution real-world fashion photography eliminated this domain gap and raised Western accuracy to **87.18%**.
2. **Fine-Tuning Outperforms Linear Probing:** Fine-tuning the top 35 layers consistently beats the frozen baseline (+5.13% accuracy, +5.29% macro F1).
3. **Balanced Cross-Domain Discrimination:** Western wear (87.18%) and ethnic wear (85.47%) operate in near parity, with T-shirt (96.0% F1), Jeans (96.0% F1), Saree (92.1% F1), Waistcoat (91.9% F1), and Lehenga (91.1% F1) achieving exceptional precision and recall.
4. **Resolving the Female Suit Bias:** Incorporating authentic Pakistani retail suits into `Shalwar Kameez` completely eliminated the open-set misclassification of Pakistani female suits as bridal Lehengas.

---

## 4. Repository Directory Structure

```
libaas-lens/
├── .venv/                         # Local Python 3.12 virtual environment
├── data/
│   └── SOURCES.md                 # Dataset provenance, citations and licences
├── libaas/
│   ├── raw/                       # Cleaned raw images before split (11 classes)
│   └── processed/                 # Balanced 70/15/15 stratified dataset (2,860 images)
│       ├── train/                 # 2,002 images (182 / class)
│       ├── val/                   # 429 images (39 / class)
│       └── test/                  # 429 images (39 / class)
├── model/                         # Production TensorFlow.js Graph Model (11 classes)
│   ├── model.json                 # Model topology and weights manifest (144 KB)
│   ├── group1-shard1of3.bin       # Weights shard 1 (4.00 MB)
│   ├── group1-shard2of3.bin       # Weights shard 2 (4.00 MB)
│   └── group1-shard3of3.bin       # Weights shard 3 (0.50 MB)
├── models/                        # Saved Keras and TensorFlow models
│   ├── baseline.keras             # Frozen MobileNetV2 baseline
│   ├── finetuned.keras            # Fine-tuned MobileNetV2
│   ├── final_model.keras          # Selected final model checkpoint
│   └── saved_model_inference/     # Clean inference SavedModel without augmentation
├── report/                        # Academic documentation & presentation materials
│   ├── REPORT.md                  # Complete university research report
│   ├── SLIDES_OUTLINE.md          # 12-slide presentation outline with speaker notes
│   └── DEFENSE_QA.md              # Technical oral defence questions and answers
├── reports/                       # Quantitative evaluation outputs and plots
│   ├── metrics.json               # Full test metrics and confusion matrix
│   ├── confusion_matrix_finetuned.png # High-res confusion matrix heatmap
│   ├── training_curves.png        # Epoch accuracy and loss curves
│   └── parity_report.json         # 22-image parity verification log
├── samples/                       # High-resolution uncompressed studio samples (11 images)
│   ├── shalwarkameez.jpg          # Authentic Pakistani 3-pc suit (800x1200)
│   ├── waistcoat.jpg              # High-res Waistcoat
│   ├── kurta.jpg                  # High-res Kurta
│   ├── sherwani.jpg               # High-res Sherwani
│   ├── lehenga.jpg                # High-res Lehenga
│   ├── saree.jpg                  # High-res Saree
│   ├── jeans.jpg                  # High-res Denim Jeans
│   ├── dress.jpg                  # High-res Western Dress
│   ├── tshirt.jpg                 # High-res T-shirt
│   ├── shirt.jpg                  # High-res Formal Shirt
│   └── jacket.jpg                 # High-res Suit Blazer
├── index.html                     # Obsidian luxury editorial web app (In-Browser AI Studio)
├── prepare_data.py                # Dataset extraction, deduplication & stratified split
├── train.py                       # Two-stage MobileNetV2 training pipeline
├── evaluate.py                    # Multi-metric test evaluation & plot generation
├── export_tfjs.py                 # Clean SavedModel & TensorFlow.js graph conversion
└── test_parity.py                 # 22-image mathematical parity verification
```

---

## 5. Quick Start & Execution

### 5.1 Local Browser Inference (Zero Setup)
Simply serve the workspace using any static web server and navigate to `http://localhost:8000`:

```bash
# Using Python's built-in HTTP server:
python -m http.server 8000
```

Open `http://localhost:8000` in Google Chrome, Microsoft Edge, or Mozilla Firefox. The web app will automatically initialize TensorFlow.js via WebGL and run client-side inference.

### 5.2 Python Pipeline Reproduction
To retrain and re-evaluate the full pipeline from source:

```bash
# 1. Activate Python virtual environment
.\.venv\Scripts\activate

# 2. Extract, deduplicate, and partition the 11-class balanced dataset
python prepare_data.py

# 3. Train Baseline and Fine-Tuned MobileNetV2 models
python train.py

# 4. Evaluate on held-out test split (generates metrics.json and confusion matrices)
python evaluate.py

# 5. Export clean SavedModel and convert to TensorFlow.js graph model
python export_tfjs.py

# 6. Verify mathematical parity across all 11 classes
python test_parity.py
```

---

## 6. Academic Team & Attribution

**Institution:** FAST National University of Computer and Emerging Sciences (FAST-NUCES), Karachi Campus  
**Department:** Department of Computer Science  
**Course:** CS4045 Deep Learning for Perception · Fall 2026

- **Afaf Shahid** (Roll No: 23K-0678, Group Coordinator) — `k230678@nu.edu.pk`
- **Ahmad Noorani** (Roll No: 23K-0547) — `k230547@nu.edu.pk`
