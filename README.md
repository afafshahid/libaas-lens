# Libaas Lens (لباس لینز)

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow 2.21](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![TensorFlow.js 4.22](https://img.shields.io/badge/TensorFlow.js-4.22-FF6F00?logo=javascript&logoColor=white)](https://www.tensorflow.org/js)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status: Submittable](https://img.shields.io/badge/Status-Complete_v1.0-2E9D6A.svg)]()

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

**Libaas Lens** bridges this perception gap with a unified deep transfer-learning model that classifies **10 apparel categories** (5 Pakistani ethnic wear, 5 Western wear) in a lightweight architecture. The final trained model is converted to a chunked **TensorFlow.js Graph Model** that executes **100% in the user's web browser**, guaranteeing complete privacy (no wardrobe photographs are ever uploaded to a server) and sub-50ms inference latency.

---

## 2. 10-Class Taxonomy

Every image is annotated by its primary visible garment. Classes are sorted in strict alphabetical order matching the model's categorical softmax output:

| Index | Garment Class | Category Domain | Primary Source Dataset |
| :---: | :--- | :---: | :--- |
| **0** | `Dress` | Western | Fashion Product Images Small (Kaggle) |
| **1** | `Jacket - Blazer` | Western | Fashion Product Images Small (Kaggle) |
| **2** | `Jeans - Trousers` | Western | Fashion Product Images Small (Kaggle) |
| **3** | `Kurta` | Ethnic | IndoFashion (`women_kurta`, `kurta_men`) |
| **4** | `Lehenga` | Ethnic | IndoFashion (`lehenga`) |
| **5** | `Waistcoat` | Ethnic | IndoFashion (`nehru_jackets`) |
| **6** | `Saree` | Ethnic | IndoFashion (`saree`) |
| **7** | `Sherwani` | Ethnic | IndoFashion (`sherwanis`) |
| **8** | `Shirt` | Western | Fashion Product Images Small (Kaggle) |
| **9** | `T-shirt` | Western | Fashion Product Images Small (Kaggle) |

*Scope Note:* *Shalwar Kameez* was substituted with *Waistcoat* from IndoFashion (`nehru_jackets`) after discovering the public Roboflow dataset provided only 93 unique base images (< 150 project threshold), ensuring an exact 260 images per class and an exact 1.0x balance ratio.

---

## 3. Empirical Performance Summary

Evaluated on the fixed, held-out **390-image test set** (39 images per class, strictly unseen during training and validation):

| Metric | Baseline Model (Frozen) | Fine-Tuned Model (Top 35 Layers) |
| :--- | :---: | :---: |
| **Overall Accuracy** | **88.21%** (344/390) | **87.95%** (343/390) |
| **Macro Precision** | 88.22% | **88.90%** |
| **Macro Recall** | **88.21%** | 87.95% |
| **Macro F1-Score** | **88.06%** | **87.71%** |
| **Western-Class Accuracy** | 92.31% | **93.33%** |
| **Ethnic-Class Accuracy** | **84.10%** | 82.56% |
| **TF.js Parity Discrepancy** | — | **$\le 0.000001$ (PASSED)** |

### Key Perception Finding
Western classes (93.33% accuracy; Jeans 100%, Dress 97.4%, T-shirt 93.8%) outperform ethnic classes (82.56% accuracy; Lehenga 100%, Waistcoat 94.9%, Sherwani 87.2%, Saree 66.7%, Kurta 64.1%) due to rigid silhouette geometries versus fluid fabric draping plasticity and collar overlaps (Kurta vs Sherwani).

---

## 4. Repository Directory Structure

```
libaas-lens/
├── .venv/                         # Local Python 3.12 virtual environment
├── data/
│   └── SOURCES.md                 # Dataset provenance, citations and licences
├── libaas/
│   ├── raw/                       # Cleaned raw images before split
│   └── processed/                 # Balanced 70/15/15 stratified dataset (2,600 images)
│       ├── train/                 # 1,820 images (182 / class)
│       ├── val/                   # 390 images (39 / class)
│       └── test/                  # 390 images (39 / class)
├── model/                         # Production TensorFlow.js Graph Model
│   ├── model.json                 # Model topology and weights manifest (144 KB)
│   ├── group1-shard1of3.bin       # Weights shard 1 (4.00 MB)
│   ├── group1-shard2of3.bin       # Weights shard 2 (4.00 MB)
│   └── group1-shard3of3.bin       # Weights shard 3 (0.49 MB)
├── models/                        # Saved Keras and TensorFlow models
│   ├── baseline.keras             # Frozen MobileNetV2 baseline
│   ├── finetuned.keras            # Fine-tuned MobileNetV2
│   ├── final_model.keras          # Selected final model checkpoint
│   └── saved_model_inference/     # Clean inference SavedModel without augmentation
├── report/                        # Academic documentation & presentation materials
│   ├── REPORT.md                  # Complete university research report
│   ├── SLIDES_OUTLINE.md          # 12-slide presentation outline with speaker notes
│   └── DEFENSE_QA.md              # 15 technical oral defence questions and answers
├── reports/                       # Quantitative evaluation outputs and plots
│   ├── metrics.json               # Full test metrics and confusion matrix
│   ├── data_summary.csv           # Class counts and balance ratios
│   ├── baseline_training.csv      # Epoch-by-epoch baseline training logs
│   ├── finetuned_training.csv     # Epoch-by-epoch fine-tuning logs
│   ├── experiments.csv            # Run comparison table
│   ├── parity_report.json         # 20-image Keras vs TF.js parity verification
│   ├── confusion_matrix_baseline.png
│   ├── confusion_matrix_finetuned.png
│   ├── training_curves.png
│   ├── sample_correct.png
│   └── sample_misclassified.png
├── samples/                       # Lightweight sample images for web classifier
├── export_tfjs.py                 # Graph export and TF.js conversion script
├── prepare_data.py                # Dataset download, deduplication and 70/15/15 split
├── train.py                       # Baseline and fine-tuning training pipeline
├── evaluate.py                    # Evaluation and metric visualization script
├── test_parity.py                 # 20-image parity test between Keras and TF.js
├── index.html                     # Redesigned single-page static web application
├── metrics.json                   # Root-level metrics reference for GitHub Pages
└── requirements.txt               # Pinned Python dependencies
```

---

## 5. Quickstart & Reproduction Guide

### Prerequisites
- Windows 11 / 10, Linux, or macOS
- Python 3.12 (or 3.10+)
- PowerShell (Windows) or Bash (Linux/macOS)

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/afafshahid/libaas-lens.git
cd libaas-lens

# Create virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation & Deduplication
```powershell
# Ingests IndoFashion and Fashion Product Images Small,
# deduplicates using SHA-256 and DCT perceptual hashing,
# and creates a stratified 70/15/15 split capped at 260 images/class:
python prepare_data.py
```

### 3. Training Baseline & Fine-Tuned Models
```powershell
# Trains frozen MobileNetV2 baseline followed by fine-tuning top 35 layers,
# saving checkpoints to models/ and logging curves to reports/:
python train.py
```

### 4. Quantitative Evaluation & Plots
```powershell
# Evaluates on the held-out 390-image test set,
# generating confusion matrices and error visualisations:
python evaluate.py
```

### 5. TensorFlow.js Model Export & Parity Test
```powershell
# Strips training augmentation layers and exports chunked TF.js graph model:
python export_tfjs.py

# Runs 20-image parity verification (verifying max discrepancy <= 2%):
python test_parity.py
```

### 6. Run Web Application Locally
Because modern browsers enforce CORS restrictions when loading binary weight shards (`.bin`) via the `file://` protocol, serve the project folder using a local HTTP server:

```powershell
python -m http.server 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in Chrome, Edge, Safari, or Firefox.

---

## 6. Citations & References

- **MobileNetV2:** Sandler et al., *"MobileNetV2: Inverted Residuals and Linear Bottlenecks"*, CVPR 2018. [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)
- **IndoFashion:** Rajput & Aneja, *"IndoFashion: Apparel Classification for Indian Ethnic Clothes"*, CVPR Workshops 2021. [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2021W/CVFAD/html/Rajput_IndoFashion_Apparel_Classification_for_Indian_Ethnic_Clothes_CVPRW_2021_paper.html)
- **Fashion Product Images Small:** Param Aggarwal, Kaggle. [Kaggle Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
- **Roboflow Pakistani Clothes:** Roboflow Universe, CC BY 4.0. [Roboflow Universe](https://universe.roboflow.com/cooking-pot/pakistani-clothes)
- **TensorFlow & Keras:** [https://www.tensorflow.org](https://www.tensorflow.org)
- **TensorFlow.js:** [https://www.tensorflow.org/js](https://www.tensorflow.org/js)

---

## 7. Project Authors & Acknowledgements

**Course:** CS4045 Deep Learning for Perception  
**Institution:** FAST-NUCES Karachi, Department of Computer Science  
**Academic Term:** Fall 2026  

- **Afaf Shahid** (23K-0678) — Group Coordinator  
  *Data pipeline, perceptual hash deduplication, baseline model, TF.js graph conversion, academic report.*
- **Ahmad Noorani** (23K-0547)  
  *Taxonomy harmonization, fine-tuning optimization, confusion matrix error analysis, web application interface, presentation slides.*
