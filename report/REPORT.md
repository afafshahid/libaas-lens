# Libaas Lens: Dual-Domain Clothing Type Classification Across Pakistani Ethnic and Western Apparel Using Deep Transfer Learning

**Course:** CS4045 Deep Learning for Perception  
**Institution:** FAST National University of Computer and Emerging Sciences (FAST-NUCES), Karachi Campus  
**Academic Term:** Fall 2026  
**Authors:**
- **Afaf Shahid** (Roll No: 23K-0678, Group Coordinator) — `k230678@nu.edu.pk`
- **Ahmad Noorani** (Roll No: 23K-0547) — `k230547@nu.edu.pk`

**Repository:** [https://github.com/afafshahid/libaas-lens](https://github.com/afafshahid/libaas-lens)  
**Live Application:** [https://afafshahid.github.io/libaas-lens/](https://afafshahid.github.io/libaas-lens/)  
**Model Architecture:** MobileNetV2 (Pretrained on ImageNet-1k)  
**Deployment Target:** On-Device Client-Side Web Inference via TensorFlow.js Graph Model  

---

## Abstract

Automated apparel visual perception represents a cornerstone of modern computer vision, powering e-commerce cataloging, visual search, and smart wardrobe systems. However, mainstream commercial and academic benchmarks exhibit severe geographic and cultural bias, heavily prioritizing Western garments (e.g., T-shirts, dresses, jeans) while neglecting traditional South Asian attire worn daily by over 1.8 billion people. In Pakistan, the sartorial reality is uniquely hybrid: individuals alternate seamlessly between traditional ethnic attire and contemporary Western fashion. 

In this work, we present **Libaas Lens** (*لباس لینز*), a unified deep transfer-learning perception framework capable of categorizing apparel into a balanced **11-class dual-domain taxonomy** spanning six Pakistani ethnic categories (*Kurta*, *Lehenga*, *Waistcoat*, *Saree*, *Shalwar Kameez*, *Sherwani*) and five Western categories (*Dress*, *Jacket - Blazer*, *Jeans - Trousers*, *Shirt*, *T-shirt*). A critical motivation for expanding to 11 classes was addressing an acute open-set failure mode: in initial 10-class models lacking a general suit category, any Pakistani stitched female outfit (2-piece or 3-piece lawn suit, frock, or long kameez) was erroneously forced into *Lehenga*. By curating authentic Pakistani designer retail imagery (Khaadi, J., Sapphire, Gul Ahmed, Alkaram) combined with IndoFashion suit sets, we established a balanced dataset of 2,860 images (260 per class) partitioned into a strict 70/15/15 stratified split.

On a held-out test split of 429 unseen images (39 per class), our fine-tuned MobileNetV2 model achieves **86.48% overall accuracy** and a **macro F1-score of 86.18%**, outperforming the frozen linear baseline (85.08% accuracy, 84.68% macro F1). Crucially, our perception analysis documents an 11.62% disparity between Western wear (92.82% accuracy) and ethnic apparel (81.20% accuracy), grounded in geometric invariance versus fluid fabric plasticity. Finally, we convert the trained model into a production-ready TensorFlow.js Graph Model with embedded preprocessing layers, achieving 100% classification parity ($\le 0.0001\%$ probability discrepancy) and enabling zero-server-upload, zero-latency inference directly in the client's web browser.

---

## 1. Introduction & Problem Statement

### 1.1 The Cultural Perception Gap in Fashion AI
Deep convolutional neural networks (CNNs) have revolutionized image classification, object detection, and visual perception. Within the fashion domain, benchmark datasets such as DeepFashion (Liu et al., 2016), Fashion-MNIST (Xiao et al., 2017), and the Kaggle Fashion Product dataset have served as standard proving grounds. However, an inspection of these benchmarks reveals a stark cultural asymmetry: almost all annotated classes represent Western attire.

In Pakistan and the wider South Asian diaspora, apparel is inherently pluralistic. The same wardrobe accommodates both Western daily wear (denim jeans, polo shirts, formal blazers) and rich ethnic traditions spanning daily cotton kurtas, embroidered festive sherwanis, bridal lehengas, formal sarees, formal waistcoats (*waskat*), and the quintessential national attire: **Shalwar Kameez** (worn in both male kurta-pajama/shalwar variants and female 2-piece/3-piece embroidered ensembles). When applied to Pakistani e-commerce storefronts or personal wardrobe inventories, commercial computer vision APIs frequently fail—misclassifying a long men's *Kurta* as a women's *Dress*, or forcing any female Pakistani festive suit into a bridal *Lehenga*.

### 1.2 Research Objectives
The primary objective of the **Libaas Lens** project is to formulate, implement, and rigorously benchmark a single, efficient deep learning classifier that bridges this divide without sacrificing inference efficiency. Specifically, the project addresses four core challenges:
1. **Taxonomic Alignment & Harmonization:** Constructing a coherent 11-class dual-domain taxonomy bridging distinct public datasets and authentic Pakistani designer catalogs into a unified multi-class single-label classification problem.
2. **Transfer Learning Optimization:** Adapting MobileNetV2 (Sandler et al., 2018), comparing a frozen linear probe baseline against fine-tuning top convolutional blocks with tailored data augmentations.
3. **Perception and Error Nuance Analysis:** Conducting an empirical evaluation comparing model performance across ethnic versus Western apparel, documenting specific visual confusions (e.g., draping variations and collar cuts) and resolving open-set false positives.
4. **Edge & Privacy-Preserving Deployment:** Delivering a client-side inference engine in TensorFlow.js that processes raw device pixels in the browser, ensuring complete user privacy without transmitting sensitive wardrobe photographs to external cloud servers.

---

## 2. Related Work

### 2.1 Deep Learning for Visual Fashion Recognition
Fashion image classification has evolved from early handcrafted descriptors (e.g., HOG, SIFT, color histograms) to deep convolutional networks. Liu et al. (2016) introduced *DeepFashion*, containing over 800,000 images annotated with clothing attributes and landmark locations. However, DeepFashion remains overwhelmingly anchored in international Western retail catalogs.

### 2.2 South Asian Apparel Datasets: IndoFashion
Addressing the regional scarcity of ethnic wear data, Rajput & Aneja (2021) published *IndoFashion: Apparel Classification for Indian Ethnic Clothes* (CVPR Workshops 2021). IndoFashion introduced 106,166 curated retail images across 15 ethnic categories sourced from Indian e-commerce portals. IndoFashion serves as the primary ethnic wear foundation for Libaas Lens.

### 2.3 MobileNetV2 and Edge Vision
MobileNetV2 (Sandler et al., 2018) introduced inverted residual blocks with linear bottlenecks and depthwise separable convolutions, dramatically reducing parameter count (3.4 million parameters) and FLOPs while preserving ImageNet classification capacity. Because real-world e-commerce visual tagging and mobile wardrobe management demand low-latency, client-side inference on consumer devices, MobileNetV2 provides an optimal trade-off between perceptual accuracy and edge compute overhead.

---

## 3. Dataset Construction, Alignment & Cleaning

### 3.1 Taxonomy Formulation
We defined a mutually exclusive 11-class taxonomy comprising six Pakistani ethnic categories and five Western categories:

| Index | Class Name | Category Domain | Primary Source Dataset |
| :---: | :--- | :---: | :--- |
| 0 | `Dress` | Western | Fashion Product Images Small (Kaggle) |
| 1 | `Jacket - Blazer` | Western | Fashion Product Images Small (Kaggle) |
| 2 | `Jeans - Trousers` | Western | Fashion Product Images Small (Kaggle) |
| 3 | `Kurta` | Ethnic | IndoFashion (`women_kurta`, `kurta_men`) |
| 4 | `Lehenga` | Ethnic | IndoFashion (`lehenga`) |
| 5 | `Nehru Jacket` (*Waistcoat*) | Ethnic | IndoFashion (`nehru_jackets`) |
| 6 | `Saree` | Ethnic | IndoFashion (`saree`) |
| 7 | `Shalwar Kameez` | Ethnic | Pakistani Fashion Retail (Khaadi, J., Sapphire, Gul Ahmed, Alkaram) + IndoFashion Suit Sets |
| 8 | `Sherwani` | Ethnic | IndoFashion (`sherwanis`) |
| 9 | `Shirt` | Western | Fashion Product Images Small (Kaggle) |
| 10 | `T-shirt` | Western | Fashion Product Images Small (Kaggle) |

*Note on Nomenclature:* Index 5 is cataloged as `Nehru Jacket` in the IndoFashion benchmark, and is presented in user interfaces and Pakistani commercial contexts as **Waistcoat** (*واسکت*).

#### Annotation & Perception Rules:
1. **Primary Garment Rule:** Each photograph is assigned exactly one label corresponding to the primary focal garment.
2. **Kurta vs. Shalwar Kameez:** Garments displaying a standalone upper tunic without an accompanying trouser/shalwar or dupatta ensemble are categorized as **Kurta**. Full two-piece or three-piece suits (tunic + shalwar/trouser + dupatta) for both men and women are categorized as **Shalwar Kameez**.
3. **Waistcoat:** Characterized by a sleeveless ethnic vest silhouette with an upright mandarin band collar and front button placket.
4. **Sherwani:** Characterized by a long, structured coat/tunic with formal collar, structured shoulders, front closure, and ceremonial fabrics.

### 3.2 Data Sourcing and Pipeline
- **IndoFashion:** Sourced from the official IndoFashion benchmark (Rajput & Aneja, 2021). JSON annotation records (`train_data.json`, `val_data.json`, `test_data.json`) were parsed to extract verified product images.
- **Fashion Product Images Small:** Sourced from Param Aggarwal's open e-commerce benchmark on Kaggle. Metadata catalog records (`styles.csv`) were indexed to isolate target `articleType` categories (`Dresses`, `Jackets`, `Blazers`, `Jeans`, `Trousers`, `Shirts`, `Tshirts`).
- **Pakistani Fashion Retail Dataset:** Authentic Pakistani designer retail images from Khaadi, Junaid Jamshed (J.), Sapphire, Gul Ahmed, and Alkaram were integrated to represent modern printed and embroidered female 2-pc/3-pc lawn suits and male shalwar kameez suits.

### 3.3 Cleaning, Deduplication & Quality Filtering
The data ingestion pipeline (`prepare_data.py`) applied automated quality verification:
1. **Corrupt File Elimination:** All candidate images were verified using PIL; unreadable, zero-byte, or invalid header files were discarded.
2. **Dimensionality Checks:** Images were checked for minimum spatial resolution ($w \ge 40$, $h \ge 40$) to prevent severe pixelation artifacts upon upsampling to $224 \times 224$.
3. **Exact and Perceptual Deduplication:** 
   - Exact duplicates were identified and removed using cryptographic SHA-256 content hashing.
   - Near-identical promotional duplicates were detected using Discrete Cosine Transform (DCT) perceptual hashing (`imagehash.phash`) with a strict Hamming distance threshold of $\le 4$.

### 3.4 Stratified Partitioning & Final Distribution
To prevent class imbalance artifacts, every class was capped at exactly **260 images**, establishing a perfectly uniform 1.0x ratio across all 11 classes (2,860 total images). Using fixed random seed 42, the dataset was split into:
- **Training Set (70%):** 2,002 images (182 per class)
- **Validation Set (15%):** 429 images (39 per class)
- **Held-Out Test Set (15%):** 429 images (39 per class)

```
libaas/processed/
├── train/          (2,002 images; 182 / class)
├── val/            (429 images; 39 / class)
└── test/           (429 images; 39 / class)
```

The split is saved permanently on disk to prevent data leakage during subsequent training and evaluation stages.

---

## 4. Methodology & Model Architecture

```
+-------------------------------------------------------------------+
|               Input RGB Image (224 x 224 x 3, [0, 255])           |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|     Embedded Keras Rescaling Layer (scale = 1/127.5, offset = -1) |
|                       Maps [0, 255] -> [-1.0, 1.0]                |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|               MobileNetV2 Backbone (ImageNet Pretrained)          |
|      Baseline: All 155 layers frozen                              |
|      Fine-Tuned: Top 35 layers unfrozen; BatchNorm in eval mode   |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                   GlobalAveragePooling2D (1280-D)                 |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                     Dropout (Rate = 0.20 / 0.30)                  |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                 Dense Output Layer (11 Units, Softmax)            |
+-------------------------------------------------------------------+
```

### 4.1 Embedded Image Preprocessing Layer
In conventional Keras deployments, `preprocess_input` is invoked as an external CPU transformation prior to tensor construction. In Libaas Lens, we explicitly instantiate a Keras `Rescaling(scale=1.0 / 127.5, offset=-1.0)` layer directly as the first internal node of the network. This design architectural choice provides critical benefits:
- Eliminates client-side preprocessing discrepancies in browser environments.
- Accepts raw `[0, 255]` pixel arrays directly from HTML5 Canvas `ImageData`.
- Ensures zero computational overhead in JavaScript execution.

### 4.2 Baseline Model (Frozen Feature Extractor)
The baseline configuration treats MobileNetV2 purely as a fixed visual feature extractor:
1. Pretrained ImageNet weights are loaded.
2. The entire backbone (155 layers) is frozen (`backbone.trainable = False`).
3. Spatial features from the final bottleneck block are aggregated via `GlobalAveragePooling2D`.
4. A regularization `Dropout(rate=0.2)` layer is applied.
5. A `Dense(11, activation="softmax")` projection maps the 1280-dimensional feature vector to class probabilities.
6. **Optimizer:** Adam with learning rate $\eta = 10^{-3}$, categorical cross-entropy loss, and EarlyStopping on validation accuracy.

### 4.3 Fine-Tuned Model (Gradual Unfreezing & Augmentation)
To specialize high-level convolutional representations to the unique textile textures, folds, and embroidery patterns of South Asian apparel, we formulate a two-stage transfer learning strategy:
1. **Data Augmentation:** Embedded training-time transformations including `RandomFlip("horizontal")`, `RandomRotation(0.08)`, `RandomZoom(0.08)`, and `RandomContrast(0.10)`.
2. **Top-Layer Unfreezing:** The top 35 layers of MobileNetV2 (corresponding to high-level bottleneck blocks 14, 15, and 16, plus the final $1\times 1$ expansion convolution) are unfrozen for gradient backpropagation.
3. **Batch Normalization Freeze:** Crucially, all `BatchNormalization` layers across the entire backbone are explicitly kept in non-trainable inference mode (`layer.trainable = False`). Unfreezing batch normalization statistics with small batch sizes ($N=32$) leads to severe variance estimation errors that corrupt pretrained ImageNet features.
4. **Optimization Schedule:** Adam with a low learning rate $\eta = 10^{-5}$, paired with `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-7)` and `EarlyStopping(patience=5)`.
5. **Inference Graph Isolation:** For export and TensorFlow.js conversion, data augmentation ops are completely stripped to ensure a purely deterministic static inference computation graph.

---

## 5. Experimental Results & Quantitative Evaluation

All evaluations were executed on the fixed, held-out 429-image test set (39 images per class). Metrics were computed using exact scikit-learn implementations and logged to `reports/metrics.json`.

### 5.1 Global Model Comparison

| Evaluation Metric | Baseline Model (Frozen) | Fine-Tuned Model (Top 35) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 85.08% (365/429) | **86.48%** (371/429) | **$+1.40\%$** |
| **Macro Precision** | 85.67% | **86.86%** | **$+1.19\%$** |
| **Macro Recall** | 85.08% | **86.48%** | **$+1.40\%$** |
| **Macro F1-Score** | 84.68% | **86.18%** | **$+1.50\%$** |
| **Ethnic-Class Accuracy** | 80.77% (189/234) | **81.20%** (190/234) | **$+0.43\%$** |
| **Western-Class Accuracy** | 90.26% (176/195) | **92.82%** (181/195) | **$+2.56\%$** |

*Key Takeaway:* Fine-tuning the top 35 layers produced consistent improvements across every global evaluation metric, delivering a +1.40% gain in overall accuracy, a +1.50% gain in macro F1, and a +2.56% gain in Western class discrimination.

### 5.2 Per-Class Precision, Recall, and F1 Breakdown

The table below details per-class performance for the **Fine-Tuned Model** on the held-out test split ($N=39$ per class):

| Garment Class | Category Domain | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dress** | Western | 0.9286 | **1.0000** | **0.9630** | 39 |
| **Jacket - Blazer** | Western | 0.9444 | 0.8718 | 0.9067 | 39 |
| **Jeans - Trousers** | Western | **1.0000** | **1.0000** | **1.0000** | 39 |
| **Kurta** | Ethnic | 0.6452 | 0.5128 | 0.5714 | 39 |
| **Lehenga** | Ethnic | 0.8444 | 0.9744 | **0.9048** | 39 |
| **Waistcoat** (*Nehru Jacket*) | Ethnic | 0.9250 | 0.9487 | **0.9367** | 39 |
| **Saree** | Ethnic | 0.9412 | 0.8205 | 0.8767 | 39 |
| **Shalwar Kameez** | Ethnic | 0.7179 | 0.7179 | 0.7179 | 39 |
| **Sherwani** | Ethnic | 0.7778 | 0.8974 | 0.8333 | 39 |
| **Shirt** | Western | 0.8462 | 0.8462 | 0.8462 | 39 |
| **T-shirt** | Western | 0.9231 | 0.9231 | 0.9231 | 39 |

### 5.3 Confusion Matrix Analysis
The full $11 \times 11$ confusion matrix for the fine-tuned model on the held-out test set is detailed below:

```
Predicted Class Index:
      [0]   [1]   [2]   [3]   [4]   [5]   [6]   [7]   [8]   [9]  [10]
[0]    39     0     0     0     0     0     0     0     0     0     0   | Dress (Western)
[1]     1    34     0     0     0     0     0     0     0     4     0   | Jacket - Blazer (Western)
[2]     0     0    39     0     0     0     0     0     0     0     0   | Jeans - Trousers (Western)
[3]     0     0     0    20     0     3     1     7     8     0     0   | Kurta (Ethnic)
[4]     0     0     0     0    38     0     0     1     0     0     0   | Lehenga (Ethnic)
[5]     0     0     0     1     0    37     0     0     1     0     0   | Waistcoat (Ethnic)
[6]     0     0     0     1     5     0    32     1     0     0     0   | Saree (Ethnic)
[7]     0     0     0     7     2     0     1    28     1     0     0   | Shalwar Kameez (Ethnic)
[8]     0     0     0     2     0     0     0     2    35     0     0   | Sherwani (Ethnic)
[9]     1     2     0     0     0     0     0     0     0    33     3   | Shirt (Western)
[10]    1     0     0     0     0     0     0     0     0     2    36   | T-shirt (Western)
```

---

## 6. Deep Perception & Qualitative Error Analysis

```
Key Visual Confusion Modes Identified:
1. Kurta (20/39)          --->  Sherwani (8), Shalwar Kameez (7), Waistcoat (3)
2. Shalwar Kameez (28/39) --->  Kurta (7), Lehenga (2), Saree (1), Sherwani (1)
3. Saree (32/39)          --->  Lehenga (5), Kurta (1), Shalwar Kameez (1)
4. Jacket (34/39)         --->  Shirt (4), Dress (1)
```

### 6.1 Resolving the Open-Set Lehenga Bias with Shalwar Kameez
In the initial 10-class model, `Lehenga` was the sole female ethnic category featuring volume, flares, and embroidery. Because female Pakistani apparel predominantly comprises 2-piece and 3-piece stitched lawn/chiffon suits, kurtis with trousers, and flared frocks with dupattas, any real-world photo of a Pakistani woman's daily or festive dress was forced into `Lehenga`. 

By curating 260 balanced images of **Shalwar Kameez** (covering both male kurta-pajama sets and authentic female Pakistani retail suits from Khaadi, J., Sapphire, Gul Ahmed, and Alkaram), Libaas Lens provided the network with the exact conceptual class required to resolve this false-positive loop. In live testing on authentic Pakistani designer catalogs, female suits achieve **99.2% to 99.9% prediction confidence as Shalwar Kameez**, completely eliminating the spurious Lehenga predictions.

### 6.2 The Ethnic vs. Western Performance Disparity
An analysis of the empirical results reveals an 11.62% perception disparity: **Western apparel achieved 92.82% accuracy, whereas ethnic apparel achieved 81.20% accuracy**. 

This disparity is grounded in geometric rigidity versus fluid plasticity:
- **Rigid Silhouette Consistency in Western Wear:** Items such as `Jeans - Trousers` achieved a perfect **100% precision and 100% recall** (39/39), and `Dress` achieved **100% recall** (39/39). Denims exhibit distinct bifurcated leg structures, pocket rivets, belt loops, and rigid seam geometries that are virtually invariant across poses.
- **Fluid Drape Plasticity in Ethnic Wear:** Ethnic garments are defined by extensive fabric draping, multi-layered pleating, and variable wrapping styles. An unstitched 6-yard `Saree` wrapped around the waist and pleated across the shoulder exhibits radical geometric variability depending on how the subject is posed.

### 6.3 Intra-Ethnic Collar & Tunic Overlap
The confusion matrix indicates mutual boundary bleed between `Kurta`, `Shalwar Kameez`, and `Sherwani`. All three garments share identical upright mandarin band collars (*bain gala*) and vertical front plackets. When downsampled to $224 \times 224$ pixels, fine textile weight differences (lightweight cotton lawn versus heavy bridal raw silk) become subtly compressed, shifting model attention to the dominant shared collar architecture.

---

## 7. TensorFlow.js Web Deployment & Parity Verification

### 7.1 Architecture of the Browser Deployment
To fulfill the project's promise of complete user privacy and zero server overhead, the trained model was ported to run entirely client-side:
1. **Model Serialization:** The trained Keras weights were converted into an inference-only architecture (stripping training-only augmentation nodes) and exported as a TensorFlow SavedModel (`models/saved_model_inference`).
2. **Graph Model Conversion:** Using `tensorflowjs_converter`, the SavedModel was transformed into a sharded TensorFlow.js Graph Model stored in `model/`:
   - `model.json` (Topology and manifest: 143.8 KB)
   - `group1-shard1of3.bin` (Weights shard 1: 4.00 MB)
   - `group1-shard2of3.bin` (Weights shard 2: 4.00 MB)
   - `group1-shard3of3.bin` (Weights shard 3: 0.50 MB)
   - **Total Footprint:** 8.64 MB, optimized for fast browser caching.

### 7.2 22-Image Parity Verification
To guarantee that browser-based TensorFlow.js inference produces predictions identical to the original native Python TensorFlow/Keras runtime, we conducted an automated 22-image parity verification (`test_parity.py`). Two random test images from each of the 11 classes were evaluated through both environments:

$$\text{Discrepancy} = \max_{i \in [1, 22]} \max_{c \in [0, 10]} |P_{\text{Keras}}(y=c \mid x_i) - P_{\text{TFJS}}(y=c \mid x_i)|$$

- **Threshold Requirement:** $\le 2.0\%$ ($0.020$)
- **Empirical Maximum Discrepancy:** **$0.000001$ ($0.0001\%$)**
- **Parity Test Status:** **PASSED (100% Top-1 Agreement across all 11 classes)**

The verification results are archived in `reports/parity_report.json`.

---

## 8. Changes from Approved Proposal

In accordance with strict academic integrity guidelines, all scope adjustments and design variations from the original proposal are explicitly recorded below:

| Proposed Item | Final Delivered Implementation | Academic & Empirical Rationale |
| :--- | :--- | :--- |
| **Shalwar Kameez Class** | Expanded into **11 Classes**, including both **Waistcoat** and **Shalwar Kameez** | The initial Roboflow dataset was insufficient (only 93 images). Rather than discarding Shalwar Kameez, we incorporated high-resolution authentic Pakistani retail images (Khaadi, J., Sapphire, Gul Ahmed, Alkaram) and IndoFashion suit sets to build a 260-image balanced class. We retained *Waistcoat* as its own category, yielding an 11-class balanced taxonomy. |
| **Own-Photo Test Split** | Public held-out test split ($N=429$); own-photo evaluation integrated into live interactive studio | Sourcing high-quality, privacy-compliant photographs of 11 distinct garment types was supplemented by an interactive live webcam and image upload interface in the web application for real-time validation. |
| **Keras Layers Model Export** | Exported as optimized **TensorFlow.js Graph Model** | The Graph Model enables operator fusion (fused Conv2D + Relu6 + BiasAdd) and smaller binary weight sharding, yielding faster browser execution and lower memory footprint. |

---

## 9. Limitations & Future Work

1. **Resolution Constraints:** Standard $224 \times 224$ MobileNetV2 spatial downsampling compresses micro-textile textures (e.g., woven jacquard, delicate zari threadwork, button engravings) that are critical for separating Kurta from Sherwani. Future work should evaluate higher-resolution backbones (e.g., EfficientNetV2-S at $384 \times 384$).
2. **Multi-Label Garment Assemblies:** In real-world wardrobes, users frequently wear assemblies (e.g., a Waistcoat over a Kurta, or a Blazer over a T-shirt). Extending Libaas Lens to multi-label classification or integrating bounding-box detection (e.g., YOLOv8-Nano) would enable simultaneous multi-garment parsing.
3. **Smartphone In-the-Wild Testing:** While retail catalog photographs provide clean white backgrounds, real-world smartphone photos introduce complex backgrounds, wrinkles, and low lighting. Domain adaptation via unsupervised contrastive learning presents a promising avenue for closing this domain shift.

---

## 10. References & Dataset Citations

1. **MobileNetV2 Architecture:**  
   Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4510–4520. [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)
2. **IndoFashion Dataset:**  
   Rajput, P., & Aneja, S. (2021). *IndoFashion: Apparel Classification for Indian Ethnic Clothes*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 3965–3971.
3. **DeepFashion Benchmark:**  
   Liu, Z., Luo, P., Qiu, S., Wang, X., & Tang, X. (2016). *DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 1096–1104.
4. **Fashion Product Images Dataset:**  
   Aggarwal, P. (2019). *Fashion Product Images Dataset (Small)*. Kaggle. [https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)
5. **Pakistani Fashion Retail Catalogs:**  
   Khaadi, Junaid Jamshed (J.), Sapphire, Gul Ahmed, and Alkaram Studio E-Commerce Collections (2023–2026).
6. **TensorFlow & Keras Ecosystem:**  
   Abadi, M., et al. (2015). *TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems*. [https://www.tensorflow.org](https://www.tensorflow.org)
7. **TensorFlow.js Framework:**  
   Smilkov, D., et al. (2019). *TensorFlow.js: Machine Learning for the Web and Mobile Devices*. Proceedings of Machine Learning and Systems (SysML). [https://www.tensorflow.org/js](https://www.tensorflow.org/js)
