# Libaas Lens: Presentation Slide Outline

**Course:** CS4045 Deep Learning for Perception · FAST-NUCES Karachi · Fall 2026  
**Project Title:** Libaas Lens (لباس لینز) — Clothing Type Classification Using Deep Transfer Learning  
**Presenters:** Afaf Shahid (23K-0678) & Ahmad Noorani (23K-0547)  
**Target Duration:** 10–12 Minutes + 3 Minutes Q&A (10–12 Slides)

---

## Slide 1: Title Slide
- **Title:** Libaas Lens (لباس لینز)
- **Subtitle:** Dual-Domain Apparel Classification Across Pakistani Ethnic & Western Fashion via Deep Transfer Learning
- **Course & Department:** CS4045 Deep Learning for Perception · Department of Computer Science
- **Institution:** FAST National University of Computer & Emerging Sciences, Karachi Campus
- **Team Members:**
  - Afaf Shahid (23K-0678, Coordinator)
  - Ahmad Noorani (23K-0547)
- **Live Demo & Repo:** `afafshahid.github.io/libaas-lens` · `github.com/afafshahid/libaas-lens`
- **Speaker Notes:**
  > "Good morning faculty and fellow peers. Today Ahmad and I are presenting Libaas Lens, our perception project addressing a fundamental cultural blindspot in modern computer vision: the automated classification of both Pakistani ethnic and Western clothing within a single edge-efficient deep model."

---

## Slide 2: Problem Statement & Motivation
- **The Cultural Perception Gap in Fashion AI:**
  - Modern vision benchmarks (DeepFashion, Fashion-MNIST) are overwhelmingly Western-centric.
  - SOTA fashion APIs routinely fail on South Asian garments (e.g., misclassifying a long men's *Kurta* as a *Dress*).
- **The Pakistani Wardrobe Reality:**
  - Hybrid dressing: Pakistani consumers seamlessly alternate between denim/shirts and traditional shalwars/kurtas/sherwanis.
  - E-commerce sellers in Karachi/Lahore require unified auto-tagging.
- **The Open-Set Female Garment Failure Mode:**
  - Initial 10-class models lacked a female suit category, forcing any Pakistani lawn suit or frock into *Lehenga*.
- **Project Goal:**
  - Classify 11 balanced apparel classes (6 Ethnic, 5 Western) in a lightweight, client-side model running 100% in the browser.
- **Speaker Notes:**
  > "If you test popular fashion classifiers on South Asian clothes, they crumble because traditional attires are absent from ImageNet and DeepFashion. In Pakistan, we don't wear only Western or only Eastern clothes—we wear both. Our goal is to build an honest, robust model that handles both with high precision."

---

## Slide 3: The 11-Class Taxonomy & Annotation Rules
- **6 Pakistani Ethnic Classes:**
  - `Kurta`, `Lehenga`, `Waistcoat` (*Nehru Jacket*), `Saree`, `Shalwar Kameez`, `Sherwani`
- **5 Western Classes:**
  - `Dress`, `Jacket - Blazer`, `Jeans - Trousers`, `Shirt`, `T-shirt`
- **Core Annotation Rules:**
  - **Single Label Rule:** Each photo is classified by its primary garment.
  - **Kurta:** Upper tunic without distinct lower trousers.
  - **Shalwar Kameez:** Complete 2-pc / 3-pc suits (tunic + shalwar/trouser + dupatta) for both men and women.
  - **Waistcoat:** Sleeveless ethnic vest with mandarin band collar (*waskat*).
  - **Sherwani:** Structured ceremonial coat with high collar and front placket.
- **Speaker Notes:**
  > "We established a clean 11-class dual-domain taxonomy. By curating authentic Pakistani designer collections (Khaadi, J., Sapphire, Gul Ahmed, Alkaram) alongside IndoFashion suits, we built a dedicated Shalwar Kameez category covering both male and female wear, completely resolving the open-set false-positive loop where women's suits were misclassified as Lehengas."

---

## Slide 4: Data Engineering, Cleaning & Deduplication
- **Data Provenance:**
  - IndoFashion (CVPRW 2021) $\rightarrow$ Curated ethnic classes.
  - Pakistani Fashion Retail (Khaadi, J., Sapphire, Gul Ahmed, Alkaram) $\rightarrow$ High-res female/male suits.
  - Kaggle Fashion Product Images Small $\rightarrow$ Clean Western staples.
  - Total: **2,860 clean, balanced images** (260/class, 1.0x balance ratio).
- **Cleaning Pipeline:**
  - Corrupt header elimination (PIL validation).
  - Minimum resolution verification ($w, h \ge 40$ px).
  - Exact deduplication via SHA-256 hashes.
  - Near-duplicate promotional pruning via 64-bit DCT perceptual hashing (`imagehash.phash`, Hamming distance $\le 4$).
- **Stratified Partitioning (70 / 15 / 15):**
  - Train: 2,002 images (182 / class)
  - Val: 429 images (39 / class)
  - Test: 429 images (39 / class)
- **Speaker Notes:**
  > "Dataset quality dictates model reliability. We eliminated corrupt files, applied perceptual hashing to prevent promotional duplicates from leaking across splits, and strictly enforced a 1.0x balance ratio with 260 images per class across all 11 classes."

---

## Slide 5: Model Architecture & Embedded Preprocessing
- **Why MobileNetV2?**
  - Inverted residual bottlenecks with depthwise separable convolutions.
  - Only 3.4M parameters; ~8.6 MB model size.
  - Runs in under 50 ms on standard laptops/phones via WebGL.
- **Architectural Innovation: Embedded Preprocessing Layer:**
  - `Rescaling(scale=1/127.5, offset=-1.0)` is baked directly into the model graph.
  - Ingests raw `[0, 255]` pixel arrays directly from HTML5 Canvas `ImageData`.
  - Zero divergence between Python training and JavaScript browser runtime.
- **Regularization:**
  - `GlobalAveragePooling2D` $\rightarrow$ `Dropout(0.3)` $\rightarrow$ `Dense(11, Softmax)`.
- **Speaker Notes:**
  > "Rather than computing preprocessing in JavaScript, we embedded the rescaling layer directly into the Keras graph. The browser passes raw RGB pixels directly to the model, eliminating floating-point mismatches and ensuring zero-overhead preprocessing."

---

## Slide 6: Transfer Learning Methodology
- **Stage 1: Frozen Feature Extractor (Baseline):**
  - Pretrained ImageNet weights loaded; all 155 backbone layers frozen.
  - Linear probe head trained with Adam ($\eta = 10^{-3}$) and EarlyStopping.
- **Stage 2: Fine-Tuning (Top 35 Layers):**
  - Unfroze top 3 bottleneck blocks (layers 120–155) to capture textile motifs, embroidery patterns, and drape textures.
  - Applied targeted data augmentations: horizontal flips, subtle rotations ($\pm 8\%$), zoom ($\pm 8\%$), and contrast adjustment.
  - Low learning rate ($\eta = 10^{-5}$) with `ReduceLROnPlateau`.
- **The Critical Rule: Batch Normalization Freeze:**
  - Kept all `BatchNormalization` layers non-trainable (`trainable = False`).
  - Prevents noisy mini-batch statistics ($N=32$) from corrupting ImageNet representations.
- **Speaker Notes:**
  > "A crucial technical insight in transfer learning is freezing Batch Normalization statistics during fine-tuning. Allowing batch norm to update on small batches rapidly corrupts pretrained activations. Freezing them ensured stable convergence."

---

## Slide 7: Quantitative Evaluation on Held-Out Test Set
- **Evaluated on 429 Unseen Test Images (39 per class):**

| Metric | Baseline Model | Fine-Tuned Model | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 85.08% | **86.48%** | **$+1.40\%$** |
| **Macro Precision** | 85.67% | **86.86%** | **$+1.19\%$** |
| **Macro Recall** | 85.08% | **86.48%** | **$+1.40\%$** |
| **Macro F1-Score** | 84.68% | **86.18%** | **$+1.50\%$** |
| **Western Accuracy** | 90.26% | **92.82%** | **$+2.56\%$** |
| **Ethnic Accuracy** | 80.77% | **81.20%** | **$+0.43\%$** |

- **Highlights:**
  - Fine-tuning yielded gains across every metric (+1.40% Accuracy, +1.50% F1).
  - Perfect performance: `Jeans - Trousers` (100% P, 100% R) and `Dress` (100% R).
- **Speaker Notes:**
  > "On our held-out test set, fine-tuning achieved 86.48% accuracy and an 86.18% macro F1-score, confirming that specialized feature adaptation substantially outperforms a frozen linear probe."

---

## Slide 8: Per-Class Performance & Confusion Matrix
- **Per-Class F1 Highlights:**
  - `Jeans - Trousers`: **100.0%** | `Dress`: **96.3%** | `Waistcoat`: **93.7%**
  - `T-shirt`: **92.3%** | `Jacket - Blazer`: **90.7%** | `Lehenga`: **90.5%**
  - `Saree`: **87.7%** | `Shirt`: **84.6%** | `Sherwani`: **83.3%**
  - `Shalwar Kameez`: **71.8%** | `Kurta`: **57.1%**
- **Key Confusion Clusters:**
  - `Kurta` $\leftrightarrow$ `Sherwani` & `Shalwar Kameez`: Shared upright mandarin collars (*bain gala*) and straight silhouettes.
  - `Saree` $\rightarrow$ `Lehenga` (5 images): Pleated bottom drape and zari border resemblance.
- **Speaker Notes:**
  > "Looking at the confusion matrix, Western garments achieve near-perfect metrics. The primary confusions occur within the ethnic domain between Kurtas, Shalwar Kameez, and Sherwanis—garments that share upright band collars and vertical tunic cuts."

---

## Slide 9: Perception Disparity: Geometric Rigidity vs. Fluid Drape
- **The 11.62% Perception Gap:**
  - Western Wear: **92.82%** accuracy.
  - Ethnic Wear: **81.20%** accuracy.
- **Physical & Visual Foundations:**
  - **Western Structural Rigidity:** Fixed silhouettes, standard seams, rigid denim twill, and stable flat-lay catalog poses.
  - **Ethnic Drape Plasticity:** Sarees and dupattas are unstitched, flexible fabrics whose visual geometry changes completely with posture, movement, and camera perspective.
  - **Collar Overlap:** Downsampling to $224 \times 224$ compresses micro-textile cues, emphasizing the shared mandarin band collar.
- **Speaker Notes:**
  > "Why does the model achieve 92.8% on Western clothes but 81.2% on ethnic wear? It comes down to geometry. Jeans and T-shirts have rigid, predictable silhouettes. In contrast, an unstitched 6-yard saree or a flowing dupatta exhibits massive geometric plasticity."

---

## Slide 10: Client-Side Edge Deployment & Parity Verification
- **Zero-Server In-Browser Inference:**
  - Converted SavedModel into a sharded TensorFlow.js Graph Model (8.64 MB total).
  - Graph optimizations: Conv2D + Relu6 + BiasAdd operator fusion.
  - Instant inference via WebGL acceleration.
- **22-Image Parity Verification:**
  - Tested 2 images per class across both native Python Keras and browser TF.js.
  - Required discrepancy threshold: $\le 2.0\%$.
  - **Empirical discrepancy:** **$0.000001$ ($0.0001\%$)** $\rightarrow$ **100% Parity PASSED**.
- **Privacy First:** Wardrobe photos never leave the user's browser.
- **Speaker Notes:**
  > "For real-world deployment, privacy is paramount. Users will not upload private wardrobe photos to an untrusted server. We converted our model to TensorFlow.js, verified mathematical parity with less than 0.0001% difference from Python, and deliver sub-50ms inference directly in the browser."

---

## Slide 11: Live Studio Demo Walkthrough
- **Demonstration Elements:**
  - High-res sample tray covering all 11 categories.
  - Real-time classification with confidence gauge and telemetry stats.
  - Webcam integration for in-the-wild wardrobe classification.
  - Interactive confusion matrix and per-class performance tables.
- **Live URL:** `https://afafshahid.github.io/libaas-lens/`
- **Speaker Notes:**
  > "Here is our live web studio. Notice the real-time classification of authentic Pakistani retail suits as Shalwar Kameez, the high precision on Western denim and jackets, and the full interactive confusion matrix rendering."

---

## Slide 12: Conclusion & Future Outlook
- **Key Contributions:**
  1. Built a balanced 11-class dual-domain dataset (2,860 images) bridging Pakistani ethnic and Western fashion.
  2. Resolved the open-set female suit false-positive mode by incorporating authentic Pakistani retail imagery.
  3. Documented and explained the 11.62% perception disparity between rigid Western apparel and plastic ethnic drapery.
  4. Delivered a 100% private, client-side web application with mathematically verified TensorFlow.js parity.
- **Future Directions:**
  - Multi-garment bounding-box detection (e.g. YOLOv8) for layered assemblies (Waistcoat over Kurta).
  - Higher-resolution backbones ($384 \times 384$) to resolve fine fabric weave distinctions.
- **Q&A:**
  - *"Thank you for your time. We are now open to your questions."*
