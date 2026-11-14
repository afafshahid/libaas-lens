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
- **Project Goal:**
  - Classify 10 apparel classes (5 Ethnic, 5 Western) in a lightweight, client-side model running 100% in the browser.
- **Speaker Notes:**
  > "If you test popular fashion classifiers on South Asian clothes, they crumble because traditional attires are absent from ImageNet and DeepFashion. In Pakistan, we don't wear only Western or only Eastern clothes—we wear both. Our goal is to build an honest, robust model that handles both with high precision."

---

## Slide 3: The 10-Class Taxonomy & Annotation Rules
- **5 Pakistani Ethnic Classes:**
  - `Kurta`, `Lehenga`, `Waistcoat`, `Saree`, `Sherwani`
- **5 Western Classes:**
  - `Dress`, `Jacket - Blazer`, `Jeans - Trousers`, `Shirt`, `T-shirt`
- **Core Annotation Rules:**
  - **Single Label Rule:** Each photo is classified by its primary garment.
  - **Kurta:** Upper tunic without distinct lower trousers.
  - **Waistcoat:** Sleeveless ethnic vest with mandarin band collar (cataloged as *nehru_jackets* in IndoFashion).
  - **Sherwani:** Structured ceremonial coat with high collar and front placket.
- **Scope Note:** Replaced *Shalwar Kameez* with *Waistcoat* after discovering the public Roboflow dataset provided only 93 unique images (< 150 threshold).
- **Speaker Notes:**
  > "We established a clean 10-class taxonomy. To ensure empirical honesty, when our initial candidate dataset for Shalwar Kameez only yielded 93 usable base images, rather than artificially inflating the data, we substituted it with Waistcoats from IndoFashion, maintaining an exact 260-image cap across all 10 classes."

---

## Slide 4: Data Engineering, Cleaning & Deduplication
- **Data Provenance:**
  - IndoFashion (CVPRW 2021) $\rightarrow$ 1,300 images (5 ethnic classes).
  - Kaggle Fashion Product Images Small $\rightarrow$ 1,300 images (5 Western classes).
  - Total: 2,600 clean, balanced images (260/class, 1.0x balance ratio).
- **Cleaning Pipeline:**
  - Corrupt header elimination (PIL validation).
  - Minimum resolution thresholding ($w \ge 40$, $h \ge 40$).
  - Exact deduplication via SHA-256.
  - Near-duplicate perceptual deduplication via DCT hash (`phash`, Hamming $\le 4$).
- **Stratified Partitioning (Seed 42):**
  - Train: 1,820 images (70%)
  - Validation: 390 images (15%)
  - Test: 390 images (15%)
- **Speaker Notes:**
  > "Data quality dictates perception quality. We removed near-duplicate catalog photos using perceptual hashing to prevent test set memorization. Every class has exactly 182 training, 39 validation, and 39 test images, completely eliminating majority-class bias."

---

## Slide 5: Deep Learning Architecture (MobileNetV2)
- **Visual Diagram:** (Include architecture flowchart from Report)
  - `Input (224x224x3)` $\rightarrow$ Embedded `Rescaling(1/127.5, -1.0)` $\rightarrow$ MobileNetV2 Backbone $\rightarrow$ `GlobalAveragePooling2D` $\rightarrow$ `Dropout(0.2-0.3)` $\rightarrow$ `Dense(10, Softmax)`.
- **Why MobileNetV2?**
  - Inverted residuals with linear bottlenecks.
  - 3.4M parameters; low compute footprint, ideal for client-side JavaScript execution.
- **Embedded Rescaling Advantage:**
  - The normalization $[0, 255] \rightarrow [-1, 1]$ is embedded *inside* the Keras computational graph.
  - No external preprocessing code needed in JavaScript—accepts raw canvas pixel arrays directly.
- **Speaker Notes:**
  > "We selected MobileNetV2 for its superior accuracy-to-parameter ratio. Crucially, we embedded the normalization layer directly inside the model graph. When this runs in the browser, JavaScript passes raw canvas pixels without manual float normalization."

---

## Slide 6: Transfer Learning & Training Strategy
- **Two Distinct Experimental Paradigms:**
  1. **Frozen Baseline:**
     - All 155 backbone layers locked.
     - Training only the Dense classification head with Adam ($\eta = 10^{-3}$).
     - Validated feature extractor capability of ImageNet weights.
  2. **Fine-Tuned Configuration:**
     - Top 35 layers unfrozen (Bottleneck blocks 14, 15, 16 + final $1\times 1$ conv).
     - Data Augmentation: Horizontal flip, $\pm 8\%$ rotation, $\pm 8\%$ zoom, $\pm 10\%$ contrast.
     - Batch Normalization explicitly frozen in inference mode.
     - Low learning rate ($\eta = 10^{-5}$) with `ReduceLROnPlateau`.
- **Slide Graphic:** Embed `reports/training_curves.png` showing smooth validation convergence.
- **Speaker Notes:**
  > "We compared a frozen baseline against a fine-tuned network. When fine-tuning, keeping Batch Normalization in inference mode was critical: updating batch statistics with a batch size of 32 degrades pretrained feature representations. With a reduced learning rate of 1e-5, the fine-tuned model converged smoothly."

---

## Slide 7: Global Quantitative Results
- **Benchmarking on 390-Image Held-Out Test Set:**
  - **Baseline Accuracy:** **88.21%** | **Macro F1:** **88.06%**
  - **Fine-Tuned Accuracy:** **87.95%** | **Macro F1:** **87.71%**
  - **Western-Class Accuracy:** **93.33%**
  - **Ethnic-Class Accuracy:** **82.56%**
- **Slide Graphic:** Embed `reports/confusion_matrix_finetuned.png`.
- **Key Empirical Takeaway:**
  - The frozen baseline achieved virtually identical overall accuracy to fine-tuning, demonstrating that high-level ImageNet spatial textures transfer remarkably well to garment silhouettes.
- **Speaker Notes:**
  > "On our held-out test set of 390 unseen images, our models achieved over 88% accuracy. But look at the sub-category breakdown: Western wear scored 93.3%, while ethnic wear scored 82.6%. This brings us to our most important perception finding."

---

## Slide 8: Perception Disparity: Structural Rigidity vs. Fluid Draping
- **Western Wear (93.33% Accuracy):**
  - Rigid seaming, fixed silhouettes, predictable contours.
  - `Jeans - Trousers`: **100% Precision, 100% Recall** (39/39).
  - `Dress`: **97.4% F1** · `T-shirt`: **93.8% F1**.
- **Ethnic Wear (82.56% Accuracy):**
  - Highly flexible fabric wrapping, unstitched draping, multi-layer pleats.
  - Saree pallu wrapping varies across regional drape traditions.
- **Slide Graphic:** Side-by-side comparison of `reports/sample_correct.png` (rigid jeans vs correctly identified lehengas).
- **Speaker Notes:**
  > "Why is Western wear easier for a CNN to classify than ethnic wear? Geometry. Jeans have rigid vertical seams, bifurcated legs, and pocket rivets that stay consistent in almost every photograph. Conversely, an ethnic Saree is six yards of unstitched fabric whose visible folds change radically with body posture."

---

## Slide 9: Error Analysis: Primary Confusion Modes
- **The 3 Major Confusion Clusters:**
  1. **Kurta vs. Sherwani (9 Misclassifications):**
     - Both share knee-length tunic silhouettes, long sleeves, and upright mandarin collars.
     - Distinctions (interfacing stiffness, metallic buttons, dense embroidery) are compressed at $224 \times 224$.
  2. **Saree vs. Lehenga (11 Misclassifications):**
     - A pleated saree fall draped around the lower body visually mimics the flare of a bridal lehenga skirt.
  3. **Jacket vs. Shirt (7 Misclassifications):**
     - Unstructured casual blazers share lapel and button placket geometries with open formal shirts.
- **Slide Graphic:** Embed `reports/sample_misclassified.png`.
- **Speaker Notes:**
  > "Examining our misclassifications, 9 Kurtas were predicted as Sherwanis. In South Asian fashion, both garments share the iconic band gala collar and side-slit tunic cut. Distinguishing between them requires fine embroidery texture, which gets smoothed out at standard 224x224 input resolution."

---

## Slide 10: Client-Side Web Deployment (TensorFlow.js)
- **Edge Deployment Architecture:**
  - Exported clean SavedModel $\rightarrow$ Converted via `tensorflowjs_converter`.
  - Chunked into 3 binary shards (`model.json` + $2\times 4\text{ MB} + 0.5\text{ MB} \approx 8.6\text{ MB}$).
- **20-Image Parity Verification (`test_parity.py`):**
  - Tested 20 random test images across all 10 classes in Python vs TF.js graph.
  - Target: Discrepancy $\le 2\%$.
  - **Empirical Max Discrepancy:** **$0.000001$ ($0.0001\%$)** $\rightarrow$ **100% Prediction Match**.
- **User Privacy:**
  - 100% of tensor operations execute inside the user's browser.
  - No wardrobe photos are ever transmitted or stored on an external cloud server.
- **Speaker Notes:**
  > "To deploy this responsibly, we converted the model to a TensorFlow.js Graph Model. We ran a strict 20-image parity test between native Keras and TF.js. The maximum probability difference was less than one in a million. Most importantly, users' personal clothing photos never leave their device."

---

## Slide 11: Live Web Application Demonstration
- **Interactive Features on `afafshahid.github.io/libaas-lens`:**
  - Drag-and-drop, file upload, mobile camera snapshot, clipboard paste (`Ctrl+V`).
  - Garment Hang-Tag with ethnic/Western badges and dynamic top-5 confidence bars.
  - Low-confidence advisory threshold (< 50%).
  - Complete embedded empirical dashboard (Interactive Confusion Matrix & Per-Class Metrics).
  - Ajrak Indigo & Madder textile theme with responsive light/dark modes.
- **Live Demo Protocol:**
  - Test 1: Uploading a traditional embroidered Kurta.
  - Test 2: Uploading a Western denim pair of Jeans.
  - Test 3: Demonstrating real-time latency (< 50ms on modern laptops).
- **Speaker Notes:**
  > "Let us briefly show you the live website running in the browser. You can drop any photo or pick from the sample chips. Here is a Kurta—classified with over 99% confidence in under 40 milliseconds on client hardware."

---

## Slide 12: Conclusion, Limitations & Future Work
- **Summary of Contributions:**
  - Balanced 10-class dataset (2,600 images) bridging Western and Pakistani ethnic fashion.
  - Validated MobileNetV2 transfer learning achieving **88.2% test accuracy**.
  - Documented geometric vs drape perception characteristics of South Asian clothing.
  - Deployed private, client-side web application with 100% mathematical parity.
- **Future Directions:**
  - Higher-resolution backbones (e.g., EfficientNetV2 at $384 \times 384$) to capture micro-embroidery textures.
  - Multi-label classification for layered outfits (e.g., Waistcoat over a Kurta).
  - In-the-wild smartphone photo collection for expanded domain adaptation.
- **Questions & Answers:**
  - Open for faculty questions.
- **Speaker Notes:**
  > "In conclusion, Libaas Lens demonstrates that culturally aware fashion perception is both feasible and deployable at the edge without heavy server infrastructure. We thank our instructor and the panel, and are now happy to take any questions."
