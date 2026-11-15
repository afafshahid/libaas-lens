# Libaas Lens: Oral Defence Questions & Technical Answers

**Course:** CS4045 Deep Learning for Perception  
**Institution:** FAST-NUCES Karachi · Fall 2026  
**Project:** Libaas Lens (Dual-Domain Clothing Type Classification)  
**Authors:** Afaf Shahid (23K-0678) & Ahmad Noorani (23K-0547)

---

### Question 1: Why did you choose MobileNetV2 over heavier backbones like ResNet50 or Vision Transformers (ViT)?
**Answer:**  
Our core system requirement was client-side browser deployment via TensorFlow.js to guarantee 100% user privacy and eliminate cloud server infrastructure costs. MobileNetV2 uses inverted residual blocks with linear bottlenecks and depthwise separable convolutions, which factorize standard convolutions into a $3\times 3$ depthwise convolution and a $1\times 1$ pointwise convolution. This reduces computational complexity from:

$$\mathcal{O}(D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F) \quad \text{to} \quad \mathcal{O}(D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F)$$

This reduces operations by roughly a factor of 8 to 9 with only 3.4 million parameters. Heavier backbones like ResNet-50 (~25M parameters, 100MB+) or Vision Transformers (86M+ parameters) would introduce massive download latencies (exceeding 100MB) and cause severe frame drops or WebGL out-of-memory errors on mobile devices, whereas our MobileNetV2 TF.js graph model is only 8.6 MB and executes in under 50 ms.

---

### Question 2: How did fine-tuning perform compared to the frozen baseline across your 11-class dataset?
**Answer:**  
On our held-out test split of 429 unseen images, fine-tuning the top 35 layers (bottlenecks 14, 15, and 16) with gentle data augmentations (flips, slight rotations, zoom, and contrast) outperformed the frozen linear probe baseline across every single evaluation metric:
- **Overall Accuracy:** Increased from **81.12% to 86.25%** (+5.13% gain).
- **Macro F1-Score:** Increased from **81.09% to 86.38%** (+5.29% gain).
- **Western Accuracy:** Increased from **83.59% to 87.18%** (+3.59% gain).
- **Ethnic Accuracy:** Rose from **79.06% to 85.47%** (+6.41% gain).

By freezing the bottom 120 layers, the network preserved general low-level edge and texture detectors, while adapting high-level convolutional kernels to domain-specific ethnic motifs, embroideries, and drape silhouettes.

---

### Question 3: Why was it essential to keep Batch Normalization layers non-trainable (`layer.trainable = False`) during fine-tuning?
**Answer:**  
During training, a `BatchNormalization` layer maintains two sets of parameters:
1. Trainable affine parameters ($\gamma$ scale, $\beta$ shift) updated via backpropagation.
2. Non-trainable running statistics ($\mu$ moving mean, $\sigma^2$ moving variance) updated using exponential moving averages of the current mini-batch.

When fine-tuning on a specialized dataset with small batch sizes ($N=32$), the mini-batch sample mean and variance fluctuate wildly and deviate significantly from the true population statistics learned across ImageNet's 1.28 million images. If Batch Normalization is set to `trainable = True`, the running statistics rapidly degrade, corrupting the activations of all downstream bottleneck layers. Keeping BatchNorm in inference mode freezes both the statistics and the affine weights, preserving the numerical stability of the pretrained backbone.

---

### Question 4: How did you handle image preprocessing so that the model runs identically in Python and the browser?
**Answer:**  
Typically, developers invoke `tf.keras.applications.mobilenet_v2.preprocess_input` in Python and re-implement the formula in JavaScript:

$$\hat{x} = \frac{x}{127.5} - 1.0 \in [-1.0, 1.0]$$

However, floating-point rounding mismatches or inconsistent image decoding between OpenCV/PIL and HTML5 Canvas often introduce inference discrepancies. In Libaas Lens, we solved this by embedding a Keras `Rescaling(scale=1.0 / 127.5, offset=-1.0)` layer directly as the first internal layer of the network. Because the rescaling node is serialized into the TensorFlow.js computational graph (`model.json`), the browser passes raw `[0, 255]` pixel tensors directly from `tf.browser.fromPixels(canvas)`. This eliminated client-side mathematical divergence and contributed to our $0.000001$ parity discrepancy.

---

### Question 5: Why did you add Shalwar Kameez alongside Waistcoat, and how did that resolve the "Lehenga false positive" problem?
**Answer:**  
In our initial 10-class taxonomy, the only ethnic female garment was *Lehenga*. When testing with real Pakistani women's clothing (e.g. 2-piece printed lawn suits, 3-piece embroidered chiffon suits, or long frocks), the network exhibited a severe failure mode: because it had no concept of a Pakistani female suit/kameez, it forced every female ethnic garment into **Lehenga** due to shared visual cues like volume, pleating, and bright embroidery!

To fix this, we curated 260 balanced images of **Shalwar Kameez**, drawing from authentic Pakistani designer retail catalogs (Khaadi, J., Sapphire, Gul Ahmed, Alkaram) for female 2-pc/3-pc suits and IndoFashion for male kurta-pajama/shalwar sets. We retained **Waistcoat** (*waskat*, 93.7% F1) as a separate category, expanding the system to 11 balanced classes. In empirical testing, Pakistani female suits that previously misclassified as Lehenga now classify as **Shalwar Kameez with 99.2% to 99.9% confidence**.

---

### Question 6: What was the purpose of perceptual hashing (`imagehash.phash`) in your data cleaning pipeline?
**Answer:**  
Public e-commerce retail datasets frequently contain near-duplicate images—for example, the exact same promotional product photo re-uploaded with different background padding, slight crops, minor color grading, or different JPEG compression rates. Standard cryptographic hashes (like SHA-256) only detect byte-for-byte identical files.

Perceptual hashing (pHash) computes the 2D Discrete Cosine Transform (DCT) of the grayscale image, isolating low spatial frequencies that capture visual structure regardless of minor compression or cropping. By computing the Hamming distance between 64-bit DCT hashes and enforcing a threshold of $\le 4$, we detected and eliminated near-duplicates across candidate pools, ensuring that no promotional duplicate of a test image could leak into the training set.

---

### Question 7: How do you explain the balance between Western wear (87.18%) and ethnic apparel (85.47%)?
**Answer:**  
Our final fine-tuned model achieved near-parity across Western (87.18%) and ethnic (85.47%) apparel (a minimal 1.71% difference). This equilibrium was made possible by:
1. **Resolution Harmonization:** Replacing low-res thumbnails with verified $400 \times 534$ real-world photos across Western classes eliminated the frequency shortcut bias that previously depressed Western performance on high-resolution test photos.
2. **Distinct Architectural Profiles:** Western clothes such as *Jeans - Trousers* (100% precision, 96% F1) and *T-shirt* (100% precision, 96% F1) have clear, rigid structural features that the network learns with minimal confusion.
3. **Ethnic Draping Adaptations:** Top-layer fine-tuning allowed the network to learn rich texture and pattern representations, boosting ethnic accuracy by +6.41% over the frozen baseline.

---

### Question 8: Why did Kurtas get confused with Sherwanis and Shalwar Kameez in the confusion matrix?
**Answer:**  
In South Asian sartorial culture, *Kurta*, *Shalwar Kameez*, and *Sherwani* share:
- A knee-length or calf-length tunic silhouette.
- Full-length straight sleeves.
- Side vents/slits (*chaak*).
- An upright mandarin/band collar (*band gala*).

In our test set, some Kurtas were predicted as Sherwani or Shalwar Kameez. The distinguishing visual cues are fabric stiffness, heavy brocade/canvas interfacing, and ornate metallic buttons. At $224\times 224$ downsampled resolution, fine fabric weight cues blur, shifting model attention to the dominant shared collar architecture.

---

### Question 9: How did you verify parity between the Python model and the browser TensorFlow.js model?
**Answer:**  
We authored an automated verification script (`test_parity.py`) evaluating 22 test images (2 from each of the 11 classes) across both the native Keras runtime and the converted TensorFlow SavedModel/TF.js inference graph. Across all 22 images and all 11 output logits:
- **Maximum Probability Discrepancy:** $0.000001$ ($0.0001\%$).
- **Parity Threshold:** Required $\le 2.0\%$.
- **Result:** **100% Top-1 Parity PASSED**.

This proves that our browser-based client application executes the exact same mathematical functions as our offline Python research pipeline.

---

### Question 10: Why were Western images initially misclassified as ethnic apparel, and how did you resolve this shortcut learning trap?
**Answer:**  
In early iterations, the Western classes (`Dress`, `Jacket - Blazer`, `Jeans - Trousers`, `Shirt`, `T-shirt`) were sourced from Kaggle's *Fashion Product Images Small*, where every image was a blurry $60 \times 80$ pixel catalog cutout. In contrast, the ethnic classes (IndoFashion and Pakistani retail) consisted of sharp, high-resolution $600 \times 800+$ photographs.

Because CNNs exploit easy high-frequency statistical shortcuts, MobileNetV2 learned that sharp edges, studio lighting, and fine fabric textures correlated with ethnic wear (`Lehenga`, `Kurta`, `Waistcoat`), while Western classes only activated on blurry, low-resolution cutouts. Consequently, when presented with real-world smartphone photos of Western clothes, the model erroneously predicted ethnic categories!

To solve this, we discarded the low-res Kaggle thumbnails and sourced 260 authentic high-resolution ($400 \times 534$) real-world photographs per Western class from Alexey Grigorev's verified clothing dataset. Retraining on this resolution-symmetrical dataset completely broke the shortcut, boosting Western test accuracy to **87.18%** and ensuring robust real-world generalization.
