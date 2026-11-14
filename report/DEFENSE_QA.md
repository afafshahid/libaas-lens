# Libaas Lens: Oral Defence Questions & Technical Answers

**Course:** CS4045 Deep Learning for Perception  
**Institution:** FAST-NUCES Karachi · Fall 2026  
**Project:** Libaas Lens (Dual-Domain Clothing Type Classification)  
**Authors:** Afaf Shahid (23K-0678) & Ahmad Noorani (23K-0547)

---

### Question 1: Why did you choose MobileNetV2 over heavier backbones like ResNet50 or Vision Transformers (ViT)?
**Answer:**  
Our system requirement was client-side browser deployment via TensorFlow.js to preserve user privacy and eliminate cloud server costs. MobileNetV2 uses inverted residual blocks with linear bottlenecks and depthwise separable convolutions, which factorize standard convolutions into a $3\times 3$ depthwise convolution and a $1\times 1$ pointwise convolution. This reduces computational complexity from:

$$\mathcal{O}(D_K \cdot D_K \cdot M \cdot N \cdot D_F \cdot D_F) \quad \text{to} \quad \mathcal{O}(D_K \cdot D_K \cdot M \cdot D_F \cdot D_F + M \cdot N \cdot D_F \cdot D_F)$$

This reduces operations by roughly a factor of 8 to 9 with only 3.4 million parameters. Heavier backbones like ResNet-50 (~25M parameters, 100MB+) or Vision Transformers (86M+ parameters) would introduce massive download latencies (exceeding 100MB) and cause severe frame drops or WebGL out-of-memory errors on mobile devices, whereas our MobileNetV2 TF.js graph model is only 8.6 MB and executes in under 50 ms.

---

### Question 2: Why did your frozen baseline achieve slightly higher overall accuracy (88.21%) than your fine-tuned model (87.95%)?
**Answer:**  
This is a well-documented phenomenon in transfer learning with small to medium datasets ($N=2,600$ images total; 182 training images per class). When the source domain (ImageNet-1k, 1.28 million natural images) is vastly larger than the target domain, ImageNet features across middle-to-high convolutional layers already encapsulate generalized edges, textures, contours, and fabric weaves.

When unfreezing the top 35 layers, even with a low learning rate ($10^{-5}$) and frozen Batch Normalization, slight feature distortion (representation drift or minor catastrophic forgetting) can occur on smaller training splits. However, fine-tuning improved the macro precision from **88.22% to 88.90%** and boosted Western class accuracy from **92.31% to 93.33%**, showing that while overall top-1 accuracy shifted by just a single misclassified image (344 vs 343 out of 390), the fine-tuned model developed sharper decision boundaries for structured categories.

---

### Question 3: Why was it essential to keep Batch Normalization layers non-trainable (`layer.trainable = False`) during fine-tuning?
**Answer:**  
During training, a `BatchNormalization` layer maintains two sets of parameters:
1. Trainable affine parameters ($\gamma$ scale, $\beta$ shift) updated via backpropagation.
2. Non-trainable running statistics ($\mu$ moving mean, $\sigma^2$ moving variance) updated using exponential moving averages of the current mini-batch.

When fine-tuning on a small custom dataset with a small batch size ($N=32$), the mini-batch sample mean and variance deviate significantly from the true population statistics learned across ImageNet's 1.28 million images. If Batch Normalization is set to `trainable = True`, the running statistics rapidly degrade, corrupting the activations of all downstream bottleneck layers. Keeping BatchNorm in inference mode freezes both the statistics and the affine weights, preserving the numerical stability of the pretrained backbone.

---

### Question 4: How did you handle image preprocessing so that the model runs identically in Python and the browser?
**Answer:**  
Typically, developers invoke `tf.keras.applications.mobilenet_v2.preprocess_input` in Python and re-implement the formula in JavaScript:

$$\hat{x} = \frac{x}{127.5} - 1.0 \in [-1.0, 1.0]$$

However, floating-point rounding mismatches or inconsistent image decoding between OpenCV/PIL and HTML5 Canvas often introduce inference discrepancies. In Libaas Lens, we solved this by embedding a Keras `Rescaling(scale=1.0 / 127.5, offset=-1.0)` layer directly as the first internal layer of the network. Because the rescaling node is serialized into the TensorFlow.js computational graph (`model.json`), the browser passes raw `[0, 255]` pixel tensors directly from `tf.browser.fromPixels(canvas)`. This eliminated client-side mathematical divergence and contributed to our $0.000001$ parity discrepancy.

---

### Question 5: Why did you substitute Shalwar Kameez with Nehru Jacket, and does this compromise your project objectives?
**Answer:**  
In our original proposal, we planned to include *Shalwar Kameez* using the Roboflow Universe "Pakistani Clothes" dataset. However, upon programmatic inspection and bounding-box verification, we found only 93 unique base images depicting complete two-piece suits. Training with only 93 images while other classes had 260 would have created severe class imbalance ($>2.5\times$ disparity) or caused massive overfitting under heavy oversampling.

Our approved proposal established a concrete contingency rule: if Shalwar Kameez yielded under 150 usable images, substitute it with *Nehru Jacket* from IndoFashion (`nehru_jackets`). Nehru Jacket is an authentic, ubiquitous Pakistani and South Asian formal ethnic waistcoat. This allowed us to preserve an exact 260-image cap across all 10 classes (a 1.0x balanced ratio), adhering strictly to empirical rigor while documenting the change transparently.

---

### Question 6: What was the purpose of perceptual hashing (`imagehash.phash`) in your data cleaning pipeline?
**Answer:**  
Public e-commerce retail datasets frequently contain near-duplicate images—for example, the exact same promotional product photo re-uploaded with different background padding, slight crops, minor color grading, or different JPEG compression rates. Standard cryptographic hashes (like SHA-256) only detect byte-for-byte identical files.

Perceptual hashing (pHash) computes the 2D Discrete Cosine Transform (DCT) of the grayscale image, isolating low spatial frequencies that capture visual structure regardless of minor compression or cropping. By computing the Hamming distance between 64-bit DCT hashes and enforcing a threshold of $\le 4$, we detected and eliminated near-duplicates across candidate pools, ensuring that no promotional duplicate of a test image could leak into the training set.

---

### Question 7: How do you explain the 10.77% performance gap between Western wear (93.33%) and ethnic apparel (82.56%)?
**Answer:**  
This gap reflects the fundamental visual perception differences between rigid structural garments and fluid draped fabrics:
1. **Geometric Invariance:** Western clothes such as *Jeans - Trousers* (100% precision and recall) and *T-shirts* have fixed seams, rigid waistbands, and standard flat-lay shapes that vary minimally with posture.
2. **Drape Plasticity:** Ethnic clothes like *Sarees* (66.67% recall) and *Kurtas* (64.10% recall) are non-rigid. A saree is six yards of unstitched fabric whose visible folds, shoulder pleats (*pallu*), and lower pleats change shape dramatically depending on body movement, wind, and wrapping style.
3. **Collar & Silhouette Overlap:** Kurtas and Sherwanis share near-identical vertical silhouettes and mandarin collars, leading to mutual confusion under $224\times 224$ downsampling.

---

### Question 8: Why did 9 out of 39 Kurtas get misclassified as Sherwanis?
**Answer:**  
In South Asian sartorial culture, both the *Kurta* and the *Sherwani* share:
- A knee-length or calf-length tunic silhouette.
- Full-length straight sleeves.
- Side vents/slits (*chaak*).
- An upright mandarin/band collar (*band gala*).

The primary visual discriminators are textile weight (heavier wool/brocade vs lightweight lawn/cotton), internal canvas interfacing, and dense metallic *zari* embroidery. At $224 \times 224$ resolution, subtle micro-textures of threadwork are smoothed by the initial stride-2 convolutions of MobileNetV2, causing the feature representations of long kurtas to overlap significantly with sherwanis in feature space.

---

### Question 9: Why did 11 out of 39 Sarees get misclassified as Lehengas?
**Answer:**  
A *Lehenga* consists of a cropped blouse (*choli*), a long flared skirt, and a draped scarf (*dupatta*). A *Saree* consists of a cropped blouse, an inner petticoat, and a long pleated drape wrapped tightly around the waist and thrown over the shoulder. When photographed from the front in a retail catalog, a saree's pleated lower half closely mimics the flared pleats of a lehenga skirt, and the draped *pallu* resembles a pinned *dupatta*. Without 3D depth cues or side-profile visibility, the 2D spatial arrangement of fabric pleats creates ambiguous visual evidence.

---

### Question 10: How did you ensure that there was no data leakage between your train, validation, and test sets?
**Answer:**  
Data leakage was prevented through three structural safeguards:
1. **Physical Directory Partitioning:** Rather than relying on dynamic runtime splitting (e.g., Keras `validation_split` or unseeded `take`/`skip` on shuffled streams), `prepare_data.py` created fixed, physically isolated folders on disk (`libaas/processed/train`, `val`, `test`).
2. **Perceptual Deduplication Before Splitting:** Deduplication via SHA-256 and DCT pHash was executed on the *entire raw corpus* before creating splits, preventing near-identical catalog variants from landing across both train and test partitions.
3. **Deterministic Random Seed:** Partitioning used a fixed seed (`seed=42`) with stratified class sampling, ensuring complete reproducibility.

---

### Question 11: What is the 20-image parity test, and why was it necessary?
**Answer:**  
When exporting a model from Keras to TensorFlow.js Graph Model format, operations undergo graph optimization, constant folding, and operator fusion (e.g., fusing Conv2D, BiasAdd, and Relu6 into single GPU kernel calls). Furthermore, weights are converted from 32-bit floats into chunked binary shard buffers.

To prove that numerical precision and prediction fidelity were preserved, our `test_parity.py` script ran 20 unseen test images (2 from each class) through both native TensorFlow in Python and the exported graph model. The maximum absolute difference across all output probability distributions was $0.000001$ ($0.0001\%$), confirming 100% top-1 prediction agreement and verifying zero deployment degradation.

---

### Question 12: Why did the TensorFlow.js conversion initially fail with an `Identity is not in graph` error, and how did you resolve it?
**Answer:**  
The fine-tuned model included Keras data augmentation layers (`RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`) defined directly within the `keras.Model` functional graph. While these layers are deactivated during training mode (`training=False`), TensorFlow's internal graph exporter generated stateful random generator variable ops (`AssignVariableOp`) for their pseudo-random seeds.

When `tensorflowjs_converter` attempted to freeze the SavedModel into a static inference graph, these stateful variable operations could not be resolved into constants, triggering a graph freeze failure. We resolved this by building a dedicated inference model topology that preserved the exact backbone and classification head weights but omitted the training-time augmentation layers, creating a clean, purely stateless inference graph that serialized without error.

---

### Question 13: What data augmentations did you use during fine-tuning, and why were they chosen?
**Answer:**  
We applied:
- `RandomFlip("horizontal")`: Garments possess left-right bilateral symmetry; horizontal flipping doubles dataset diversity without corrupting semantic labels.
- `RandomRotation(0.08)` ($\approx \pm 14.4^\circ$): Simulates slight camera tilt and clothing hanger angles.
- `RandomZoom(0.08)` ($\pm 8\%$): Simulates distance variation and cropping differences in consumer photographs.
- `RandomContrast(0.10)`: Simulates varying ambient illumination and exposure levels across indoor retail lighting.

We deliberately avoided vertical flipping (clothes are never worn upside down) and extreme shear or affine warping, which would distort garment proportions (e.g., turning a shirt into a dress).

---

### Question 14: How does your web app handle low-confidence predictions or non-clothing inputs?
**Answer:**  
The web application inspects the top-1 softmax probability output:
- If the top probability is $\ge 50\%$, the predicted garment hang-tag is displayed with standard confidence badges.
- If the top probability falls below $50\%$ ($P < 0.50$), an automated **Advisory Warning Banner** is triggered: *"Low prediction confidence (< 50%). Make sure garment is clearly centered."*
- In addition, the interface dynamically renders the full top-5 probability distribution bars, allowing the user to observe whether the model is debating between closely related categories (e.g., Kurta 48% vs Sherwani 42%).

---

### Question 15: What are the primary limitations of Libaas Lens, and what would you implement next?
**Answer:**  
1. **Single-Garment Assumption:** Libaas Lens currently assumes one primary garment per image. Real-world outfits are multi-layered (e.g., a Nehru Jacket worn over a Kurta). Our next step would be incorporating an object detection head (e.g., YOLOv8-Nano) to predict bounding boxes and simultaneous multi-garment labels.
2. **Catalog Domain Shift:** Our training data comes from clean e-commerce catalogs with white or neutral backgrounds. Performance on crumpled clothes lying on a bed or captured under harsh shadows requires domain adaptation via contrastive learning (SimCLR/MoCo) on in-the-wild smartphone photos.
3. **Resolution Bottleneck:** Elevating input resolution to $384 \times 384$ using an architecture like EfficientNetV2-S would allow the network to resolve micro-embroidery stitches that disambiguate Kurtas from Sherwanis.
