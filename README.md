# Multicut-Inspired Region-Adaptive Image Compression

This repository contains a **research-oriented implementation of a region-adaptive image compression pipeline**
inspired by **Multicut / correlation clustering**, but optimized for **computational efficiency and interpretability**.

The method performs **content-adaptive bit-depth selection per region** based on an explicit
**rate–distortion objective**, and produces a globally consistent partition using a **graph-based merging strategy**.

This project reflects the **latest version** of the implementation and matches the final presentation and report.

---

## ✨ Key Features

- **Region-adaptive compression**
  - Different image regions are encoded with different bit-depths.
- **Multicut-inspired formulation**
  - Regions are represented as nodes in a Region Adjacency Graph (RAG).
  - Edge merge costs are derived from region-level rate–distortion costs.
- **Efficient implementation**
  - Global 4×4 DCT + quantization + IDCT is executed **only once per channel**.
  - Region costs are evaluated using **mergeable sufficient statistics**, avoiding repeated pixel access.
- **Discrete optimization**
  - Bit-depth is selected from a discrete candidate set (e.g. {3,4,5,6}).
- **Scalable batch evaluation**
  - Designed to process large image collections efficiently.

---

## 📌 Method Overview

The pipeline consists of the following steps:

1. **Superpixel Segmentation**
   - The input image is partitioned into superpixels (SLIC, with grid fallback).

2. **Region Adjacency Graph (RAG) Construction**
   - An 8-neighborhood adjacency graph is built over superpixels.

3. **Global DCT Reconstruction (Once per Channel)**
   - The image is transformed using 4×4 DCT blocks.
   - Quantization and inverse DCT are applied globally to obtain a reconstructed image.

4. **Candidate Bit-Depth Evaluation**
   - A discrete set of bit-depths (e.g. b ∈ {3,4,5,6}) is evaluated.
   - Posterization is implemented via precomputed LUTs.

5. **Region Cost Definition**
   - For each region S and bit-depth b, the cost is defined as:
     \[
     C(S,b) = R(S,b) + \lambda D(S,b)
     \]
     where:
     - R(S,b) is a histogram-based entropy proxy (rate term),
     - D(S,b) is the mean squared error over region pixels (distortion term).

6. **Best Bit-Depth Selection**
   - For each region:
     \[
     b^*(S) = \arg\min_b C(S,b)
     \]

7. **Edge Merge Cost Computation**
   - For each adjacent region pair (u,v):
     \[
     \Delta C(u,v) = C(u \cup v) - C(u) - C(v)
     \]

8. **Graph-Based Region Merging**
   - Regions are merged according to edge costs using a greedy graph contraction strategy.
   - The result is a globally consistent partition.

9. **Final Bit-Depth Selection per Final Region**
   - Statistics from merged superpixels are aggregated.
   - The optimal bit-depth is re-selected for each final region.

10. **Final Reconstruction**
    - Region-wise posterization is applied to the global reconstructed image.

---

## 📊 Rate–Distortion Modeling

### Distortion Term

For a region S and bit-depth b, distortion is measured as mean squared error:
\[
D(S,b) = \frac{1}{|S|} \sum_{p \in S} (I(p) - \hat{I}_b(p))^2
\]

### Rate Term

The rate term is approximated using a histogram-based entropy proxy:
\[
R(S,b) = N \log N - \sum_i n_i \log n_i
\]
where nᵢ denotes histogram bin counts of quantized intensities.

---

## 🧮 Implementation Highlights

- **Mergeable sufficient statistics**
  - Pixel count
  - Squared error sums
  - Intensity histograms
- **No repeated DCT per region**
- **No per-edge pixel scans**
- **Union–Find for final region labeling**

---

## 📁 Output and Evaluation

For each image, the pipeline outputs:

- A compressed image stored using a standard container format (PNG).
- Per-image metrics written to `metrics.csv`:
  - original size
  - compressed size
  - compression ratio
- Aggregated statistics written to `summary.csv`.

An optional decoding step verifies output correctness by reconstructing images into raw pixel formats.

---

## 🚀 How to Run

```bash
python main.py --input_root path/to/images --output_root path/to/output
