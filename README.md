# Multicut-Inspired Region-Adaptive Image Compression

This repository contains a **research-oriented implementation of a region-adaptive image compression pipeline**
inspired by **Multicut / correlation clustering**, and implemented with a strong focus on
**computational efficiency, interpretability, and reproducibility**.

The method performs **content-adaptive bit-depth selection per image region**
based on an explicit **rate–distortion objective**, and produces a globally consistent
segmentation using a **graph-based greedy merging strategy (GAEC)**.

This repository reflects the **latest version of the project**, and matches the
final presentation and report.

---

## ✨ Key Features

- **Region-adaptive compression**
  - Different image regions are encoded with different bit-depths.
- **Multicut-inspired formulation**
  - Regions are represented as nodes in a Region Adjacency Graph (RAG).
  - Edge merge costs are derived from region-level rate–distortion costs.
- **Efficient implementation**
  - Global 4×4 DCT + quantization + IDCT is executed **only once per channel**.
  - Region costs are evaluated using **mergeable sufficient statistics**,
    avoiding repeated pixel access.
- **Discrete optimization**
  - Bit-depth is selected from a discrete candidate set (e.g. {3,4,5,6,7}).
- **Greedy Additive Edge Contraction (GAEC)**
  - A standard and scalable multicut approximation algorithm.
- **Scalable batch evaluation**
  - Designed to process large image collections efficiently.
- **Quantitative evaluation**
  - Automatic generation of per-image and global CSV statistics.

---

## 📌 Method Overview

The pipeline consists of the following steps:

1. **Superpixel Segmentation**
   - The input image is partitioned into superpixels using SLIC
     (with a grid-based fallback).

2. **Region Adjacency Graph (RAG) Construction**
   - An 8-neighborhood adjacency graph is built over superpixels.

3. **Global DCT Reconstruction (Once per Channel)**
   - The image is transformed using 4×4 DCT blocks.
   - Quantization and inverse DCT are applied globally to obtain
     a reconstructed reference image.

4. **Candidate Bit-Depth Evaluation**
   - A discrete set of bit-depths (e.g. b ∈ {3,4,5,6,7}) is evaluated.
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

8. **Graph-Based Region Merging (GAEC)**
   - Regions are merged greedily according to negative edge costs.
   - Edge costs are updated additively after each contraction.

9. **Final Bit-Depth Selection per Final Region**
   - Statistics from merged superpixels are aggregated.
   - The optimal bit-depth is re-selected for each final region.

10. **Final Reconstruction**
    - Region-wise posterization is applied to the globally reconstructed image.

---

## 🔍 Compression and Decompression Result

The following example illustrates the effect of the proposed compression method.
The left image shows the original input image, while the right image shows the
reconstructed result after compression and decompression.

<p align="center">
  <img src="figures/results/example_original.png" width="45%">
  <img src="figures/results/example_reconstructed.png" width="45%">
</p>

**Left:** Original image  
**Right:** Image after compression and decompression using the proposed method

Note that the reconstruction is **lossy** due to region-wise bit-depth reduction.
The goal is not exact pixel recovery, but preservation of perceptually important
structures under reduced storage cost.

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
R(S,b) = N \log_2 N - \sum_i n_i \log_2 n_i
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
- **Union–Find with controlled root selection**
- **Priority-queue-based GAEC implementation**

---

## 📁 Output and Evaluation

For each image, the pipeline produces:

- A compressed image stored using a standard container format
  (PNG / JPEG / WebP).
- Per-image metrics written to `metrics.csv`:
  - original file size
  - compressed file size
  - saved percentage
- Aggregated statistics written to `summary.csv`:
  - global saved percentage
  - mean / median saved percentage
  - win-rate compared to baselines

An optional decoding step reconstructs images into raw formats
(e.g. BMP) to verify correctness.

---

## 📊 Quantitative Evaluation

<p align="center">
  <img src="figures/csv_statistics.png" width="70%">
</p>

---

## 🚀 How to Run

```bash
python main.py
