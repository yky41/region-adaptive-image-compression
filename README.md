# Multicut-Inspired Region-Adaptive Image Compression

This repository contains a **research-oriented implementation of a region-adaptive image compression pipeline**
inspired by **Multicut / correlation clustering**, designed for **computational efficiency, interpretability, and scalability**.

The method performs **content-adaptive bit-depth selection per region** based on an explicit
**rate–distortion objective**, and produces a globally consistent partition using a **graph-based merging strategy (GAEC)**.

This repository reflects the **latest version** of the implementation and is consistent with the final presentation and report.

---

## ✨ Key Features

- **Region-adaptive compression**
  - Different image regions are encoded with different bit-depths.
- **Multicut-inspired formulation**
  - Regions are nodes in a Region Adjacency Graph (RAG).
  - Edge merge costs are derived from region-level rate–distortion costs.
- **Efficient implementation**
  - Global 4×4 DCT + quantization + IDCT is executed **only once per channel**.
  - Region costs are evaluated using **mergeable sufficient statistics**.
- **Discrete optimization**
  - Bit-depth is selected from a discrete candidate set (e.g. {3,4,5,6}).
- **Scalable batch evaluation**
  - Designed for large-scale image collections.

---

## 📌 Method Overview

![Pipeline overview](figures/overview_pipeline.png)

The pipeline consists of the following steps:

1. Superpixel segmentation (SLIC, with grid fallback)
2. Region Adjacency Graph (8-neighborhood)
3. Global 4×4 DCT reconstruction (once per channel)
4. Candidate bit-depth evaluation via LUT-based posterization
5. Region cost definition using a rate–distortion objective
6. Per-region optimal bit-depth selection
7. Edge cost computation from merged region costs
8. Graph-based region merging (GAEC)
9. Final bit-depth selection per merged region
10. Final region-wise reconstruction

---

## 🔍 Visual Results

![Reconstruction comparison](figures/example_reconstruction.png)

The proposed method preserves structural details in complex regions
while aggressively reducing bit-depth in homogeneous areas.

---

## 📊 Rate–Distortion Modeling

### Distortion Term

For a region \(S\) and bit-depth \(b\), distortion is measured as mean squared error:

\[
D(S,b) = \frac{1}{|S|} \sum_{p \in S} \left(I(p) - \hat{I}_b(p)\right)^2
\]

### Rate Term

The rate term is approximated using a histogram-based entropy proxy:

\[
R(S,b) = N \log N - \sum_i n_i \log n_i
\]

where \(n_i\) denotes histogram bin counts of quantized intensities.

---

## 🧮 Implementation Highlights

- Mergeable sufficient statistics:
  - Pixel count
  - Squared error sums
  - Intensity histograms
- No repeated DCT per region
- No per-edge pixel scans
- Union–Find for final region labeling
- GAEC for approximate multicut optimization

---

## 📁 Output and Evaluation

For each image, the pipeline outputs:

- A compressed image (PNG container)
- Per-image statistics in `metrics.csv`
- Aggregated statistics in `summary.csv`

---

## 📊 Quantitative Evaluation

![CSV statistics](figures/csv_statistics.png)

---

## 🚀 How to Run

```bash
python main.py --input_root path/to/images --output_root path/to/output
