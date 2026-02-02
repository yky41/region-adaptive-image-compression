# Multicut-Inspired Region-Adaptive Image Compression

This repository provides a **research-oriented implementation of a region-adaptive image compression framework**
inspired by **Multicut / correlation clustering**, with a strong emphasis on:

- explicit **rate–distortion modeling**
- **region-wise adaptive bit-depth selection**
- **efficient and scalable implementation**

The implementation corresponds to the **latest version** of the project and is fully aligned with
the final presentation, experimental setup, and report.

---

## 📌 Motivation

Conventional image compression methods apply **uniform encoding parameters** over the entire image.
However, natural images exhibit **strong spatial heterogeneity**:

- smooth regions tolerate aggressive compression,
- textured or edge-rich regions require higher fidelity.

This project explores a **region-adaptive compression strategy** that:
- partitions the image into regions,
- selects an optimal bit-depth per region,
- enforces global consistency via a graph-based merging scheme.

---

## 🧠 Core Idea (High-Level)

1. Partition the image into small regions (superpixels).
2. Model compression quality using a **rate–distortion objective**.
3. Decide whether neighboring regions should be merged based on cost differences.
4. Solve the resulting partitioning problem approximately using **GAEC**.
5. Apply region-wise bit-depth reduction via **lookup tables (LUTs)**.

---

## ✨ Key Features

- **Region-adaptive compression**
  - Each final region is assigned its own optimal bit-depth.
- **Multicut-inspired formulation**
  - Regions are nodes in a Region Adjacency Graph (RAG).
  - Merge decisions are driven by additive cost differences.
- **Efficient implementation**
  - Global 4×4 DCT + quantization + IDCT is executed **only once per channel**.
  - Region costs are computed using **mergeable sufficient statistics**.
- **Discrete optimization**
  - Bit-depth is selected from a discrete candidate set (e.g. {3,4,5,6,7}).
- **GAEC solver**
  - Greedy Additive Edge Contraction is used as a scalable multicut approximation.
- **Batch processing & evaluation**
  - Designed for large image collections.
  - Automatically produces CSV summaries.

---

## 🔧 Method Pipeline (Detailed)

### Step 1: Superpixel Segmentation
The input image is first segmented into superpixels using **SLIC**.
If SLIC is unavailable, a grid-based fallback is used.

Each superpixel serves as an initial region node.

---

### Step 2: Region Adjacency Graph (RAG)
An **8-neighborhood RAG** is constructed over superpixels.

- Nodes: superpixels
- Edges: spatial adjacency

---

### Step 3: Global DCT Reconstruction (Once per Channel)

The image is processed using **4×4 block DCT**:

1. Forward DCT
2. Quantization
3. Inverse DCT

This step is executed **once globally per channel**, producing a reconstructed reference image.
No per-region DCT is performed.

---

### Step 4: Candidate Bit-Depth Evaluation

A discrete set of candidate bit-depths is considered:

\[
b \in \{3,4,5,6,7\}
\]

For each bit-depth:
- a **posterization LUT** is constructed,
- reconstructed intensities are quantized using the LUT.

---

### Step 5: Rate–Distortion Cost Definition

For a region \( S \) and bit-depth \( b \), the cost is defined as:

\[
C(S,b) = R(S,b) + \lambda D(S,b)
\]

#### Distortion term
\[
D(S,b) = \frac{1}{|S|} \sum_{p \in S} (I(p) - \hat{I}_b(p))^2
\]

#### Rate term (entropy proxy)
\[
R(S,b) = N \log_2 N - \sum_i n_i \log_2 n_i
\]

where \( n_i \) are histogram bin counts of quantized intensities.

---

### Step 6: Best Bit-Depth per Superpixel
For each superpixel:
\[
b^*(S) = \arg\min_b C(S,b)
\]

---

### Step 7: Edge Merge Cost
For each adjacent region pair \( (u,v) \):

\[
\Delta C(u,v) = C(u \cup v) - C(u) - C(v)
\]

- Negative cost → merging is beneficial
- Positive cost → keep regions separate

---

### Step 8: Graph-Based Region Merging (GAEC)

The multicut problem is approximately solved using
**Greedy Additive Edge Contraction (GAEC)**:

- repeatedly contract the most negative edge,
- update neighboring edge costs additively,
- stop when no negative edge remains.

---

### Step 9: Final Bit-Depth Selection
After merging:
- sufficient statistics are aggregated per final region,
- the optimal bit-depth is re-selected.

---

### Step 10: Final Reconstruction
The final image is reconstructed by applying
region-wise posterization on the global reconstructed image.

---

## 🖼️ Compression and Decompression Example

The following example shows the effect of the proposed method.

<p align="center">
  <img src="figures/results/example_original.png" width="45%">
  <img src="figures/results/example_reconstructed.png" width="45%">
</p>

**Left:** Original image  
**Right:** Image after compression and decompression

The reconstruction is **lossy** by design due to adaptive bit-depth reduction.
The goal is to preserve perceptually important structures while reducing storage cost.

---

## 📊 Output Files

For each image, the pipeline produces:

- **Compressed image**
  - Stored using standard container formats (PNG / JPEG / WebP).
- **metrics.csv**
  - One row per image
  - Includes original size, compressed size, and saved percentage.
- **summary.csv**
  - Aggregated statistics over the dataset
  - Mean, median, global saved percentage, and win-rate.

---

## 📁 Project Structure

```text
.
├─ main.py
├─ README.md
│
├─ figures/
│   ├─ csv_statistics.png
│   └─ results/
│       ├─ example_original.png
│       └─ example_reconstructed.png
