# -*- coding: utf-8 -*-
"""
Multicut + Baselines (WIDE CSV ONLY + OVERALL SUMMARY)
+ 8邻域
+ GAEC (Greedy Additive Edge Contraction) 作为 Multicut 近似求解器（替代 Triangle Repair）

[FAST VERSION]
- 全图每个通道只做一次 DCT/IDCT
- 区域/边代价用统计量合并（不做 per-edge DCT）

输出：
1) metrics.csv（宽表，一张图一行）
dataset, rel_path, in_format, orig_bytes,
multicut_out_bytes, multicut_saved_percent,
baseline_png_out_bytes, baseline_png_saved_percent,
baseline_jpeg_out_bytes, baseline_jpeg_saved_percent,
baseline_webp_out_bytes, baseline_webp_saved_percent

2) summary.csv（整体对比，4行：multicut/png/jpeg/webp）
包含：total_out_bytes/global_saved_percent/mean/median/win_rate 等整体指标
"""

from __future__ import annotations
import os, math, shutil, heapq
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ---------------------------
# 4x4 DCT/IDCT (no SciPy)
# ---------------------------
def dct_matrix(N: int) -> np.ndarray:
    M = np.zeros((N, N), dtype=np.float64)
    alpha0 = math.sqrt(1.0 / N)
    alpha = math.sqrt(2.0 / N)
    for k in range(N):
        for n in range(N):
            if k == 0:
                M[k, n] = alpha0
            else:
                M[k, n] = alpha * math.cos(math.pi * (2*n + 1) * k / (2*N))
    return M

_DCT4 = dct_matrix(4)
_IDCT4 = _DCT4.T

Q4_DEFAULT = np.array([
    [8, 16, 19, 22],
    [16, 16, 22, 24],
    [19, 22, 26, 27],
    [22, 24, 27, 29]
], dtype=np.float64)

def pad_to_multiple_of_4(img2d: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = img2d.shape[:2]
    ph = (4 - (h % 4)) % 4
    pw = (4 - (w % 4)) % 4
    if ph == 0 and pw == 0:
        return img2d, (0, 0)
    out = np.pad(img2d, ((0, ph), (0, pw)), mode="edge")
    return out, (ph, pw)

def dct_reconstruct_full_channel_u8(ch_u8: np.ndarray, Q4: np.ndarray) -> np.ndarray:
    """
    全图通道一次性：4x4 block DCT -> 量化 -> 反量化 -> IDCT
    用 einsum 批处理 blocks，避免 Python 循环。
    """
    ch = ch_u8.astype(np.float64, copy=False)
    pad, (ph, pw) = pad_to_multiple_of_4(ch)
    hp, wp = pad.shape
    bh, bw = hp // 4, wp // 4

    blocks = pad.reshape(bh, 4, bw, 4).transpose(0, 2, 1, 3)  # (bh,bw,4,4)

    # DCT: M @ block @ M.T
    C = np.einsum("ij,abjk,kl->abil", _DCT4, blocks, _DCT4.T)

    Cq = np.rint(C / Q4).astype(np.int32)
    Cdeq = Cq.astype(np.float64) * Q4

    # IDCT: M.T @ coeff @ M
    recon = np.einsum("ij,abjk,kl->abil", _DCT4.T, Cdeq, _DCT4)
    recon = recon.transpose(0, 2, 1, 3).reshape(hp, wp)

    recon = recon[:hp-ph if ph else hp, :wp-pw if pw else wp]
    return np.clip(np.rint(recon), 0, 255).astype(np.uint8)


# ---------------------------
# Step 1: SLIC superpixels (or fallback)
# ---------------------------
def compute_superpixels(image: np.ndarray, n_segments: int = 200, compactness: float = 10.0) -> np.ndarray:
    """
    返回 labels (H,W), int32, 从 0 开始。
    """
    H, W = image.shape[:2]
    try:
        from skimage.segmentation import slic
        img = image.astype(np.float32)

        # skimage 新版需要 channel_axis 参数
        if img.ndim == 3:
            labels = slic(img, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1)
        else:
            labels = slic(img, n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=None)
        return labels.astype(np.int32)

    except Exception:
        # fallback grid
        grid = int(math.sqrt(n_segments))
        grid = max(1, grid)
        gh = max(1, H // grid)
        gw = max(1, W // grid)
        labels = np.zeros((H, W), dtype=np.int32)
        lab = 0
        for r0 in range(0, H, gh):
            for c0 in range(0, W, gw):
                r1 = min(H, r0 + gh)
                c1 = min(W, c0 + gw)
                labels[r0:r1, c0:c1] = lab
                lab += 1
        uniq = np.unique(labels)
        remap = {u: i for i, u in enumerate(uniq.tolist())}
        out = np.vectorize(remap.get)(labels).astype(np.int32)
        return out


# ---------------------------
# Step 2: Build RAG adjacency (8-neighborhood)
# edges are unordered sets {u,v} -> store as (min,max)
# ---------------------------
def build_rag(labels: np.ndarray, neighborhood: int = 8) -> Tuple[int, List[Tuple[int,int]]]:
    H, W = labels.shape
    K = int(labels.max()) + 1
    edges = set()

    def add_edge(u: int, v: int):
        if u == v:
            return
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))

    for i in range(H):
        for j in range(W):
            u = int(labels[i, j])
            if j + 1 < W:
                add_edge(u, int(labels[i, j+1]))
            if i + 1 < H:
                add_edge(u, int(labels[i+1, j]))
            if neighborhood == 8:
                if i + 1 < H and j + 1 < W:
                    add_edge(u, int(labels[i+1, j+1]))
                if i + 1 < H and j - 1 >= 0:
                    add_edge(u, int(labels[i+1, j-1]))

    return K, sorted(edges)


# ---------------------------
# FAST: build flat indices per label in ONE pass (sort bucket)
# idx_sorted[starts[i]:ends[i]] are flat positions for label i
# ---------------------------
def build_indices_by_label(labels: np.ndarray, K: int):
    flat = labels.ravel().astype(np.int32, copy=False)
    idx = np.arange(flat.size, dtype=np.int32)

    order = np.argsort(flat, kind="mergesort")   # stable sort
    flat_sorted = flat[order]
    idx_sorted = idx[order]

    labs = np.arange(K, dtype=np.int32)
    starts = np.searchsorted(flat_sorted, labs, side="left")
    ends   = np.searchsorted(flat_sorted, labs, side="right")
    return idx_sorted, starts, ends


# ---------------------------
# posterize LUT (FAST)
# ---------------------------
def build_posterize_lut(bits: int) -> np.ndarray:
    bits = int(bits)
    if bits >= 8:
        return np.arange(256, dtype=np.uint8)
    bits = max(1, bits)
    if bits == 1:
        step = 255.0
    else:
        levels = (1 << bits) - 1
        step = 255.0 / levels
    x = np.arange(256, dtype=np.float64)
    y = np.rint(np.rint(x / step) * step)
    y = np.clip(y, 0, 255).astype(np.uint8)
    return y


# ---------------------------
# Save helpers
# ---------------------------
def save_as_format(
    pil_img: Image.Image,
    out_path: str,
    out_format: str,
    png_compress_level: int = 9,
    jpeg_quality: int = 75,
    webp_quality: int = 75,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fmt = (out_format or "").upper()
    if fmt == "PNG":
        pil_img.save(out_path, format="PNG", optimize=True, compress_level=png_compress_level)
    elif fmt in ("JPG", "JPEG"):
        pil_img.convert("RGB").save(out_path, format="JPEG", quality=jpeg_quality, optimize=True, progressive=True)
    elif fmt == "WEBP":
        pil_img.save(out_path, format="WEBP", quality=webp_quality, method=6)
    else:
        pil_img.save(out_path, format=fmt if fmt else None)

def save_png_best(pil_img: Image.Image, out_path: str, compress_level: int = 9) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img = pil_img
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGBA")

    tmp_dir = os.path.dirname(out_path)
    base = os.path.basename(out_path)

    tmp_rgb = os.path.join(tmp_dir, base + ".__tmp_rgb.png")
    tmp_pal = os.path.join(tmp_dir, base + ".__tmp_pal.png")

    img.save(tmp_rgb, format="PNG", optimize=True, compress_level=compress_level)

    try:
        pal = img.convert(
            "P",
            palette=Image.Palette.ADAPTIVE,
            colors=256,
            dither=Image.Dither.NONE
        )
        pal.save(tmp_pal, format="PNG", optimize=True, compress_level=compress_level)

        b_rgb = os.path.getsize(tmp_rgb)
        b_pal = os.path.getsize(tmp_pal)

        if b_pal <= b_rgb:
            os.replace(tmp_pal, out_path)
            if os.path.exists(tmp_rgb):
                os.remove(tmp_rgb)
        else:
            os.replace(tmp_rgb, out_path)
            if os.path.exists(tmp_pal):
                os.remove(tmp_pal)

    except Exception:
        os.replace(tmp_rgb, out_path)
        if os.path.exists(tmp_pal):
            try:
                os.remove(tmp_pal)
            except Exception:
                pass


# ---------------------------
# FAST entropy from hist counts
# returns H*N = N log2 N - sum n_i log2 n_i
# ---------------------------
def entropy_bits_from_hist(hist: np.ndarray) -> float:
    N = float(hist.sum())
    if N <= 0:
        return 0.0
    nz = hist > 0
    hn = float(N * math.log2(N) - np.sum(hist[nz] * np.log2(hist[nz])))
    return hn


# ---------------------------
# Union-Find (可控根节点：union_keep 保证 keep 做 root)
# ---------------------------
class UnionFindKeep:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, x: int) -> int:
        # 路径压缩
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return int(x)

    def union_keep(self, keep: int, other: int) -> int:
        """
        强制把 other_root 挂到 keep_root 上，保证 keep_root 仍为根。
        返回合并后的 root（= keep_root）。
        """
        rk = self.find(keep)
        ro = self.find(other)
        if rk == ro:
            return rk
        self.parent[ro] = rk
        self.size[rk] += self.size[ro]
        return rk


# ---------------------------
# GAEC: Greedy Additive Edge Contraction (老师点名的标准近似算法)
# ---------------------------
def gaec_unionfind(
    K: int,
    edge_costs: List[Tuple[float, int, int]],
    stop_when_nonnegative: bool = True,
) -> UnionFindKeep:
    """
    edge_costs: (c, u, v). c<0 => encourage merge. c>=0 => discourage merge.
    GAEC:
      - repeatedly pick most negative edge between current components
      - contract that edge
      - update costs additively: c_{A∪B, X} = c_{A,X} + c_{B,X}
      - stop when best edge >= 0 (optional)
    """
    uf = UnionFindKeep(K)

    # adjacency cost dict per component root:
    # adj[u][v] = cost, maintained symmetric
    adj: List[Dict[int, float]] = [defaultdict(float) for _ in range(K)]
    for c, u, v in edge_costs:
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        adj[a][b] += float(c)
        adj[b][a] += float(c)

    # heap keeps candidates (cost, u, v) with u < v, cost is "current at push time" (may become stale)
    heap: List[Tuple[float, int, int]] = []
    for u in range(K):
        for v, c in adj[u].items():
            if u < v:
                heapq.heappush(heap, (c, u, v))

    while heap:
        c, u, v = heapq.heappop(heap)
        ru, rv = uf.find(u), uf.find(v)
        if ru == rv:
            continue

        a, b = (ru, rv) if ru < rv else (rv, ru)
        cur = adj[a].get(b, None)
        if cur is None:
            continue
        # stale check
        if abs(cur - c) > 1e-12:
            continue

        if stop_when_nonnegative and cur >= 0.0:
            break

        # choose keep root by adjacency size (merge small into large)
        if len(adj[ru]) < len(adj[rv]):
            keep, kill = rv, ru
        else:
            keep, kill = ru, rv

        # remove keep<->kill edge first if exists
        kk_a, kk_b = (keep, kill) if keep < kill else (kill, keep)
        if kill in adj[keep]:
            del adj[keep][kill]
        if keep in adj[kill]:
            del adj[kill][keep]

        # contract: force keep as root
        uf.union_keep(keep, kill)

        # move all neighbors of kill into keep additively
        kill_items = list(adj[kill].items())
        for t, c_kt in kill_items:
            rt = uf.find(t)
            if rt == keep or rt == kill:
                continue

            # remove kill<->rt
            if rt in adj[kill]:
                del adj[kill][rt]
            if kill in adj[rt]:
                del adj[rt][kill]

            # add cost to keep<->rt
            newc = adj[keep].get(rt, 0.0) + float(c_kt)
            adj[keep][rt] = newc
            adj[rt][keep] = newc

            x, y = (keep, rt) if keep < rt else (rt, keep)
            heapq.heappush(heap, (adj[x][y], x, y))

        adj[kill].clear()

    return uf


# ---------------------------
# FAST region stats per superpixel and per bits
# Store: N, sumsq_total, hists_per_channel (C x 256)
# ---------------------------
@dataclass
class BitsStats:
    N: int
    sumsq_total: float
    hists: np.ndarray  # shape (C,256), int64

@dataclass
class BestCost:
    C_total: float
    best_bits: int


def compute_superpixel_stats_all_bits_from_flatidx(
    orig_channels_u8: List[np.ndarray],
    recon_channels_u8: List[np.ndarray],
    flat_idx: np.ndarray,
    candidate_bits: Tuple[int, ...],
    luts: Dict[int, np.ndarray],
) -> Dict[int, BitsStats]:
    """
    flat_idx: indices into ravel() space (0..H*W-1)
    """
    N = int(flat_idx.size)
    C = len(orig_channels_u8)
    out: Dict[int, BitsStats] = {}

    if N == 0:
        z = np.zeros((C, 256), dtype=np.int64)
        for bits in candidate_bits:
            out[int(bits)] = BitsStats(N=0, sumsq_total=0.0, hists=z.copy())
        return out

    orig_vals = [np.take(ch.ravel(), flat_idx) for ch in orig_channels_u8]
    rec_vals  = [np.take(ch.ravel(), flat_idx) for ch in recon_channels_u8]

    for bits in candidate_bits:
        lut = luts[int(bits)]
        sumsq_total = 0.0
        hists = np.zeros((C, 256), dtype=np.int64)

        for ci in range(C):
            rq = lut[rec_vals[ci]]
            diff = orig_vals[ci].astype(np.float64) - rq.astype(np.float64)
            sumsq_total += float(np.dot(diff, diff))
            hists[ci] = np.bincount(rq, minlength=256).astype(np.int64)

        out[int(bits)] = BitsStats(N=N, sumsq_total=sumsq_total, hists=hists)

    return out


def best_cost_from_stats(
    stats_by_bits: Dict[int, BitsStats],
    lam: float,
    C: int,
    candidate_bits: Tuple[int, ...],
) -> BestCost:
    best_C = float("inf")
    best_bits = int(candidate_bits[0])
    for bits in candidate_bits:
        st = stats_by_bits[int(bits)]
        if st.N <= 0:
            C_total = 0.0
        else:
            D_mse = st.sumsq_total / (st.N * C)
            R_bits = 0.0
            for ci in range(C):
                R_bits += entropy_bits_from_hist(st.hists[ci])
            C_total = float(R_bits + lam * D_mse)

        if C_total < best_C:
            best_C = C_total
            best_bits = int(bits)

    return BestCost(C_total=best_C, best_bits=best_bits)


def merge_two_stats(a: BitsStats, b: BitsStats) -> BitsStats:
    return BitsStats(
        N=int(a.N + b.N),
        sumsq_total=float(a.sumsq_total + b.sumsq_total),
        hists=(a.hists + b.hists)
    )


# ---------------------------
# Multicut compress (FAST + GAEC)
# ---------------------------
@dataclass
class MulticutResult:
    recon_u8: np.ndarray


def multicut_compress_image_fast(
    img_u8: np.ndarray,
    n_superpixels: int = 200,
    lam: float = 0.05,
    keep_alpha_lossless: bool = True,
    candidate_bits: Tuple[int, ...] = (3, 4, 5, 6, 7),
    rag_neighborhood: int = 8,
    solver: str = "gaec",   # "gaec"（老师建议） 或 "triangle"(已删，不推荐)
) -> MulticutResult:
    alpha = None
    if img_u8.ndim == 2:
        orig_channels_u8 = [img_u8.astype(np.uint8)]
        mode = "L"
    else:
        C0 = img_u8.shape[2]
        if C0 == 4 and keep_alpha_lossless:
            orig_channels_u8 = [img_u8[..., 0].astype(np.uint8),
                                img_u8[..., 1].astype(np.uint8),
                                img_u8[..., 2].astype(np.uint8)]
            alpha = img_u8[..., 3].copy()
            mode = "RGBA"
        elif C0 == 2 and keep_alpha_lossless:
            orig_channels_u8 = [img_u8[..., 0].astype(np.uint8)]
            alpha = img_u8[..., 1].copy()
            mode = "LA"
        else:
            orig_channels_u8 = [img_u8[..., c].astype(np.uint8) for c in range(C0)]
            mode = "RGB" if C0 == 3 else None

    C = len(orig_channels_u8)

    # 1) SLIC
    labels_super = compute_superpixels(img_u8, n_segments=n_superpixels)

    # 2) 8-neighborhood RAG
    K, edges = build_rag(labels_super, neighborhood=rag_neighborhood)

    # 2.5) Build flat indices per superpixel
    idx_sorted, starts, ends = build_indices_by_label(labels_super, K)

    # 3) Global DCT recon once per channel
    recon_channels_u8 = [dct_reconstruct_full_channel_u8(ch, Q4_DEFAULT) for ch in orig_channels_u8]

    # LUT cache
    luts = {int(b): build_posterize_lut(int(b)) for b in candidate_bits}

    # 4) per-superpixel stats for all bits
    stats_super: List[Dict[int, BitsStats]] = []
    best_super: List[BestCost] = []
    for v in range(K):
        flat_idx = idx_sorted[starts[v]:ends[v]]
        st = compute_superpixel_stats_all_bits_from_flatidx(
            orig_channels_u8, recon_channels_u8,
            flat_idx,
            candidate_bits=candidate_bits,
            luts=luts
        )
        stats_super.append(st)
        best_super.append(best_cost_from_stats(st, lam=lam, C=C, candidate_bits=candidate_bits))

    # 5) edge costs via stats merge
    edge_costs: List[Tuple[float, int, int]] = []
    for (u, v) in edges:
        best_merge_C = float("inf")
        for bits in candidate_bits:
            st_uv = merge_two_stats(stats_super[u][int(bits)], stats_super[v][int(bits)])
            if st_uv.N <= 0:
                C_uv = 0.0
            else:
                D_mse = st_uv.sumsq_total / (st_uv.N * C)
                R_bits = 0.0
                for ci in range(C):
                    R_bits += entropy_bits_from_hist(st_uv.hists[ci])
                C_uv = float(R_bits + lam * D_mse)

            if C_uv < best_merge_C:
                best_merge_C = C_uv

        # c_e < 0 => merging improves total cost
        c_e = float(best_merge_C - (best_super[u].C_total + best_super[v].C_total))
        edge_costs.append((c_e, u, v))

    # 6) Solve multicut approximately by GAEC (老师建议)
    if solver.lower() != "gaec":
        raise ValueError("本版本为满足老师要求，默认只保留 GAEC。请用 solver='gaec'。")
    uf = gaec_unionfind(K=K, edge_costs=edge_costs, stop_when_nonnegative=True)

    # 7) components -> final labels (用映射数组，避免 N 次全图 bool mask)
    comp: Dict[int, List[int]] = defaultdict(list)
    for v in range(K):
        comp[uf.find(v)].append(v)
    comps = list(comp.values())
    K_final = len(comps)

    map_super_to_final = np.empty(K, dtype=np.int32)
    for new_id, super_ids in enumerate(comps):
        for sid in super_ids:
            map_super_to_final[sid] = new_id
    labels_final = map_super_to_final[labels_super].astype(np.int32)

    # 8) choose bits per final region using merged stats (NO pixel access)
    region_bits: Dict[int, int] = {}
    for seg_id, super_ids in enumerate(comps):
        best_C = float("inf")
        best_bits = int(candidate_bits[len(candidate_bits)//2])

        for bits in candidate_bits:
            agg: Optional[BitsStats] = None
            for sid in super_ids:
                if agg is None:
                    agg = stats_super[sid][int(bits)]
                else:
                    agg = merge_two_stats(agg, stats_super[sid][int(bits)])

            if agg is None or agg.N <= 0:
                C_tot = 0.0
            else:
                D_mse = agg.sumsq_total / (agg.N * C)
                R_bits = 0.0
                for ci in range(C):
                    R_bits += entropy_bits_from_hist(agg.hists[ci])
                C_tot = float(R_bits + lam * D_mse)

            if C_tot < best_C:
                best_C = C_tot
                best_bits = int(bits)

        region_bits[seg_id] = best_bits

    # 9) reconstruct final image from global recon + region bits
    idx2, s2, e2 = build_indices_by_label(labels_final, K_final)
    out_channels = [rc.copy() for rc in recon_channels_u8]

    for seg_id, bits in region_bits.items():
        flat_idx = idx2[s2[seg_id]:e2[seg_id]]
        if flat_idx.size == 0:
            continue
        lut = luts[int(bits)]
        for ci in range(C):
            flat = out_channels[ci].ravel()
            vals = flat[flat_idx]
            flat[flat_idx] = lut[vals]

    if img_u8.ndim == 2:
        recon = out_channels[0]
    else:
        recon = np.stack(out_channels, axis=-1)
        if alpha is not None:
            recon = np.concatenate([recon, alpha[..., None]], axis=-1)

    return MulticutResult(recon_u8=recon)


# ---------------------------
# Batch entry (WIDE CSV ONLY + summary.csv)
# ---------------------------
def run(
    input_root: str,
    output_root: str,
    max_images: int = 100,
    only_subfolder: Optional[str] = None,

    lam: float = 0.05,
    n_super_small: int = 80,
    n_super_big: int = 200,

    candidate_bits: Tuple[int, ...] = (3, 4, 5, 6, 7),
    posterize_bits_alpha: int = 8,

    png_compress_level: int = 9,
    jpeg_quality: int = 75,
    webp_quality: int = 75,

    rag_neighborhood: int = 8,
):
    os.makedirs(output_root, exist_ok=True)
    metrics_path = os.path.join(output_root, "metrics.csv")
    summary_path = os.path.join(output_root, "summary.csv")

    def saved_percent(orig_b: int, out_b: int) -> float:
        if orig_b <= 0:
            return 0.0
        return (1.0 - (out_b / orig_b)) * 100.0

    n_images = 0
    total_orig_bytes = 0
    total_out = {"multicut": 0, "png": 0, "jpeg": 0, "webp": 0}
    saved_list = {"multicut": [], "png": [], "jpeg": [], "webp": []}
    win_count = {"multicut": 0, "png": 0, "jpeg": 0, "webp": 0}

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(
            "dataset,rel_path,in_format,orig_bytes,"
            "multicut_out_bytes,multicut_saved_percent,"
            "baseline_png_out_bytes,baseline_png_saved_percent,"
            "baseline_jpeg_out_bytes,baseline_jpeg_saved_percent,"
            "baseline_webp_out_bytes,baseline_webp_saved_percent\n"
        )

        subfolders = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
        subfolders.sort()

        count = 0
        for dataset in subfolders:
            if only_subfolder and dataset != only_subfolder:
                continue
            if count >= max_images:
                break

            base_dir = os.path.join(input_root, dataset)
            print(f"\n=== Dataset: {dataset} ===")

            for root, _, files in os.walk(base_dir):
                if count >= max_images:
                    break
                for fname in files:
                    if count >= max_images:
                        break
                    if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
                        continue

                    in_path = os.path.join(root, fname)
                    rel = os.path.relpath(in_path, base_dir).replace("\\", "/")
                    orig_bytes = os.path.getsize(in_path)

                    pil = Image.open(in_path)
                    in_format = (pil.format or "").upper()
                    mode = pil.mode

                    # 按你原逻辑：跳过 BMP/TIFF 输入
                    if in_format in ("BMP", "TIFF", "TIF"):
                        count += 1
                        continue

                    if mode == "P":
                        pil = pil.convert("RGBA")
                        mode = pil.mode
                    if mode not in ("L", "LA", "RGB", "RGBA"):
                        pil = pil.convert("RGB")
                        mode = pil.mode

                    img = np.array(pil)
                    n_sp = n_super_small if max(img.shape[0], img.shape[1]) <= 128 else n_super_big

                    out_dir = os.path.join(output_root, dataset, os.path.dirname(rel))
                    os.makedirs(out_dir, exist_ok=True)
                    stem, _ = os.path.splitext(os.path.basename(rel))

                    # ---------------------------
                    # multicut (FAST + GAEC)
                    # ---------------------------
                    res = multicut_compress_image_fast(
                        img,
                        n_superpixels=n_sp,
                        lam=lam,
                        keep_alpha_lossless=True,
                        candidate_bits=candidate_bits,
                        rag_neighborhood=rag_neighborhood,
                        solver="gaec",
                    )
                    recon = res.recon_u8

                    # alpha posterize optional
                    if recon.ndim == 3 and recon.shape[2] == 4 and posterize_bits_alpha < 8:
                        a = recon[..., 3].copy()
                        lut_a = build_posterize_lut(int(posterize_bits_alpha))
                        a = lut_a[a]
                        recon = np.dstack([recon[..., :3], a])

                    out_path_multicut = os.path.join(out_dir, stem + "_multicut." + in_format.lower())
                    pil_recon = Image.fromarray(recon, mode=mode)

                    if in_format == "PNG":
                        save_png_best(pil_recon, out_path_multicut, compress_level=png_compress_level)
                    else:
                        save_as_format(
                            pil_recon, out_path_multicut, out_format=in_format,
                            png_compress_level=png_compress_level,
                            jpeg_quality=jpeg_quality,
                            webp_quality=webp_quality,
                        )

                    out_bytes_m = os.path.getsize(out_path_multicut)

                    # 保守：如果反而变大，就直接复制原图
                    if out_bytes_m >= orig_bytes:
                        shutil.copy2(in_path, out_path_multicut)
                        out_bytes_m = os.path.getsize(out_path_multicut)

                    sp_m = saved_percent(orig_bytes, out_bytes_m)

                    # ---------------------------
                    # baselines
                    # ---------------------------
                    out_path_png = os.path.join(out_dir, stem + "_baseline_png.png")
                    save_png_best(pil, out_path_png, compress_level=png_compress_level)
                    out_bytes_png = os.path.getsize(out_path_png)
                    sp_png = saved_percent(orig_bytes, out_bytes_png)

                    out_path_jpg = os.path.join(out_dir, stem + "_baseline_jpeg.jpg")
                    save_as_format(
                        pil, out_path_jpg, "JPEG",
                        png_compress_level=png_compress_level,
                        jpeg_quality=jpeg_quality,
                        webp_quality=webp_quality,
                    )
                    out_bytes_jpg = os.path.getsize(out_path_jpg)
                    sp_jpg = saved_percent(orig_bytes, out_bytes_jpg)

                    out_path_webp = os.path.join(out_dir, stem + "_baseline_webp.webp")
                    save_as_format(
                        pil, out_path_webp, "WEBP",
                        png_compress_level=png_compress_level,
                        jpeg_quality=jpeg_quality,
                        webp_quality=webp_quality,
                    )
                    out_bytes_webp = os.path.getsize(out_path_webp)
                    sp_webp = saved_percent(orig_bytes, out_bytes_webp)

                    f.write(
                        f"{dataset},{rel},{in_format},{orig_bytes},"
                        f"{out_bytes_m},{sp_m:.2f},"
                        f"{out_bytes_png},{sp_png:.2f},"
                        f"{out_bytes_jpg},{sp_jpg:.2f},"
                        f"{out_bytes_webp},{sp_webp:.2f}\n"
                    )

                    n_images += 1
                    total_orig_bytes += orig_bytes

                    total_out["multicut"] += out_bytes_m
                    total_out["png"] += out_bytes_png
                    total_out["jpeg"] += out_bytes_jpg
                    total_out["webp"] += out_bytes_webp

                    saved_list["multicut"].append(sp_m)
                    saved_list["png"].append(sp_png)
                    saved_list["jpeg"].append(sp_jpg)
                    saved_list["webp"].append(sp_webp)

                    out_map = {
                        "multicut": out_bytes_m,
                        "png": out_bytes_png,
                        "jpeg": out_bytes_jpg,
                        "webp": out_bytes_webp
                    }
                    min_bytes = min(out_map.values())
                    for k, v in out_map.items():
                        if v == min_bytes:
                            win_count[k] += 1

                    print(f"  {rel}  done")
                    count += 1

    def safe_mean(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def safe_median(xs: List[float]) -> float:
        return float(np.median(xs)) if xs else 0.0

    def global_saved_percent(total_orig: int, total_out_b: int) -> float:
        if total_orig <= 0:
            return 0.0
        return (1.0 - (total_out_b / total_orig)) * 100.0

    with open(summary_path, "w", encoding="utf-8") as fsum:
        fsum.write(
            "method,n_images,total_orig_bytes,total_out_bytes,"
            "global_saved_percent,mean_saved_percent,median_saved_percent,"
            "win_count,win_rate_percent\n"
        )
        for method_key, method_name in [
            ("multicut", "multicut_gaec"),
            ("png", "baseline_png"),
            ("jpeg", "baseline_jpeg"),
            ("webp", "baseline_webp"),
        ]:
            tot_out_b = int(total_out[method_key])
            gsp = global_saved_percent(total_orig_bytes, tot_out_b)
            mean_sp = safe_mean(saved_list[method_key])
            med_sp = safe_median(saved_list[method_key])
            wc = int(win_count[method_key])
            wr = (wc / n_images * 100.0) if n_images > 0 else 0.0

            fsum.write(
                f"{method_name},{n_images},{total_orig_bytes},{tot_out_b},"
                f"{gsp:.2f},{mean_sp:.2f},{med_sp:.2f},"
                f"{wc},{wr:.2f}\n"
            )

    print("\nDone.")
    print("metrics.csv :", metrics_path)
    print("summary.csv :", summary_path)


# ---------------------------
# Decompress outputs
# ---------------------------
def decompress_outputs(
    compressed_root: str,
    decompressed_root: str,
    out_format: str = "BMP",
    keep_structure: bool = True,
    skip_non_images: bool = True,
):
    out_format = (out_format or "").upper().strip()
    assert out_format in ("BMP", "PNG", "TIFF"), "out_format 只建议 BMP/PNG/TIFF"

    valid_ext = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")
    os.makedirs(decompressed_root, exist_ok=True)

    n = 0
    for root, _, files in os.walk(compressed_root):
        for fname in files:
            in_path = os.path.join(root, fname)
            if skip_non_images and (not fname.lower().endswith(valid_ext)):
                continue

            rel = os.path.relpath(in_path, compressed_root)
            rel_dir = os.path.dirname(rel)

            stem, _ = os.path.splitext(os.path.basename(rel))
            out_dir = os.path.join(decompressed_root, rel_dir) if keep_structure else decompressed_root
            os.makedirs(out_dir, exist_ok=True)

            out_ext = ".bmp" if out_format == "BMP" else (".png" if out_format == "PNG" else ".tiff")
            out_path = os.path.join(out_dir, stem + out_ext)

            try:
                img = Image.open(in_path)
                img.load()

                if out_format == "BMP":
                    img.save(out_path, format="BMP")
                elif out_format == "PNG":
                    img.save(out_path, format="PNG", optimize=False, compress_level=0)
                else:
                    img.save(out_path, format="TIFF")

                n += 1
            except Exception as e:
                print(f"[decompress] skip failed: {in_path}  err={e}")

    print(f"\n[decompress] Done. decoded & saved: {n}")
    print("[decompress] decompressed_root:", decompressed_root)


if __name__ == "__main__":
    INPUT_ROOT  = r"C:\Users\lenovo\Desktop\images"
    OUTPUT_ROOT = r"C:\Users\lenovo\Desktop\result2\screenshot_game_compressed_out"

    MAX_IMAGES = 2000
    ONLY_SUBFOLDER = "screenshot_game"

    LAM = 0.05
    CANDIDATE_BITS = (3, 4, 5, 6, 7)
    POSTERIZE_BITS_ALPHA = 8

    JPEG_QUALITY = 75
    WEBP_QUALITY = 75
    PNG_COMPRESS_LEVEL = 9

    RAG_NEIGHBORHOOD = 8

    run(
        input_root=INPUT_ROOT,
        output_root=OUTPUT_ROOT,
        max_images=MAX_IMAGES,
        only_subfolder=ONLY_SUBFOLDER,
        lam=LAM,
        candidate_bits=CANDIDATE_BITS,
        posterize_bits_alpha=POSTERIZE_BITS_ALPHA,
        png_compress_level=PNG_COMPRESS_LEVEL,
        jpeg_quality=JPEG_QUALITY,
        webp_quality=WEBP_QUALITY,
        rag_neighborhood=RAG_NEIGHBORHOOD,
    )

    DECOMPRESSED_ROOT = r"C:\Users\lenovo\Desktop\result2\screenshot_game_decompressed_out"
    decompress_outputs(
        compressed_root=OUTPUT_ROOT,
        decompressed_root=DECOMPRESSED_ROOT,
        out_format="BMP",
        keep_structure=True,
        skip_non_images=True,
    )
