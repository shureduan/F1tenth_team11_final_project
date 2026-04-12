#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raceline visualization — computed directly from:
  raceline_output/raceline.csv          (x_m, y_m, s_m, kappa_radpm, vx_mps)
  raceline_output/left_boundary.csv     (x, y)
  raceline_output/right_boundary.csv    (x, y)

Local features (d_left, d_right, delta_kappa, delta_psi, d_inner) are read directly
from the finalized local feature CSV in the local_result folder (use-version file).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.spatial import KDTree



def load_local_feature_csv(path):
    df = pd.read_csv(path)
    required = ["segment_id", "kappa", "delta_kappa", "delta_psi", "d_left", "d_right", "d_inner"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"local feature CSV missing required column: {col}")
    return df
# ── paths ──────────────────────────────────────────────────────────────────────
RACELINE_DIR = "/Users/shure_duan/VScode/f1tenth/raceline_generation/raceline_output"
OUT_DIR      = "/Users/shure_duan/VScode/f1tenth/neural_input/local_input/local_result"

RACELINE_CSV  = os.path.join(RACELINE_DIR, "raceline.csv")
LEFT_BD_CSV   = os.path.join(RACELINE_DIR, "left_boundary.csv")
RIGHT_BD_CSV  = os.path.join(RACELINE_DIR, "right_boundary.csv")
LOCAL_FEATURE_CSV = "/Users/shure_duan/VScode/f1tenth/neural_input/local_input/local_result/x_local_features_u.csv"

OUT_FIG1 = os.path.join(OUT_DIR, "segment_abs_curvature_track_heatmap_u.png")
OUT_FIG2 = os.path.join(OUT_DIR, "segment_inner_space_dkappa_track_panels_u.png")


# ── feature computation ────────────────────────────────────────────────────────

def compute_features(x, y, kappa, left_xy, right_xy):
    """
    Given raceline (x, y, kappa) and boundary arrays, compute:
      d_left, d_right         : Euclidean distance to nearest boundary point
      left_bd_idx, right_bd_idx : index of that nearest boundary point
      delta_kappa             : per-point change in curvature (circular diff)
      delta_psi               : per-point change in heading angle (circular)
    """
    n = len(x)

    # KDTree nearest-neighbour for boundary distances
    ltree = KDTree(left_xy)
    rtree = KDTree(right_xy)

    d_left,  left_bd_idx  = ltree.query(np.column_stack([x, y]))
    d_right, right_bd_idx = rtree.query(np.column_stack([x, y]))

    # delta_kappa (circular)
    kappa_next = np.roll(kappa, -1)
    delta_kappa = kappa_next - kappa

    # heading angle psi from consecutive points (circular)
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    psi = np.arctan2(y_next - y, x_next - x)

    # delta_psi with angle wrapping
    psi_next = np.roll(psi, -1)
    delta_psi = psi_next - psi
    delta_psi = (delta_psi + np.pi) % (2 * np.pi) - np.pi   # wrap to [-π, π]

    return (d_left, d_right,
            left_bd_idx.astype(int), right_bd_idx.astype(int),
            delta_kappa, delta_psi)


# ── helpers (unchanged from original) ─────────────────────────────────────────

def style_axis(ax, title):
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def add_margin_from_xy(ax, x, y, ratio=0.04):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin - ratio * dx, xmax + ratio * dx)
    ax.set_ylim(ymin - ratio * dy, ymax + ratio * dy)


def normalize01(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    vmin = np.nanmin(v)
    vmax = np.nanmax(v)
    return (v - vmin) / (vmax - vmin + eps)


def compute_forward_tangent_unit_vectors(x, y):
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    dx = x_next - x
    dy = y_next - y
    n = np.sqrt(dx**2 + dy**2) + 1e-12
    return dx / n, dy / n


def compute_inner_space(kappa, d_left, d_right):
    return np.where(kappa >= 0.0, d_left, d_right)


def cyclic_short_arc_indices(i, j, n):
    fwd = (j - i) % n
    bwd = (i - j) % n
    if fwd <= bwd:
        if i <= j:
            return np.arange(i, j + 1)
        return np.r_[np.arange(i, n), np.arange(0, j + 1)]
    else:
        if j <= i:
            return np.arange(i, j - 1, -1)
        return np.r_[np.arange(i, -1, -1), np.arange(n - 1, j - 1, -1)]


def build_track_sector_polygons(left_bd, right_bd, left_idx, right_idx):
    nseg = len(left_idx)
    polys = []
    nl = len(left_bd)
    nr = len(right_bd)
    for i in range(nseg):
        j = (i + 1) % nseg
        li = int(left_idx[i]) % nl
        lj = int(left_idx[j]) % nl
        ri = int(right_idx[i]) % nr
        rj = int(right_idx[j]) % nr
        left_arc  = left_bd[cyclic_short_arc_indices(li, lj, nl)]
        right_arc = right_bd[cyclic_short_arc_indices(ri, rj, nr)]
        poly = np.vstack([left_arc, right_arc[::-1]])
        polys.append(poly.astype(float))
    return polys


def circular_gaussian_smooth(v, sigma=2.2, radius=None):
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n == 0 or sigma <= 0:
        return v.copy()
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)
    vp = np.r_[v[-radius:], v, v[:radius]]
    out = np.convolve(vp, kernel, mode="valid")
    return out[:n]




def resample_closed_array(arr, n_out):
    arr = np.asarray(arr)
    n_in = len(arr)
    if n_in == n_out:
        return arr.copy()
    if n_in < 2:
        raise ValueError("Need at least 2 points to resample a closed array.")
    idx = np.linspace(0.0, n_in, n_out, endpoint=False)
    i0 = np.floor(idx).astype(int) % n_in
    i1 = (i0 + 1) % n_in
    t = idx - np.floor(idx)
    if arr.ndim == 1:
        return (1.0 - t) * arr[i0] + t * arr[i1]
    return (1.0 - t)[:, None] * arr[i0] + t[:, None] * arr[i1]

def truncate_colormap(cmap_name="turbo", minval=0.10, maxval=1.0, n=256):
    base = plt.get_cmap(cmap_name)
    new_colors = base(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(f"{cmap_name}_trunc", new_colors)


def add_track_heatmap(ax, left_bd, right_bd, left_idx, right_idx, values,
                      cmap="turbo", vmin=None, vmax=None, zorder=2):
    polys = build_track_sector_polygons(left_bd, right_bd, left_idx, right_idx)
    vals  = np.asarray(values, dtype=float)
    if len(vals) != len(polys):
        raise ValueError(f"Value count ({len(vals)}) != segment count ({len(polys)}).")
    if vmin is None: vmin = np.nanmin(vals)
    if vmax is None: vmax = np.nanmax(vals)
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1e-6
    norm = Normalize(vmin=vmin, vmax=vmax)
    pc = PolyCollection(polys, array=vals, cmap=cmap, norm=norm,
                        edgecolors="face", linewidths=0.2, zorder=zorder)
    ax.add_collection(pc)
    return pc


def draw_track_outline(ax, left_xy, right_xy, color="black", lw=1.0, alpha=0.95, zorder=4):
    for xy in (left_xy, right_xy):
        xc = np.r_[xy[:, 0], xy[0, 0]]
        yc = np.r_[xy[:, 1], xy[0, 1]]
        ax.plot(xc, yc, color=color, lw=lw, alpha=alpha, zorder=zorder)


def draw_centerline(ax, x, y, color="white", lw=0.8, alpha=0.7, zorder=5):
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]],
            color=color, lw=lw, alpha=alpha, zorder=zorder)


def build_dense_adaptive_arrows(x, y, delta_psi):
    a0 = normalize01(np.abs(delta_psi))
    a  = np.sqrt(10.0 * a0)
    a  = a / (np.nanmax(a) + 1e-12)

    ux, uy = compute_forward_tangent_unit_vectors(x, y)
    X, Y, U, V = [], [], [], []

    n = len(x)
    i = 0
    while i < n:
        ai = a[i]

        X.append(x[i])
        Y.append(y[i])
        U.append(ux[i])
        V.append(uy[i])

        if ai < 0.12:
            i += 16
        elif ai < 0.22:
            i += 12
        elif ai < 0.38:
            i += 6
        elif ai < 0.60:
            i += 3
        elif ai < 0.82:
            i += 2
        else:
            i += 1

    return np.array(X), np.array(Y), np.array(U), np.array(V)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    for path in [RACELINE_CSV, LEFT_BD_CSV, RIGHT_BD_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    rl  = pd.read_csv(RACELINE_CSV)
    lbd = pd.read_csv(LEFT_BD_CSV)
    rbd = pd.read_csv(RIGHT_BD_CSV)

    x_full     = rl["x_m"].to_numpy(float)
    y_full     = rl["y_m"].to_numpy(float)
    kappa_full = rl["kappa_radpm"].to_numpy(float)

    left_xy  = lbd[["x", "y"]].to_numpy(float)
    right_xy = rbd[["x", "y"]].to_numpy(float)

    print(f"Raceline : {len(x_full)} pts")
    print(f"Left bd  : {len(left_xy)} pts")
    print(f"Right bd : {len(right_xy)} pts")

    # Read finalized local-input features directly from saved feature space
    local_df = load_local_feature_csv(LOCAL_FEATURE_CSV).sort_values("segment_id").reset_index(drop=True)
    n_vis = len(local_df)

    # Use the local-feature segmentation as the visualization base.
    # If the raceline is denser than the local feature CSV, downsample it uniformly
    # around the closed loop so both spaces line up by segment count.
    if n_vis != len(x_full):
        print(f"[INFO] Resampling visualization base from {len(x_full)} raceline points to {n_vis} local-feature segments.")
    x = resample_closed_array(x_full, n_vis)
    y = resample_closed_array(y_full, n_vis)
    kappa_geom = resample_closed_array(kappa_full, n_vis)

    d_left = local_df["d_left"].to_numpy(dtype=float)
    d_right = local_df["d_right"].to_numpy(dtype=float)
    delta_kappa = local_df["delta_kappa"].to_numpy(dtype=float)
    delta_psi = local_df["delta_psi"].to_numpy(dtype=float)
    inner_space = local_df["d_inner"].to_numpy(dtype=float)

    # Map each visualization segment to boundary indices using nearest neighbors.
    ltree = KDTree(left_xy)
    rtree = KDTree(right_xy)
    _, left_idx = ltree.query(np.column_stack([x, y]))
    _, right_idx = rtree.query(np.column_stack([x, y]))
    left_idx = left_idx.astype(int)
    right_idx = right_idx.astype(int)

    all_xy = np.vstack([left_xy, right_xy])

    # Smooth |kappa| for heatmap
    abs_kappa_vis = circular_gaussian_smooth(np.abs(kappa_geom), sigma=1.6)
    abs_kappa_vis[abs_kappa_vis < 0.015] = 0.0

    turbo_light = truncate_colormap("turbo", 0.10, 1.0)

    # ── Fig 1: absolute curvature heatmap ──────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    pc1 = add_track_heatmap(
        ax1, left_xy, right_xy, left_idx, right_idx,
        abs_kappa_vis, cmap=turbo_light, vmin=0.0, vmax=0.35, zorder=2,
    )
    cbar1 = fig1.colorbar(pc1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Absolute curvature $|\\kappa|$ [1/m]")

    draw_track_outline(ax1, left_xy, right_xy, color="black", lw=1.0, zorder=4)
    draw_centerline(ax1, x, y, color="white", lw=0.9, alpha=0.65, zorder=5)

    Xa, Ya, Ua, Va = build_dense_adaptive_arrows(x, y, delta_psi)
    diag = np.sqrt((all_xy[:, 0].max() - all_xy[:, 0].min())**2 +
                   (all_xy[:, 1].max() - all_xy[:, 1].min())**2)
    arrow_len = 0.008 * diag
    ax1.quiver(Xa, Ya, Ua * arrow_len, Va * arrow_len,
               angles="xy", scale_units="xy", scale=1.0,
               color="black", width=0.0014,
               headwidth=4.0, headlength=5.2, headaxislength=4.8,
               alpha=0.95, zorder=6)

    style_axis(ax1, "Track-based Absolute Curvature Heatmap")
    add_margin_from_xy(ax1, all_xy[:, 0], all_xy[:, 1], ratio=0.04)
    plt.tight_layout()
    fig1.savefig(OUT_FIG1, dpi=300, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved: {OUT_FIG1}")

    # ── Fig 2: inner space + delta_kappa panels ─────────────────────────────────
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    panel_specs = [
        ("Inner-line space", inner_space, "viridis", "Inner distance [m]"),
        ("Curvature variation $\\Delta\\kappa$",
         delta_kappa, "coolwarm", "$\\Delta\\kappa$ [1/m$^2$]"),
    ]
    for ax, (title, vals, cmap, cbar_label) in zip(axes.ravel(), panel_specs):
        pc = add_track_heatmap(ax, left_xy, right_xy, left_idx, right_idx,
                               vals, cmap=cmap, zorder=2)
        cbar = fig2.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        draw_track_outline(ax, left_xy, right_xy, color="black", lw=1.0, zorder=4)
        draw_centerline(ax, x, y, color="white", lw=0.8, alpha=0.6, zorder=5)
        style_axis(ax, title)
        add_margin_from_xy(ax, all_xy[:, 0], all_xy[:, 1], ratio=0.04)

    plt.tight_layout()
    fig2.savefig(OUT_FIG2, dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {OUT_FIG2}")


if __name__ == "__main__":
    main()
