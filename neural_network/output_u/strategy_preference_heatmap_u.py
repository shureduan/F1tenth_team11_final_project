#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment-based strategy preference heatmap for F1TENTH track visualization.

This script uses the same segment-style visualization logic as the local-input
visualizer, but colors each segment by strategy preference score:

- positive value -> stronger apex-defense preference
- negative value -> stronger best-line preference
- value near 0   -> neutral

Expected inputs:
  1) defense_preference_prediction_u.csv
  2) x_local_features_u.csv               (for segment count / segment_id alignment)
  3) raceline.csv                         (geometry base)
  4) left_boundary.csv
  5) right_boundary.csv

Author: ChatGPT
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib import patheffects
from scipy.spatial import KDTree
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# =============================================================================
# Fixed paths
# =============================================================================
BASE_DIR = "/Users/shure_duan/VScode/f1tenth"
RACELINE_DIR = os.path.join(BASE_DIR, "raceline_generation", "raceline_output")
LOCAL_DIR = os.path.join(BASE_DIR, "neural_input", "local_input", "local_result")
NN_OUT_DIR = os.path.join(BASE_DIR, "neural_network", "output_u")

RACELINE_CSV = os.path.join(RACELINE_DIR, "raceline.csv")
LEFT_BD_CSV = os.path.join(RACELINE_DIR, "left_boundary.csv")
RIGHT_BD_CSV = os.path.join(RACELINE_DIR, "right_boundary.csv")
LOCAL_FEATURE_CSV = os.path.join(LOCAL_DIR, "x_local_features_u.csv")
PREFERENCE_CSV = os.path.join(NN_OUT_DIR, "defense_preference_prediction_u.csv")

OUT_FIG = os.path.join(NN_OUT_DIR, "strategy_preference_track_heatmap_u.png")
OUT_FIG_PDF = os.path.join(NN_OUT_DIR, "strategy_preference_track_heatmap_u.pdf")


# =============================================================================
# Loading
# =============================================================================
def load_local_feature_csv(path):
    df = pd.read_csv(path)
    required = ["segment_id"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"local feature CSV missing required column: {col}")
    return df


def load_preference_csv(path):
    df = pd.read_csv(path)
    required = ["segment_id", "D_pred"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"preference CSV missing required column: {col}")
    return df


# =============================================================================
# Helpers
# =============================================================================
def style_axis(ax, title):
    ax.set_title(title, fontsize=16, weight="semibold", pad=12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]", fontsize=11)
    ax.set_ylabel("y [m]", fontsize=11)



def add_margin_from_xy(ax, x, y, ratio=0.04):
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    dx = xmax - xmin
    dy = ymax - ymin
    ax.set_xlim(xmin - ratio * dx, xmax + ratio * dx)
    ax.set_ylim(ymin - ratio * dy, ymax + ratio * dy)



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
        left_arc = left_bd[cyclic_short_arc_indices(li, lj, nl)]
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



def build_preference_cmap():
    # Blue (best-line) -> light gray (neutral) -> red (apex-defense)
    colors = [
        "#2c7fb8",  # strong negative
        "#7fcdbb",  # mild negative
        "#f3f3f3",  # neutral
        "#fdae61",  # mild positive
        "#d7191c",  # strong positive
    ]
    return LinearSegmentedColormap.from_list("strategy_pref_diverging", colors, N=256)



def add_track_heatmap(ax, left_bd, right_bd, left_idx, right_idx, values,
                      cmap, norm, zorder=2):
    polys = build_track_sector_polygons(left_bd, right_bd, left_idx, right_idx)
    vals = np.asarray(values, dtype=float)
    if len(vals) != len(polys):
        raise ValueError(f"Value count ({len(vals)}) != segment count ({len(polys)}).")
    pc = PolyCollection(
        polys,
        array=vals,
        cmap=cmap,
        norm=norm,
        edgecolors="face",
        linewidths=0.25,
        zorder=zorder,
    )
    ax.add_collection(pc)
    return pc



def draw_track_outline(ax, left_xy, right_xy, color="#222222", lw=1.1, alpha=0.95, zorder=4):
    for xy in (left_xy, right_xy):
        xc = np.r_[xy[:, 0], xy[0, 0]]
        yc = np.r_[xy[:, 1], xy[0, 1]]
        ax.plot(xc, yc, color=color, lw=lw, alpha=alpha, zorder=zorder)



def draw_centerline(ax, x, y, color="white", lw=0.9, alpha=0.55, zorder=5):
    ax.plot(
        np.r_[x, x[0]],
        np.r_[y, y[0]],
        color=color,
        lw=lw,
        alpha=alpha,
        zorder=zorder,
    )



def add_custom_preference_legend(ax, cmap, norm):
    # Background box in upper right
    box = inset_axes(ax, width="30%", height="25%", loc="upper right", borderpad=1.0)
    box.set_facecolor((1, 1, 1, 0.92))
    for spine in box.spines.values():
        spine.set_edgecolor("#666666")
        spine.set_linewidth(0.9)
    box.set_xticks([])
    box.set_yticks([])
    box.set_xlim(0, 1)
    box.set_ylim(0, 1)

    box.text(0.08, 0.88, "Strategy preference", fontsize=11, weight="bold", ha="left", va="center")

    # Gradient bar
    grad_ax = inset_axes(
        box, width="84%", height="18%", loc="center",
        bbox_to_anchor=(0.08, 0.44, 0.84, 0.18),
        bbox_transform=box.transAxes, borderpad=0
    )
    gradient = np.linspace(norm.vmin, norm.vmax, 512).reshape(1, -1)
    grad_ax.imshow(gradient, aspect="auto", cmap=cmap, norm=norm, extent=[norm.vmin, norm.vmax, 0, 1])
    grad_ax.set_yticks([])
    grad_ax.set_xticks([norm.vmin, 0.0, norm.vmax])
    grad_ax.tick_params(axis="x", labelsize=8, length=2)
    for spine in grad_ax.spines.values():
        spine.set_edgecolor("#777777")
        spine.set_linewidth(0.7)

    box.text(0.08, 0.32, "Positive → stronger apex-defense preference", fontsize=8.7, ha="left", va="center")
    box.text(0.08, 0.20, "Negative → stronger best-line preference", fontsize=8.7, ha="left", va="center")
    box.text(0.08, 0.08, "Near 0 → neutral", fontsize=8.7, ha="left", va="center")


# =============================================================================
# Main
# =============================================================================
def main():
    for path in [RACELINE_CSV, LEFT_BD_CSV, RIGHT_BD_CSV, LOCAL_FEATURE_CSV, PREFERENCE_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

    os.makedirs(NN_OUT_DIR, exist_ok=True)

    # Geometry
    rl = pd.read_csv(RACELINE_CSV)
    lbd = pd.read_csv(LEFT_BD_CSV)
    rbd = pd.read_csv(RIGHT_BD_CSV)

    x_full = rl["x_m"].to_numpy(float)
    y_full = rl["y_m"].to_numpy(float)
    left_xy = lbd[["x", "y"]].to_numpy(float)
    right_xy = rbd[["x", "y"]].to_numpy(float)

    # Segment alignment base
    local_df = load_local_feature_csv(LOCAL_FEATURE_CSV).sort_values("segment_id").reset_index(drop=True)
    pref_df = load_preference_csv(PREFERENCE_CSV).sort_values("segment_id").reset_index(drop=True)

    if len(local_df) != len(pref_df):
        raise ValueError(
            f"Segment count mismatch: local features have {len(local_df)} rows, "
            f"but preference CSV has {len(pref_df)} rows."
        )

    if not np.array_equal(local_df["segment_id"].to_numpy(), pref_df["segment_id"].to_numpy()):
        raise ValueError("segment_id ordering mismatch between local feature CSV and preference CSV.")

    n_vis = len(pref_df)
    x = resample_closed_array(x_full, n_vis)
    y = resample_closed_array(y_full, n_vis)

    # Map segments to boundaries using nearest boundary point to segment center
    ltree = KDTree(left_xy)
    rtree = KDTree(right_xy)
    _, left_idx = ltree.query(np.column_stack([x, y]))
    _, right_idx = rtree.query(np.column_stack([x, y]))
    left_idx = left_idx.astype(int)
    right_idx = right_idx.astype(int)

    # Preference values
    pref_raw = pref_df["D_pred"].to_numpy(float)
    pref_vis = circular_gaussian_smooth(pref_raw, sigma=1.2)

    # Symmetric diverging color scale around zero
    vmax = np.nanpercentile(np.abs(pref_vis), 98)
    vmax = max(vmax, 1e-3)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = build_preference_cmap()

    all_xy = np.vstack([left_xy, right_xy])

    # Figure
    fig, ax = plt.subplots(figsize=(12.5, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f8f8")

    pc = add_track_heatmap(
        ax=ax,
        left_bd=left_xy,
        right_bd=right_xy,
        left_idx=left_idx,
        right_idx=right_idx,
        values=pref_vis,
        cmap=cmap,
        norm=norm,
        zorder=2,
    )

    draw_track_outline(ax, left_xy, right_xy, color="#1f1f1f", lw=1.1, alpha=0.95, zorder=4)
    draw_centerline(ax, x, y, color="white", lw=0.8, alpha=0.55, zorder=5)

    # Main colorbar
    cbar = fig.colorbar(pc, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Strategy preference score", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # Title + subtitle
    title = ax.set_title(
        "Track Segment Strategy Preference Heatmap",
        fontsize=17,
        weight="semibold",
        pad=16,
    )
    title.set_path_effects([patheffects.withStroke(linewidth=3, foreground="white", alpha=0.35)])

    ax.text(
        0.015, 0.985,
        "Red = stronger apex-defense preference   |   Blue = stronger best-line preference",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10.5,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.28", facecolor=(1, 1, 1, 0.78), edgecolor="none"),
        zorder=10,
    )

    add_custom_preference_legend(ax, cmap, norm)

    style_axis(ax, "Track Segment Strategy Preference Heatmap")
    add_margin_from_xy(ax, all_xy[:, 0], all_xy[:, 1], ratio=0.04)

    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=320, bbox_inches="tight")
    fig.savefig(OUT_FIG_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PNG: {OUT_FIG}")
    print(f"Saved PDF: {OUT_FIG_PDF}")


if __name__ == "__main__":
    main()
