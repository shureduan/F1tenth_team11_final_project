#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Look-ahead context strategy map — computed directly from:
  raceline_output/raceline.csv         (x_m, y_m, s_m, kappa_radpm, vx_mps)
  raceline_output/left_boundary.csv    (x, y)
  raceline_output/right_boundary.csv   (x, y)

All context features are recomputed here; no dependency on x_context_features.csv.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from scipy.spatial import KDTree

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = "/Users/shure_duan/VScode/f1tenth"
RACELINE_DIR = os.path.join(BASE_DIR, "raceline_generation", "raceline_output")
RESULT_DIR   = os.path.join(BASE_DIR, "neural_input", "look_ahead_context_input", "look_ahead_result")

RACELINE_CSV  = os.path.join(RACELINE_DIR, "raceline.csv")
LEFT_BD_CSV   = os.path.join(RACELINE_DIR, "left_boundary.csv")
RIGHT_BD_CSV  = os.path.join(RACELINE_DIR, "right_boundary.csv")

OUT_PNG = os.path.join(RESULT_DIR, "x_context_strategy_map.png")
OUT_PDF = os.path.join(RESULT_DIR, "x_context_strategy_map.pdf")

LOOK_AHEAD = 30          # number of raceline points to look ahead
SUBSAMPLE_N = 500        # downsample raceline to this many pts before computing features
STRAIGHT_QUANTILE = 0.25  # |kappa| below this percentile = straight
TURN_QUANTILE     = 0.60  # |kappa| above this percentile = turning
ALPHA             = 0.5   # weighting for compound indicator


# ── feature computation ────────────────────────────────────────────────────────

def _longest_consecutive_ones(b):
    best = curr = 0
    for v in b:
        if v:
            curr += 1
            best = max(best, curr)
        else:
            curr = 0
    return best


def compute_context_features(x, y, kappa, left_xy, right_xy, look_ahead=30):
    n = len(x)

    # Nearest boundary
    ltree = KDTree(left_xy)
    rtree = KDTree(right_xy)
    d_left,  left_bd_idx  = ltree.query(np.column_stack([x, y]))
    d_right, right_bd_idx = rtree.query(np.column_stack([x, y]))

    left_proj  = left_xy[left_bd_idx]
    right_proj = right_xy[right_bd_idx]
    widths = np.linalg.norm(left_proj - right_proj, axis=1)

    # Arc-length per segment (chord length)
    dx = np.roll(x, -1) - x
    dy = np.roll(y, -1) - y
    ds = np.sqrt(dx**2 + dy**2)

    abs_k = np.abs(kappa)

    # Quantile-based thresholds (robust across different track scales)
    kappa_straight = float(np.quantile(abs_k, STRAIGHT_QUANTILE))
    kappa_turn     = float(np.quantile(abs_k, TURN_QUANTILE))

    # Look-ahead features (circular)
    mean_curv    = np.zeros(n)
    max_curv     = np.zeros(n)
    straight_ind = np.zeros(n)
    compound_ind = np.zeros(n)
    accum_head   = np.zeros(n)

    for i in range(n):
        idxs = np.arange(i, i + look_ahead) % n
        window_k_abs    = abs_k[idxs]
        window_k_signed = kappa[idxs]
        window_ds       = ds[idxs]

        mean_curv[i] = window_k_abs.mean()
        max_curv[i]  = window_k_abs.max()

        # straight indicator: longest consecutive straight run (per spec)
        b = (window_k_abs < kappa_straight).astype(int)
        straight_ind[i] = _longest_consecutive_ones(b) / look_ahead

        # compound indicator: alpha*(n_turn/W) + (1-alpha)*(n_switch / max(n_turn-1,1))
        # sign switches counted only among turning segments (per spec)
        turn_mask = window_k_abs > kappa_turn
        n_turn = int(np.sum(turn_mask))
        if n_turn > 1:
            turn_signs = np.sign(window_k_signed[turn_mask])
            n_switch = int(np.sum(np.diff(turn_signs) != 0))
        else:
            n_switch = 0
        turn_term   = n_turn / float(look_ahead)
        switch_term = n_switch / float(max(n_turn - 1, 1)) if n_turn > 0 else 0.0
        compound_ind[i] = ALPHA * turn_term + (1.0 - ALPHA) * switch_term

        # accumulated heading change: sum(|kappa| * ds)  (per spec)
        accum_head[i] = float(np.sum(window_k_abs * window_ds))

    return {
        "left_proj":    left_proj,
        "right_proj":   right_proj,
        "left_bd_idx":  left_bd_idx,
        "right_bd_idx": right_bd_idx,
        "widths":       widths,
        "mean_curv":    mean_curv,
        "max_curv":     max_curv,
        "straight_ind": straight_ind,
        "compound_ind": compound_ind,
        "accum_head":   accum_head,
    }


# ── geometry helpers ───────────────────────────────────────────────────────────

def compute_tangents(xy):
    prev_xy = np.roll(xy, 1, axis=0)
    next_xy = np.roll(xy, -1, axis=0)
    t = next_xy - prev_xy
    n = np.linalg.norm(t, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return t / n


def compute_normals(tangents):
    return np.column_stack([-tangents[:, 1], tangents[:, 0]])


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
    nl, nr = len(left_bd), len(right_bd)
    polys = []
    for i in range(nseg):
        j = (i + 1) % nseg
        li, lj = int(left_idx[i]) % nl, int(left_idx[j]) % nl
        ri, rj = int(right_idx[i]) % nr, int(right_idx[j]) % nr
        left_arc  = left_bd[cyclic_short_arc_indices(li, lj, nl)]
        right_arc = right_bd[cyclic_short_arc_indices(ri, rj, nr)]
        polys.append(np.vstack([left_arc, right_arc[::-1]]).astype(float))
    return polys


def find_true_runs(mask):
    mask = np.asarray(mask, dtype=bool)
    n = len(mask)
    if not np.any(mask):
        return []
    idx = np.where(mask)[0]
    runs, current = [], [idx[0]]
    for k in range(1, len(idx)):
        if idx[k] == idx[k - 1] + 1:
            current.append(idx[k])
        else:
            runs.append(np.array(current, dtype=int))
            current = [idx[k]]
    runs.append(np.array(current, dtype=int))
    if len(runs) >= 2 and runs[0][0] == 0 and runs[-1][-1] == n - 1:
        merged = np.concatenate([runs[-1], runs[0]])
        runs = [merged] + runs[1:-1]
    return runs


def draw_track_outline(ax, left_xy, right_xy, color="black", lw=1.0, alpha=0.95, zorder=4):
    for xy in (left_xy, right_xy):
        ax.plot(np.r_[xy[:, 0], xy[0, 0]], np.r_[xy[:, 1], xy[0, 1]],
                color=color, lw=lw, alpha=alpha, zorder=zorder)


def draw_green_chevrons(ax, center, tangent, width, color="#39D353", alpha=0.96, lw=1.0):
    total_len = 1.6 * width
    n_chev = 4
    spacing = total_len / (n_chev + 0.6)
    side = 0.5 * width          # lateral half-span = track half-width
    start = center - 0.5 * total_len * tangent
    perp = np.array([-tangent[1], tangent[0]])
    for k in range(n_chev):
        base = start + (k + 0.2) * spacing * tangent
        tip  = base + 0.48 * spacing * tangent
        left  = base - side * perp
        right = base + side * perp
        for pt in (left, right):
            ax.plot([pt[0], tip[0]], [pt[1], tip[1]],
                    color=color, alpha=alpha, linewidth=lw,
                    solid_capstyle="round", zorder=7)


def draw_compound_arc_frames(ax, runs, left_proj, right_proj, normals,
                              expand_frac=0.15, color="darkorange", lw=2.5, alpha=0.88,
                              hatch_lw=0.9, hatch_alpha=0.55, hatch_step=2, zorder=6):
    """Orange closed arc outline + perpendicular fill lines for each compound corner run."""
    for run in runs:
        if len(run) < 2:
            continue
        lp = left_proj[run]
        rp = right_proj[run]
        nl = normals[run]

        avg_w = float(np.mean(np.linalg.norm(lp - rp, axis=1)))
        d = expand_frac * avg_w

        lp_out = lp + d * nl
        rp_out = rp - d * nl

        # Closed outline
        outline = np.vstack([lp_out, rp_out[::-1], lp_out[[0]]])
        ax.plot(outline[:, 0], outline[:, 1],
                color=color, linewidth=lw, alpha=alpha,
                solid_capstyle="round", solid_joinstyle="round",
                zorder=zorder)

        # Perpendicular fill lines (cross-hatch from left to right boundary)
        for k in range(0, len(run), hatch_step):
            ax.plot([lp_out[k, 0], rp_out[k, 0]],
                    [lp_out[k, 1], rp_out[k, 1]],
                    color=color, linewidth=hatch_lw, alpha=hatch_alpha,
                    solid_capstyle="round", zorder=zorder)


def add_margin_from_xy(ax, x, y, ratio=0.04):
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    ax.set_xlim(xmin - ratio*(xmax-xmin), xmax + ratio*(xmax-xmin))
    ax.set_ylim(ymin - ratio*(ymax-ymin), ymax + ratio*(ymax-ymin))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    for p in [RACELINE_CSV, LEFT_BD_CSV, RIGHT_BD_CSV]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    rl  = pd.read_csv(RACELINE_CSV)
    lbd = pd.read_csv(LEFT_BD_CSV)
    rbd = pd.read_csv(RIGHT_BD_CSV)

    x_raw     = rl["x_m"].to_numpy(float)
    y_raw     = rl["y_m"].to_numpy(float)
    kappa_raw = rl["kappa_radpm"].to_numpy(float)

    # Downsample to SUBSAMPLE_N evenly-spaced points so look-ahead covers ~30m
    idx_sub = np.round(np.linspace(0, len(x_raw) - 1, SUBSAMPLE_N)).astype(int)
    x     = x_raw[idx_sub]
    y     = y_raw[idx_sub]
    kappa = kappa_raw[idx_sub]

    left_xy  = lbd[["x", "y"]].to_numpy(float)
    right_xy = rbd[["x", "y"]].to_numpy(float)

    print(f"Raceline: {len(x)} pts (subsampled from {len(x_raw)})  |  "
          f"Left bd: {len(left_xy)}  |  Right bd: {len(right_xy)}")

    feat = compute_context_features(x, y, kappa, left_xy, right_xy, LOOK_AHEAD)

    left_proj   = feat["left_proj"]
    right_proj  = feat["right_proj"]
    left_idx    = feat["left_bd_idx"]
    right_idx   = feat["right_bd_idx"]
    widths      = feat["widths"]
    mean_curv   = feat["mean_curv"]
    straight_ind = feat["straight_ind"]
    compound_ind = feat["compound_ind"]
    accum_head  = feat["accum_head"]

    # Thresholds
    q_compound = np.quantile(compound_ind, 0.90)
    # Straight: fire if longest consecutive straight run >= 10 out of 30 segments
    q_straight = 10.0 / LOOK_AHEAD
    # Accum heading: top 25% to catch sharp corners like hairpins
    q_accum    = np.quantile(accum_head, 0.75)

    compound_mask = compound_ind >= q_compound
    straight_mask = straight_ind >= q_straight
    accum_mask    = accum_head   >= q_accum

    # Tangents / normals along raceline
    rl_xy     = np.column_stack([x, y])
    tangents  = compute_tangents(rl_xy)
    normals   = compute_normals(tangents)

    polys = build_track_sector_polygons(left_xy, right_xy, left_idx, right_idx)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")

    norm = Normalize(vmin=np.nanmin(mean_curv), vmax=np.nanmax(mean_curv))
    pc = PolyCollection(polys, array=mean_curv, cmap="inferno", norm=norm,
                        edgecolors="face", linewidths=0.2, zorder=2)
    ax.add_collection(pc)

    draw_track_outline(ax, left_xy, right_xy, color="black", lw=1.0, zorder=4)

    # Raceline as centerline (white)
    ax.plot(np.r_[x, x[0]], np.r_[y, y[0]],
            color="white", lw=0.9, alpha=0.72, zorder=5)

    # Compound arc frames (orange)
    compound_runs = find_true_runs(compound_mask)
    draw_compound_arc_frames(ax, compound_runs, left_proj, right_proj, normals)

    # Straight chevrons (green) — subsample to avoid clutter
    straight_idx = np.where(straight_mask)[0]
    if len(straight_idx) > 0:
        keep, last = [], -999
        for idx in straight_idx:
            if idx - last >= 5:
                keep.append(idx)
                last = idx
        for i in keep:
            draw_green_chevrons(ax, rl_xy[i], tangents[i], widths[i])

    # Accumulated heading change markers (red !)
    for run in find_true_runs(accum_mask):
        mid = run[len(run) // 2]
        pos = rl_xy[mid] + 0.72 * widths[mid] * normals[mid]
        ax.text(pos[0], pos[1], "!", color="red", fontsize=20,
                fontweight="bold", ha="center", va="center", zorder=8)

    # Direction arrows (sparse)
    n = len(x)
    step = max(1, n // 36)
    for i in range(0, n, step):
        scale = 0.35 * widths[i]
        p0 = rl_xy[i] - 0.25 * scale * tangents[i]
        p1 = rl_xy[i] + 0.75 * scale * tangents[i]
        ax.annotate("", xy=(p1[0], p1[1]), xytext=(p0[0], p0[1]),
                    arrowprops=dict(arrowstyle="->", color="deepskyblue",
                                   lw=1.0, alpha=0.70),
                    zorder=5)

    cbar = fig.colorbar(pc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(f"Look-Ahead Mean Curvature ({LOOK_AHEAD} segments)", fontsize=12)

    ax.set_title(
        "Look-Ahead Context Strategy Map\n"
        "Base: mean curvature ahead | Orange frame: compound corners | "
        "Green >>>>: long straight ahead | Red !: high accumulated heading change",
        fontsize=13, pad=14,
    )

    all_xy = np.vstack([left_xy, right_xy, rl_xy])
    add_margin_from_xy(ax, all_xy[:, 0], all_xy[:, 1], ratio=0.04)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")


if __name__ == "__main__":
    main()
