#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# =========================
# Fixed folder paths
# =========================
RACELINE_DIR = "/Users/shure_duan/VScode/f1tenth/raceline_generation/raceline_output"
OUT_DIR = "/Users/shure_duan/VScode/f1tenth/neural_input/local_input/local_result"

RACELINE_CSV = os.path.join(RACELINE_DIR, "raceline.csv")
LEFT_BD_CSV = os.path.join(RACELINE_DIR, "left_boundary.csv")
RIGHT_BD_CSV = os.path.join(RACELINE_DIR, "right_boundary.csv")

OUTPUT_FEATURE_CSV = os.path.join(OUT_DIR, "x_local_features_u.csv")
OUTPUT_FEATURE_NPY = os.path.join(OUT_DIR, "x_local_u.npy")


def load_raceline_csv(path):
    df = pd.read_csv(path)
    required = ["x_m", "y_m", "s_m", "kappa_radpm"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"raceline.csv missing required column: {col}")
    return df


def load_boundary_csv(path):
    df = pd.read_csv(path)
    required = ["x", "y"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"boundary CSV missing required column: {col}")
    return df


def remove_consecutive_duplicate_points(df, xcol, ycol, scol=None, eps=1e-12):
    pts = df[[xcol, ycol]].to_numpy(dtype=float)
    keep = np.ones(len(df), dtype=bool)
    if len(df) > 1:
        d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep[1:] = d > eps
    df2 = df.loc[keep].copy().reset_index(drop=True)

    if scol is not None and len(df2) > 1:
        s = df2[scol].to_numpy(dtype=float)
        if np.any(np.diff(s) <= 0):
            order = np.argsort(s)
            df2 = df2.iloc[order].reset_index(drop=True)
            s = df2[scol].to_numpy(dtype=float)
            keep_s = np.ones(len(df2), dtype=bool)
            keep_s[1:] = np.diff(s) > eps
            df2 = df2.loc[keep_s].copy().reset_index(drop=True)
    return df2


def interp_closed_scalar(s_old, v_old, s_new):
    L = float(s_old[-1])
    ds0 = float(s_old[1] - s_old[0]) if len(s_old) > 1 else 0.0
    s_ext = np.concatenate([s_old, [L + ds0]])
    v_ext = np.concatenate([v_old, [v_old[0]]])
    return np.interp(s_new, s_ext, v_ext)


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def compute_features_from_visual_v2_logic(x, y, kappa, left_xy, right_xy):
    """
    Exact feature logic copied from x_local_visualize_v2.py:
      - d_left / d_right: Euclidean distance to nearest boundary point
      - delta_kappa: circular forward difference
      - delta_psi: forward heading difference with wrapping

    For d_inner, keep the same value range but reverse the side choice so the
    heatmap color relation flips without changing to a width-complement metric.
    """
    ltree = cKDTree(left_xy)
    rtree = cKDTree(right_xy)

    d_left, left_bd_idx = ltree.query(np.column_stack([x, y]))
    d_right, right_bd_idx = rtree.query(np.column_stack([x, y]))

    kappa_next = np.roll(kappa, -1)
    delta_kappa = kappa_next - kappa

    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    psi = np.arctan2(y_next - y, x_next - x)

    psi_next = np.roll(psi, -1)
    delta_psi = wrap_to_pi(psi_next - psi)

    # Original visualize_v2 inner-side logic would be:
    # np.where(kappa >= 0.0, d_left, d_right)
    # You asked for the color relation to be reversed while staying in the same
    # ~0-2 m scale, so we swap the selected side instead of using width-complement.
    d_inner = np.where(kappa >= 0.0, d_right, d_left)

    return (
        d_left.astype(float),
        d_right.astype(float),
        left_bd_idx.astype(int),
        right_bd_idx.astype(int),
        delta_kappa.astype(float),
        delta_psi.astype(float),
        d_inner.astype(float),
    )


def compute_local_features(
    raceline_csv,
    left_boundary_csv,
    right_boundary_csv,
    n_segments=600,
):
    raceline_df = load_raceline_csv(raceline_csv)
    left_df = load_boundary_csv(left_boundary_csv)
    right_df = load_boundary_csv(right_boundary_csv)

    raceline_df = remove_consecutive_duplicate_points(raceline_df, "x_m", "y_m", "s_m")

    x_old = raceline_df["x_m"].to_numpy(dtype=float)
    y_old = raceline_df["y_m"].to_numpy(dtype=float)
    s_old = raceline_df["s_m"].to_numpy(dtype=float)
    kappa_old = raceline_df["kappa_radpm"].to_numpy(dtype=float)

    left_xy = left_df[["x", "y"]].to_numpy(dtype=float)
    right_xy = right_df[["x", "y"]].to_numpy(dtype=float)

    total_len = float(s_old[-1])
    s_new = np.linspace(0.0, total_len, int(n_segments), endpoint=False)

    x = interp_closed_scalar(s_old, x_old, s_new)
    y = interp_closed_scalar(s_old, y_old, s_new)
    kappa = interp_closed_scalar(s_old, kappa_old, s_new)

    (
        d_left,
        d_right,
        left_bd_idx,
        right_bd_idx,
        delta_kappa,
        delta_psi,
        d_inner,
    ) = compute_features_from_visual_v2_logic(x, y, kappa, left_xy, right_xy)

    x_local = np.column_stack([
        kappa,
        delta_kappa,
        delta_psi,
        d_left,
        d_right,
        d_inner,
    ])

    df = pd.DataFrame({
        "segment_id": np.arange(int(n_segments)),
        "kappa": kappa,
        "delta_kappa": delta_kappa,
        "delta_psi": delta_psi,
        "d_left": d_left,
        "d_right": d_right,
        "d_inner": d_inner,
        "left_bd_idx": left_bd_idx,
        "right_bd_idx": right_bd_idx,
        "s_m": s_new,
        "x_m": x,
        "y_m": y,
    })

    return x_local, df


if __name__ == "__main__":
    for path in [RACELINE_CSV, LEFT_BD_CSV, RIGHT_BD_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    os.makedirs(OUT_DIR, exist_ok=True)

    x_local, df_local = compute_local_features(
        raceline_csv=RACELINE_CSV,
        left_boundary_csv=LEFT_BD_CSV,
        right_boundary_csv=RIGHT_BD_CSV,
        n_segments=600,
    )

    print("\nDone.")
    print("x_local shape:", x_local.shape)
    print(df_local.head())

    df_local.to_csv(OUTPUT_FEATURE_CSV, index=False)
    np.save(OUTPUT_FEATURE_NPY, x_local)

    print(f"\nSaved CSV to: {OUTPUT_FEATURE_CSV}")
    print(f"Saved NPY to: {OUTPUT_FEATURE_NPY}")
