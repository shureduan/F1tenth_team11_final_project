import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# =========================
# Fixed folder paths
# =========================
BASE_DIR = "/Users/shure_duan/VScode/f1tenth/raceline_generation/raceline_output"
OUT_DIR = "/Users/shure_duan/VScode/f1tenth/neural_input/local_input/local_result"

RACELINE_CSV = os.path.join(BASE_DIR, "raceline.csv")
CENTERLINE_CSV = os.path.join(BASE_DIR, "centerline.csv")
LEFT_BOUNDARY_CSV = os.path.join(BASE_DIR, "left_boundary.csv")
RIGHT_BOUNDARY_CSV = os.path.join(BASE_DIR, "right_boundary.csv")

OUTPUT_FEATURE_CSV = os.path.join(OUT_DIR, "x_local_features.csv")
OUTPUT_FEATURE_NPY = os.path.join(OUT_DIR, "x_local.npy")


# =========================
# Loading
# =========================
def load_raceline_csv(path):
    df = pd.read_csv(path)
    required = ["x_m", "y_m", "s_m", "kappa_radpm"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"raceline.csv missing required column: {col}")
    return df


def load_centerline_csv(path):
    df = pd.read_csv(path)
    required = ["x", "y", "s"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"centerline.csv missing required column: {col}")
    return df


def load_boundary_csv(path):
    df = pd.read_csv(path)
    required = ["x", "y"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"boundary CSV missing required column: {col}")
    return df


# =========================
# Helpers
# =========================
def remove_consecutive_duplicate_points(df, xcol, ycol, scol=None, eps=1e-12):
    pts = df[[xcol, ycol]].to_numpy(dtype=float)
    keep = np.ones(len(df), dtype=bool)
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep[1:] = d > eps

    df2 = df.loc[keep].copy().reset_index(drop=True)

    if scol is not None:
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
    """
    Periodic interpolation for scalar values on a closed loop.
    """
    L = s_old[-1]
    s_ext = np.concatenate([s_old, [L + (s_old[1] - s_old[0])]])
    v_ext = np.concatenate([v_old, [v_old[0]]])
    return np.interp(s_new, s_ext, v_ext)


def interp_closed_xy(s_old, x_old, y_old, n_segments):
    """
    Resample closed curve by arc length using existing s_m.
    """
    L = s_old[-1]
    ds = L / n_segments
    s_new = np.linspace(0.0, L, n_segments, endpoint=False)

    x_new = interp_closed_scalar(s_old, x_old, s_new)
    y_new = interp_closed_scalar(s_old, y_old, s_new)
    return s_new, x_new, y_new, ds, L


def circular_gradient(values, ds):
    """
    Gradient on a closed loop (uniform spacing).
    """
    return (np.roll(values, -1) - np.roll(values, 1)) / (2.0 * ds)


def circular_gradient_nonuniform(values, s, L):
    """
    Central-difference gradient on a closed loop with non-uniform spacing.
    L: total arc length (period).
    """
    s_next = np.roll(s, -1).copy()
    s_next[-1] = s[0] + L
    s_prev = np.roll(s, 1).copy()
    s_prev[0] = s[-1] - L
    v_next = np.roll(values, -1)
    v_prev = np.roll(values, 1)
    return (v_next - v_prev) / (s_next - s_prev)


def adaptive_arc_sample(s_old, kappa_old, n_segments, kappa_weight=5.0):
    """
    Non-uniform arc-length resampling concentrated at high-curvature regions.

    Builds a density function  d(s) = 1 + kappa_weight * |kappa(s)|  on a
    fine grid, computes its CDF, then inverts it at n_segments uniform
    targets so that high-curvature sections get proportionally more points.

    Returns s_new [n_segments].
    """
    L = s_old[-1]
    n_fine = 10 * n_segments
    s_fine = np.linspace(0.0, L, n_fine, endpoint=False)
    kappa_fine = np.interp(s_fine, s_old, kappa_old)

    density = 1.0 + kappa_weight * np.abs(kappa_fine)

    cum = np.concatenate([[0.0], np.cumsum(density)])
    cum /= cum[-1]
    s_fine_ext = np.concatenate([s_fine, [L]])

    targets = np.linspace(0.0, 1.0, n_segments, endpoint=False)
    s_new = np.interp(targets, cum, s_fine_ext)
    return s_new


def compute_heading(x, y):
    # arctan2 is scale-invariant, so uniform ds=1 is fine for angle computation
    dx = circular_gradient(x, 1.0)
    dy = circular_gradient(y, 1.0)
    psi = np.arctan2(dy, dx)
    psi = np.unwrap(psi)
    return psi


def project_boundaries_to_normals(x, y, psi, left_bd, right_bd, k_search=50):
    """
    For each raceline point, find the boundary point that is most directly
    perpendicular to the raceline (minimum tangential offset among candidates
    on the correct side).

    Returns
    -------
    left_proj     : [N, 2]  projected left boundary points
    right_proj    : [N, 2]  projected right boundary points
    d_left        : [N]     perpendicular distance to left boundary
    d_right       : [N]     perpendicular distance to right boundary
    left_bd_idx   : [N]     index into left_bd for each projected point
    right_bd_idx  : [N]     index into right_bd for each projected point
    """
    left_tree = cKDTree(left_bd)
    right_tree = cKDTree(right_bd)

    n = len(x)
    left_proj = np.zeros((n, 2))
    right_proj = np.zeros((n, 2))
    d_left_out = np.zeros(n)
    d_right_out = np.zeros(n)
    left_bd_idx = np.zeros(n, dtype=int)
    right_bd_idx = np.zeros(n, dtype=int)

    k_l = min(k_search, len(left_bd))
    k_r = min(k_search, len(right_bd))

    for i in range(n):
        px, py = x[i], y[i]
        tx = np.cos(psi[i])
        ty = np.sin(psi[i])

        # left normal: 90° CCW from tangent
        nx, ny = -ty, tx

        # --- Left boundary ---
        _, idxs = left_tree.query([px, py], k=k_l)
        cands = left_bd[idxs]
        dv = cands - np.array([px, py])
        norm_comp = dv[:, 0] * nx + dv[:, 1] * ny
        tang_comp = np.abs(dv[:, 0] * tx + dv[:, 1] * ty)

        on_side = norm_comp > 0
        if on_side.sum() > 0:
            best = np.where(on_side)[0][np.argmin(tang_comp[on_side])]
        else:
            best = np.argmin(tang_comp)

        left_proj[i] = cands[best]
        d_left_out[i] = np.abs(norm_comp[best])
        left_bd_idx[i] = int(idxs[best])

        # --- Right boundary: normal is 90° CW from tangent ---
        rnx, rny = ty, -tx
        _, idxs = right_tree.query([px, py], k=k_r)
        cands = right_bd[idxs]
        dv = cands - np.array([px, py])
        norm_comp = dv[:, 0] * rnx + dv[:, 1] * rny
        tang_comp = np.abs(dv[:, 0] * tx + dv[:, 1] * ty)

        on_side = norm_comp > 0
        if on_side.sum() > 0:
            best = np.where(on_side)[0][np.argmin(tang_comp[on_side])]
        else:
            best = np.argmin(tang_comp)

        right_proj[i] = cands[best]
        d_right_out[i] = np.abs(norm_comp[best])
        right_bd_idx[i] = int(idxs[best])

    return left_proj, right_proj, d_left_out, d_right_out, left_bd_idx, right_bd_idx


# =========================
# Main feature builder
# =========================
def compute_local_features(
    raceline_csv,
    centerline_csv,
    left_boundary_csv,
    right_boundary_csv,
    n_segments=300,
    kappa_weight=5.0,
):
    """
    kappa_weight: controls adaptive density. Higher = more points at curves.
                  Set to 0 for uniform spacing.
    """
    # load
    raceline_df = load_raceline_csv(raceline_csv)
    centerline_df = load_centerline_csv(centerline_csv)
    left_df = load_boundary_csv(left_boundary_csv)
    right_df = load_boundary_csv(right_boundary_csv)

    # remove duplicate raceline points
    raceline_df = remove_consecutive_duplicate_points(
        raceline_df, xcol="x_m", ycol="y_m", scol="s_m"
    )

    x_old = raceline_df["x_m"].to_numpy(dtype=float)
    y_old = raceline_df["y_m"].to_numpy(dtype=float)
    s_old = raceline_df["s_m"].to_numpy(dtype=float)
    kappa_old = raceline_df["kappa_radpm"].to_numpy(dtype=float)

    centerline_df = remove_consecutive_duplicate_points(
        centerline_df, xcol="x", ycol="y", scol="s"
    )
    cx_old = centerline_df["x"].to_numpy(dtype=float)
    cy_old = centerline_df["y"].to_numpy(dtype=float)
    cs_old = centerline_df["s"].to_numpy(dtype=float)

    left_bd = left_df[["x", "y"]].to_numpy(dtype=float)
    right_bd = right_df[["x", "y"]].to_numpy(dtype=float)

    print("Loaded files successfully:")
    print(f"  raceline: {raceline_df.shape}")
    print(f"  centerline: {centerline_df.shape}")
    print(f"  left boundary: {left_bd.shape}")
    print(f"  right boundary: {right_bd.shape}")

    total_len = float(s_old[-1])
    centerline_total_len = float(cs_old[-1])

    # Adaptive arc-length resampling for raceline (UNCHANGED)
    print(f"[INFO] Adaptive resampling (n={n_segments}, kappa_weight={kappa_weight}) ...")
    s = adaptive_arc_sample(s_old, kappa_old, n_segments, kappa_weight=kappa_weight)

    # raceline sampled features (UNCHANGED)
    x = interp_closed_scalar(s_old, x_old, s)
    y = interp_closed_scalar(s_old, y_old, s)

    # centerline sampled independently for boundary-side projection (UNCHANGED)
    s_center = np.linspace(0.0, centerline_total_len, n_segments, endpoint=False)
    cx = interp_closed_scalar(cs_old, cx_old, s_center)
    cy = interp_closed_scalar(cs_old, cy_old, s_center)

    # =========================================================
    # ONLY CHANGE:
    # kappa and delta_kappa now come from centerline
    # =========================================================
    psi_center = compute_heading(cx, cy)
    ds_center = centerline_total_len / n_segments
    kappa = circular_gradient(psi_center, ds_center)
    delta_kappa = circular_gradient_nonuniform(kappa, s_center, centerline_total_len)

    # heading and delta psi for raceline features (UNCHANGED)
    psi = compute_heading(x, y)
    delta_psi = circular_gradient_nonuniform(psi, s, total_len)

    # Local arc-length per raceline segment (UNCHANGED)
    s_next = np.roll(s, -1).copy()
    s_next[-1] = s[0] + total_len
    ds_arr = s_next - s

    # boundary projection from centerline (UNCHANGED)
    print("Projecting boundaries along normals ...")
    left_proj, right_proj, d_left, d_right, left_bd_idx, right_bd_idx = \
        project_boundaries_to_normals(cx, cy, psi_center, left_bd, right_bd)

    # local width (sum of perpendicular half-widths)
    width = d_left + d_right

    # x_local = [kappa_i, delta_kappa_i, w_i, dL_i, dR_i, delta_psi_i]
    x_local = np.column_stack([
        kappa,
        delta_kappa,
        width,
        d_left,
        d_right,
        delta_psi
    ])

    df = pd.DataFrame({
        "segment_id": np.arange(n_segments),
        "s": s,
        "ds": ds_arr,
        "x": x,
        "y": y,
        "kappa": kappa,
        "delta_kappa": delta_kappa,
        "width": width,
        "d_left": d_left,
        "d_right": d_right,
        "delta_psi": delta_psi,
        "psi": psi,
        # Projected boundary points for perpendicular cross-section visualization
        "left_proj_x": left_proj[:, 0],
        "left_proj_y": left_proj[:, 1],
        "right_proj_x": right_proj[:, 0],
        "right_proj_y": right_proj[:, 1],
        # Indices into the raw boundary arrays
        "left_bd_idx": left_bd_idx,
        "right_bd_idx": right_bd_idx,
    })

    return x_local, df


# =========================
# Run directly
# =========================
if __name__ == "__main__":
    for path in [RACELINE_CSV, CENTERLINE_CSV, LEFT_BOUNDARY_CSV, RIGHT_BOUNDARY_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    x_local, df_local = compute_local_features(
        raceline_csv=RACELINE_CSV,
        centerline_csv=CENTERLINE_CSV,
        left_boundary_csv=LEFT_BOUNDARY_CSV,
        right_boundary_csv=RIGHT_BOUNDARY_CSV,
        n_segments=600,
        kappa_weight=5.0,
    )

    print("\nDone.")
    print("x_local shape:", x_local.shape)
    print(df_local.head())

    df_local.to_csv(OUTPUT_FEATURE_CSV, index=False)
    np.save(OUTPUT_FEATURE_NPY, x_local)

    print(f"\nSaved CSV to: {OUTPUT_FEATURE_CSV}")
    print(f"Saved NPY to: {OUTPUT_FEATURE_NPY}")