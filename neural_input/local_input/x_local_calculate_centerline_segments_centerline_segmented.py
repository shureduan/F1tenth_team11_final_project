import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# =========================
# Fixed folder paths
# =========================
# Three reference lines come from boundary_output
BOUNDARY_DIR = "/Users/shure_duan/VScode/f1tenth/boundary_output"
# Raceline still comes from raceline_generation/raceline_output
RACELINE_DIR = "/Users/shure_duan/VScode/f1tenth/raceline_generation/raceline_output"
OUT_DIR = "/Users/shure_duan/VScode/f1tenth/neural_input/local_input/local_result"

RACELINE_CSV = os.path.join(RACELINE_DIR, "raceline.csv")
CENTERLINE_CSV = os.path.join(BOUNDARY_DIR, "centerline.csv")
INNER_BOUNDARY_CSV = os.path.join(BOUNDARY_DIR, "inner_boundary.csv")
OUTER_BOUNDARY_CSV = os.path.join(BOUNDARY_DIR, "outer_boundary.csv")

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
    """Periodic interpolation for scalar values on a closed loop."""
    L = s_old[-1]
    s_ext = np.concatenate([s_old, [L + (s_old[1] - s_old[0])]])
    v_ext = np.concatenate([v_old, [v_old[0]]])
    return np.interp(s_new, s_ext, v_ext)


def circular_gradient(values, ds):
    """Gradient on a closed loop (uniform spacing)."""
    return (np.roll(values, -1) - np.roll(values, 1)) / (2.0 * ds)


def circular_gradient_nonuniform(values, s, L):
    """Central-difference gradient on a closed loop with non-uniform spacing."""
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
    Segment logic stays exactly the same as before.
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
    dx = circular_gradient(x, 1.0)
    dy = circular_gradient(y, 1.0)
    psi = np.arctan2(dy, dx)
    psi = np.unwrap(psi)
    return psi


def build_cross_section_features_from_two_boundaries(x, y, psi, bd1, bd2, k_search=50):
    """
    Use the two reference boundaries (inner / outer) directly, but determine
    left/right at each raceline segment from the raceline tangent/normal.

    This avoids assuming that 'inner' is globally left or globally right.
    """
    tree1 = cKDTree(bd1)
    tree2 = cKDTree(bd2)

    n = len(x)
    left_proj = np.zeros((n, 2))
    right_proj = np.zeros((n, 2))
    d_left = np.zeros(n)
    d_right = np.zeros(n)
    left_src = np.empty(n, dtype=object)
    right_src = np.empty(n, dtype=object)
    left_bd_idx = np.zeros(n, dtype=int)
    right_bd_idx = np.zeros(n, dtype=int)

    k1 = min(k_search, len(bd1))
    k2 = min(k_search, len(bd2))

    for i in range(n):
        px, py = x[i], y[i]
        tx = np.cos(psi[i])
        ty = np.sin(psi[i])
        lnx, lny = -ty, tx   # left normal
        rnx, rny = ty, -tx   # right normal

        _, idxs1 = tree1.query([px, py], k=k1)
        _, idxs2 = tree2.query([px, py], k=k2)
        idxs1 = np.atleast_1d(idxs1)
        idxs2 = np.atleast_1d(idxs2)

        cands1 = bd1[idxs1]
        cands2 = bd2[idxs2]

        pts = np.vstack([cands1, cands2])
        src = np.array(["inner"] * len(cands1) + ["outer"] * len(cands2), dtype=object)
        src_idx = np.concatenate([idxs1, idxs2])

        dv = pts - np.array([px, py])
        tang = np.abs(dv[:, 0] * tx + dv[:, 1] * ty)
        left_norm = dv[:, 0] * lnx + dv[:, 1] * lny
        right_norm = dv[:, 0] * rnx + dv[:, 1] * rny

        left_valid = left_norm > 0
        right_valid = right_norm > 0

        if np.any(left_valid):
            cand_idx = np.where(left_valid)[0]
            best = cand_idx[np.argmin(tang[cand_idx])]
        else:
            best = np.argmax(left_norm)
        left_proj[i] = pts[best]
        d_left[i] = abs(left_norm[best])
        left_src[i] = src[best]
        left_bd_idx[i] = int(src_idx[best])

        if np.any(right_valid):
            cand_idx = np.where(right_valid)[0]
            best = cand_idx[np.argmin(tang[cand_idx])]
        else:
            best = np.argmax(right_norm)
        right_proj[i] = pts[best]
        d_right[i] = abs(right_norm[best])
        right_src[i] = src[best]
        right_bd_idx[i] = int(src_idx[best])

    return left_proj, right_proj, d_left, d_right, left_src, right_src, left_bd_idx, right_bd_idx


# =========================
# Main feature builder
# =========================
def compute_local_features(
    raceline_csv,
    centerline_csv,
    inner_boundary_csv,
    outer_boundary_csv,
    n_segments=300,
    kappa_weight=5.0,
):
    # load
    raceline_df = load_raceline_csv(raceline_csv)
    centerline_df = load_centerline_csv(centerline_csv)
    inner_df = load_boundary_csv(inner_boundary_csv)
    outer_df = load_boundary_csv(outer_boundary_csv)

    raceline_df = remove_consecutive_duplicate_points(
        raceline_df, xcol="x_m", ycol="y_m", scol="s_m"
    )
    centerline_df = remove_consecutive_duplicate_points(
        centerline_df, xcol="x", ycol="y", scol="s"
    )

    x_old = raceline_df["x_m"].to_numpy(dtype=float)
    y_old = raceline_df["y_m"].to_numpy(dtype=float)
    s_old = raceline_df["s_m"].to_numpy(dtype=float)
    kappa_old = raceline_df["kappa_radpm"].to_numpy(dtype=float)

    cx_old = centerline_df["x"].to_numpy(dtype=float)
    cy_old = centerline_df["y"].to_numpy(dtype=float)
    cs_old = centerline_df["s"].to_numpy(dtype=float)

    inner_bd = inner_df[["x", "y"]].to_numpy(dtype=float)
    outer_bd = outer_df[["x", "y"]].to_numpy(dtype=float)

    print("Loaded files successfully:")
    print(f"  raceline: {raceline_df.shape}")
    print(f"  centerline: {centerline_df.shape}")
    print(f"  inner boundary: {inner_bd.shape}")
    print(f"  outer boundary: {outer_bd.shape}")

    raceline_total_len = float(s_old[-1])
    centerline_total_len = float(cs_old[-1])

    # =========================================================
    # Segment logic: KEEP EXACTLY THE SAME, BUT APPLY IT ON CENTERLINE.
    # First compute centerline curvature on its native arc-length, then do
    # adaptive sampling on centerline. After that, map segment progress to
    # raceline for raceline-based features.
    # =========================================================
    psi_center_old = compute_heading(cx_old, cy_old)
    kappa_center_old = circular_gradient_nonuniform(psi_center_old, cs_old, centerline_total_len)

    print(f"[INFO] Adaptive resampling on centerline (n={n_segments}, kappa_weight={kappa_weight}) ...")
    s_center = adaptive_arc_sample(cs_old, kappa_center_old, n_segments, kappa_weight=kappa_weight)

    s_center_next = np.roll(s_center, -1).copy()
    s_center_next[-1] = s_center[0] + centerline_total_len
    ds_arr = s_center_next - s_center

    # Centerline samples: used for curvature and curvature-rate
    cx = interp_closed_scalar(cs_old, cx_old, s_center)
    cy = interp_closed_scalar(cs_old, cy_old, s_center)
    psi_center = compute_heading(cx, cy)
    kappa = circular_gradient_nonuniform(psi_center, s_center, centerline_total_len)
    delta_kappa = circular_gradient_nonuniform(kappa, s_center, centerline_total_len)

    # =========================================================
    # Map centerline segments to raceline by normalized progress alpha.
    # Raceline samples are then used for heading change and left/right space.
    # =========================================================
    alpha = s_center / centerline_total_len
    s_race = alpha * raceline_total_len

    x = interp_closed_scalar(s_old, x_old, s_race)
    y = interp_closed_scalar(s_old, y_old, s_race)
    psi = compute_heading(x, y)
    delta_psi = circular_gradient_nonuniform(psi, s_race, raceline_total_len)

    # =========================================================
    # Left/right local space: use RACELINE heading + the two reference boundaries
    # =========================================================
    print("Projecting boundaries along raceline normals ...")
    (
        left_proj,
        right_proj,
        d_left,
        d_right,
        left_src,
        right_src,
        left_bd_idx,
        right_bd_idx,
    ) = build_cross_section_features_from_two_boundaries(x, y, psi, inner_bd, outer_bd)

    width = d_left + d_right

    # x_local = [kappa_i, delta_kappa_i, w_i, dL_i, dR_i, delta_psi_i]
    x_local = np.column_stack([
        kappa,
        delta_kappa,
        width,
        d_left,
        d_right,
        delta_psi,
    ])

    df = pd.DataFrame({
        "segment_id": np.arange(n_segments),
        "s": s_center,
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
        "center_x": cx,
        "center_y": cy,
        "center_s": s_center,
        "raceline_s": s_race,
        "left_proj_x": left_proj[:, 0],
        "left_proj_y": left_proj[:, 1],
        "right_proj_x": right_proj[:, 0],
        "right_proj_y": right_proj[:, 1],
        "left_boundary_source": left_src,
        "right_boundary_source": right_src,
        "left_bd_idx": left_bd_idx,
        "right_bd_idx": right_bd_idx,
    })

    return x_local, df


# =========================
# Run directly
# =========================
if __name__ == "__main__":
    for path in [RACELINE_CSV, CENTERLINE_CSV, INNER_BOUNDARY_CSV, OUTER_BOUNDARY_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    os.makedirs(OUT_DIR, exist_ok=True)

    x_local, df_local = compute_local_features(
        raceline_csv=RACELINE_CSV,
        centerline_csv=CENTERLINE_CSV,
        inner_boundary_csv=INNER_BOUNDARY_CSV,
        outer_boundary_csv=OUTER_BOUNDARY_CSV,
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
