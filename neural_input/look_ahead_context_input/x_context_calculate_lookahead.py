import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# =========================
# Fixed folder paths
# =========================
BASE_DIR = "/Users/shure_duan/VScode/f1tenth/raceline_generation/raceline_output"
OUT_DIR = "/Users/shure_duan/VScode/f1tenth/neural_input/look_ahead_context_input/look_ahead_result"

RACELINE_CSV = os.path.join(BASE_DIR, "raceline.csv")
CENTERLINE_CSV = os.path.join(BASE_DIR, "centerline.csv")
LEFT_BOUNDARY_CSV = os.path.join(BASE_DIR, "left_boundary.csv")
RIGHT_BOUNDARY_CSV = os.path.join(BASE_DIR, "right_boundary.csv")

OUTPUT_FEATURE_CSV = os.path.join(OUT_DIR, "x_context_features.csv")
OUTPUT_FEATURE_NPY = os.path.join(OUT_DIR, "x_context.npy")


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
    L = float(s_old[-1])
    ds0 = float(s_old[1] - s_old[0])
    s_ext = np.concatenate([s_old, [L + ds0]])
    v_ext = np.concatenate([v_old, [v_old[0]]])
    return np.interp(s_new, s_ext, v_ext)


def circular_gradient(values, ds):
    return (np.roll(values, -1) - np.roll(values, 1)) / (2.0 * ds)


def circular_gradient_nonuniform(values, s, L):
    s_next = np.roll(s, -1).copy()
    s_next[-1] = s[0] + L
    s_prev = np.roll(s, 1).copy()
    s_prev[0] = s[-1] - L
    v_next = np.roll(values, -1)
    v_prev = np.roll(values, 1)
    return (v_next - v_prev) / (s_next - s_prev)


def compute_heading(x, y):
    dx = circular_gradient(x, 1.0)
    dy = circular_gradient(y, 1.0)
    psi = np.arctan2(dy, dx)
    psi = np.unwrap(psi)
    return psi


def count_sign_switches(values, valid_mask):
    signed_vals = values[valid_mask]
    if len(signed_vals) <= 1:
        return 0

    signs = np.sign(signed_vals).astype(float)

    for i in range(len(signs)):
        if signs[i] == 0:
            if i > 0 and signs[i - 1] != 0:
                signs[i] = signs[i - 1]
            else:
                for j in range(i + 1, len(signs)):
                    if signs[j] != 0:
                        signs[i] = signs[j]
                        break

    switches = 0
    for i in range(1, len(signs)):
        if signs[i] != 0 and signs[i - 1] != 0 and signs[i] != signs[i - 1]:
            switches += 1
    return switches


def project_boundaries_to_normals(x, y, psi, left_bd, right_bd, k_search=50):
    """
    Same segmentation logic as x_local:
    for each centerline segment point, project left/right boundary along local normal.
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

        # left normal
        nx, ny = -ty, tx

        _, idxs = left_tree.query([px, py], k=k_l)
        cands = left_bd[idxs]
        dv = cands - np.array([px, py])
        norm_comp = dv[:, 0] * nx + dv[:, 1] * ny
        tang_comp = np.abs(dv[:, 0] * tx + dv[:, 1] * ty)

        on_side = norm_comp > 0
        if np.any(on_side):
            best = np.where(on_side)[0][np.argmin(tang_comp[on_side])]
        else:
            best = np.argmin(tang_comp)

        left_proj[i] = cands[best]
        d_left_out[i] = np.abs(norm_comp[best])
        left_bd_idx[i] = int(idxs[best])

        # right normal
        rnx, rny = ty, -tx
        _, idxs = right_tree.query([px, py], k=k_r)
        cands = right_bd[idxs]
        dv = cands - np.array([px, py])
        norm_comp = dv[:, 0] * rnx + dv[:, 1] * rny
        tang_comp = np.abs(dv[:, 0] * tx + dv[:, 1] * ty)

        on_side = norm_comp > 0
        if np.any(on_side):
            best = np.where(on_side)[0][np.argmin(tang_comp[on_side])]
        else:
            best = np.argmin(tang_comp)

        right_proj[i] = cands[best]
        d_right_out[i] = np.abs(norm_comp[best])
        right_bd_idx[i] = int(idxs[best])

    return left_proj, right_proj, d_left_out, d_right_out, left_bd_idx, right_bd_idx


def longest_consecutive_ones(arr):
    best = 0
    curr = 0
    for v in arr:
        if v == 1:
            curr += 1
            best = max(best, curr)
        else:
            curr = 0
    return best


# =========================
# Main feature builder
# =========================
def compute_lookahead_context_features(
    raceline_csv,
    centerline_csv,
    left_boundary_csv,
    right_boundary_csv,
    n_segments=600,
    window_size=30,
    alpha=0.5,
    straight_quantile=0.25,
    turn_quantile=0.60,
):
    raceline_df = load_raceline_csv(raceline_csv)
    centerline_df = load_centerline_csv(centerline_csv)
    left_df = load_boundary_csv(left_boundary_csv)
    right_df = load_boundary_csv(right_boundary_csv)

    raceline_df = remove_consecutive_duplicate_points(
        raceline_df, xcol="x_m", ycol="y_m", scol="s_m"
    )
    centerline_df = remove_consecutive_duplicate_points(
        centerline_df, xcol="x", ycol="y", scol="s"
    )

    x_old = raceline_df["x_m"].to_numpy(dtype=float)
    y_old = raceline_df["y_m"].to_numpy(dtype=float)
    s_old = raceline_df["s_m"].to_numpy(dtype=float)

    cx_old = centerline_df["x"].to_numpy(dtype=float)
    cy_old = centerline_df["y"].to_numpy(dtype=float)
    cs_old = centerline_df["s"].to_numpy(dtype=float)

    left_bd = left_df[["x", "y"]].to_numpy(dtype=float)
    right_bd = right_df[["x", "y"]].to_numpy(dtype=float)

    centerline_total_len = float(cs_old[-1])

    # same uniform 600-segment centerline logic as x_local
    s_center = np.linspace(0.0, centerline_total_len, n_segments, endpoint=False)
    cx = interp_closed_scalar(cs_old, cx_old, s_center)
    cy = interp_closed_scalar(cs_old, cy_old, s_center)

    psi_center = compute_heading(cx, cy)
    ds_center = centerline_total_len / n_segments

    # signed curvature from centerline
    kappa_signed = circular_gradient(psi_center, ds_center)
    kappa_abs = np.abs(kappa_signed)

    ds_arr = np.full(n_segments, ds_center, dtype=float)

    print("Projecting boundaries along centerline normals ...")
    left_proj, right_proj, d_left, d_right, left_bd_idx, right_bd_idx = \
        project_boundaries_to_normals(cx, cy, psi_center, left_bd, right_bd)

    width = d_left + d_right

    # thresholds
    kappa_straight = float(np.quantile(kappa_abs, straight_quantile))
    kappa_turn = float(np.quantile(kappa_abs, turn_quantile))

    mean_curvature = np.zeros(n_segments, dtype=float)
    max_curvature = np.zeros(n_segments, dtype=float)
    straight_indicator = np.zeros(n_segments, dtype=float)
    compound_indicator = np.zeros(n_segments, dtype=float)
    accumulated_heading_change = np.zeros(n_segments, dtype=float)

    for i in range(n_segments):
        idx = (i + np.arange(window_size)) % n_segments

        k_win_abs = kappa_abs[idx]
        k_win_signed = kappa_signed[idx]
        ds_win = ds_arr[idx]

        # 1) mean curvature
        mean_curvature[i] = np.mean(k_win_abs)

        # 2) max curvature
        max_curvature[i] = np.max(k_win_abs)

        # 3) straight indicator: longest consecutive straight run in full window (per spec)
        b = (k_win_abs < kappa_straight).astype(int)
        straight_indicator[i] = longest_consecutive_ones(b) / float(window_size)

        # 4) compound / continuous corner indicator
        turn_mask = k_win_abs > kappa_turn
        n_turn = int(np.sum(turn_mask))
        n_switch = count_sign_switches(k_win_signed, turn_mask)

        turn_term = n_turn / float(window_size)
        switch_term = (n_switch / float(max(n_turn - 1, 1))) if n_turn > 0 else 0.0
        compound_indicator[i] = alpha * turn_term + (1.0 - alpha) * switch_term

        # 5) accumulated heading change
        accumulated_heading_change[i] = np.sum(k_win_abs * ds_win)

    x_context = np.column_stack([
        mean_curvature,
        max_curvature,
        straight_indicator,
        compound_indicator,
        accumulated_heading_change,
    ])

    df = pd.DataFrame({
        "segment_id": np.arange(n_segments),
        "s_center": s_center,
        "centerline_x": cx,
        "centerline_y": cy,
        "psi_center": psi_center,
        "kappa_signed_centerline": kappa_signed,
        "kappa_abs_centerline": kappa_abs,
        "width": width,
        "d_left": d_left,
        "d_right": d_right,

        "left_proj_x": left_proj[:, 0],
        "left_proj_y": left_proj[:, 1],
        "right_proj_x": right_proj[:, 0],
        "right_proj_y": right_proj[:, 1],
        "left_bd_idx": left_bd_idx,
        "right_bd_idx": right_bd_idx,

        "mean_curvature_30": mean_curvature,
        "max_curvature_30": max_curvature,
        "straight_indicator_30": straight_indicator,
        "compound_indicator_30": compound_indicator,
        "accum_heading_change_30": accumulated_heading_change,
    })

    meta = {
        "kappa_straight": kappa_straight,
        "kappa_turn": kappa_turn,
        "window_size": window_size,
        "alpha": alpha,
        "straight_quantile": straight_quantile,
        "turn_quantile": turn_quantile,
    }

    return x_context, df, meta


# =========================
# Run directly
# =========================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    for path in [RACELINE_CSV, CENTERLINE_CSV, LEFT_BOUNDARY_CSV, RIGHT_BOUNDARY_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

    x_context, df_context, meta = compute_lookahead_context_features(
        raceline_csv=RACELINE_CSV,
        centerline_csv=CENTERLINE_CSV,
        left_boundary_csv=LEFT_BOUNDARY_CSV,
        right_boundary_csv=RIGHT_BOUNDARY_CSV,
        n_segments=600,
        window_size=30,
        alpha=0.5,
        straight_quantile=0.25,
        turn_quantile=0.60,
    )

    print("\nDone.")
    print("x_context shape:", x_context.shape)
    print(df_context.head())

    df_context.to_csv(OUTPUT_FEATURE_CSV, index=False)
    np.save(OUTPUT_FEATURE_NPY, x_context)

    print(f"\nSaved CSV to: {OUTPUT_FEATURE_CSV}")
    print(f"Saved NPY to: {OUTPUT_FEATURE_NPY}")
    print(f"Thresholds used: {meta}")