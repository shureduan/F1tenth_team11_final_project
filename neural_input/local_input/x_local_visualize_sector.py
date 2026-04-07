import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

RACELINE_DIR = "/Users/shure_duan/VScode/f1tenth/raceline_output"

XLOCAL_CSV = "/Users/shure_duan/VScode/f1tenth/neural_input/local_input/local_result.csv"
LEFT_BOUNDARY_CSV = os.path.join(RACELINE_DIR, "left_boundary.csv")
RIGHT_BOUNDARY_CSV = os.path.join(RACELINE_DIR, "right_boundary.csv")

OUT_FIG1 = os.path.join(RACELINE_DIR, "segment_abs_curvature_track_heatmap.png")
OUT_FIG2 = os.path.join(RACELINE_DIR, "segment_inner_space_dkappa_track_panels.png")


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

        left_arc = left_bd[cyclic_short_arc_indices(li, lj, nl)]
        right_arc = right_bd[cyclic_short_arc_indices(ri, rj, nr)]

        poly = np.vstack([left_arc, right_arc[::-1]])
        polys.append(poly.astype(float))
    return polys


def circular_gaussian_smooth(v, sigma=2.2, radius=None):
    v = np.asarray(v, dtype=float)
    n = len(v)
    if n == 0:
        return v.copy()
    if sigma <= 0:
        return v.copy()
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * sigma)))

    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= np.sum(kernel)

    vp = np.r_[v[-radius:], v, v[:radius]]
    out = np.convolve(vp, kernel, mode="valid")
    return out[:n]


def percentile_sqrt_boost(values):
    """
    Convert values to percentile score in [0, 100], then map with:
        score = 13 * sqrt(pct) - 30
    So:
        64 -> 74
        100 -> 100
    Low values are clipped to [0, 100].
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 1:
        v = v.ravel()

    out = np.full_like(v, np.nan, dtype=float)
    finite_mask = np.isfinite(v)

    if not np.any(finite_mask):
        return out

    vf = v[finite_mask]
    n = len(vf)

    if n == 1:
        pct = np.array([100.0], dtype=float)
    else:
        order = np.argsort(vf, kind="mergesort")
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        pct = 100.0 * ranks / (n - 1)

    boosted = 13.0 * np.sqrt(pct) - 30.0
    boosted = np.clip(boosted, 0.0, 100.0)
    out[finite_mask] = boosted
    return out


def add_track_heatmap(ax, left_bd, right_bd, left_idx, right_idx, values,
                      cmap="turbo", vmin=None, vmax=None, zorder=2):
    polys = build_track_sector_polygons(left_bd, right_bd, left_idx, right_idx)
    vals = np.asarray(values, dtype=float)

    if len(vals) != len(polys):
        raise ValueError(f"Value count ({len(vals)}) does not match segment count ({len(polys)}).")

    if vmin is None:
        vmin = np.nanmin(vals)
    if vmax is None:
        vmax = np.nanmax(vals)

    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6

    norm = Normalize(vmin=vmin, vmax=vmax)
    pc = PolyCollection(
        polys,
        array=vals,
        cmap=cmap,
        norm=norm,
        edgecolors="face",
        linewidths=0.2,
        zorder=zorder,
    )
    ax.add_collection(pc)
    return pc


def draw_track_outline(ax, left_xy, right_xy, color="black", lw=1.0, alpha=0.95, zorder=4):
    lx = np.r_[left_xy[:, 0], left_xy[0, 0]]
    ly = np.r_[left_xy[:, 1], left_xy[0, 1]]
    rx = np.r_[right_xy[:, 0], right_xy[0, 0]]
    ry = np.r_[right_xy[:, 1], right_xy[0, 1]]
    ax.plot(lx, ly, color=color, lw=lw, alpha=alpha, zorder=zorder)
    ax.plot(rx, ry, color=color, lw=lw, alpha=alpha, zorder=zorder)


def draw_centerline(ax, x, y, color="white", lw=0.8, alpha=0.7, zorder=5):
    xc = np.r_[x, x[0]]
    yc = np.r_[y, y[0]]
    ax.plot(xc, yc, color=color, lw=lw, alpha=alpha, zorder=zorder)


def build_dense_adaptive_arrows(x, y, delta_psi):
    a0 = normalize01(np.abs(delta_psi))

    # contrast enhancement: density weight *10, then sqrt, then renormalize
    a = np.sqrt(10.0 * a0)
    a = a / (np.nanmax(a) + 1e-12)

    ux, uy = compute_forward_tangent_unit_vectors(x, y)
    X, Y, U, V = [], [], [], []
    n = len(x)
    i = 0

    while i < n:
        ai = a[i]
        j = (i + 1) % n

        if ai < 0.12:
            X.append(x[i]); Y.append(y[i]); U.append(ux[i]); V.append(uy[i]); i += 3
        elif ai < 0.22:
            X.append(x[i]); Y.append(y[i]); U.append(ux[i]); V.append(uy[i]); i += 2
        elif ai < 0.38:
            X.append(x[i]); Y.append(y[i]); U.append(ux[i]); V.append(uy[i]); i += 1
        elif ai < 0.60:
            X.append(x[i]); Y.append(y[i]); U.append(ux[i]); V.append(uy[i])
            xm = 0.5 * (x[i] + x[j]); ym = 0.5 * (y[i] + y[j])
            um = x[j] - x[i]; vm = y[j] - y[i]
            nn = np.sqrt(um**2 + vm**2) + 1e-12
            X.append(xm); Y.append(ym); U.append(um / nn); V.append(vm / nn)
            i += 1
        elif ai < 0.82:
            X.append(x[i]); Y.append(y[i]); U.append(ux[i]); V.append(uy[i])
            x1 = x[i] + (x[j] - x[i]) / 3.0; y1 = y[i] + (y[j] - y[i]) / 3.0
            x2 = x[i] + 2.0 * (x[j] - x[i]) / 3.0; y2 = y[i] + 2.0 * (y[j] - y[i]) / 3.0
            um = x[j] - x[i]; vm = y[j] - y[i]
            nn = np.sqrt(um**2 + vm**2) + 1e-12
            um /= nn; vm /= nn
            X.extend([x1, x2]); Y.extend([y1, y2]); U.extend([um, um]); V.extend([vm, vm])
            i += 1
        else:
            X.append(x[i]); Y.append(y[i]); U.append(ux[i]); V.append(uy[i])
            x1 = x[i] + 0.25 * (x[j] - x[i]); y1 = y[i] + 0.25 * (y[j] - y[i])
            x2 = x[i] + 0.50 * (x[j] - x[i]); y2 = y[i] + 0.50 * (y[j] - y[i])
            x3 = x[i] + 0.75 * (x[j] - x[i]); y3 = y[i] + 0.75 * (y[j] - y[i])
            um = x[j] - x[i]; vm = y[j] - y[i]
            nn = np.sqrt(um**2 + vm**2) + 1e-12
            um /= nn; vm /= nn
            X.extend([x1, x2, x3]); Y.extend([y1, y2, y3]); U.extend([um, um, um]); V.extend([vm, vm, vm])
            i += 1

    return np.array(X), np.array(Y), np.array(U), np.array(V)


def main():
    for path in [XLOCAL_CSV, LEFT_BOUNDARY_CSV, RIGHT_BOUNDARY_CSV]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(XLOCAL_CSV)
    left_df = pd.read_csv(LEFT_BOUNDARY_CSV)
    right_df = pd.read_csv(RIGHT_BOUNDARY_CSV)

    required = [
        "x", "y", "kappa", "delta_kappa", "d_left", "d_right", "delta_psi",
        "left_bd_idx", "right_bd_idx"
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"x_local_features.csv missing required column: {col}")

    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    kappa = df["kappa"].to_numpy(dtype=float)
    delta_kappa = df["delta_kappa"].to_numpy(dtype=float)
    d_left = df["d_left"].to_numpy(dtype=float)
    d_right = df["d_right"].to_numpy(dtype=float)
    delta_psi = df["delta_psi"].to_numpy(dtype=float)
    left_idx = df["left_bd_idx"].to_numpy(dtype=int)
    right_idx = df["right_bd_idx"].to_numpy(dtype=int)

    left_xy_raw = left_df[["x", "y"]].to_numpy(dtype=float)
    right_xy_raw = right_df[["x", "y"]].to_numpy(dtype=float)

    # Stronger smoothing to reduce blocky jumps in curvature color
    abs_kappa = np.abs(kappa)
    abs_kappa_smooth = circular_gaussian_smooth(abs_kappa, sigma=2.2)
    abs_kappa_smooth = circular_gaussian_smooth(abs_kappa_smooth, sigma=1.4)
    abs_kappa_boost = percentile_sqrt_boost(abs_kappa_smooth)

    inner_space = compute_inner_space(kappa, d_left, d_right)
    all_xy = np.vstack([left_xy_raw, right_xy_raw])

    fig1, ax1 = plt.subplots(figsize=(12, 10))
    pc1 = add_track_heatmap(
        ax1,
        left_xy_raw, right_xy_raw,
        left_idx, right_idx,
        abs_kappa_boost,
        cmap="turbo",
        vmin=0.0,
        vmax=100.0,
        zorder=2,
    )
    cbar1 = fig1.colorbar(pc1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Boosted absolute curvature score [0-100]")

    draw_track_outline(ax1, left_xy_raw, right_xy_raw, color="black", lw=1.0, alpha=0.95, zorder=4)
    draw_centerline(ax1, x, y, color="white", lw=0.9, alpha=0.65, zorder=5)

    Xa, Ya, Ua, Va = build_dense_adaptive_arrows(x, y, delta_psi)
    diag = np.sqrt((np.max(all_xy[:, 0]) - np.min(all_xy[:, 0]))**2 + (np.max(all_xy[:, 1]) - np.min(all_xy[:, 1]))**2)
    arrow_len = 0.008 * diag
    ax1.quiver(
        Xa, Ya, Ua * arrow_len, Va * arrow_len,
        angles="xy", scale_units="xy", scale=1.0,
        color="black", width=0.0014,
        headwidth=4.0, headlength=5.2, headaxislength=4.8,
        alpha=0.95, zorder=6,
    )
    style_axis(ax1, "Track-based Absolute Curvature Heatmap")
    add_margin_from_xy(ax1, all_xy[:, 0], all_xy[:, 1], ratio=0.04)
    plt.tight_layout()
    fig1.savefig(OUT_FIG1, dpi=300, bbox_inches="tight")
    plt.close(fig1)

    fig2, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    panel_specs = [
        ("Inner-line space", inner_space, "viridis", "Inner distance [m]"),
        ("Curvature variation $\\Delta\\kappa_{center}$", delta_kappa, "coolwarm", "$\\Delta\\kappa$ [1/m$^2$]"),
    ]
    for ax, (title, vals, cmap, cbar_label) in zip(np.ravel(axes), panel_specs):
        pc = add_track_heatmap(ax, left_xy_raw, right_xy_raw, left_idx, right_idx, vals, cmap=cmap, zorder=2)
        cbar = fig2.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label)
        draw_track_outline(ax, left_xy_raw, right_xy_raw, color="black", lw=1.0, alpha=0.95, zorder=4)
        draw_centerline(ax, x, y, color="white", lw=0.8, alpha=0.6, zorder=5)
        style_axis(ax, title)
        add_margin_from_xy(ax, all_xy[:, 0], all_xy[:, 1], ratio=0.04)

    plt.tight_layout()
    fig2.savefig(OUT_FIG2, dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print("Saved:")
    print(f"  {OUT_FIG1}")
    print(f"  {OUT_FIG2}")


if __name__ == "__main__":
    main()