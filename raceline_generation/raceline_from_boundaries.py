#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate minimum-curvature raceline directly from boundary CSVs.

Input:  boundary_output/
          inner_boundary.csv   [x, y]
          outer_boundary.csv   [x, y]
          centerline.csv       [x, y, s]

Output: raceline_output/
          raceline.csv
          raceline_debug.png
          raceline_velocity.png
          summary.txt
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

# ── quadprog shim: inject before tph imports it ───────────────────────────────
try:
    import quadprog as _qp_check  # noqa: F401
except (ImportError, OSError):
    _qp_mod = types.ModuleType("quadprog")

    def _solve_qp(G, a, C=None, b=None, meq=0):
        from scipy.optimize import minimize as _min
        G = np.asarray(G, dtype=float)
        a = np.asarray(a, dtype=float)
        n = len(a)
        cons = []
        if C is not None and b is not None:
            C = np.asarray(C, dtype=float)
            b = np.asarray(b, dtype=float)
            m = C.shape[1]
            for i in range(meq):
                cons.append({"type": "eq",
                             "fun": lambda x, _i=i: float(C[:, _i] @ x - b[_i]),
                             "jac": lambda x, _i=i: C[:, _i]})
            for i in range(meq, m):
                cons.append({"type": "ineq",
                             "fun": lambda x, _i=i: float(C[:, _i] @ x - b[_i]),
                             "jac": lambda x, _i=i: C[:, _i]})
        res = _min(lambda x: 0.5 * x @ G @ x - a @ x, np.zeros(n),
                   jac=lambda x: G @ x - a, constraints=cons, method="SLSQP",
                   options={"ftol": 1e-10, "maxiter": 2000, "disp": False})
        if not res.success:
            res = _min(lambda x: 0.5 * x @ G @ x - a @ x, np.zeros(n),
                       jac=lambda x: G @ x - a, constraints=cons, method="SLSQP",
                       options={"ftol": 1e-7, "maxiter": 5000, "disp": False})
        return (res.x, None, None, None, None, None)

    _qp_mod.solve_qp = _solve_qp
    sys.modules["quadprog"] = _qp_mod
    print("[INFO] quadprog not available — using scipy SLSQP fallback.")

import trajectory_planning_helpers as tph  # noqa: E402


# ── helpers ───────────────────────────────────────────────────────────────────

def moving_average_circular(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    pad = k // 2
    padded = np.concatenate([x[-pad:], x, x[:pad]])
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(padded, kernel, mode="same")[pad: pad + len(x)]


# ── loading ───────────────────────────────────────────────────────────────────

def load_boundaries(boundary_dir: str):
    inner = np.loadtxt(os.path.join(boundary_dir, "inner_boundary.csv"),
                       delimiter=",", skiprows=1)
    outer = np.loadtxt(os.path.join(boundary_dir, "outer_boundary.csv"),
                       delimiter=",", skiprows=1)
    cl_data = np.loadtxt(os.path.join(boundary_dir, "centerline.csv"),
                         delimiter=",", skiprows=1)
    centerline = cl_data[:, :2]
    return inner[:, :2], outer[:, :2], centerline


# ── reftrack construction ─────────────────────────────────────────────────────

def _signed_area(pts: np.ndarray) -> float:
    """Shoelace formula — positive = CCW, negative = CW."""
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def shrink_boundary(pts: np.ndarray, centerline: np.ndarray,
                    shrink_dist: float) -> np.ndarray:
    """
    Move each boundary point toward the nearest centerline point by shrink_dist.
    This offsets the boundary inward (toward the track corridor) creating a
    safety buffer so the raceline never touches the physical wall.
    """
    cl_tree = KDTree(centerline)
    _, idx = cl_tree.query(pts)
    nearest_cl = centerline[idx]
    directions = nearest_cl - pts
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.maximum(norms, 1e-8)
    return pts + directions * shrink_dist


def compute_reftrack(centerline: np.ndarray,
                     inner_pts: np.ndarray,
                     outer_pts: np.ndarray,
                     safety_margin: float = 0.15,
                     subsample: int = 400) -> np.ndarray:
    """
    Build reftrack [N, 4] = [x, y, w_right, w_left] from boundary CSVs.

    Strategy: determine whether the centerline is CCW or CW, then assign
    inner / outer boundary distances to the correct side.

    For a CCW loop the right-hand normal points INWARD, so:
        w_right = distance to inner boundary   (inward side)
        w_left  = distance to outer boundary   (outward side)

    For a CW loop the mapping is reversed.

    This avoids the ambiguity of per-point signed-projection checks and
    is robust to highly curved sections.
    """
    # Down-sample centerline to a manageable number of points.
    if subsample and len(centerline) > subsample:
        idx = np.round(np.linspace(0, len(centerline) - 1, subsample)).astype(int)
        centerline = centerline[idx]

    n = len(centerline)

    # Determine track winding
    area = _signed_area(centerline)
    ccw = area > 0
    print(f"[INFO] Centerline signed area = {area:.1f} m² → "
          f"{'CCW (right normal = inward)' if ccw else 'CW (right normal = outward)'}")

    # KD-trees for fast nearest-neighbour queries
    inner_tree = KDTree(inner_pts)
    outer_tree = KDTree(outer_pts)

    # Distance from every centerline point to each boundary
    d_inner, _ = inner_tree.query(centerline)
    d_outer, _ = outer_tree.query(centerline)

    if ccw:
        # Right normal points inward → inner is on the right
        hw_right = d_inner
        hw_left  = d_outer
    else:
        # Right normal points outward → outer is on the right
        hw_right = d_outer
        hw_left  = d_inner

    # Safety margin subtracted, hard floor so QP stays feasible
    hw_right = np.maximum(hw_right - safety_margin, 0.4)
    hw_left  = np.maximum(hw_left  - safety_margin, 0.4)

    # Smooth to remove sample-spacing noise
    hw_right = moving_average_circular(hw_right, 11)
    hw_left  = moving_average_circular(hw_left,  11)
    hw_right = np.maximum(hw_right, 0.4)
    hw_left  = np.maximum(hw_left,  0.4)

    print(f"[INFO] Reftrack: {n} pts | "
          f"w_right {hw_right.min():.2f}–{hw_right.max():.2f} m | "
          f"w_left {hw_left.min():.2f}–{hw_left.max():.2f} m")

    return np.column_stack([centerline, hw_right, hw_left])


# ── minimum-curvature optimisation ───────────────────────────────────────────

def _refresh_halfwidths(rt_xy: np.ndarray, _normvec: np.ndarray,
                        inner_tree: KDTree, outer_tree: KDTree,
                        margin: float, ccw: bool) -> tuple[np.ndarray, np.ndarray]:
    """Recompute half-widths at given positions using KDTree Euclidean distances.
    For CCW: normvec points right = inward → hw_right = inner dist, hw_left = outer dist.
    """
    d_inner, _ = inner_tree.query(rt_xy)
    d_outer, _ = outer_tree.query(rt_xy)
    if ccw:
        hw_r, hw_l = d_inner, d_outer
    else:
        hw_r, hw_l = d_outer, d_inner
    hw_r = np.maximum(hw_r - margin, 0.10)
    hw_l = np.maximum(hw_l - margin, 0.10)
    hw_r = moving_average_circular(hw_r, 11)
    hw_l = moving_average_circular(hw_l, 11)
    return np.maximum(hw_r, 0.10), np.maximum(hw_l, 0.10)


def run_min_curv(reftrack: np.ndarray,
                 stepsize_reg: float = 1.0,
                 stepsize_interp: float = 0.1,
                 kappa_bound: float = 0.4,
                 w_veh: float = 0.4,
                 iters: int = 3,  # kept for CLI compat, unused with SLSQP fallback
                 inner_tree=None,
                 outer_tree=None,
                 margin: float = 0.40,
                 ccw: bool = True) -> dict:

    print("[INFO] Resampling reftrack ...")
    rt = tph.interp_track.interp_track(track=reftrack, stepsize=stepsize_reg)

    # Gentle smoothing — too large a kernel causes normal crossings at tight corners
    k_sm = max(3, int(stepsize_reg * 3))
    for col in (0, 1):
        rt[:, col] = moving_average_circular(rt[:, col], k_sm)
    print(f"[INFO]   {len(rt)} pts after resample (smooth k={k_sm}).")

    # Initial splines + normal vectors
    refpath_cl = np.vstack([rt[:, :2], rt[0, :2]])
    coeffs_x, coeffs_y, A, normvec = tph.calc_splines.calc_splines(path=refpath_cl)

    cross = tph.check_normals_crossing.check_normals_crossing(
        track=rt, normvec_normalized=normvec, horizon=10)
    if cross:
        print("[WARN] Normal vectors crossing on initial reftrack.")

    # ── Single QP solve (SLSQP fallback is not stable enough for re-linearisation) ──
    print("[INFO] Min-curv QP ...")
    if inner_tree is not None and outer_tree is not None:
        hw_r, hw_l = _refresh_halfwidths(rt[:, :2], normvec,
                                          inner_tree, outer_tree, margin, ccw)
        rt[:, 2] = hw_r
        rt[:, 3] = hw_l

    alpha_opt = np.zeros(len(rt))
    try:
        alpha_opt, curv_err = tph.opt_min_curv.opt_min_curv(
            reftrack=rt,
            normvectors=normvec,
            A=A,
            kappa_bound=kappa_bound,
            w_veh=w_veh,
            print_debug=True,
            closed=True,
        )
        print(f"[INFO] QP curv_error={curv_err:.6f}")
    except Exception as exc:
        print(f"[WARN] QP failed: {exc}. Using centerline.")

    # Light smoothing of alpha (quadprog gives clean solutions; k=11 removes spline-edge noise)
    alpha_opt = moving_average_circular(alpha_opt, 11)
    a_max = np.maximum(rt[:, 2] - w_veh / 2.0, 0.0)
    a_min = -np.maximum(rt[:, 3] - w_veh / 2.0, 0.0)
    alpha_opt = np.clip(alpha_opt, a_min, a_max)

    # ── Build final interpolated raceline ──────────────────────────────────────
    print("[INFO] Building final raceline ...")
    (raceline, _A_rl, _cx_rl, _cy_rl,
     spl_idx, _t, s_rl, _slen, el_rl) = tph.create_raceline.create_raceline(
        refline=rt[:, :2],
        normvectors=normvec,
        alpha=alpha_opt,
        stepsize_interp=stepsize_interp,
    )

    # Post-smooth xy to remove any residual kinks from spline boundary effects
    for col in (0, 1):
        raceline[:, col] = moving_average_circular(raceline[:, col], 7)

    # Curvature (numerical)
    _, kappa = tph.calc_head_curv_num.calc_head_curv_num(
        path=raceline,
        el_lengths=el_rl,
        is_closed=True,
        stepsize_psi_preview=1.0,
        stepsize_psi_review=1.0,
        stepsize_curv_preview=2.0,
        stepsize_curv_review=2.0,
    )
    kappa = moving_average_circular(kappa, 5)

    # Velocity profile (F1TENTH params)
    v_max = 8.0
    ax_max = 4.0
    ay_max = 4.0
    ggv = np.array([[0.0, ax_max, ay_max], [v_max, ax_max, ay_max]])
    ax_machines = np.array([[0.0, ax_max], [v_max, ax_max]])
    vx = tph.calc_vel_profile.calc_vel_profile(
        ax_max_machines=ax_machines,
        kappa=kappa,
        el_lengths=el_rl,
        closed=True,
        drag_coeff=0.004,
        m_veh=3.5,
        ggv=ggv,
        v_max=v_max,
    )

    return {
        "raceline_xy": raceline,
        "s_raceline": s_rl,
        "kappa_raceline": kappa,
        "vx_profile": vx,
        "el_lengths": el_rl,
        "reftrack_interp": rt,
        "normvec": normvec,
        "alpha_opt": alpha_opt,
    }


# ── boundary-violation check ──────────────────────────────────────────────────

def check_and_clip(result: dict,
                   inner_pts: np.ndarray,
                   outer_pts: np.ndarray,
                   w_veh: float = 0.4) -> dict:
    rl = result["raceline_xy"]
    inner_tree = KDTree(inner_pts)
    outer_tree = KDTree(outer_pts)
    d_in, _ = inner_tree.query(rl)
    d_out, _ = outer_tree.query(rl)
    print(f"[CHECK] Min dist to inner: {d_in.min():.3f} m  |  Min dist to outer: {d_out.min():.3f} m")
    half_w = w_veh / 2.0
    bad = int(np.sum((d_in < half_w) | (d_out < half_w)))
    if bad:
        print(f"[WARN]  {bad} points within {half_w:.2f} m of a wall.")
    else:
        print("[CHECK] All raceline points are within track boundaries.")
    return result


# ── output ────────────────────────────────────────────────────────────────────

def save_outputs(outdir: str, result: dict,
                 inner_pts: np.ndarray, outer_pts: np.ndarray,
                 left_bnd: np.ndarray, right_bnd: np.ndarray) -> None:
    """
    left_bnd  = shrunk outer boundary  (car's left  wall, CCW track)
    right_bnd = shrunk inner boundary  (car's right wall, CCW track)
    """
    os.makedirs(outdir, exist_ok=True)

    rl = result["raceline_xy"]
    s  = result["s_raceline"]
    kappa = result["kappa_raceline"]
    vx = result["vx_profile"]
    el = result["el_lengths"]
    ref = result["reftrack_interp"]

    lap_time = float(np.sum(el / np.maximum(vx, 0.01)))
    print(f"[INFO] Estimated lap time : {lap_time:.2f} s")
    print(f"[INFO] Raceline length    : {s[-1]:.2f} m")
    print(f"[INFO] Max |curvature|    : {np.abs(kappa).max():.4f} 1/m")
    print(f"[INFO] Speed range        : {vx.min():.2f} – {vx.max():.2f} m/s")

    # Raceline CSV --------------------------------------------------------------
    with open(os.path.join(outdir, "raceline.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_m", "y_m", "s_m", "kappa_radpm", "vx_mps"])
        writer.writerows(np.column_stack([rl, s, kappa, vx]).tolist())

    # Left / right boundary CSVs ------------------------------------------------
    with open(os.path.join(outdir, "left_boundary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(left_bnd.tolist())

    with open(os.path.join(outdir, "right_boundary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(right_bnd.tolist())

    print(f"[INFO] Saved left_boundary.csv ({len(left_bnd)} pts) and "
          f"right_boundary.csv ({len(right_bnd)} pts)")

    # Summary -------------------------------------------------------------------
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Min-curvature raceline (from boundary CSVs)\n")
        f.write(f"Raceline points  : {len(rl)}\n")
        f.write(f"Raceline length  : {s[-1]:.3f} m\n")
        f.write(f"Lap time est.    : {lap_time:.2f} s\n")
        f.write(f"Speed range      : {vx.min():.2f} – {vx.max():.2f} m/s\n")
        f.write(f"Max |curvature|  : {np.abs(kappa).max():.4f} 1/m\n")

    # Overlay plot (world coordinates) ------------------------------------------
    # Show physical walls (faded) + effective shrunk walls + raceline
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.plot(outer_pts[:, 0], outer_pts[:, 1], color="deepskyblue", lw=1.0,
            ls="--", alpha=0.4, label="outer wall (physical)")
    ax.plot(inner_pts[:, 0], inner_pts[:, 1], color="orange", lw=1.0,
            ls="--", alpha=0.4, label="inner wall (physical)")
    ax.plot(left_bnd[:, 0], left_bnd[:, 1], color="deepskyblue", lw=1.8,
            label="left boundary (effective)")
    ax.plot(right_bnd[:, 0], right_bnd[:, 1], color="orange", lw=1.8,
            label="right boundary (effective)")
    ax.plot(ref[:, 0], ref[:, 1], color="yellow", lw=1.0, ls="--", alpha=0.6,
            label="centerline (ref)")
    ax.plot(rl[:, 0], rl[:, 1], color="red", lw=2.0, label="min-curvature raceline")
    ax.set_aspect("equal")
    ax.legend(fontsize=9, loc="best")
    ax.set_title("Minimum-Curvature Raceline (from boundary CSVs)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "raceline_debug.png"), dpi=160)
    plt.close(fig)

    # Velocity-coloured scatter -------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.plot(left_bnd[:, 0], left_bnd[:, 1], color="deepskyblue", lw=1.2, alpha=0.7)
    ax.plot(right_bnd[:, 0], right_bnd[:, 1], color="orange", lw=1.2, alpha=0.7)
    sc = ax.scatter(rl[:, 0], rl[:, 1], c=vx, cmap="RdYlGn", s=6, zorder=3)
    plt.colorbar(sc, ax=ax, label="velocity [m/s]")
    ax.set_aspect("equal")
    ax.set_title("Raceline — velocity profile")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "raceline_velocity.png"), dpi=160)
    plt.close(fig)

    # Curvature + speed profiles ------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(s, kappa, color="tab:blue")
    axes[0].set_ylabel("curvature [1/m]")
    axes[0].set_title("Raceline profiles")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(s, vx, color="tab:red")
    axes[1].set_ylabel("velocity [m/s]")
    axes[1].set_xlabel("arc length [m]")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "curvature_velocity.png"), dpi=150)
    plt.close(fig)

    print(f"[INFO] Outputs saved to: {outdir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Min-curvature raceline from boundary CSVs.")
    p.add_argument("--boundary_dir", default="/Users/shure_duan/VScode/f1tenth/boundary_output",
                   help="Folder containing inner/outer/centerline CSVs")
    p.add_argument("--outdir", default="/Users/shure_duan/VScode/f1tenth/raceline_generation/raceline_output",
                   help="Output folder")
    p.add_argument("--margin",  type=float, default=0.40,
                   help="Safety margin subtracted from half-widths [m] (default 0.40)")
    p.add_argument("--w_veh",   type=float, default=0.35,
                   help="Vehicle width [m] (default 0.35)")
    p.add_argument("--stepsize_reg",    type=float, default=1.0,
                   help="Uniform resample spacing for QP [m] (default 1.0)")
    p.add_argument("--stepsize_interp", type=float, default=0.1,
                   help="Final raceline interpolation spacing [m] (default 0.1)")
    p.add_argument("--kappa_bound",     type=float, default=0.4,
                   help="Max curvature [1/m] (default 0.4)")
    p.add_argument("--iters",  type=int,   default=3,
                   help="QP re-linearisation iterations (default 3)")
    p.add_argument("--subsample", type=int, default=400,
                   help="Downsample centerline to this many pts before QP (default 400)")
    return p.parse_args()


def main():
    args = parse_args()

    print("[INFO] Loading boundary CSVs ...")
    inner_pts, outer_pts, centerline = load_boundaries(args.boundary_dir)
    print(f"[INFO]   inner={len(inner_pts)} outer={len(outer_pts)} cl={len(centerline)} pts")

    # Shrink physical boundaries inward by margin → effective corridor walls
    print(f"[INFO] Shrinking boundaries inward by {args.margin:.2f} m ...")
    shrunk_inner = shrink_boundary(inner_pts, centerline, args.margin)
    shrunk_outer = shrink_boundary(outer_pts, centerline, args.margin)

    area = _signed_area(centerline)
    ccw = area > 0
    # For CCW: right normal = inward → right boundary = (shrunk) inner, left = (shrunk) outer
    right_bnd = shrunk_inner if ccw else shrunk_outer
    left_bnd  = shrunk_outer if ccw else shrunk_inner

    print("[INFO] Computing reftrack half-widths from shrunk boundaries ...")
    # Use the shrunk boundaries as the effective walls for the QP
    reftrack = compute_reftrack(centerline, shrunk_inner, shrunk_outer,
                                safety_margin=0.0,   # margin already applied in shrink
                                subsample=args.subsample)

    inner_tree_shrunk = KDTree(shrunk_inner)
    outer_tree_shrunk = KDTree(shrunk_outer)

    result = run_min_curv(
        reftrack=reftrack,
        stepsize_reg=args.stepsize_reg,
        stepsize_interp=args.stepsize_interp,
        kappa_bound=args.kappa_bound,
        w_veh=args.w_veh,
        iters=args.iters,
        inner_tree=inner_tree_shrunk,
        outer_tree=outer_tree_shrunk,
        margin=0.0,   # margin already applied
        ccw=ccw,
    )

    print("[INFO] Verifying boundary compliance (against physical walls) ...")
    # Check against physical (un-shrunk) boundaries — the real walls
    result = check_and_clip(result, inner_pts, outer_pts, w_veh=args.w_veh)

    save_outputs(args.outdir, result, inner_pts, outer_pts, left_bnd, right_bnd)


if __name__ == "__main__":
    main()
