#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a minimum-curvature raceline from a ROS/F1TENTH map (PNG + YAML).

Algorithm based on:
  Heilmeier et al., "Minimum Curvature Trajectory Planning and Control for an
  Autonomous Racecar", Vehicle System Dynamics, 2020.
  https://github.com/TUMFTM/global_racetrajectory_optimization

Pipeline
--------
1. Load PNG + YAML map.
2. Extract drivable corridor mask (handles both occupancy maps and boundary-line
   maps like the Shanghai drawing).
3. Skeletonize corridor -> centerline reference points (8-connectivity).
4. Compute left/right half-widths from the distance transform.
5. Build reftrack array [x, y, w_right, w_left] in world coordinates.
6. Run trajectory_planning_helpers spline approximation + smoothing.
7. Solve minimum-curvature QP (tph.opt_min_curv).
8. Compute curvature profile and velocity profile.
9. Save CSV / NPZ outputs and debug plots.

Dependencies
------------
pip install numpy scipy pyyaml opencv-python scikit-image matplotlib
pip install trajectory_planning_helpers   # installs quadprog as well
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Optional

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import ndimage
from skimage.morphology import skeletonize

# quadprog has an ABI issue on Python 3.13. Inject a scipy SLSQP shim before tph imports it.
import sys as _sys, types as _types
try:
    import quadprog as _qp_check  # noqa: F401
except (ImportError, OSError):
    _qp_mod = _types.ModuleType("quadprog")
    def _solve_qp(G, a, C=None, b=None, meq=0):
        import numpy as _np
        from scipy.optimize import minimize as _min
        G = _np.asarray(G, dtype=float); a = _np.asarray(a, dtype=float); n = len(a)
        cons = []
        if C is not None and b is not None:
            C = _np.asarray(C, dtype=float); b = _np.asarray(b, dtype=float); m = C.shape[1]
            for i in range(meq):
                cons.append({"type":"eq",  "fun":lambda x,_i=i: float(C[:,_i]@x-b[_i]), "jac":lambda x,_i=i: C[:,_i]})
            for i in range(meq, m):
                cons.append({"type":"ineq","fun":lambda x,_i=i: float(C[:,_i]@x-b[_i]), "jac":lambda x,_i=i: C[:,_i]})
        res = _min(lambda x: 0.5*x@G@x-a@x, _np.zeros(n), jac=lambda x: G@x-a,
                   constraints=cons, method="SLSQP", options={"ftol":1e-10,"maxiter":2000,"disp":False})
        if not res.success:
            res = _min(lambda x: 0.5*x@G@x-a@x, _np.zeros(n), jac=lambda x: G@x-a,
                       constraints=cons, method="SLSQP", options={"ftol":1e-7,"maxiter":5000,"disp":False})
        return (res.x, None, None, None, None, None)
    _qp_mod.solve_qp = _solve_qp
    _sys.modules["quadprog"] = _qp_mod
    print("[INFO] quadprog not available — using scipy SLSQP fallback.")

import trajectory_planning_helpers as tph


# ============================================================
# Map loading / preprocessing
# ============================================================

def load_ros_map(yaml_path: str, png_path: Optional[str] = None):
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if png_path is None:
        img_path = cfg["image"]
        if not os.path.isabs(img_path):
            img_path = os.path.join(os.path.dirname(yaml_path), img_path)
    else:
        img_path = png_path

    occ_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if occ_img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    resolution = float(cfg["resolution"])           # m/pixel
    origin = tuple(cfg["origin"])                   # (ox, oy, oz)
    negate = int(cfg.get("negate", 0))
    occupied_thresh = float(cfg.get("occupied_thresh", 0.65))
    free_thresh = float(cfg.get("free_thresh", 0.196))

    return occ_img, resolution, origin, negate, occupied_thresh, free_thresh


# ============================================================
# Corridor extraction
# ============================================================

def looks_like_boundary_line_map(occ_img: np.ndarray) -> bool:
    dark_ratio = float(np.mean(occ_img < 80))
    bright_ratio = float(np.mean(occ_img > 220))
    return dark_ratio < 0.08 and bright_ratio > 0.75


def build_track_mask_from_line_map(occ_img: np.ndarray) -> np.ndarray:
    """Extract corridor between two boundary lines."""
    boundary = (occ_img < 200).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.dilate(boundary, kernel, iterations=1)

    white = (1 - boundary).astype(np.uint8)
    labeled, num = ndimage.label(white)

    h, w = occ_img.shape
    corners = [labeled[0, 0], labeled[0, w-1], labeled[h-1, 0], labeled[h-1, w-1]]
    outside = max(set(corners), key=corners.count)

    best_mask, best_score = None, -1.0
    for lab in range(1, num + 1):
        if lab == outside:
            continue
        comp = (labeled == lab).astype(np.uint8)
        area = int(comp.sum())
        if area < 100:
            continue
        skel_len = int(skeletonize(comp > 0).sum())
        dist = ndimage.distance_transform_edt(comp > 0)
        mean_hw = float(dist[comp > 0].mean())
        score = skel_len - 30.0 * mean_hw
        if score > best_score:
            best_score = score
            best_mask = comp

    if best_mask is None:
        raise RuntimeError("Failed to isolate track corridor from boundary-line map.")

    best_mask = ndimage.binary_opening(best_mask, structure=np.ones((3, 3))).astype(np.uint8)
    best_mask = ndimage.binary_closing(best_mask, structure=np.ones((3, 3))).astype(np.uint8)
    return best_mask


def build_track_mask_from_occupancy_map(occ_img: np.ndarray, negate: int,
                                        free_thresh: float) -> np.ndarray:
    img = occ_img.astype(np.float32) / 255.0
    occ_prob = 1.0 - img if negate == 0 else img
    mask = (occ_prob < free_thresh).astype(np.uint8)

    labeled, num = ndimage.label(mask)
    if num > 1:
        sizes = ndimage.sum(mask, labeled, index=np.arange(1, num + 1))
        largest = int(np.argmax(sizes)) + 1
        mask = (labeled == largest).astype(np.uint8)

    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3))).astype(np.uint8)
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3))).astype(np.uint8)
    return mask


def build_track_mask(occ_img: np.ndarray, negate: int, free_thresh: float) -> np.ndarray:
    if looks_like_boundary_line_map(occ_img):
        print("[INFO] Detected boundary-line track map.")
        return build_track_mask_from_line_map(occ_img)
    else:
        print("[INFO] Detected occupancy-style map.")
        return build_track_mask_from_occupancy_map(occ_img, negate, free_thresh)


# ============================================================
# Centerline extraction
# ============================================================

def extract_centerline_pixels(track_mask: np.ndarray) -> np.ndarray:
    """
    Skeletonize the corridor mask using 8-connectivity (skeletons are
    diagonally connected, so 4-connectivity would fragment the result).
    Returns ordered pixel coordinates [row, col].
    """
    skel = skeletonize(track_mask > 0).astype(np.uint8)
    struct8 = np.ones((3, 3), dtype=int)
    labeled, num = ndimage.label(skel, structure=struct8)
    if num == 0:
        raise RuntimeError("No skeleton found in track corridor.")

    sizes = [(labeled == i).sum() for i in range(1, num + 1)]
    best_lab = int(np.argmax(sizes)) + 1
    best_len = sizes[best_lab - 1]
    print(f"[INFO] Skeleton: {num} components, largest has {best_len} pixels.")

    if best_len < 50:
        raise RuntimeError("Skeleton too small â check that the track corridor was extracted correctly.")

    return np.argwhere(labeled == best_lab)  # [N, 2] as (row, col)


def order_skeleton_graph(pts_rc: np.ndarray) -> np.ndarray:
    """
    Traverse the skeleton as a graph using 8-connectivity.
    At junction pixels (degree > 2) always continue along the branch that
    keeps the largest unvisited sub-path, which avoids the short spurious spurs.
    Returns pts_rc reordered as a closed loop.
    """
    pts = pts_rc.astype(np.int32)
    # Build index map: pixel (r,c) -> index in pts array
    idx_map = {}
    for i, (r, c) in enumerate(pts):
        idx_map[(int(r), int(c))] = i

    n = len(pts)
    # Adjacency: 8-connected neighbours that are also skeleton pixels
    def neighbors(r, c):
        nb = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if (nr, nc) in idx_map:
                    nb.append(idx_map[(nr, nc)])
        return nb

    # Start from degree-1 endpoint if any; otherwise leftmost point
    degrees = np.zeros(n, dtype=int)
    adj = [[] for _ in range(n)]
    for i, (r, c) in enumerate(pts):
        nbs = neighbors(r, c)
        adj[i] = nbs
        degrees[i] = len(nbs)

    endpoints = np.where(degrees == 1)[0]
    start = int(endpoints[0]) if len(endpoints) > 0 else int(np.argmin(pts[:, 1]))

    visited = np.zeros(n, dtype=bool)
    order = [start]
    visited[start] = True

    cur = start
    for _ in range(n - 1):
        nbs = [nb for nb in adj[cur] if not visited[nb]]
        if not nbs:
            break
        # Among unvisited neighbors, prefer those with more remaining unvisited neighbors
        # (greedy: avoids dead-end spurs)
        scores = [len([x for x in adj[nb] if not visited[x]]) for nb in nbs]
        nxt = nbs[int(np.argmax(scores))]
        order.append(nxt)
        visited[nxt] = True
        cur = nxt

    ordered = pts[order].astype(np.float32)
    return ordered


def moving_average_circular(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    pad = k // 2
    padded = np.concatenate([x[-pad:], x, x[:pad]])
    kernel = np.ones(k, dtype=np.float64) / k
    smoothed = np.convolve(padded, kernel, mode="same")
    return smoothed[pad: pad + len(x)]


def pixel_to_world(pts_rc: np.ndarray, resolution: float,
                   origin: tuple, img_h: int) -> np.ndarray:
    rows = pts_rc[:, 0]
    cols = pts_rc[:, 1]
    x = origin[0] + cols * resolution
    y = origin[1] + (img_h - 1 - rows) * resolution
    return np.stack([x, y], axis=1)


def sample_half_widths(track_mask: np.ndarray, xy_world: np.ndarray,
                       resolution: float, origin: tuple) -> np.ndarray:
    """Sample distance-to-wall in meters at each centerline world point."""
    dist_pix = ndimage.distance_transform_edt(track_mask > 0)
    h, w = track_mask.shape
    vals = []
    for x, y in xy_world:
        col = int(round((x - origin[0]) / resolution))
        row = int(round(h - 1 - (y - origin[1]) / resolution))
        row = int(np.clip(row, 0, h - 1))
        col = int(np.clip(col, 0, w - 1))
        vals.append(float(dist_pix[row, col]) * resolution)
    return np.array(vals, dtype=np.float64)


# ============================================================
# Build reftrack [x, y, w_right, w_left]
# ============================================================

def build_reftrack(yaml_path: str, png_path: Optional[str],
                   margin: float, ds_skeleton: float,
                   smooth_k: int):
    """
    Returns reftrack array [N, 4] = [x_m, y_m, w_tr_right_m, w_tr_left_m]
    This is the format expected by trajectory_planning_helpers.
    The array is UNCLOSED (last point != first point).
    """
    occ_img, resolution, origin, negate, _, free_thresh = load_ros_map(yaml_path, png_path)
    track_mask = build_track_mask(occ_img, negate, free_thresh)

    # Skeleton -> ordered centerline pixels
    pts_rc = extract_centerline_pixels(track_mask)
    ordered = order_skeleton_graph(pts_rc)

    # Sample EDT on original (unsmoothed) skeleton for conservative half-widths
    # Use a local minimum over a window to be conservative at corners
    dist_pix = ndimage.distance_transform_edt(track_mask > 0)
    hw_raw_pix = np.array([dist_pix[int(r), int(c)] for r, c in ordered])
    # Local minimum over window ~1.5m / resolution pixels to catch tight corners
    win = max(3, int(1.5 / resolution))
    hw_min_pix = np.array([hw_raw_pix[max(0,i-win):i+win+1].min() for i in range(len(hw_raw_pix))])
    hw_min_pix = moving_average_circular(hw_min_pix, 11)

    # Smooth the centerline
    rows_sm = moving_average_circular(ordered[:, 0], smooth_k)
    cols_sm = moving_average_circular(ordered[:, 1], smooth_k)
    ordered_sm = np.stack([rows_sm, cols_sm], axis=1)

    # To world
    xy_world = pixel_to_world(ordered_sm, resolution, origin, occ_img.shape[0])

    # Thin out to ds_skeleton spacing (avoids excessive points)
    step_pix = max(1, int(ds_skeleton / resolution))
    xy_world = xy_world[::step_pix]
    hw_sub = hw_min_pix[::step_pix]

    # Also sample EDT at the actual smoothed centerline positions â
    # smoothing can shift points away from the skeleton ridge, so EDT
    # there may be smaller than what the ridge-based hw_sub says.
    # Take the minimum of both to be conservative.
    hw_at_smooth = sample_half_widths(track_mask, xy_world, resolution, origin)
    hw = np.minimum(hw_sub * resolution, hw_at_smooth)
    hw = np.maximum(hw - margin, 0.05)

    reftrack = np.column_stack([xy_world, hw, hw])  # symmetric widths from centerline
    print(f"[INFO] Reftrack: {len(reftrack)} points, "
          f"track width range {2*hw.min():.2f}â{2*hw.max():.2f} m.")
    return reftrack, occ_img, track_mask, resolution, origin


# ============================================================
# Minimum curvature optimization
# ============================================================

# alias so run_min_curv can use the same helper
_moving_average_circular = moving_average_circular


def run_min_curv(reftrack: np.ndarray, stepsize_prep: float,
                 stepsize_reg: float, stepsize_interp: float,
                 kappa_bound: float, w_veh: float,
                 iters_mincurv: int,
                 track_mask: Optional[np.ndarray] = None,
                 resolution: float = 0.1,
                 origin: tuple = (0.0, 0.0, 0.0)) -> dict:
    """
    Run the TUMFTM minimum-curvature pipeline:
      1. Spline approximation / smoothing
      2. Normal vectors
      3. opt_min_curv QP
      4. Raceline interpolation
      5. Curvature + velocity profile
    """
    # ----- Linear interpolation to uniform stepsize + moving-average smoothing -----
    # (tph.spline_approximation has scipy compatibility issues; interp_track is stable)
    print("[INFO] Interpolating reftrack to uniform stepsize ...")
    reftrack_interp = tph.interp_track.interp_track(track=reftrack, stepsize=stepsize_reg)

    # Smooth x/y with circular moving average (widths stay as-is)
    k_smooth = max(3, int(stepsize_reg * 3))
    for col in [0, 1]:
        reftrack_interp[:, col] = _moving_average_circular(reftrack_interp[:, col], k_smooth)

    print(f"[INFO] Reftrack after interp: {len(reftrack_interp)} points.")

    # ----- Compute spline coefficients -----
    refpath_cl = np.vstack([reftrack_interp[:, :2], reftrack_interp[0, :2]])
    coeffs_x, coeffs_y, A, normvec = tph.calc_splines.calc_splines(path=refpath_cl)

    # Check normals not crossing
    normals_cross = tph.check_normals_crossing.check_normals_crossing(
        track=reftrack_interp, normvec_normalized=normvec, horizon=10)
    if normals_cross:
        print("[WARN] Some spline normals are crossing â consider increasing s_reg.")

    # ----- Recompute half-widths by ray-marching along actual normal vectors -----
    # This guarantees QP constraints are tight against the real wall, not just EDT at centerline.
    if track_mask is not None:
        h_mask, w_mask = track_mask.shape
        margin_used = 0.0  # margin was already applied in build_reftrack; apply again here

        def _ray_dist(start_xy, dir_xy, mask, res, orig, max_d=5.0, step=0.04):
            """Distance along dir until mask == 0 (wall)."""
            x0, y0 = start_xy
            dx, dy = dir_xy
            d = step
            while d <= max_d:
                px, py = x0 + dx * d, y0 + dy * d
                c = int(round((px - orig[0]) / res))
                r = int(round(h_mask - 1 - (py - orig[1]) / res))
                r = int(np.clip(r, 0, h_mask - 1))
                c = int(np.clip(c, 0, w_mask - 1))
                if mask[r, c] == 0:
                    return max(d - step, 0.0)
                d += step
            return max_d

        hw_right_new = np.zeros(len(reftrack_interp))
        hw_left_new  = np.zeros(len(reftrack_interp))
        for i in range(len(reftrack_interp)):
            pt = reftrack_interp[i, :2]
            nv = normvec[i]
            hw_right_new[i] = max(_ray_dist(pt,  nv, track_mask, resolution, origin) - 0.06, 0.05)
            hw_left_new[i]  = max(_ray_dist(pt, -nv, track_mask, resolution, origin) - 0.06, 0.05)

        reftrack_interp[:, 2] = hw_right_new
        reftrack_interp[:, 3] = hw_left_new
        print(f"[INFO] Ray-marched half-widths: right {hw_right_new.min():.2f}â{hw_right_new.max():.2f} m, "
              f"left {hw_left_new.min():.2f}â{hw_left_new.max():.2f} m")

    # ----- Minimum curvature QP (with simple restarts if infeasible) -----
    print(f"[INFO] Running minimum curvature QP ({iters_mincurv} iterations) ...")
    alpha_opt = None
    for it in range(iters_mincurv):
        try:
            alpha_opt, curv_err = tph.opt_min_curv.opt_min_curv(
                reftrack=reftrack_interp,
                normvectors=normvec,
                A=A,
                kappa_bound=kappa_bound,
                w_veh=w_veh,
                print_debug=(it == 0),
                closed=True,
            )
            print(f"[INFO]   iter {it+1}: curv_error={curv_err:.4f}")
            if curv_err < 0.01:
                print("[INFO]   Converged.")
                break
            # Update reference around solution for next linearization
            reftrack_interp[:, :2] += normvec * alpha_opt[:, None]
            reftrack_interp[:, 2] -= alpha_opt
            reftrack_interp[:, 3] += alpha_opt
            reftrack_interp[:, 2:] = np.clip(reftrack_interp[:, 2:], 0.02, None)
            refpath_cl = np.vstack([reftrack_interp[:, :2], reftrack_interp[0, :2]])
            coeffs_x, coeffs_y, A, normvec = tph.calc_splines.calc_splines(path=refpath_cl)
            alpha_opt = np.zeros(reftrack_interp.shape[0])
        except Exception as e:
            print(f"[WARN]   iter {it+1} failed: {e}. Stopping iterations.")
            if alpha_opt is None:
                alpha_opt = np.zeros(reftrack_interp.shape[0])
            break

    if alpha_opt is None:
        alpha_opt = np.zeros(reftrack_interp.shape[0])

    reftrack_final = reftrack_interp
    normvec_final = normvec

    # ----- Create final raceline -----
    print("[INFO] Interpolating final raceline ...")
    (raceline, A_rl, coeffs_x_rl, coeffs_y_rl,
     spl_idx, t_vals, s_rl, spl_len, el_len_cl) = tph.create_raceline.create_raceline(
        refline=reftrack_final[:, :2],
        normvectors=normvec_final,
        alpha=alpha_opt,
        stepsize_interp=stepsize_interp,
    )

    # el_len_cl has N elements for N raceline points (closed distances)
    el_rl = el_len_cl  # shape [N]

    # ----- Post-processing: pull any out-of-bounds raceline points back inside -----
    if track_mask is not None:
        dist_pix = ndimage.distance_transform_edt(track_mask > 0)
        h_mask, w_mask = track_mask.shape
        clipped = 0
        for i in range(len(raceline)):
            x, y = raceline[i]
            col = int(round((x - origin[0]) / resolution))
            row = int(round(h_mask - 1 - (y - origin[1]) / resolution))
            row = int(np.clip(row, 0, h_mask - 1))
            col = int(np.clip(col, 0, w_mask - 1))
            clearance = dist_pix[row, col] * resolution
            if clearance < 0.03:   # point is too close to / outside the wall
                # Binary-search alpha toward zero until inside
                idx_ref = spl_idx[i]
                nv_at = normvec_final[min(idx_ref, len(normvec_final) - 1)]
                ref_pt = reftrack_final[min(idx_ref, len(reftrack_final) - 1), :2]
                a = alpha_opt[min(idx_ref, len(alpha_opt) - 1)]
                for _ in range(10):
                    a *= 0.7
                    nx, ny = ref_pt + nv_at * a
                    nc = int(round((nx - origin[0]) / resolution))
                    nr = int(round(h_mask - 1 - (ny - origin[1]) / resolution))
                    nr = int(np.clip(nr, 0, h_mask - 1))
                    nc = int(np.clip(nc, 0, w_mask - 1))
                    if dist_pix[nr, nc] * resolution >= 0.05:
                        raceline[i] = [nx, ny]
                        clipped += 1
                        break
        if clipped > 0:
            print(f"[INFO] Clipped {clipped} out-of-bounds raceline points back inside corridor.")

    # ----- Curvature: numerical computation on dense raceline -----
    # Use preview/review of 1.0m on a 0.1m-spaced path
    _, kappa_rl = tph.calc_head_curv_num.calc_head_curv_num(
        path=raceline,
        el_lengths=el_rl,
        is_closed=True,
        stepsize_psi_preview=1.0,
        stepsize_psi_review=1.0,
        stepsize_curv_preview=2.0,
        stepsize_curv_review=2.0,
    )
    # Smooth to remove remaining noise
    kappa_rl = _moving_average_circular(kappa_rl, 5)

    # ----- Velocity profile (F1TENTH params) -----
    v_max = 8.0          # m/s
    ax_max = 4.0         # m/s2
    ay_max = 4.0         # m/s2
    m_veh = 3.5          # kg
    drag_coeff = 0.004

    ggv = np.array([[0.0, ax_max, ay_max],
                    [v_max, ax_max, ay_max]])
    ax_max_machines = np.array([[0.0, ax_max],
                                [v_max, ax_max]])

    vx_profile = tph.calc_vel_profile.calc_vel_profile(
        ax_max_machines=ax_max_machines,
        kappa=kappa_rl,
        el_lengths=el_rl,
        closed=True,
        drag_coeff=drag_coeff,
        m_veh=m_veh,
        ggv=ggv,
        v_max=v_max,
    )

    return {
        "raceline_xy": raceline,
        "s_raceline": s_rl,
        "kappa_raceline": kappa_rl,
        "vx_profile": vx_profile,
        "el_lengths": el_rl,
        "reftrack_interp": reftrack_final,
        "normvec": normvec_final,
        "alpha_opt": alpha_opt,
    }


# ============================================================
# Save / visualize
# ============================================================

def world_to_pixel(xy: np.ndarray, resolution: float, origin: tuple,
                   img_h: int) -> np.ndarray:
    cols = (xy[:, 0] - origin[0]) / resolution
    rows = img_h - 1 - (xy[:, 1] - origin[1]) / resolution
    return np.stack([cols, rows], axis=1)


def save_outputs(outdir: str, result: dict, occ_img: np.ndarray,
                 track_mask: np.ndarray, resolution: float,
                 origin: tuple) -> None:
    os.makedirs(outdir, exist_ok=True)

    rl = result["raceline_xy"]
    s = result["s_raceline"]
    kappa = result["kappa_raceline"]
    vx = result["vx_profile"]
    ref = result["reftrack_interp"]

    lap_time = float(np.sum(result["el_lengths"] / np.maximum(vx, 0.01)))
    print(f"[INFO] Estimated lap time: {lap_time:.2f} s")
    print(f"[INFO] Raceline length: {s[-1]:.2f} m")
    print(f"[INFO] Max curvature: {np.abs(kappa).max():.4f} 1/m")

    # CSV
    csv_path = os.path.join(outdir, "raceline.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x_m", "y_m", "s_m", "kappa_radpm", "vx_mps"])
        writer.writerows(np.column_stack([rl, s, kappa, vx]).tolist())

    # NPZ
    np.savez(
        os.path.join(outdir, "raceline_data.npz"),
        raceline_xy=rl,
        s_raceline=s,
        kappa_raceline=kappa,
        vx_profile=vx,
        reftrack_interp=ref,
        alpha_opt=result["alpha_opt"],
    )

    # Summary
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Minimum-curvature raceline generation summary\n")
        f.write("Algorithm: TUMFTM/CommonRoad min-curvature QP\n")
        f.write(f"Raceline points: {len(rl)}\n")
        f.write(f"Raceline length: {s[-1]:.3f} m\n")
        f.write(f"Estimated lap time: {lap_time:.2f} s\n")
        f.write(f"Max speed: {vx.max():.2f} m/s\n")
        f.write(f"Min speed: {vx.min():.2f} m/s\n")
        f.write(f"Max |curvature|: {np.abs(kappa).max():.4f} 1/m\n")

    # --- Extract true track wall contours from the corridor mask ---
    # Use OpenCV to find contours of the corridor (these ARE the actual walls)
    mask_u8 = (track_mask > 0).astype(np.uint8) * 255
    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # RETR_CCOMP: hierarchy[0][i][3]==-1 means outer contour, else inner
    wall_outer, wall_inner = [], []
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            pts_cnt = cnt.reshape(-1, 2).astype(float)  # (col, row)
            if hierarchy[0][i][3] == -1:
                wall_outer.append(pts_cnt)
            else:
                wall_inner.append(pts_cnt)

    # --- Pixel coords for raceline and smoothed centerline ---
    h = occ_img.shape[0]
    pix_rl = world_to_pixel(rl, resolution, origin, h)
    pix_ref = world_to_pixel(ref[:, :2], resolution, origin, h)

    def _plot_walls(ax):
        for cnt in wall_outer:
            ax.plot(cnt[:, 0], cnt[:, 1], color="deepskyblue", lw=1.2)
        for cnt in wall_inner:
            ax.plot(cnt[:, 0], cnt[:, 1], color="orange", lw=1.2)

    # Overlay on original map
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(occ_img, cmap="gray", origin="upper")
    _plot_walls(ax)
    ax.plot(pix_ref[:, 0], pix_ref[:, 1], color="yellow", lw=1.5, label="centerline")
    ax.plot(pix_rl[:, 0], pix_rl[:, 1], color="red", lw=2.0, label="min-curvature raceline")
    # Legend proxies
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="deepskyblue", lw=1.2, label="outer wall"),
        Line2D([0], [0], color="orange",      lw=1.2, label="inner wall"),
        Line2D([0], [0], color="yellow",       lw=1.5, label="centerline"),
        Line2D([0], [0], color="red",          lw=2.0, label="min-curvature raceline"),
    ], fontsize=8)
    ax.set_title("Minimum-Curvature Raceline (TUMFTM/CommonRoad algorithm)")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "raceline_overlay_on_map.png"), dpi=180)
    plt.close(fig)

    # Overlay on track mask
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(track_mask, cmap="gray", origin="upper")
    _plot_walls(ax)
    ax.plot(pix_ref[:, 0], pix_ref[:, 1], color="yellow", lw=1.5, label="centerline")
    ax.plot(pix_rl[:, 0], pix_rl[:, 1], color="red", lw=2.0, label="min-curvature raceline")
    ax.legend(handles=[
        Line2D([0], [0], color="deepskyblue", lw=1.2, label="outer wall"),
        Line2D([0], [0], color="orange",      lw=1.2, label="inner wall"),
        Line2D([0], [0], color="yellow",       lw=1.5, label="centerline"),
        Line2D([0], [0], color="red",          lw=2.0, label="min-curvature raceline"),
    ], fontsize=8)
    ax.set_title("Raceline over extracted corridor mask")
    ax.axis("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "raceline_overlay_on_mask.png"), dpi=180)
    plt.close(fig)

    # Curvature + velocity
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(s, kappa, color="tab:blue")
    axes[0].set_ylabel("curvature [1/m]")
    axes[0].set_title("Raceline curvature and velocity profiles")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(s, vx, color="tab:red")
    axes[1].set_ylabel("velocity [m/s]")
    axes[1].set_xlabel("arc length s [m]")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "curvature_velocity.png"), dpi=150)
    plt.close(fig)

    print(f"[INFO] Outputs saved to: {outdir}")


# ============================================================
# Main CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate minimum-curvature raceline from PNG/YAML map "
                    "(TUMFTM/CommonRoad algorithm via trajectory_planning_helpers).")
    p.add_argument("--yaml", required=True, help="Path to map YAML")
    p.add_argument("--png", default=None, help="Optional explicit PNG path")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--margin", type=float, default=0.15,
                   help="Safety margin from wall [m] (default 0.15)")
    p.add_argument("--ds_skeleton", type=float, default=0.5,
                   help="Skeleton point spacing before spline fitting [m] (default 0.5)")
    p.add_argument("--stepsize_prep", type=float, default=1.0,
                   help="Stepsize for initial spline resampling [m] (default 1.0)")
    p.add_argument("--stepsize_reg", type=float, default=3.0,
                   help="Stepsize for spline regularization [m] (default 3.0)")
    p.add_argument("--stepsize_interp", type=float, default=0.1,
                   help="Stepsize for final raceline interpolation [m] (default 0.1)")
    p.add_argument("--kappa_bound", type=float, default=0.4,
                   help="Max curvature bound [1/m] (default 0.4)")
    p.add_argument("--w_veh", type=float, default=0.4,
                   help="Vehicle width [m] (default 0.4)")
    p.add_argument("--smooth_k", type=int, default=15,
                   help="Moving-average kernel for skeleton smoothing (default 15)")
    p.add_argument("--iters_mincurv", type=int, default=3,
                   help="Number of QP linearization iterations (default 3)")
    return p.parse_args()


def main():
    args = parse_args()

    reftrack, occ_img, track_mask, resolution, origin = build_reftrack(
        yaml_path=args.yaml,
        png_path=args.png,
        margin=args.margin,
        ds_skeleton=args.ds_skeleton,
        smooth_k=args.smooth_k,
    )

    result = run_min_curv(
        reftrack=reftrack,
        stepsize_prep=args.stepsize_prep,
        stepsize_reg=args.stepsize_reg,
        stepsize_interp=args.stepsize_interp,
        kappa_bound=args.kappa_bound,
        w_veh=args.w_veh,
        iters_mincurv=args.iters_mincurv,
        track_mask=track_mask,
        resolution=resolution,
        origin=origin,
    )

    save_outputs(
        outdir=args.outdir,
        result=result,
        occ_img=occ_img,
        track_mask=track_mask,
        resolution=resolution,
        origin=origin,
    )


if __name__ == "__main__":
    main()
