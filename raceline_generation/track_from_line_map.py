#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
track_from_line_map.py

Purpose
-------
Build a clean track geometry representation from a line-drawing track map
(two black boundary lines on bright background), such as your Shanghai map.

Input
-----
- map PNG
- map YAML

Output
------
- left_boundary.csv
- right_boundary.csv
- refline.csv
- track_geom.npz
- track_debug.png
- track_mask.png

Main idea
---------
1. Read PNG + YAML
2. Detect black boundary lines
3. Build corridor mask between outer/inner contours
4. Build an initial refline from contour midpoints
5. Compute normals on the refline
6. Ray-cast along +/- normal to hit true boundary lines
7. Rebuild smooth paired left/right boundaries
8. Rebuild smooth final refline
9. Save everything

Dependencies
------------
pip install numpy scipy pyyaml opencv-python matplotlib

Example
-------
python track_from_line_map.py \
    --yaml /Users/shure_duan/VScode/f1tenth/map/Shanghai_map.yaml \
    --png /Users/shure_duan/VScode/f1tenth/map/Shanghai_map.png \
    --outdir /Users/shure_duan/VScode/f1tenth/track_geom_output \
    --ds 0.10
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import ndimage
from scipy.interpolate import splprep, splev


# ============================================================
# Utilities
# ============================================================

def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def moving_average_circular(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x.copy()
    pad = k // 2
    x_pad = np.concatenate([x[-pad:], x, x[:pad]])
    kernel = np.ones(k, dtype=np.float64) / float(k)
    y = np.convolve(x_pad, kernel, mode="same")
    return y[pad:pad + len(x)]


def path_length(xy: np.ndarray) -> float:
    xy = np.asarray(xy, dtype=np.float64)
    return float(
        np.sum(
            np.hypot(
                np.roll(xy[:, 0], -1) - xy[:, 0],
                np.roll(xy[:, 1], -1) - xy[:, 1],
            )
        )
    )


def signed_area_closed(xy: np.ndarray) -> float:
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def reorder_closed_curve_start(xy: np.ndarray) -> np.ndarray:
    """
    Reorder a closed curve so the start point is stable.
    Choose leftmost point, then upper as tie-break.
    """
    idx = np.lexsort((xy[:, 1], xy[:, 0]))[0]
    return np.vstack([xy[idx:], xy[:idx]])


def resample_closed_polyline(xy: np.ndarray, n: int) -> np.ndarray:
    """
    Uniformly resample a closed polyline by arc length.
    """
    xy = np.asarray(xy, dtype=np.float64)
    xy_closed = np.vstack([xy, xy[0]])

    seg = np.hypot(np.diff(xy_closed[:, 0]), np.diff(xy_closed[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]

    if total < 1e-8:
        raise RuntimeError("Closed polyline has near-zero length.")

    target = np.linspace(0.0, total, n, endpoint=False)
    x_new = np.interp(target, s, xy_closed[:, 0])
    y_new = np.interp(target, s, xy_closed[:, 1])
    return np.stack([x_new, y_new], axis=1)


def resample_closed_curve_spline(xy: np.ndarray, n: int, smoothing: float = 1.0) -> np.ndarray:
    """
    Smooth a closed curve with periodic spline, then sample n points.
    """
    xy = np.asarray(xy, dtype=np.float64)
    xy_closed = np.vstack([xy, xy[0]])

    seg = np.hypot(np.diff(xy_closed[:, 0]), np.diff(xy_closed[:, 1]))
    keep = np.concatenate([[True], seg > 1e-6])
    xy_closed = xy_closed[keep]

    if len(xy_closed) < 10:
        raise RuntimeError("Too few unique points to fit periodic spline.")

    tck, _ = splprep([xy_closed[:, 0], xy_closed[:, 1]], s=smoothing, per=True)
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    x_new, y_new = splev(u, tck)
    return np.stack([x_new, y_new], axis=1)


def contour_cv_to_xy(contour: np.ndarray) -> np.ndarray:
    """
    OpenCV contour -> [N,2] in image pixel coordinates (x=col, y=row)
    """
    return contour[:, 0, :].astype(np.float64)


def pixel_xy_to_world(xy_pix: np.ndarray, meta, img_h: int) -> np.ndarray:
    """
    Input pixel coordinates in image convention:
      x_pix = col
      y_pix = row
    Convert to ROS world coordinates.
    """
    cols = xy_pix[:, 0]
    rows = xy_pix[:, 1]
    x = meta.origin[0] + cols * meta.resolution
    y = meta.origin[1] + (img_h - 1 - rows) * meta.resolution
    return np.stack([x, y], axis=1)


def world_to_pixel(xy_world: np.ndarray, meta, img_h: int) -> np.ndarray:
    cols = (xy_world[:, 0] - meta.origin[0]) / meta.resolution
    rows = img_h - 1 - (xy_world[:, 1] - meta.origin[1]) / meta.resolution
    return np.stack([cols, rows], axis=1)


def save_csv(path: str, header: List[str], data: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data.tolist())


# ============================================================
# Data structures
# ============================================================

@dataclass
class MapMeta:
    resolution: float
    origin: Tuple[float, float, float]
    negate: int
    occupied_thresh: float
    free_thresh: float


@dataclass
class TrackGeom:
    left_bound_xy: np.ndarray
    right_bound_xy: np.ndarray
    refline_xy: np.ndarray
    heading: np.ndarray
    normals_xy: np.ndarray
    curvature: np.ndarray
    s: np.ndarray
    width: np.ndarray
    track_mask: np.ndarray
    boundary_mask: np.ndarray
    occ_img: np.ndarray
    meta: MapMeta


# ============================================================
# Map loading
# ============================================================

def load_ros_map(yaml_path: str, png_path: Optional[str] = None) -> Tuple[np.ndarray, MapMeta]:
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

    meta = MapMeta(
        resolution=float(cfg["resolution"]),
        origin=tuple(cfg["origin"]),
        negate=int(cfg.get("negate", 0)),
        occupied_thresh=float(cfg.get("occupied_thresh", 0.65)),
        free_thresh=float(cfg.get("free_thresh", 0.196)),
    )
    return occ_img, meta


# ============================================================
# Line-map track extraction
# ============================================================

def build_boundary_mask_from_line_map(occ_img: np.ndarray) -> np.ndarray:
    """
    Build a binary mask of black boundary lines from a line-drawing map.
    1 = boundary line
    0 = others
    """
    boundary = (occ_img < 200).astype(np.uint8)
    boundary = cv2.dilate(boundary, np.ones((3, 3), np.uint8), iterations=1)
    return boundary


def build_line_map_contours_and_mask(
    occ_img: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For a line-drawing track map:
    - detect black boundary lines
    - find contour hierarchy
    - identify the two contours that bound the drivable corridor
    - return:
        outer_boundary_xy_pix
        inner_boundary_xy_pix
        track_mask
        boundary_mask
    """
    boundary_mask = build_boundary_mask_from_line_map(occ_img)
    boundary_img = boundary_mask.astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(boundary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) < 4:
        raise RuntimeError("Failed to recover nested contours from line map.")

    hierarchy = hierarchy[0]

    best_pair = None
    best_gap = -1.0

    for i, h in enumerate(hierarchy):
        child = h[2]
        if child == -1:
            continue

        area_i = abs(cv2.contourArea(contours[i]))
        area_c = abs(cv2.contourArea(contours[child]))
        gap = area_i - area_c

        if gap < 1000:
            continue

        if gap > best_gap:
            best_gap = gap
            best_pair = (i, child)

    if best_pair is None:
        raise RuntimeError("Could not identify a valid contour pair for the track corridor.")

    outer_idx, inner_idx = best_pair
    outer_xy_pix = contour_cv_to_xy(contours[outer_idx])
    inner_xy_pix = contour_cv_to_xy(contours[inner_idx])

    h, w = occ_img.shape
    outer_fill = np.zeros((h, w), dtype=np.uint8)
    inner_fill = np.zeros((h, w), dtype=np.uint8)

    cv2.drawContours(outer_fill, contours, outer_idx, 1, thickness=-1)
    cv2.drawContours(inner_fill, contours, inner_idx, 1, thickness=-1)

    track_mask = (outer_fill > 0).astype(np.uint8)
    track_mask[inner_fill > 0] = 0

    track_mask = ndimage.binary_opening(track_mask, structure=np.ones((3, 3))).astype(np.uint8)
    track_mask = ndimage.binary_closing(track_mask, structure=np.ones((3, 3))).astype(np.uint8)

    return outer_xy_pix, inner_xy_pix, track_mask, boundary_mask


# ============================================================
# Initial centerline from contour midpoint
# ============================================================

def build_initial_midpoint_track(
    outer_xy_pix: np.ndarray,
    inner_xy_pix: np.ndarray,
    meta: MapMeta,
    img_h: int,
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build aligned left/right boundaries and midpoint centerline from two contours.
    """
    if np.sign(signed_area_closed(outer_xy_pix)) != np.sign(signed_area_closed(inner_xy_pix)):
        inner_xy_pix = inner_xy_pix[::-1]

    outer_xy_pix = reorder_closed_curve_start(outer_xy_pix)
    inner_xy_pix = reorder_closed_curve_start(inner_xy_pix)

    outer_rs = resample_closed_polyline(outer_xy_pix, n_samples)
    inner_rs = resample_closed_polyline(inner_xy_pix, n_samples)

    # Find best circular shift for inner boundary
    best_shift = 0
    best_cost = np.inf
    for shift in range(n_samples):
        inner_shift = np.roll(inner_rs, shift, axis=0)
        cost = np.mean(np.sum((outer_rs - inner_shift) ** 2, axis=1))
        if cost < best_cost:
            best_cost = cost
            best_shift = shift

    inner_rs = np.roll(inner_rs, best_shift, axis=0)

    outer_world = pixel_xy_to_world(outer_rs, meta, img_h)
    inner_world = pixel_xy_to_world(inner_rs, meta, img_h)
    center_world = 0.5 * (outer_world + inner_world)

    return outer_world, inner_world, center_world


# ============================================================
# Reference line geometry
# ============================================================

def compute_curve_geometry(xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(xy)
    ext = np.vstack([xy[-1], xy, xy[0], xy[1]])
    dx = 0.5 * (ext[2:, 0] - ext[:-2, 0])
    dy = 0.5 * (ext[2:, 1] - ext[:-2, 1])

    heading = np.arctan2(dy[:n], dx[:n])
    normals = np.stack([-np.sin(heading), np.cos(heading)], axis=1)

    ds_step = np.hypot(
        np.roll(xy[:, 0], -1) - xy[:, 0],
        np.roll(xy[:, 1], -1) - xy[:, 1],
    )
    dtheta = wrap_to_pi(np.roll(heading, -1) - heading)
    curvature = dtheta / np.maximum(ds_step, 1e-6)
    curvature = moving_average_circular(curvature, 9)

    s = np.zeros(n, dtype=np.float64)
    s[1:] = np.cumsum(ds_step[:-1])
    return heading, normals, curvature, s


# ============================================================
# Boundary pairing by normal ray-cast
# ============================================================

def sample_boundaries_by_normal_raycast(
    refline_xy: np.ndarray,
    normals_xy: np.ndarray,
    boundary_mask: np.ndarray,
    track_mask: np.ndarray,
    meta: MapMeta,
    max_search_m: float = 30.0,
    step_m: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each refline point, ray-cast along +/- normal direction until hitting
    the true black boundary line.
    """
    h, w = boundary_mask.shape
    max_steps = int(max_search_m / step_m)

    left_pts = []
    right_pts = []

    for p, n in zip(refline_xy, normals_xy):
        # +normal
        left_hit = None
        prev_inside = p.copy()
        for k in range(1, max_steps + 1):
            q = p + n * (k * step_m)
            col = int(round((q[0] - meta.origin[0]) / meta.resolution))
            row = int(round(h - 1 - (q[1] - meta.origin[1]) / meta.resolution))

            if row < 0 or row >= h or col < 0 or col >= w:
                break

            if track_mask[row, col] > 0:
                prev_inside = q.copy()

            if boundary_mask[row, col] > 0:
                left_hit = prev_inside.copy()
                break

        if left_hit is None:
            left_hit = prev_inside.copy()
        left_pts.append(left_hit)

        # -normal
        right_hit = None
        prev_inside = p.copy()
        for k in range(1, max_steps + 1):
            q = p - n * (k * step_m)
            col = int(round((q[0] - meta.origin[0]) / meta.resolution))
            row = int(round(h - 1 - (q[1] - meta.origin[1]) / meta.resolution))

            if row < 0 or row >= h or col < 0 or col >= w:
                break

            if track_mask[row, col] > 0:
                prev_inside = q.copy()

            if boundary_mask[row, col] > 0:
                right_hit = prev_inside.copy()
                break

        if right_hit is None:
            right_hit = prev_inside.copy()
        right_pts.append(right_hit)

    return np.asarray(left_pts), np.asarray(right_pts)


# ============================================================
# Final smoothing and reconstruction
# ============================================================

def rebuild_track_geometry(
    left_bound_xy: np.ndarray,
    right_bound_xy: np.ndarray,
    n_samples: int,
    smoothing: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Smooth paired left/right boundaries and rebuild midpoint refline.
    """
    left_rs = resample_closed_curve_spline(left_bound_xy, n_samples, smoothing=smoothing)
    right_rs = resample_closed_curve_spline(right_bound_xy, n_samples, smoothing=smoothing)
    refline = 0.5 * (left_rs + right_rs)
    refline = resample_closed_curve_spline(refline, n_samples, smoothing=smoothing)
    return left_rs, right_rs, refline


# ============================================================
# Main track builder
# ============================================================

def build_track_from_line_map(
    yaml_path: str,
    png_path: Optional[str],
    ds: float,
) -> TrackGeom:
    occ_img, meta = load_ros_map(yaml_path, png_path)

    outer_xy_pix, inner_xy_pix, track_mask, boundary_mask = build_line_map_contours_and_mask(occ_img)

    approx_len_pix = max(path_length(outer_xy_pix), path_length(inner_xy_pix))
    n_samples_init = max(700, int(approx_len_pix / max(ds / meta.resolution, 1.0)))

    left_world_raw, right_world_raw, refline_raw = build_initial_midpoint_track(
        outer_xy_pix=outer_xy_pix,
        inner_xy_pix=inner_xy_pix,
        meta=meta,
        img_h=occ_img.shape[0],
        n_samples=n_samples_init,
    )

    n_final = max(600, int(path_length(refline_raw) / max(ds, 1e-3)))
    refline = resample_closed_curve_spline(refline_raw, n_final, smoothing=1.0)

    heading_tmp, normals_tmp, _, _ = compute_curve_geometry(refline)

    left_bound, right_bound = sample_boundaries_by_normal_raycast(
        refline_xy=refline,
        normals_xy=normals_tmp,
        boundary_mask=boundary_mask,
        track_mask=track_mask,
        meta=meta,
        max_search_m=30.0,
        step_m=max(meta.resolution * 0.5, ds * 0.25),
    )

    left_bound, right_bound, refline = rebuild_track_geometry(
        left_bound_xy=left_bound,
        right_bound_xy=right_bound,
        n_samples=n_final,
        smoothing=1.0,
    )

    heading, normals, curvature, s = compute_curve_geometry(refline)
    width = np.linalg.norm(left_bound - right_bound, axis=1)

    return TrackGeom(
        left_bound_xy=left_bound,
        right_bound_xy=right_bound,
        refline_xy=refline,
        heading=heading,
        normals_xy=normals,
        curvature=curvature,
        s=s,
        width=width,
        track_mask=track_mask,
        boundary_mask=boundary_mask,
        occ_img=occ_img,
        meta=meta,
    )


# ============================================================
# Save / visualize
# ============================================================

def save_outputs(outdir: str, track: TrackGeom) -> None:
    os.makedirs(outdir, exist_ok=True)

    save_csv(
        os.path.join(outdir, "left_boundary.csv"),
        ["x", "y"],
        track.left_bound_xy,
    )

    save_csv(
        os.path.join(outdir, "right_boundary.csv"),
        ["x", "y"],
        track.right_bound_xy,
    )

    save_csv(
        os.path.join(outdir, "refline.csv"),
        ["x", "y", "s", "curvature", "width"],
        np.column_stack([
            track.refline_xy,
            track.s,
            track.curvature,
            track.width,
        ]),
    )

    np.savez(
        os.path.join(outdir, "track_geom.npz"),
        left_bound_xy=track.left_bound_xy,
        right_bound_xy=track.right_bound_xy,
        refline_xy=track.refline_xy,
        heading=track.heading,
        normals_xy=track.normals_xy,
        curvature=track.curvature,
        s=track.s,
        width=track.width,
        occ_img=track.occ_img,
        track_mask=track.track_mask,
        boundary_mask=track.boundary_mask,
        resolution=track.meta.resolution,
        origin=np.array(track.meta.origin),
    )

    img_h = track.occ_img.shape[0]
    pix_left = world_to_pixel(track.left_bound_xy, track.meta, img_h)
    pix_right = world_to_pixel(track.right_bound_xy, track.meta, img_h)
    pix_ref = world_to_pixel(track.refline_xy, track.meta, img_h)

    # Save mask only
    mask_img = (track.track_mask * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(outdir, "track_mask.png"), mask_img)

    # Debug plot on original map
    plt.figure(figsize=(10, 10))
    plt.imshow(track.occ_img, cmap="gray", origin="upper")
    plt.plot(pix_left[:, 0], pix_left[:, 1], color="deepskyblue", linewidth=1.0, label="left boundary")
    plt.plot(pix_right[:, 0], pix_right[:, 1], color="orange", linewidth=1.0, label="right boundary")
    plt.plot(pix_ref[:, 0], pix_ref[:, 1], color="red", linewidth=2.0, label="refline")
    plt.legend()
    plt.title("Extracted track geometry")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "track_debug.png"), dpi=180)
    plt.close()

    # Curvature plot
    plt.figure(figsize=(10, 4))
    plt.plot(track.s, track.curvature)
    plt.xlabel("arc length s [m]")
    plt.ylabel("curvature [1/m]")
    plt.title("Reference line curvature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "refline_curvature.png"), dpi=180)
    plt.close()

    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Track geometry extraction summary\n")
        f.write(f"Refline length [m]: {path_length(track.refline_xy):.3f}\n")
        f.write(f"Mean width [m]: {np.mean(track.width):.3f}\n")
        f.write(f"Min width [m]: {np.min(track.width):.3f}\n")
        f.write(f"Max width [m]: {np.max(track.width):.3f}\n")
        f.write(f"Number of samples: {len(track.refline_xy)}\n")


# ============================================================
# Main CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract clean track geometry from a line-drawing map.")
    p.add_argument("--yaml", required=True, help="Path to map YAML")
    p.add_argument("--png", default=None, help="Optional explicit PNG path")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--ds", type=float, default=0.10, help="Approximate arc-length sample step [m]")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    track = build_track_from_line_map(
        yaml_path=args.yaml,
        png_path=args.png,
        ds=args.ds,
    )

    save_outputs(args.outdir, track)

    print("=== Done ===")
    print(f"refline points : {len(track.refline_xy)}")
    print(f"refline length : {path_length(track.refline_xy):.3f} m")
    print(f"mean width     : {np.mean(track.width):.3f} m")
    print(f"saved to       : {args.outdir}")


if __name__ == "__main__":
    main()