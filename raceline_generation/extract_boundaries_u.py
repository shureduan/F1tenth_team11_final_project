#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_boundaries_from_track_mask.py

Purpose
-------
Extract the two true closed boundaries of a track corridor from a line-drawing
track map or from the already recovered corridor mask.

This script ONLY does:
1. read PNG + YAML
2. build a correct track corridor mask
3. extract the two real closed boundaries of that corridor
4. save them

It does NOT do:
- refline generation
- boundary pairing
- raceline optimization

Input
-----
- map PNG
- map YAML

Output
------
- track_mask.png
- outer_boundary.csv
- inner_boundary.csv
- boundary_debug.png
- boundary_geom.npz
- summary.txt

Dependencies
------------
pip install numpy scipy pyyaml opencv-python matplotlib
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


# ============================================================
# Utilities
# ============================================================

def path_length_closed(xy: np.ndarray) -> float:
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
class BoundaryGeom:
    outer_boundary_xy: np.ndarray
    inner_boundary_xy: np.ndarray
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
# Track mask extraction
# ============================================================

def looks_like_boundary_line_map(occ_img: np.ndarray) -> bool:
    """
    Heuristic detector for line-drawing track maps.
    """
    dark_ratio = float(np.mean(occ_img < 80))
    bright_ratio = float(np.mean(occ_img > 220))
    return dark_ratio < 0.08 and bright_ratio > 0.75


def build_boundary_mask_from_line_map(occ_img: np.ndarray) -> np.ndarray:
    """
    Build a binary mask of black boundary lines from a line-drawing map.
    1 = boundary line
    0 = others
    """
    boundary = (occ_img < 200).astype(np.uint8)
    boundary = cv2.dilate(boundary, np.ones((3, 3), np.uint8), iterations=1)
    return boundary


def build_track_mask_from_line_map(occ_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the corridor mask from a line-drawing track map.

    Returns
    -------
    track_mask    : 1 = corridor
    boundary_mask : 1 = black boundary line
    """
    boundary_mask = build_boundary_mask_from_line_map(occ_img)
    boundary_img = boundary_mask.astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(boundary_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) < 4:
        raise RuntimeError("Failed to recover contour hierarchy from line map.")

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
    h, w = occ_img.shape
    outer_fill = np.zeros((h, w), dtype=np.uint8)
    inner_fill = np.zeros((h, w), dtype=np.uint8)

    cv2.drawContours(outer_fill, contours, outer_idx, 1, thickness=-1)
    cv2.drawContours(inner_fill, contours, inner_idx, 1, thickness=-1)

    track_mask = (outer_fill > 0).astype(np.uint8)
    track_mask[inner_fill > 0] = 0

    track_mask = ndimage.binary_opening(track_mask, structure=np.ones((3, 3))).astype(np.uint8)
    track_mask = ndimage.binary_closing(track_mask, structure=np.ones((3, 3))).astype(np.uint8)

    return track_mask, boundary_mask


def build_track_mask_from_occupancy_map(occ_img: np.ndarray, meta: MapMeta) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standard ROS occupancy-map interpretation.
    Returns:
      track_mask
      boundary_mask (estimated from mask gradient)
    """
    img = occ_img.astype(np.float32) / 255.0
    occ_prob = 1.0 - img if meta.negate == 0 else img

    track_mask = (occ_prob < meta.free_thresh).astype(np.uint8)

    labeled, num = ndimage.label(track_mask)
    if num > 1:
        sizes = ndimage.sum(track_mask, labeled, index=np.arange(1, num + 1))
        largest = int(np.argmax(sizes)) + 1
        track_mask = (labeled == largest).astype(np.uint8)

    track_mask = ndimage.binary_opening(track_mask, structure=np.ones((3, 3))).astype(np.uint8)
    track_mask = ndimage.binary_closing(track_mask, structure=np.ones((3, 3))).astype(np.uint8)

    # estimated boundary from mask edge
    eroded = ndimage.binary_erosion(track_mask, structure=np.ones((3, 3))).astype(np.uint8)
    boundary_mask = ((track_mask > 0) & (eroded == 0)).astype(np.uint8)

    return track_mask, boundary_mask


# ============================================================
# Boundary extraction directly from track_mask
# ============================================================

def extract_two_boundaries_from_track_mask(track_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the two true closed boundaries of the corridor directly from the mask.

    Strategy
    --------
    1. Find all contours of the binary corridor mask.
    2. Keep contours with non-trivial area/perimeter.
    3. Choose the two largest closed contours.
    4. Return them as outer and inner according to area.

    Returns
    -------
    outer_xy_pix : [N,2]
    inner_xy_pix : [M,2]
    """
    mask_u8 = (track_mask.astype(np.uint8) * 255)

    contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    if len(contours) < 2:
        raise RuntimeError("Failed to extract two contours from track mask.")

    candidates = []
    for c in contours:
        xy = contour_cv_to_xy(c)
        if len(xy) < 20:
            continue
        area = abs(cv2.contourArea(c))
        peri = cv2.arcLength(c, closed=True)
        if area < 100 or peri < 50:
            continue
        candidates.append((area, peri, xy))

    if len(candidates) < 2:
        raise RuntimeError("Not enough valid boundary contours found from track mask.")

    # Sort by area descending
    candidates.sort(key=lambda t: t[0], reverse=True)

    # Usually the two largest valid contours are exactly the corridor's outer/inner boundaries
    c1 = candidates[0][2]
    c2 = candidates[1][2]

    a1 = abs(signed_area_closed(c1))
    a2 = abs(signed_area_closed(c2))

    if a1 >= a2:
        outer_xy_pix, inner_xy_pix = c1, c2
    else:
        outer_xy_pix, inner_xy_pix = c2, c1

    outer_xy_pix = reorder_closed_curve_start(outer_xy_pix)
    inner_xy_pix = reorder_closed_curve_start(inner_xy_pix)

    return outer_xy_pix, inner_xy_pix


# ============================================================
# Main builder
# ============================================================

def build_boundary_geom(yaml_path: str, png_path: Optional[str], n_samples: int) -> BoundaryGeom:
    occ_img, meta = load_ros_map(yaml_path, png_path)

    if looks_like_boundary_line_map(occ_img):
        track_mask, boundary_mask = build_track_mask_from_line_map(occ_img)
    else:
        track_mask, boundary_mask = build_track_mask_from_occupancy_map(occ_img, meta)

    outer_xy_pix, inner_xy_pix = extract_two_boundaries_from_track_mask(track_mask)

    outer_xy_pix = resample_closed_polyline(outer_xy_pix, n_samples)
    inner_xy_pix = resample_closed_polyline(inner_xy_pix, n_samples)

    outer_xy_world = pixel_xy_to_world(outer_xy_pix, meta, occ_img.shape[0])
    inner_xy_world = pixel_xy_to_world(inner_xy_pix, meta, occ_img.shape[0])

    return BoundaryGeom(
        outer_boundary_xy=outer_xy_world,
        inner_boundary_xy=inner_xy_world,
        track_mask=track_mask,
        boundary_mask=boundary_mask,
        occ_img=occ_img,
        meta=meta,
    )


# ============================================================
# Save / visualize
# ============================================================

def save_outputs(outdir: str, geom: BoundaryGeom) -> None:
    os.makedirs(outdir, exist_ok=True)

    save_csv(
        os.path.join(outdir, "outer_boundary.csv"),
        ["x", "y"],
        geom.outer_boundary_xy,
    )

    save_csv(
        os.path.join(outdir, "inner_boundary.csv"),
        ["x", "y"],
        geom.inner_boundary_xy,
    )

    np.savez(
        os.path.join(outdir, "boundary_geom.npz"),
        outer_boundary_xy=geom.outer_boundary_xy,
        inner_boundary_xy=geom.inner_boundary_xy,
        track_mask=geom.track_mask,
        boundary_mask=geom.boundary_mask,
        occ_img=geom.occ_img,
        resolution=geom.meta.resolution,
        origin=np.array(geom.meta.origin),
    )

    mask_img = (geom.track_mask * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(outdir, "track_mask.png"), mask_img)

    img_h = geom.occ_img.shape[0]
    pix_outer = world_to_pixel(geom.outer_boundary_xy, geom.meta, img_h)
    pix_inner = world_to_pixel(geom.inner_boundary_xy, geom.meta, img_h)

    plt.figure(figsize=(10, 10))
    plt.imshow(geom.occ_img, cmap="gray", origin="upper")
    plt.plot(pix_outer[:, 0], pix_outer[:, 1], color="deepskyblue", linewidth=1.5, label="outer boundary")
    plt.plot(pix_inner[:, 0], pix_inner[:, 1], color="orange", linewidth=1.5, label="inner boundary")
    plt.legend()
    plt.title("Extracted true boundaries from track mask")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "boundary_debug.png"), dpi=180)
    plt.close()

    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("Boundary extraction summary\n")
        f.write(f"Outer boundary length [m]: {path_length_closed(geom.outer_boundary_xy):.3f}\n")
        f.write(f"Inner boundary length [m]: {path_length_closed(geom.inner_boundary_xy):.3f}\n")
        f.write(f"Outer boundary points: {len(geom.outer_boundary_xy)}\n")
        f.write(f"Inner boundary points: {len(geom.inner_boundary_xy)}\n")


# ============================================================
# Main CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract the two true closed boundaries from a track mask.")
    p.add_argument("--yaml", required=True, help="Path to map YAML")
    p.add_argument("--png", default=None, help="Optional explicit PNG path")
    p.add_argument("--outdir", required=True, help="Output directory")
    p.add_argument("--n_samples", type=int, default=2000, help="Number of resampled points per boundary")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    geom = build_boundary_geom(
        yaml_path=args.yaml,
        png_path=args.png,
        n_samples=args.n_samples,
    )

    save_outputs(args.outdir, geom)

    print("=== Done ===")
    print(f"outer points : {len(geom.outer_boundary_xy)}")
    print(f"inner points : {len(geom.inner_boundary_xy)}")
    print(f"saved to     : {args.outdir}")


if __name__ == "__main__":
    main()