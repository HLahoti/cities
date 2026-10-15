"""Rasterising simple vector geometry onto the grid.

Coordinates are (x, y) in cell units: x = column, y = row (origin top-left).
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path


def polygon_mask(shape, coords) -> np.ndarray:
    """Cells whose centres fall inside the polygon."""
    h, w = shape
    pts = np.asarray(coords, dtype=float)
    path = Path(pts)
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
    mask = np.zeros(shape, dtype=bool)
    if x1 < x0 or y1 < y0:
        return mask
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    inside = path.contains_points(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    mask[y0:y1 + 1, x0:x1 + 1] = inside
    return mask


def line_cells(p0, p1) -> tuple[np.ndarray, np.ndarray]:
    """8-connected cells along a segment (dense sampling; no gaps)."""
    (x0, y0), (x1, y1) = p0, p1
    n = int(np.ceil(max(abs(x1 - x0), abs(y1 - y0)))) * 2 + 1
    t = np.linspace(0.0, 1.0, n)
    xs = np.rint(x0 + (x1 - x0) * t).astype(int)
    ys = np.rint(y0 + (y1 - y0) * t).astype(int)
    rc = np.unique(np.c_[ys, xs], axis=0)
    return rc[:, 0], rc[:, 1]


def polyline_mask(shape, coords, width: float = 1.0) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    pts = np.asarray(coords, dtype=float)
    for a, b in zip(pts[:-1], pts[1:]):
        r, c = line_cells(a, b)
        ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
        mask[r[ok], c[ok]] = True
    if width > 1:
        from scipy import ndimage
        mask = ndimage.binary_dilation(mask, iterations=int(round((width - 1) / 2)) or 1)
    return mask


def circle_mask(shape, center, radius: float) -> np.ndarray:
    rr, cc = np.indices(shape)
    return np.hypot(cc - center[0], rr - center[1]) <= radius


def rect_mask(shape, x0, y0, x1, y1) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    m[max(int(y0), 0):int(y1) + 1, max(int(x0), 0):int(x1) + 1] = True
    return m


def feature_mask(shape, feat: dict, scale: float = 1.0) -> np.ndarray:
    """Mask for a feature dict; ``scale`` converts file units to cells."""
    g = feat["geometry"]
    if g == "polygon":
        return polygon_mask(shape, np.asarray(feat["coords"]) / scale)
    if g == "polyline":
        # width is always in cells
        return polyline_mask(shape, np.asarray(feat["coords"]) / scale, feat.get("width", 1))
    if g == "circle":
        return circle_mask(shape, np.asarray(feat["center"]) / scale, feat["radius"] / scale)
    if g == "point":
        m = np.zeros(shape, dtype=bool)
        x, y = np.rint(np.asarray(feat["coords"]) / scale).astype(int)
        if 0 <= y < shape[0] and 0 <= x < shape[1]:
            m[y, x] = True
        return m
    if g == "rect":
        x0, y0, x1, y1 = np.asarray(feat["bounds"]) / scale
        return rect_mask(shape, x0, y0, x1, y1)
    raise ValueError(f"Unknown geometry '{g}'")


def burn(arr: np.ndarray, mask: np.ndarray, value: float, op: str = "set") -> None:
    if op == "set":
        arr[mask] = value
    elif op == "max":
        arr[mask] = np.maximum(arr[mask], value)
    elif op == "min":
        arr[mask] = np.minimum(arr[mask], value)
    elif op == "add":
        arr[mask] = arr[mask] + value
    else:
        raise ValueError(f"Unknown burn op '{op}'")
