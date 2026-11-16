"""Scores for generated clips, from NumPy and OpenCV only. CPU.

structure score
    Edge agreement between a generated frame and the simulator's edge frame for the same instant,
    with a 3-pixel tolerance, divided by the agreement of the same generated frame with the
    simulator's edges from a different clip (the mismatched baseline). 1.0 means the output
    follows our geometry no better than it follows an unrelated scene.

ripple score
    Strength of oriented, stripe-like texture on surfaces that are flat and plain in the scene
    (water, paving, road, lawn), away from every real edge. Stripes have a strong local gradient
    that points the same way over a neighbourhood; grain and noise do not. The score is the mean,
    over those pixels, of gradient strength times its local coherence, in grey levels per pixel.
    0 is a smooth surface.
"""
from __future__ import annotations

import cv2
import numpy as np

TOL_PX = 3
CANNY = (50, 150)
FLAT_CLASSES = ("ground", "road", "sidewalk", "water", "boardwalk", "pier", "median", "lawn", "plaza", "parking", "path", "park")


def edges_of(frame_bgr: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (0, 0), 1.0)
    return cv2.Canny(g, *CANNY) > 0


def agreement(a: np.ndarray, b: np.ndarray, tol: int = TOL_PX) -> float:
    """Mean of: share of a's edge pixels within tol of an edge of b, and the reverse."""
    if not a.any() or not b.any():
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    da = cv2.dilate(a.astype(np.uint8), k) > 0
    db = cv2.dilate(b.astype(np.uint8), k) > 0
    return 0.5 * (float((a & db).sum()) / float(a.sum()) + float((b & da).sum()) / float(b.sum()))


def structure_score(gen_bgr: np.ndarray, sim_edges: np.ndarray, other_edges: np.ndarray) -> dict:
    e = edges_of(gen_bgr)
    own, other = agreement(e, sim_edges), agreement(e, other_edges)
    return {"agreement": own, "baseline": other, "score": own / other if other > 0 else float("inf")}


def ripple_score(gen_bgr: np.ndarray, flat_mask: np.ndarray, sim_edges: np.ndarray, away_px: int = 8, sigma: float = 8.0) -> float:
    g = cv2.GaussianBlur(cv2.cvtColor(gen_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0, cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    jxx, jyy, jxy = (cv2.GaussianBlur(v, (0, 0), sigma) for v in (gx * gx, gy * gy, gx * gy))
    tr = jxx + jyy
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / np.maximum(tr, 1e-6)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * away_px + 1, 2 * away_px + 1))
    keep = flat_mask & ~(cv2.dilate(sim_edges.astype(np.uint8), k) > 0)
    keep &= cv2.erode(flat_mask.astype(np.uint8), k) > 0
    if keep.sum() < 500:
        return float("nan")
    return float((np.sqrt(tr) * coherence)[keep].mean())
