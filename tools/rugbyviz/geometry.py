"""Homography estimation from scratch: DLT + Hartley normalisation + RANSAC.

This is a teaching implementation of what cv2.findHomography does internally.
It is validated against OpenCV in tools/test_geometry.py.

The whole file rests on one idea: a homography maps points on a plane between
two views, and in *homogeneous coordinates* that map is a plain matrix multiply.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


# ---------------------------------------------------------------------------
# Homogeneous coordinates
# ---------------------------------------------------------------------------

def to_homogeneous(pts: np.ndarray) -> np.ndarray:
    """(N,2) -> (N,3) by appending a 1.

    A 2D point (x, y) becomes (x, y, 1). The trick is that any scalar multiple
    of that triple denotes the SAME 2D point: (2x, 2y, 2) is still (x, y).
    That redundancy is what lets a single 3x3 matrix express perspective.
    """
    return np.hstack([pts, np.ones((len(pts), 1))])


def from_homogeneous(pts: np.ndarray) -> np.ndarray:
    """(N,3) -> (N,2) by dividing through by the third coordinate.

    This division is where perspective actually happens. An affine transform
    leaves w == 1 and nothing interesting occurs; a homography makes w vary
    per point, and dividing by a position-dependent number is precisely what
    makes distant things shrink and parallel lines converge.
    """
    w = pts[:, 2:3]
    # w == 0 means the point mapped to infinity (a vanishing point).
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return pts[:, :2] / w


def apply_homography(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map (N,2) points through a 3x3 homography."""
    return from_homogeneous(to_homogeneous(np.asarray(pts, dtype=np.float64)) @ H.T)


# ---------------------------------------------------------------------------
# Hartley normalisation
# ---------------------------------------------------------------------------

def normalisation_matrix(pts: np.ndarray) -> np.ndarray:
    """Similarity transform putting the centroid at the origin and the mean
    distance from the origin at sqrt(2).

    Why this exists: raw pixel coordinates on a 1920x1080 image are ~1e3, so
    the DLT design matrix contains entries of order 1 (the -1 column), 1e3
    (x, y) and 1e6 (u*x). That spread of six orders of magnitude makes the
    matrix badly conditioned, and the SVD answer gets swamped by floating
    point error. Rescaling everything to order 1 fixes it.

    This is not a nicety. Skipping it visibly degrades the result -- see the
    ablation in tools/test_geometry.py.
    """
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    mean_dist = np.sqrt((centred ** 2).sum(axis=1)).mean()
    mean_dist = max(mean_dist, 1e-12)
    s = np.sqrt(2) / mean_dist
    return np.array([[s, 0.0, -s * centroid[0]],
                     [0.0, s, -s * centroid[1]],
                     [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Direct Linear Transform
# ---------------------------------------------------------------------------

def dlt_homography(src: np.ndarray, dst: np.ndarray, normalise: bool = True) -> np.ndarray:
    """Least-squares homography from >= 4 correspondences.

    Derivation. We want H with  dst ~ H @ src  (~ meaning "up to scale").
    Writing src = (x, y, 1) and dst = (u, v):

        u = (h11 x + h12 y + h13) / (h31 x + h32 y + h33)
        v = (h21 x + h22 y + h23) / (h31 x + h32 y + h33)

    Those are nonlinear in the unknowns because of the division. Multiply out
    the denominator and they become linear:

        u (h31 x + h32 y + h33) - (h11 x + h12 y + h13) = 0
        v (h31 x + h32 y + h33) - (h21 x + h22 y + h23) = 0

    Two linear equations per correspondence, nine unknowns, so four points
    give eight equations. Stack them as A h = 0 and solve.

    Note it is A h = 0, not A h = b -- a homogeneous system. h = 0 is always a
    solution and we do not want it, and any scalar multiple of a real solution
    is equally valid (H is only defined up to scale). So we seek the unit
    vector h minimising ||A h||, which is the right singular vector of A with
    the smallest singular value: the last row of Vt from the SVD.

    `normalise=False` exists only to demonstrate why Hartley normalisation
    matters. Do not use it for real work.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) != len(dst):
        raise ValueError("src and dst must have the same length")
    if len(src) < 4:
        raise ValueError(f"need >= 4 correspondences, got {len(src)}")

    if normalise:
        T_src = normalisation_matrix(src)
        T_dst = normalisation_matrix(dst)
        s = apply_homography(T_src, src)
        d = apply_homography(T_dst, dst)
    else:
        T_src = T_dst = np.eye(3)
        s, d = src, dst

    x, y = s[:, 0], s[:, 1]
    u, v = d[:, 0], d[:, 1]
    z = np.zeros(len(src))
    o = np.ones(len(src))

    # Two rows per correspondence, interleaved.
    A = np.empty((2 * len(src), 9))
    A[0::2] = np.stack([-x, -y, -o, z, z, z, u * x, u * y, u], axis=1)
    A[1::2] = np.stack([z, z, z, -x, -y, -o, v * x, v * y, v], axis=1)

    # Smallest singular vector = last row of Vt.
    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    # Undo the normalisation. We solved in normalised coordinates:
    #   d_norm = H_norm @ s_norm,  d_norm = T_dst @ d,  s_norm = T_src @ s
    #   => T_dst d = H_norm T_src s  =>  d = inv(T_dst) H_norm T_src s
    H = np.linalg.inv(T_dst) @ H_norm @ T_src

    # Fix the arbitrary scale so H[2,2] == 1, matching OpenCV's convention.
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


# ---------------------------------------------------------------------------
# RANSAC
# ---------------------------------------------------------------------------

def _degenerate(p: np.ndarray, rel_tol: float = 1e-3) -> bool:
    """True if any 3 of the 4 sample points are (near) collinear.

    Four points define a homography only if no three are collinear. Three
    collinear points carry less independent information than they appear to,
    and the resulting system is rank deficient. It is cheaper to reject the
    sample than to let it produce a garbage model.

    Test is the triangle area via the 2D cross product, made scale-free by
    dividing by the longest side so the tolerance means something.
    """
    for a, b, c in combinations(range(len(p)), 3):
        ab, ac = p[b] - p[a], p[c] - p[a]
        # 2D cross product (a scalar). np.cross dropped 2-vector support in
        # NumPy 2.0, so write it out.
        area = abs(float(ab[0] * ac[1] - ab[1] * ac[0]))
        scale = max(np.linalg.norm(ab), np.linalg.norm(ac), 1e-12)
        if area / (scale ** 2) < rel_tol:
            return True
    return False


def ransac_homography(src: np.ndarray, dst: np.ndarray,
                      threshold: float = 3.0,
                      max_iters: int = 2000,
                      confidence: float = 0.995,
                      seed: int | None = 0):
    """Robust homography. Returns (H, inlier_mask).

    Why robust fitting is mandatory here: SIFT matching produces a meaningful
    fraction of wrong correspondences (repeated structure -- a row of similar
    parked cars -- is the classic culprit). Plain least squares minimises total
    squared error, so a handful of matches wrong by 500 px will drag the fit
    badly. One bad match can ruin the answer.

    RANSAC instead: guess from a minimal sample, count how many other points
    agree, keep the best-supported guess. Wrong matches do not agree with each
    other, so they never form a large consensus and simply get outvoted.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = len(src)
    if n < 4:
        raise ValueError(f"need >= 4 correspondences, got {n}")

    rng = np.random.default_rng(seed)
    best_mask = np.zeros(n, dtype=bool)
    best_count = 0
    thr_sq = threshold ** 2
    budget = max_iters

    i = 0
    while i < min(budget, max_iters):
        i += 1
        idx = rng.choice(n, 4, replace=False)
        if _degenerate(src[idx]) or _degenerate(dst[idx]):
            continue
        try:
            H = dlt_homography(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.all(np.isfinite(H)):
            continue

        err = ((apply_homography(H, src) - dst) ** 2).sum(axis=1)
        mask = err < thr_sq
        count = int(mask.sum())

        if count > best_count:
            best_count, best_mask = count, mask
            # Adaptive stopping. Once we know roughly what fraction w of the
            # data is inlying, the chance a random 4-sample is all inliers is
            # w**4, so the number of trials needed to see one such sample with
            # probability `confidence` is log(1-conf)/log(1-w**4).
            w = count / n
            denom = np.log(max(1e-12, 1.0 - w ** 4))
            if denom < 0:
                budget = min(max_iters, int(np.log(1.0 - confidence) / denom) + 1)

    if best_count < 4:
        raise RuntimeError("RANSAC failed: no consensus set of >= 4 points")

    # Final refit on ALL inliers. The minimal 4-point sample only located the
    # right consensus set; refitting on every agreeing point averages out
    # their individual noise and is what gets you sub-pixel accuracy.
    H = dlt_homography(src[best_mask], dst[best_mask])
    return H, best_mask


def reprojection_error(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Per-point euclidean distance between H(src) and dst, in pixels."""
    return np.linalg.norm(apply_homography(H, src) - dst, axis=1)
