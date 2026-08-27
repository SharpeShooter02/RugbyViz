"""Validate the from-scratch homography code against OpenCV.

Three tests, in increasing realism:
  1. exact  -- noise-free synthetic data; we should recover H to ~1e-10
  2. noisy  -- synthetic + gaussian noise + deliberate outliers; RANSAC must
               reject the outliers and land close to OpenCV
  3. real   -- actual SIFT matches from the match footage

Plus an ablation showing what happens without Hartley normalisation.

Important: a homography is only defined up to scale, so comparing H matrices
elementwise is meaningless. Two matrices differing by a factor of 7 describe
the identical transform. We always compare by pushing points through them and
measuring where they land, in pixels.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from rugbyviz.geometry import (  # noqa: E402
    apply_homography, dlt_homography, ransac_homography, reprojection_error,
)

rng = np.random.default_rng(42)
IMG_W, IMG_H = 1920, 1080


def random_homography() -> np.ndarray:
    """A plausible camera-motion homography: modest rotation, scale, translation
    and a touch of perspective."""
    ang = rng.uniform(-0.15, 0.15)
    sc = rng.uniform(0.85, 1.15)
    H = np.array([
        [sc * np.cos(ang), -sc * np.sin(ang), rng.uniform(-200, 200)],
        [sc * np.sin(ang), sc * np.cos(ang), rng.uniform(-120, 120)],
        [rng.uniform(-8e-5, 8e-5), rng.uniform(-8e-5, 8e-5), 1.0],
    ])
    return H / H[2, 2]


def agreement(Ha: np.ndarray, Hb: np.ndarray) -> float:
    """Median pixel disagreement between two homographies over the image.

    This is the only sane way to compare two homographies.
    """
    grid = np.stack(np.meshgrid(np.linspace(0, IMG_W, 30),
                                np.linspace(0, IMG_H, 30)), -1).reshape(-1, 2)
    return float(np.median(np.linalg.norm(
        apply_homography(Ha, grid) - apply_homography(Hb, grid), axis=1)))


def test_exact() -> None:
    print("1. EXACT  (noise-free, 4 and 20 points)")
    for n in (4, 20):
        H_true = random_homography()
        src = rng.uniform([0, 0], [IMG_W, IMG_H], size=(n, 2))
        dst = apply_homography(H_true, src)

        H_ours = dlt_homography(src, dst)
        H_cv, _ = cv2.findHomography(src, dst, 0)

        print(f"   n={n:<3} ours vs truth: {agreement(H_ours, H_true):.2e} px"
              f"   ours vs opencv: {agreement(H_ours, H_cv):.2e} px")
    print()


def test_normalisation_ablation() -> None:
    """Does Hartley normalisation actually buy anything?

    Three point distributions of increasing nastiness. Conditioning only bites
    when the design matrix entries span many orders of magnitude, which needs
    either large coordinates or points clustered far from the origin.
    """
    print("2. ABLATION  (does Hartley normalisation matter?)")
    cases = {
        "well spread over frame": lambda n: rng.uniform([0, 0], [IMG_W, IMG_H], (n, 2)),
        "clustered in a corner ": lambda n: rng.uniform([1750, 950], [1900, 1050], (n, 2)),
        "large coords (~1e6)   ": lambda n: rng.uniform([1e6, 1e6], [1e6 + 400, 1e6 + 300], (n, 2)),
    }
    for label, gen in cases.items():
        H_true = random_homography()
        src = gen(30)
        dst = apply_homography(H_true, src) + rng.normal(0, 0.3, (30, 2))
        on = dlt_homography(src, dst, normalise=True)
        off = dlt_homography(src, dst, normalise=False)
        # Compare over the region the points actually occupy, not the frame.
        lo, hi = src.min(0), src.max(0)
        grid = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], 20),
                                    np.linspace(lo[1], hi[1], 20)), -1).reshape(-1, 2)
        e_on = np.median(np.linalg.norm(apply_homography(on, grid) - apply_homography(H_true, grid), axis=1))
        e_off = np.median(np.linalg.norm(apply_homography(off, grid) - apply_homography(H_true, grid), axis=1))
        print(f"   {label}  with: {e_on:>10.5f} px   without: {e_off:>12.5f} px")
    print()


def test_noisy_with_outliers() -> None:
    print("3. NOISY + OUTLIERS  (200 points, 30% outliers, sigma=0.5px)")
    n, frac = 200, 0.30
    H_true = random_homography()
    src = rng.uniform([0, 0], [IMG_W, IMG_H], size=(n, 2))
    dst = apply_homography(H_true, src) + rng.normal(0, 0.5, (n, 2))

    n_out = int(n * frac)
    out_idx = rng.choice(n, n_out, replace=False)
    dst[out_idx] = rng.uniform([0, 0], [IMG_W, IMG_H], size=(n_out, 2))
    truth_inlier = np.ones(n, bool)
    truth_inlier[out_idx] = False

    H_ours, mask_ours = ransac_homography(src, dst, threshold=3.0)
    H_cv, mask_cv = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    mask_cv = mask_cv.ravel().astype(bool)

    # Non-robust fit on the same data, to show what RANSAC is buying us.
    H_naive = dlt_homography(src, dst)

    tp = int((mask_ours & truth_inlier).sum())
    fp = int((mask_ours & ~truth_inlier).sum())
    print(f"   ours   inliers={mask_ours.sum():<4} (correct={tp}, wrong={fp})"
          f"  err vs truth={agreement(H_ours, H_true):.4f} px")
    print(f"   opencv inliers={mask_cv.sum():<4}"
          f"{'':<22}err vs truth={agreement(H_cv, H_true):.4f} px")
    print(f"   ours vs opencv: {agreement(H_ours, H_cv):.4f} px")
    print(f"   no RANSAC (plain least squares): {agreement(H_naive, H_true):.1f} px  <-- ruined")
    print()


def test_real_frames() -> None:
    print("4. REAL FRAMES  (SIFT matches from the match footage)")
    pairs = [("pair_a0", "pair_a1"), ("pair_b0", "pair_b1"), ("pair_c0", "pair_c1")]
    root = Path("data/derived/frames")
    sift = cv2.SIFT_create(nfeatures=4000)

    for a_name, b_name in pairs:
        a = cv2.imread(str(root / f"{a_name}.jpg"))
        b = cv2.imread(str(root / f"{b_name}.jpg"))
        if a is None or b is None:
            print(f"   {a_name}: frames missing, skipped")
            continue

        mask = np.zeros(a.shape[:2], np.uint8)
        mask[:430, :] = 255  # static background only
        ka, da = sift.detectAndCompute(a, mask)
        kb, db = sift.detectAndCompute(b, mask)
        good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2)
                if m.distance < 0.75 * n.distance]
        src = np.float64([ka[m.queryIdx].pt for m in good])
        dst = np.float64([kb[m.trainIdx].pt for m in good])

        H_ours, mask_ours = ransac_homography(src, dst, threshold=3.0)
        H_cv, mask_cv = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        mask_cv = mask_cv.ravel().astype(bool)

        e_ours = np.median(reprojection_error(H_ours, src[mask_ours], dst[mask_ours]))
        e_cv = np.median(reprojection_error(H_cv, src[mask_cv], dst[mask_cv]))
        print(f"   {a_name}->{b_name}  matches={len(good):<5}"
              f" ours: {mask_ours.sum():>4} inliers, {e_ours:.3f} px  |"
              f" opencv: {mask_cv.sum():>4} inliers, {e_cv:.3f} px  |"
              f" agree to {agreement(H_ours, H_cv):.3f} px")
    print()


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    test_exact()
    test_normalisation_ablation()
    test_noisy_with_outliers()
    test_real_frames()
