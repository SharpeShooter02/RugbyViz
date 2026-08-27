"""Why did the mosaic canvas explode? Look at direct-to-reference registrations
only (the trustworthy ones) and characterise the camera motion."""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

SRC = Path("data/derived/mosaic_src")
paths = sorted(SRC.glob("*.jpg"))
sift = cv2.SIFT_create(nfeatures=3000)

t0 = time.perf_counter()
kps, descs = [], []
for p in paths:
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    k, d = sift.detectAndCompute(img, None)
    kps.append(np.float64([q.pt for q in k]) if k else np.empty((0, 2)))
    descs.append(d if d is not None else np.empty((0, 128), np.float32))
print(f"features done ({time.perf_counter()-t0:.0f}s)")

ref = len(paths) // 2
h, w = cv2.imread(str(paths[0])).shape[:2]
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
corners = np.float64([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

rows = []
for i in range(len(paths)):
    if i == ref:
        rows.append((i, 99999, 0.0, 1.0, 0.0, 0.0))
        continue
    raw = flann.knnMatch(descs[i], descs[ref], k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 40:
        rows.append((i, len(good), np.nan, np.nan, np.nan, np.nan))
        continue
    src = np.float64([kps[i][m.queryIdx] for m in good])
    dst = np.float64([kps[ref][m.trainIdx] for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None:
        rows.append((i, len(good), np.nan, np.nan, np.nan, np.nan))
        continue
    n = int(mask.ravel().sum())
    c = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    area_ratio = abs(cv2.contourArea(c.astype(np.float32))) / (w * h)
    dx = float(c[:, 0].mean() - w / 2)
    # perspective magnitude: how strong the projective part is
    persp = float(np.hypot(H[2, 0], H[2, 1]) * w)
    rows.append((i, n, dx, area_ratio, persp, float(c[:, 0].max() - c[:, 0].min())))

ok = [r for r in rows if not np.isnan(r[2])]
print(f"\ndirect registrations: {len(ok)}/{len(paths)}")

dxs = np.array([r[2] for r in ok])
ar = np.array([r[3] for r in ok])
pe = np.array([r[4] for r in ok])
wd = np.array([r[5] for r in ok])

print(f"\nhorizontal offset from reference (px in ref frame):")
print(f"   min {dxs.min():10.0f}   max {dxs.max():10.0f}   span {dxs.max()-dxs.min():10.0f}")
print(f"\narea ratio (warped frame area / original):")
print(f"   min {ar.min():10.3f}   max {ar.max():10.3f}")
print(f"\nprojective strength |h31,h32|*width  (0 = affine, >1 = severe):")
print(f"   min {pe.min():10.3f}   median {np.median(pe):8.3f}   max {pe.max():10.3f}")
print(f"\nwarped frame width (px):")
print(f"   min {wd.min():10.0f}   median {np.median(wd):8.0f}   max {wd.max():10.0f}")

print("\nworst 12 by warped width:")
for r in sorted(ok, key=lambda r: -r[5])[:12]:
    print(f"   {paths[r[0]].name}  inliers={r[1]:<5} dx={r[2]:>9.0f} "
          f"area={r[3]:>8.3f} persp={r[4]:>7.3f} width={r[5]:>10.0f}")

# How far apart in time can two frames still register directly?
reg_idx = np.array([r[0] for r in ok])
print(f"\nregistered frame indices span: {reg_idx.min()} .. {reg_idx.max()}")
gaps = [i for i in range(len(paths)) if i not in set(reg_idx)]
print(f"unregistered count: {len(gaps)}")
if gaps:
    print(f"unregistered indices (first 30): {gaps[:30]}")
