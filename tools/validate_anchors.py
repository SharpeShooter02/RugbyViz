"""Cross-validate the mosaic's per-frame homographies against each other.

build_mosaic.py accepted a homography if it was geometrically plausible on its
own. That is not enough: a frame can register plausibly and still be placed
wrongly on the mosaic, and every later frame that anchors to it inherits the
error. One such anchor dragged 35 s of the match sideways far enough that the
entire spectator crowd fell inside the pitch bounds.

The test here is agreement, not plausibility. For anchor i with stored
homography H_i, register i against a neighbour j and compute

    H_i_via_j = H_j  @  H_(i -> j)

If H_i is right, that composition should reproduce it. Disagreement measured
in mosaic pixels over the frame corners. An anchor that disagrees with several
independent neighbours is the one at fault.

Writes data/derived/positions/anchor_blacklist.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

MOSAIC_META = Path("data/derived/mosaic/mosaic_meta.json")
ANCHOR_DIR = Path("data/derived/mosaic_src")
CACHE = Path("data/derived/positions/anchor_features.npz")
OUT = Path("data/derived/positions/anchor_blacklist.json")

N_NEIGHBOURS = 4
MIN_INLIERS = 60
TOL_PX = 40.0          # median corner disagreement above this = suspect


def main() -> None:
    meta = json.loads(MOSAIC_META.read_text())
    homs = meta["homographies"]
    names = sorted(homs, key=lambda n: int("".join(c for c in n if c.isdigit())))
    idx = {n: i for i, n in enumerate(names)}

    if not CACHE.exists():
        raise SystemExit("run tools/track_positions.py once first to build "
                         f"{CACHE} (anchor SIFT features)")
    z = np.load(CACHE, allow_pickle=True)
    kps, descs = list(z["kps"]), list(z["descs"])

    img = cv2.imread(str(ANCHOR_DIR / names[0]))
    h, w = img.shape[:2]
    corners = np.float64([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=40))

    print(f"cross-validating {len(names)} anchors against "
          f"{N_NEIGHBOURS} neighbours each\n")
    disagreement: dict[str, list[float]] = {n: [] for n in names}
    t0 = time.perf_counter()

    for i, ni in enumerate(names):
        Hi = np.array(homs[ni])
        base = cv2.perspectiveTransform(corners, Hi).reshape(-1, 2)
        for off in (-2, -1, 1, 2)[:N_NEIGHBOURS]:
            j = i + off
            if not (0 <= j < len(names)):
                continue
            if len(descs[i]) < 2 or len(descs[j]) < 2:
                continue
            raw = flann.knnMatch(descs[i], descs[j], k=2)
            good = [m for m, n2 in (p for p in raw if len(p) == 2)
                    if m.distance < 0.75 * n2.distance]
            if len(good) < MIN_INLIERS:
                continue
            src = np.float64([kps[i][m.queryIdx] for m in good])
            dst = np.float64([kps[j][m.trainIdx] for m in good])
            Hij, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if Hij is None or mask.ravel().sum() < MIN_INLIERS:
                continue
            via = np.array(homs[names[j]]) @ Hij
            got = cv2.perspectiveTransform(corners, via).reshape(-1, 2)
            if not np.all(np.isfinite(got)):
                continue
            disagreement[ni].append(float(np.median(np.linalg.norm(got - base, axis=1))))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(names)} ({time.perf_counter()-t0:.0f}s)", flush=True)

    scores = {n: float(np.median(v)) for n, v in disagreement.items() if v}
    unchecked = [n for n in names if n not in scores]
    vals = np.array(list(scores.values()))

    print(f"\ncorner disagreement with neighbours (mosaic px):")
    for p in (50, 75, 90, 95, 99):
        print(f"   p{p:<3} {np.percentile(vals, p):8.1f}")
    print(f"   max  {vals.max():8.1f}")

    bad = sorted([n for n, s in scores.items() if s > TOL_PX],
                 key=lambda n: -scores[n])
    print(f"\nsuspect anchors (> {TOL_PX:.0f} px): {len(bad)} of {len(scores)}")
    for n in bad[:20]:
        print(f"   {n}  {scores[n]:9.1f} px  "
              f"({len(disagreement[n])} neighbours agreed to disagree)")
    if unchecked:
        print(f"\n{len(unchecked)} anchors had too few matches to check: "
              f"{unchecked[:8]}{' ...' if len(unchecked) > 8 else ''}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "tolerance_px": TOL_PX,
        "blacklist": bad + unchecked,
        "scores": scores,
    }, indent=2))
    print(f"\nwrote {OUT}  ({len(bad) + len(unchecked)} anchors blacklisted)")


if __name__ == "__main__":
    main()
