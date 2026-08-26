# Findings

## 2025-08-26 — Source footage characterisation

`a-side-vs-msu-2025-09-13.mp4`

| Property | Value |
|---|---|
| Resolution | 1920x1080 |
| Frame rate | 30000/1001 (29.97 fps) |
| Duration | 5761.7 s (~96 min) |
| Frames | 172,673 |
| Codec | h264 / yuv420p |
| Bitrate | ~4.67 Mbit/s |

This is Veo's **auto-follow crop**, not the stitched panorama. The full panorama
is not available through the download we have.

Duration exceeds an 80-minute match, so the file includes warm-up and/or
half-time. Period boundaries must be marked before any time-based stat is valid.

## 2025-08-26 — Camera motion is 8-DOF, not 3-DOF

**Question.** Veo's auto-follow view is a virtual camera panning over a stitched
panorama. If it were a plain crop-and-zoom of a fixed image, per-frame
calibration would be a 3-DOF problem (x, y, zoom) — much easier than full
homography. Is it?

**Method.** `tools/dof_test.py`. For frame pairs 2 s apart: mask to the top 430 px
(static background only — trees, houses, fence, parked cars; excludes grass and
moving players), extract SIFT keypoints, match with Lowe ratio test at 0.75,
fit each model with RANSAC, measure median and p90 reprojection error on the
homography inlier set.

**Result.** Median reprojection error, pixels:

| Pair | motion | 2-DOF trans | 4-DOF sim | 6-DOF affine | 8-DOF homography |
|---|---|---|---|---|---|
| a (t=900)  | 16 px pan          |  0.78 |  0.59 |  0.50 | **0.34** |
| b (t=2100) | 14 px pan + zoom   |  5.88 |  0.95 |  0.96 | **0.28** |
| c (t=4500) | 319 px pan, 3.5% zoom | 14.88 | 15.71 | 13.19 | **0.30** |

Match counts 749–1789 per pair, ~95% homography inliers.

**Conclusion.** The 3-DOF hypothesis is **rejected**. Similarity transforms are
adequate only for small motions; on a real pan they degrade to ~16 px median /
62 px p90, while homography stays at ~0.3 px across all cases. Veo renders a
rectified virtual view off a cylindrical stitch, so perspective warp varies with
pan angle. Use `cv2.findHomography`.

This is a good outcome: registration is sub-pixel and extremely well-conditioned
on this footage, thanks to a feature-rich static background.

## 2025-08-26 — The pitch is unmarked

Sampled frames at t = 300, 900, 1500, 2100, 3300, 3900, 4500, 5100.

A faint halfway line is visible at t=300. At t=900 and t=3300 there are **no
field markings at all** — bare grass with a few orange cones near the touchline.

**Implication.** The standard approach to sports pitch calibration (detect
lines, fit a known pitch template) is not viable here. Instead:

- frame -> mosaic changes every frame -> must be automatic (solved, see above)
- mosaic -> pitch never changes -> do it **by hand, once per match**

Expect a few metres of calibration error given there is little to click besides
cones and corner flags. Acceptable for thirds-of-pitch territory; not acceptable
for precise metres-gained.

## Open risks

1. **No ground truth.** No manual stats exist for any match. Ruck detection
   cannot be validated or tuned until ~20 min of play is hand-tagged.
2. **Mosaic drift.** Chaining thousands of homographies accumulates error and
   fails silently — territory degrades over the match with no error raised.
   Mitigate by anchoring to keyframes and re-solving globally.
3. **Small-object detection.** Far-touchline players are ~25-35 px tall.
   Untested. May require tiled inference.
4. **Python 3.13** is ahead of much of the ML ecosystem. May need a 3.12 venv
   when PyTorch and a detector are added.
