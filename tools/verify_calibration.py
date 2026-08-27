"""Draw the known pitch geometry back onto the mosaic using the saved calibration.

If the calibration is right, the drawn lines land on the real markings in the
mosaic. If they float off into the trees, the clicks were wrong.

This is the check that matters -- residuals can look small while the whole
solution is subtly wrong (e.g. FAR and NEAR touchlines swapped).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

MOSAIC = Path("data/derived/mosaic/mosaic.jpg")
CONFIG = Path("config/a-side-vs-msu-2025-09-13.json")
OUT = Path("data/derived/mosaic/calibration_check.jpg")
OUT_BEV = Path("data/derived/mosaic/birdseye.jpg")

if not CONFIG.exists():
    sys.exit(f"missing {CONFIG} - run tools/calibrate_pitch.py first")

cfg = json.loads(CONFIG.read_text())
H = np.array(cfg["mosaic_to_pitch"], np.float64)   # mosaic px -> pitch m
H_inv = np.linalg.inv(H)                           # pitch m -> mosaic px
L, W = cfg["pitch_length_m"], cfg["pitch_width_m"]

img = cv2.imread(str(MOSAIC))


def to_mosaic(pts_m: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.float64(pts_m).reshape(-1, 1, 2), H_inv).reshape(-1, 2)


def draw_line(a, b, colour, thick=3, n=60):
    """Draw a pitch-space line as a polyline so any residual curvature shows."""
    ts = np.linspace(0, 1, n)[:, None]
    pts = to_mosaic(np.array(a) * (1 - ts) + np.array(b) * ts)
    ok = np.all(np.isfinite(pts), axis=1)
    pts = pts[ok].astype(np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts], False, colour, thick, cv2.LINE_AA)


GREEN, YELLOW, CYAN, RED = (0, 255, 0), (0, 255, 255), (255, 255, 0), (0, 0, 255)

# touchlines and try lines
draw_line((0, 0), (L, 0), GREEN, 4)
draw_line((0, W), (L, W), GREEN, 4)
draw_line((0, 0), (0, W), RED, 4)
draw_line((L, 0), (L, W), RED, 4)
# halfway, 22s, 10m lines
draw_line((L / 2, 0), (L / 2, W), YELLOW, 3)
for x in (22, L - 22):
    draw_line((x, 0), (x, W), YELLOW, 3)
for x in (L / 2 - 10, L / 2 + 10, 5, L - 5):
    draw_line((x, 0), (x, W), CYAN, 1)

# the clicked points
for p in cfg["points"]:
    m = np.int32(p["mosaic"])
    colour = (0, 255, 255) if p["inlier"] else (0, 0, 255)
    cv2.drawMarker(img, tuple(m), colour, cv2.MARKER_TILTED_CROSS, 34, 3)
    cv2.putText(img, f'{p["pitch"][0]:.0f},{p["pitch"][1]:.0f}',
                (m[0] + 16, m[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, colour, 3)

cv2.imwrite(str(OUT), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
small = cv2.resize(img, (1800, int(1800 * img.shape[0] / img.shape[1])))
cv2.imwrite(str(OUT.with_name("calibration_check_small.jpg")), small)

# Bird's-eye rectification: warp the mosaic into true pitch coordinates.
# A correct calibration makes the pitch a clean rectangle here.
PPM = 10  # pixels per metre
bev_H = np.array([[PPM, 0, 0], [0, PPM, 0], [0, 0, 1]], np.float64) @ H
bev = cv2.warpPerspective(cv2.imread(str(MOSAIC)), bev_H,
                          (int(L * PPM), int(W * PPM)))
for x in (0, 22, 50, 78, 100):
    cv2.line(bev, (int(x * PPM), 0), (int(x * PPM), int(W * PPM)), YELLOW, 2)
cv2.rectangle(bev, (0, 0), (int(L * PPM) - 1, int(W * PPM) - 1), GREEN, 3)
cv2.imwrite(str(OUT_BEV), bev, [cv2.IMWRITE_JPEG_QUALITY, 92])

print(f"median residual : {cfg['median_residual_m']:.2f} m")
print(f"points          : {len(cfg['points'])} "
      f"({sum(1 for p in cfg['points'] if p['inlier'])} inliers)")
print(f"overlay         : {OUT}")
print(f"bird's-eye      : {OUT_BEV}")
print("\nCheck the overlay: green touchlines and yellow halfway/22s should sit")
print("on the real markings. In the bird's-eye view the pitch should be a")
print("clean rectangle with the halfway line straight down the middle.")
