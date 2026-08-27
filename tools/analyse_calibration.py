"""How trustworthy is the current calibration, and where?

Residual-on-clicked-points only tells you how well the fit explains the points
you gave it. It says nothing about regions you did not click, which are pure
extrapolation. This quantifies that.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

cfg = json.loads(Path("config/a-side-vs-msu-2025-09-13.json").read_text())
H = np.array(cfg["mosaic_to_pitch"])          # mosaic px -> pitch m
H_inv = np.linalg.inv(H)
L, W = cfg["pitch_length_m"], cfg["pitch_width_m"]

clicked_y = sorted({round(p["pitch"][1], 1) for p in cfg["points"]})
clicked_x = sorted({round(p["pitch"][0], 1) for p in cfg["points"]})
print(f"clicked y values : {clicked_y}")
print(f"clicked x values : {clicked_x}")
print(f"y coverage       : {min(clicked_y):.0f} - {max(clicked_y):.0f} m "
      f"of 0 - {W:.0f}  ({W - max(clicked_y):.0f} m extrapolated)\n")


def to_mosaic(x, y):
    return cv2.perspectiveTransform(np.float64([[[x, y]]]), H_inv).reshape(2)


def to_pitch(mx, my):
    return cv2.perspectiveTransform(np.float64([[[mx, my]]]), H).reshape(2)


# --- scale: how many mosaic pixels represent one metre, across the width ----
print("resolution across the pitch width (at halfway):")
print(f"  {'pitch y':>8}  {'px per metre':>13}  {'1 px =':>10}  region")
for y in (0, 5, 15, 25, 35, 45, 55, 62, 68, 70):
    # Use a symmetric interval and divide by its ACTUAL length. Clamping the
    # endpoints at the pitch edge without adjusting the divisor understates
    # px/m at y=0 and y=70 by 2x, which looks like a non-monotonic scale and
    # invites a false diagnosis of a vanishing-line collapse.
    lo, hispan = max(0.0, y - 0.5), min(float(W), y + 0.5)
    a, b = to_mosaic(50, lo), to_mosaic(50, hispan)
    ppm = float(np.linalg.norm(b - a)) / (hispan - lo)
    tag = "clicked" if any(abs(y - c) < 0.6 for c in clicked_y) else (
        "EXTRAPOLATED" if y > max(clicked_y) else "interpolated")
    print(f"  {y:>8.0f}  {ppm:>13.1f}  {1/max(ppm,1e-9):>9.2f}m  {tag}")

# --- sensitivity: shift each clicked point by 3 px, see what moves ----------
print("\nsensitivity: re-solve with each point nudged 3 px, measure the change")
src = np.float64([p["mosaic"] for p in cfg["points"]])
dst = np.float64([p["pitch"] for p in cfg["points"]])
probes = np.float64([[50, 35], [50, 70], [50, 0], [0, 35], [100, 35], [50, 65]])
probe_names = ["centre of pitch", "halfway x NEAR touch", "halfway x FAR touch",
               "LEFT try centre", "RIGHT try centre", "halfway x near 5m"]

base = cv2.perspectiveTransform(
    np.array([to_mosaic(*p) for p in probes]).reshape(-1, 1, 2), H).reshape(-1, 2)

shifts = {n: [] for n in probe_names}
rng = np.random.default_rng(0)
for trial in range(40):
    jitter = rng.normal(0, 3.0, src.shape)
    Hj, _ = cv2.findHomography(src + jitter, dst, 0)
    if Hj is None:
        continue
    got = cv2.perspectiveTransform(
        np.array([to_mosaic(*p) for p in probes]).reshape(-1, 1, 2), Hj).reshape(-1, 2)
    for n, b, g in zip(probe_names, base, got):
        shifts[n].append(float(np.linalg.norm(g - b)))

print(f"  {'probe location':<24}{'median shift':>14}{'p90':>10}")
for n in probe_names:
    v = np.array(shifts[n])
    if len(v):
        print(f"  {n:<24}{np.median(v):>11.2f} m{np.percentile(v,90):>9.2f} m")

# --- where does the mapping stop being valid? ------------------------------
# A homography sends one line in the source plane to infinity: the vanishing
# line, where the projective denominator w = h20*x + h21*y + h22 hits zero.
# Beyond it, points fold through infinity and coordinates are meaningless.
#
# Here that line is real physics, not a fitting error: the camera stands ON the
# near touchline near halfway, and a camera cannot image the ground it occupies.
print("\nvalid region (where the projective denominator stays away from zero)")
hi = H_inv
gx, gy = np.meshgrid(np.linspace(0, L, 201), np.linspace(0, W, 141))
w = hi[2, 0] * gx + hi[2, 1] * gy + hi[2, 2]
ref = hi[2, 0] * 50.0 + hi[2, 1] * 35.0 + hi[2, 2]   # denominator mid-pitch
rel = w / ref
bad = rel < 0.02          # sign flip or near-zero => at/over the vanishing line
print(f"  fraction of the pitch surface that is degenerate: {100*bad.mean():.1f}%")

# Largest y that is valid across the WHOLE length, not just at halfway.
valid_rows = [float(y) for y, row in zip(np.linspace(0, W, 141), bad) if not row.any()]
max_valid = max(valid_rows) if valid_rows else 0.0
worst_x = float(gx[0][np.argmin(rel.min(axis=0))])
print(f"  valid at every x up to y = {max_valid:.1f} m of {W:.0f}")
print(f"  weakest column is x = {worst_x:.0f} m (min relative denominator "
      f"{rel.min():.3f})")
if max_valid < W:
    print(f"  -> pitch y > {max_valid:.0f} m is unreliable near the touchline ends.")
    print(f"     That is where the camera stands, so players there are largely")
    print(f"     out of frame anyway. Flag rather than trust those positions.")

cfg["max_valid_y_m"] = round(min(max_valid, W), 1)
cfg["degenerate_fraction"] = round(float(bad.mean()), 4)
Path("config/a-side-vs-msu-2025-09-13.json").write_text(json.dumps(cfg, indent=2))
print(f"  recorded max_valid_y_m = {cfg['max_valid_y_m']} in the config")

# --- what does the pitch rectangle map back to? ----------------------------
print("\nsanity: pitch corners -> mosaic -> pitch round trip")
for name, (x, y) in [("far-left try corner", (0, 0)), ("far-right try corner", (L, 0)),
                     ("near-left try corner", (0, W)), ("near-right try corner", (L, W))]:
    m = to_mosaic(x, y)
    back = to_pitch(*m)
    inside = 0 <= m[0] < cfg["mosaic_size"][0] and 0 <= m[1] < cfg["mosaic_size"][1]
    print(f"  {name:<22} mosaic=({m[0]:8.0f},{m[1]:8.0f})  "
          f"{'in canvas' if inside else 'OUTSIDE CANVAS'}")
