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
    a, b = to_mosaic(50, max(0, y - 0.5)), to_mosaic(50, min(W, y + 0.5))
    ppm = float(np.linalg.norm(b - a))
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

# --- what does the pitch rectangle map back to? ----------------------------
print("\nsanity: pitch corners -> mosaic -> pitch round trip")
for name, (x, y) in [("far-left try corner", (0, 0)), ("far-right try corner", (L, 0)),
                     ("near-left try corner", (0, W)), ("near-right try corner", (L, W))]:
    m = to_mosaic(x, y)
    back = to_pitch(*m)
    inside = 0 <= m[0] < cfg["mosaic_size"][0] and 0 <= m[1] < cfg["mosaic_size"][1]
    print(f"  {name:<22} mosaic=({m[0]:8.0f},{m[1]:8.0f})  "
          f"{'in canvas' if inside else 'OUTSIDE CANVAS'}")
