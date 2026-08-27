"""Headless smoke test of the calibration path.

calibrate_pitch.py needs a human at the mouse, so this exercises everything
around the clicking: the viewer coordinate maths, the offscreen render, the
solve/save, and the verification overlay. If this passes, a real click session
will not hit a crash halfway through and lose the work.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

spec = importlib.util.spec_from_file_location("calib", ROOT / "tools" / "calibrate_pitch.py")
calib = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calib)

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------------------
print("\n[1] mosaic loads")
mos = cv2.imread(str(calib.MOSAIC))
check("mosaic readable", mos is not None)
if mos is None:
    sys.exit("cannot continue without a mosaic")
H_img, W_img = mos.shape[:2]
print(f"      {W_img} x {H_img}")

# ---------------------------------------------------------------------------
print("\n[2] viewer coordinate maths")
v = calib.Viewer(mos)
ok = True
for scale in (0.1, 0.5, 1.0, 4.0):
    v.scale = scale
    v.cx, v.cy = W_img * 0.37, H_img * 0.61
    for px, py in [(0, 0), (750, 430), (1499, 859)]:
        mx, my = v.view_to_img(px, py)
        bx, by = v.img_to_view(mx, my)
        if abs(bx - px) > 1e-6 or abs(by - py) > 1e-6:
            ok = False
check("view<->image round-trip exact", ok)

v.scale = 1.0
v.cx, v.cy = W_img / 2, H_img / 2
mx, my = v.view_to_img(calib.VIEW_W / 2, calib.VIEW_H / 2)
check("view centre maps to pan centre", abs(mx - W_img / 2) < 1e-9 and abs(my - H_img / 2) < 1e-9)

# ---------------------------------------------------------------------------
print("\n[3] offscreen render (no window)")
pts = [(1000.0, 900.0, "test", 50.0, 0.0)]
for scale in (0.05, 0.13, 1.0, 8.0):
    v.scale = scale
    v.cx, v.cy = W_img / 2, H_img / 2
    frame = v.render(pts, 0)
    check(f"render at {scale}x", frame.shape == (calib.VIEW_H, calib.VIEW_W, 3)
          and frame.any(), f"nonzero px={int((frame > 0).sum())}")

# edge cases: panned hard into a corner, and past the end of the landmark list
v.scale = 6.0
v.cx, v.cy = 0, 0
check("render at top-left corner", v.render(pts, 0).shape == (calib.VIEW_H, calib.VIEW_W, 3))
v.cx, v.cy = W_img, H_img
check("render at bottom-right corner", v.render(pts, 0).shape == (calib.VIEW_H, calib.VIEW_W, 3))
check("render past last landmark",
      v.render(pts, len(calib.LANDMARKS)).shape == (calib.VIEW_H, calib.VIEW_W, 3))

# ---------------------------------------------------------------------------
print("\n[4] solve + save, using a known synthetic homography")
# Invent a ground-truth pitch->mosaic mapping, generate exact clicks from it,
# and check the solver recovers it. This tests the code, not the mosaic.
L, W = calib.PITCH_LENGTH, calib.PITCH_WIDTH
pitch_quad = np.float32([[0, 0], [L, 0], [L, W], [0, W]])
mosaic_quad = np.float32([[900, 800], [10600, 780], [9900, 1900], [1500, 2000]])
H_p2m = cv2.getPerspectiveTransform(pitch_quad, mosaic_quad)

synth = []
for lbl, px, py in calib.LANDMARKS:
    m = cv2.perspectiveTransform(np.float32([[[px, py]]]), H_p2m).reshape(2)
    synth.append((float(m[0]), float(m[1]), lbl, px, py))

orig_match, orig_dir = calib.MATCH, calib.CONFIG_DIR
calib.MATCH = "_smoketest"
saved = calib.save(synth, mos.shape)
check("save() returned True", saved)

cfg_path = orig_dir / "_smoketest.json"
check("config written", cfg_path.exists())
if cfg_path.exists():
    cfg = json.loads(cfg_path.read_text())
    check("median residual ~0 on exact data", cfg["median_residual_m"] < 0.01,
          f"{cfg['median_residual_m']:.2e} m")
    H_solved = np.array(cfg["mosaic_to_pitch"])
    # Round-trip a grid of pitch points through mosaic and back.
    grid = np.float32([[x, y] for x in np.linspace(0, L, 11) for y in np.linspace(0, W, 8)])
    to_m = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H_p2m)
    back = cv2.perspectiveTransform(to_m, H_solved).reshape(-1, 2)
    err = np.linalg.norm(back - grid, axis=1)
    check("pitch->mosaic->pitch round-trip", err.max() < 0.01, f"max {err.max():.2e} m")

# ---------------------------------------------------------------------------
print("\n[5] under-determined input is rejected")
check("3 points refused", calib.save(synth[:3], mos.shape) is False
      if len(synth) >= 3 else True)

# ---------------------------------------------------------------------------
print("\n[6] verify_calibration.py end-to-end")
target = orig_dir / f"{orig_match}.json"
had_real = target.exists()
if not had_real and cfg_path.exists():
    target.write_text(cfg_path.read_text())
r = subprocess.run([sys.executable, "tools/verify_calibration.py"],
                   cwd=ROOT, capture_output=True, text=True)
print("      " + "\n      ".join(r.stdout.strip().splitlines()[:6]))
if r.returncode != 0:
    print("      STDERR: " + r.stderr.strip()[:500])
check("verify_calibration exit 0", r.returncode == 0)
for f in ("calibration_check.jpg", "birdseye.jpg"):
    p = ROOT / "data/derived/mosaic" / f
    check(f"produced {f}", p.exists() and p.stat().st_size > 1000)

# cleanup: never leave synthetic calibration lying around
if not had_real:
    target.unlink(missing_ok=True)
cfg_path.unlink(missing_ok=True)
calib.MATCH = orig_match

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILED: {fails}"))
sys.exit(1 if fails else 0)
