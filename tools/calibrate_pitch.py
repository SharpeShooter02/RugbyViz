r"""Click known pitch landmarks on the mosaic to solve the mosaic -> pitch homography.

Run:
    .\.venv\Scripts\python.exe tools\calibrate_pitch.py

Controls
    left click        place the current landmark
    scroll wheel      zoom in / out (zooms toward the cursor)
    right drag        pan
    n / space         skip this landmark (not visible in the mosaic)
    u                 undo last placed point
    s                 solve + save
    q / esc           quit without saving

Pitch coordinate system (World Rugby full size, metres):

    y=0   far touchline
          +----------------------------------------------------+
          |        |              |              |        |     |
   x=0 try|  x=22  |    x=50      |    x=78      | x=100 try    |
          |        |  halfway     |              |        |     |
          +----------------------------------------------------+
    y=70  near touchline

"left" and "right" mean as they appear in the mosaic image, not any real
compass direction. Which end is x=0 is arbitrary; it only has to be consistent.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

MOSAIC = Path("data/derived/mosaic/mosaic.jpg")
META = Path("data/derived/mosaic/mosaic_meta.json")
CONFIG_DIR = Path("config")
MATCH = "a-side-vs-msu-2025-09-13"

PITCH_LENGTH = 100.0   # try line to try line
PITCH_WIDTH = 70.0

POST_HALF_GAP = 2.8    # rugby posts stand 5.6 m apart, centred on the width

# The Veo camera sits on the near touchline near halfway, so it cannot see the
# near touchline underneath itself. Landmarks are therefore chosen to give
# spread in y WITHOUT relying on the near touchline:
#
#   y =  0    far touchline                      (visible along its length)
#   y =  5    far 5m dashed line                 (parallel to the touchline)
#   y = 15    far 15m dashed line                        "
#   y = 32.2  far-side goalpost base
#   y = 37.8  near-side goalpost base
#   y = 55    near 15m dashed line               (visible at oblique angle)
#   y = 70    near touchline                     (mostly NOT visible)
#
# The "+" cross marks visible on the pitch are exactly where those dashed
# lines cross the 22s, the 10m lines and halfway. Each is a precise point.
#
# Ordered most-reliable first, so a partial session still yields good points.
FAR_5, FAR_15, NEAR_15, NEAR_5 = 5.0, 15.0, 55.0, 65.0
POST_FAR = PITCH_WIDTH / 2 - POST_HALF_GAP      # 32.2
POST_NEAR = PITCH_WIDTH / 2 + POST_HALF_GAP     # 37.8

LANDMARKS = [
    # -- goalposts: rigid, unambiguous, and the only easy mid-width points ----
    ("RIGHT goalpost - FAR-side post base", PITCH_LENGTH, POST_FAR),
    ("RIGHT goalpost - NEAR-side post base", PITCH_LENGTH, POST_NEAR),
    ("LEFT goalpost - FAR-side post base", 0.0, POST_FAR),
    ("LEFT goalpost - NEAR-side post base", 0.0, POST_NEAR),

    # -- far touchline: the one line visible end to end --------------------
    ("halfway  x  FAR touchline", 50.0, 0.0),
    ("LEFT 22  x  FAR touchline", 22.0, 0.0),
    ("RIGHT 22 x  FAR touchline", 78.0, 0.0),

    # -- cross marks: dashed 15m line crossing the big lines ---------------
    ("halfway  x  FAR 15m dashed line", 50.0, FAR_15),
    ("LEFT 22  x  FAR 15m dashed line", 22.0, FAR_15),
    ("RIGHT 22 x  FAR 15m dashed line", 78.0, FAR_15),

    # -- cross marks: dashed 5m line ---------------------------------------
    ("halfway  x  FAR 5m dashed line", 50.0, FAR_5),
    ("LEFT 22  x  FAR 5m dashed line", 22.0, FAR_5),
    ("RIGHT 22 x  FAR 5m dashed line", 78.0, FAR_5),

    # -- 10m lines (dashed, 10m either side of halfway) --------------------
    ("LEFT 10m x  FAR touchline", 40.0, 0.0),
    ("RIGHT 10m x  FAR touchline", 60.0, 0.0),

    # -- corners and try lines ---------------------------------------------
    ("LEFT corner flag - FAR side", 0.0, 0.0),
    ("RIGHT corner flag - FAR side", PITCH_LENGTH, 0.0),

    # -- near side: rare and precious. Any one of these is worth a lot ------
    # Without points beyond y=55 the homography's vanishing line falls inside
    # the pitch and the last few metres map to nonsense, so the near 5m dashes
    # matter more than their count suggests.
    ("LEFT 22  x  NEAR 5m dashed line", 22.0, NEAR_5),
    ("RIGHT 22 x  NEAR 5m dashed line", 78.0, NEAR_5),
    ("halfway  x  NEAR 5m dashed line", 50.0, NEAR_5),
    ("LEFT 10m x  NEAR 5m dashed line", 40.0, NEAR_5),
    ("RIGHT 10m x  NEAR 5m dashed line", 60.0, NEAR_5),
    ("LEFT 22  x  NEAR 15m dashed line", 22.0, NEAR_15),
    ("RIGHT 22 x  NEAR 15m dashed line", 78.0, NEAR_15),
    ("halfway  x  NEAR 15m dashed line", 50.0, NEAR_15),
    ("LEFT 22  x  NEAR touchline", 22.0, PITCH_WIDTH),
    ("RIGHT 22 x  NEAR touchline", 78.0, PITCH_WIDTH),
    ("LEFT corner flag - NEAR side", 0.0, PITCH_WIDTH),
    ("RIGHT corner flag - NEAR side", PITCH_LENGTH, PITCH_WIDTH),
]

WIN = "calibrate pitch  |  click landmark, scroll=zoom, right-drag=pan, u=undo, n=skip, s=save, q=quit"
VIEW_W, VIEW_H = 1500, 860


class Viewer:
    def __init__(self, img: np.ndarray):
        self.img = img
        self.H, self.W = img.shape[:2]
        self.scale = min(VIEW_W / self.W, VIEW_H / self.H)
        self.cx, self.cy = self.W / 2, self.H / 2
        self.dragging = False
        self.drag_from = (0, 0)
        self.cursor = (0, 0)

    # -- coordinate mapping -------------------------------------------------
    def view_to_img(self, px: float, py: float) -> tuple[float, float]:
        return (self.cx + (px - VIEW_W / 2) / self.scale,
                self.cy + (py - VIEW_H / 2) / self.scale)

    def img_to_view(self, mx: float, my: float) -> tuple[float, float]:
        return ((mx - self.cx) * self.scale + VIEW_W / 2,
                (my - self.cy) * self.scale + VIEW_H / 2)

    def clamp(self) -> None:
        self.scale = float(np.clip(self.scale, 0.05, 20.0))
        self.cx = float(np.clip(self.cx, 0, self.W))
        self.cy = float(np.clip(self.cy, 0, self.H))

    # -- rendering ----------------------------------------------------------
    def render(self, points, idx) -> np.ndarray:
        canvas = np.zeros((VIEW_H, VIEW_W, 3), np.uint8)
        x0, y0 = self.view_to_img(0, 0)
        x1, y1 = self.view_to_img(VIEW_W, VIEW_H)
        ix0, iy0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
        ix1, iy1 = min(self.W, int(np.ceil(x1))), min(self.H, int(np.ceil(y1)))
        if ix1 > ix0 and iy1 > iy0:
            crop = self.img[iy0:iy1, ix0:ix1]
            dw = int(round((ix1 - ix0) * self.scale))
            dh = int(round((iy1 - iy0) * self.scale))
            if dw > 0 and dh > 0:
                interp = cv2.INTER_NEAREST if self.scale > 1.5 else cv2.INTER_AREA
                res = cv2.resize(crop, (dw, dh), interpolation=interp)
                vx, vy = self.img_to_view(ix0, iy0)
                vx, vy = int(round(vx)), int(round(vy))
                sx0, sy0 = max(0, vx), max(0, vy)
                sx1, sy1 = min(VIEW_W, vx + dw), min(VIEW_H, vy + dh)
                if sx1 > sx0 and sy1 > sy0:
                    canvas[sy0:sy1, sx0:sx1] = res[sy0 - vy:sy1 - vy, sx0 - vx:sx1 - vx]

        for i, (mx, my, lbl, _, _) in enumerate(points):
            vx, vy = self.img_to_view(mx, my)
            if -50 < vx < VIEW_W + 50 and -50 < vy < VIEW_H + 50:
                p = (int(vx), int(vy))
                cv2.drawMarker(canvas, p, (0, 255, 255), cv2.MARKER_CROSS, 22, 2)
                cv2.circle(canvas, p, 11, (0, 255, 255), 1)
                cv2.putText(canvas, str(i + 1), (p[0] + 13, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # crosshair at cursor for precise placement
        cxp, cyp = self.cursor
        cv2.line(canvas, (cxp, 0), (cxp, VIEW_H), (90, 90, 90), 1)
        cv2.line(canvas, (0, cyp), (VIEW_W, cyp), (90, 90, 90), 1)

        # header
        cv2.rectangle(canvas, (0, 0), (VIEW_W, 78), (0, 0, 0), -1)
        if idx < len(LANDMARKS):
            lbl, lx, ly = LANDMARKS[idx]
            done = any(p[2] == lbl for p in points)
            head = f"{'ALREADY PLACED - click only to move it:' if done else 'CLICK:'}  {lbl}"
            cv2.putText(canvas, head, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        (120, 255, 120) if done else (0, 255, 255), 2)
            cv2.putText(canvas, f"pitch coords ({lx:.0f} m, {ly:.0f} m)   "
                                f"[{idx+1}/{len(LANDMARKS)}]   placed: {len(points)}"
                                f"   zoom {self.scale:.2f}x   "
                                f"{'n=keep and move on' if done else 'n=skip'}",
                        (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        else:
            cv2.putText(canvas, f"All landmarks done - {len(points)} placed. "
                                f"Press 's' to solve and save.", (16, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return canvas


def main() -> None:
    if not MOSAIC.exists():
        sys.exit(f"missing {MOSAIC} - run tools/build_mosaic.py first")
    img = cv2.imread(str(MOSAIC))
    print(f"mosaic {img.shape[1]} x {img.shape[0]}")

    v = Viewer(img)
    points: list[tuple[float, float, str, float, float]] = []
    idx = 0

    # Resume an existing session: keep points already placed and jump to the
    # first landmark that has none, so adding a few extras does not mean
    # re-clicking everything.
    existing = CONFIG_DIR / f"{MATCH}.json"
    if existing.exists():
        cfg = json.loads(existing.read_text())
        if cfg.get("mosaic_size") == [img.shape[1], img.shape[0]]:
            for p in cfg["points"]:
                points.append((p["mosaic"][0], p["mosaic"][1], p["label"],
                               p["pitch"][0], p["pitch"][1]))
            done = {p[2] for p in points}
            print(f"resuming: loaded {len(points)} existing points from {existing}")
            print("          press 'n' through the ones already placed, or 'u' to redo them")
            while idx < len(LANDMARKS) and LANDMARKS[idx][0] in done:
                idx += 1
            print(f"          next unplaced landmark: "
                  f"{LANDMARKS[idx][0] if idx < len(LANDMARKS) else '(none - press s)'}")
        else:
            print(f"ignoring {existing}: it was clicked on a different mosaic size "
                  f"{cfg.get('mosaic_size')} vs {[img.shape[1], img.shape[0]]}")

    def on_mouse(event, x, y, flags, _):
        nonlocal idx
        v.cursor = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and idx < len(LANDMARKS):
            mx, my = v.view_to_img(x, y)
            lbl, px, py = LANDMARKS[idx]
            # One point per landmark: clicking a landmark that already has one
            # replaces it rather than adding a duplicate.
            was = any(p[2] == lbl for p in points)
            points[:] = [p for p in points if p[2] != lbl]
            points.append((mx, my, lbl, px, py))
            print(f"  {'replaced' if was else 'placed  '} {lbl:<34} "
                  f"mosaic=({mx:8.1f},{my:8.1f})  pitch=({px:5.1f},{py:5.1f})")
            idx += 1
        elif event == cv2.EVENT_RBUTTONDOWN:
            v.dragging = True
            v.drag_from = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            v.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and v.dragging:
            dx, dy = x - v.drag_from[0], y - v.drag_from[1]
            v.cx -= dx / v.scale
            v.cy -= dy / v.scale
            v.drag_from = (x, y)
            v.clamp()
        elif event == cv2.EVENT_MOUSEWHEEL:
            before = v.view_to_img(x, y)
            v.scale *= 1.25 if flags > 0 else 1 / 1.25
            v.clamp()
            after = v.view_to_img(x, y)
            v.cx += before[0] - after[0]
            v.cy += before[1] - after[1]
            v.clamp()

    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)
    print("\nclick each prompted landmark; 'n' to skip one you cannot see; 's' to save\n")

    while True:
        cv2.imshow(WIN, v.render(points, idx))
        k = cv2.waitKey(16) & 0xFF
        if k in (ord("q"), 27):
            print("quit without saving")
            break
        if k in (ord("n"), ord(" ")) and idx < len(LANDMARKS):
            print(f"  skipped {LANDMARKS[idx][0]}")
            idx += 1
        elif k == ord("u"):
            # Step back to the nearest EARLIER landmark that actually has a
            # point, and remove that one. Popping the end of the list is wrong
            # once points have been loaded from a saved session, because list
            # order no longer tracks landmark order.
            labels = {p[2] for p in points}
            j = idx - 1
            while j >= 0 and LANDMARKS[j][0] not in labels:
                j -= 1
            if j < 0:
                print("  nothing to undo before this landmark")
            else:
                lbl = LANDMARKS[j][0]
                points[:] = [p for p in points if p[2] != lbl]
                idx = j
                print(f"  undid {lbl}  (re-click it, or 'n' to leave it unplaced)")
        elif k == ord("s"):
            if len(points) < 4:
                print(f"  need at least 4 points, have {len(points)}")
                continue
            if save(points, img.shape):
                break
    cv2.destroyAllWindows()


def save(points, shape) -> bool:
    if len(points) < 4:
        print(f"  need at least 4 points to solve a homography, have {len(points)}")
        return False

    src = np.float64([[p[0], p[1]] for p in points])   # mosaic pixels
    dst = np.float64([[p[3], p[4]] for p in points])   # pitch metres

    if len(points) == 4:
        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        mask = np.ones(len(points), bool)
    else:
        H, m = cv2.findHomography(src, dst, cv2.RANSAC, 2.0)
        if H is None:
            print("  ERROR: could not solve a homography from those points")
            return False
        mask = m.ravel().astype(bool)

    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst, axis=1)

    print("\n  residuals (metres):")
    for p, e, ok in zip(points, err, mask):
        flag = "" if ok else "   <-- OUTLIER, check this click"
        print(f"    {p[2]:<34} {e:6.2f} m{flag}")
    print(f"  median {np.median(err):.2f} m   max {np.max(err):.2f} m")
    if np.median(err) > 3.0:
        print("  WARNING: median residual > 3 m. Points are probably mislabelled -")
        print("           check you did not swap FAR and NEAR touchlines.")

    # Spread across the pitch WIDTH is the weak axis with this camera: the near
    # touchline is hidden under the camera, so points tend to pile onto the far
    # touchline. Points along a single line cannot determine a homography, and
    # a narrow band determines it only near that band -- everything else is
    # extrapolation, which degrades fast.
    y_used = dst[:, 1]
    y_span = float(y_used.max() - y_used.min())
    x_span = float(dst[:, 0].max() - dst[:, 0].min())
    print(f"\n  coverage: x span {x_span:.0f} m of {PITCH_LENGTH:.0f}, "
          f"y span {y_span:.0f} m of {PITCH_WIDTH:.0f}")
    print(f"  distinct y values: {sorted(set(np.round(y_used, 1)))}")
    if len(set(np.round(y_used, 1))) < 2:
        print("  ERROR: every point is on one line across the pitch. A homography")
        print("         cannot be solved from collinear points - the result is")
        print("         meaningless. You need points at a second distance from")
        print("         the far touchline (goalpost bases are the easiest).")
    elif y_span < 20:
        print(f"  WARNING: only {y_span:.0f} m of width covered. Positions near the")
        print("           near touchline will be extrapolated and may be off by")
        print("           many metres. Add goalpost bases or any near-side point.")
    if x_span < 40:
        print(f"  WARNING: only {x_span:.0f} m of length covered - same problem along"
              " the pitch.")

    CONFIG_DIR.mkdir(exist_ok=True)
    out = CONFIG_DIR / f"{MATCH}.json"
    out.write_text(json.dumps({
        "match": MATCH,
        "created": datetime.now().isoformat(timespec="seconds"),
        "pitch_length_m": PITCH_LENGTH,
        "pitch_width_m": PITCH_WIDTH,
        "mosaic_size": [shape[1], shape[0]],
        "mosaic_to_pitch": H.tolist(),
        "points": [{"mosaic": [p[0], p[1]], "pitch": [p[3], p[4]],
                    "label": p[2], "residual_m": float(e), "inlier": bool(ok)}
                   for p, e, ok in zip(points, err, mask)],
        "median_residual_m": float(np.median(err)),
    }, indent=2))
    print(f"\n  saved {out}")
    print("  now run:  .\\.venv\\Scripts\\python.exe tools\\verify_calibration.py")
    return True


if __name__ == "__main__":
    main()
