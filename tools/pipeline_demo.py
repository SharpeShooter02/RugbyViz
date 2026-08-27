"""End-to-end: video frame -> detections -> pitch metres -> crowd filter -> bird's-eye.

Composes the two homographies we have built:

    frame pixels --(H_frame_to_mosaic)--> mosaic pixels --(H_mosaic_to_pitch)--> metres

Then rejects any detection whose FEET land outside the playing area. That is the
crowd filter: spectators stand beyond the far touchline, so they fall out
geometrically without any appearance model.

Run:
    .venv/Scripts/python.exe tools/pipeline_demo.py [n_frames]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MOSAIC_META = Path("data/derived/mosaic/mosaic_meta.json")
CONFIG = Path("config/a-side-vs-msu-2025-09-13.json")
FRAME_DIR = Path("data/derived/mosaic_src")
OUT = Path("data/derived/pipeline")
OUT.mkdir(parents=True, exist_ok=True)

CONF = 0.25
IMGSZ = 1920
# In-goal areas extend past the try lines, so x gets a generous margin.
#
# y margins are ASYMMETRIC and that is deliberate. The far touchline resolves
# at ~1.6 px/m, so a player standing on it can measure a metre or two either
# side -- but the spectators are only 3-5 m beyond it, which is under 8 pixels.
# A loose far margin therefore swallows the entire crowd. The near touchline
# resolves at ~190 px/m, where a few metres of margin costs nothing.
MARGIN_X = 12.0
# 2.0 chosen from the histogram in sweep(): the spectator line forms a dense
# band from -8 m to -2 m peaking at -5 m, then falls off a cliff. Cutting at
# -2 m removes it while keeping the handful of detections that sit just off
# the touchline, which at 1.6 px/m are consistent with real players.
MARGIN_Y_FAR = 2.0     # beyond the far touchline (y < 0)
MARGIN_Y_NEAR = 4.0    # beyond the near touchline (y > W)
EXPECTED_PLAYERS = 30          # 15 a side
PPM = 9                        # bird's-eye pixels per metre


def load() -> tuple[dict, np.ndarray, float, float]:
    meta = json.loads(MOSAIC_META.read_text())
    cfg = json.loads(CONFIG.read_text())
    return meta, np.array(cfg["mosaic_to_pitch"]), cfg["pitch_length_m"], cfg["pitch_width_m"]


def feet_of(boxes: np.ndarray) -> np.ndarray:
    """Bottom-centre of each box - where the player meets the ground.

    The homography maps the GROUND PLANE, so the point we transform has to be
    on the ground. Using the box centre would place everyone several metres
    further away than they are.
    """
    return np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], axis=1)


def kit_colour(img: np.ndarray, box: np.ndarray) -> str:
    """Crude red-vs-dark split on the torso. Only a sanity signal: if the
    filter works we expect roughly even numbers of the two kits."""
    x1, y1, x2, y2 = box.astype(int)
    h = y2 - y1
    ty1, ty2 = y1 + int(0.2 * h), y1 + int(0.55 * h)
    patch = img[max(0, ty1):max(ty1 + 1, ty2), max(0, x1):max(x1 + 1, x2)]
    if patch.size == 0:
        return "?"
    b, g, r = patch.reshape(-1, 3).mean(axis=0)
    if r > g * 1.25 and r > b * 1.25 and r > 60:
        return "red"
    return "dark"


def main() -> None:
    n_frames = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    meta, H_m2p, L, W = load()
    homs = meta["homographies"]
    names = sorted(homs)
    picks = [names[int(i)] for i in np.linspace(0, len(names) - 1, n_frames)]

    model = YOLO("yolo11x.pt")
    rows = []
    all_on, all_off = [], []
    raw_pitch = []

    print(f"pitch {L:.0f} x {W:.0f} m   margins x{MARGIN_X:.0f} far{MARGIN_Y_FAR:.0f} near{MARGIN_Y_NEAR:.0f}   "
          f"{len(names)} registered frames available\n")
    hdr = (f"{'frame':<12}{'detected':>9}{'on pitch':>10}{'rejected':>10}"
           f"{'red':>6}{'dark':>6}{'x span':>9}{'y span':>9}")
    print(hdr); print("-" * len(hdr))

    for name in picks:
        img = cv2.imread(str(FRAME_DIR / name))
        if img is None:
            continue
        H_f2m = np.array(homs[name])
        H_f2p = H_m2p @ H_f2m           # frame pixels -> pitch metres, one matrix

        r = model.predict(img, imgsz=IMGSZ, conf=CONF, classes=[0], verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        if not len(boxes):
            continue
        pitch = cv2.perspectiveTransform(
            feet_of(boxes).reshape(-1, 1, 2).astype(np.float64), H_f2p).reshape(-1, 2)

        on = ((pitch[:, 0] > -MARGIN_X) & (pitch[:, 0] < L + MARGIN_X) &
              (pitch[:, 1] > -MARGIN_Y_FAR) & (pitch[:, 1] < W + MARGIN_Y_NEAR))
        raw_pitch.append(pitch)
        kits = [kit_colour(img, b) for b in boxes[on]]
        nred = kits.count("red")
        p_on = pitch[on]
        xs = f"{p_on[:,0].min():.0f}-{p_on[:,0].max():.0f}" if len(p_on) else "-"
        ys = f"{p_on[:,1].min():.0f}-{p_on[:,1].max():.0f}" if len(p_on) else "-"
        print(f"{name:<12}{len(boxes):>9}{int(on.sum()):>10}{int((~on).sum()):>10}"
              f"{nred:>6}{len(kits)-nred:>6}{xs:>9}{ys:>9}")
        rows.append((name, len(boxes), int(on.sum()), int((~on).sum()), nred, len(kits) - nred))
        all_on.append(p_on); all_off.append(pitch[~on])

        vis = img.copy()
        for b, keep, pt in zip(boxes, on, pitch):
            c = (0, 220, 0) if keep else (0, 0, 235)
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 2)
            if keep:
                cv2.putText(vis, f"{pt[0]:.0f},{pt[1]:.0f}", (int(b[0]), int(b[1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
        cv2.putText(vis, f"{name}   green=on pitch ({int(on.sum())})   "
                         f"red=rejected ({int((~on).sum())})", (18, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
        cv2.imwrite(str(OUT / f"{Path(name).stem}_filtered.jpg"), vis,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])

        # bird's-eye for this frame
        bev = draw_pitch(L, W)
        for pt, kit in zip(p_on, kits):
            px, py = int((pt[0] + MARGIN_X) * PPM), int((pt[1] + MARGIN_Y_NEAR) * PPM)
            col = (40, 40, 235) if kit == "red" else (60, 60, 60)
            cv2.circle(bev, (px, py), 7, col, -1)
            cv2.circle(bev, (px, py), 7, (255, 255, 255), 1)
        cv2.putText(bev, f"{name}  {len(p_on)} players", (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(OUT / f"{Path(name).stem}_birdseye.jpg"), bev)

    summarise(rows, all_on, all_off, L, W)
    sweep(raw_pitch, L, W)


def draw_pitch(L: float, W: float) -> np.ndarray:
    w = int((L + 2 * MARGIN_X) * PPM)
    h = int((W + 2 * MARGIN_Y_NEAR) * PPM)
    img = np.full((h, w, 3), (35, 95, 45), np.uint8)
    ox, oy = int(MARGIN_X * PPM), int(MARGIN_Y_NEAR * PPM)

    def line(x1, y1, x2, y2, c=(235, 235, 235), t=2):
        cv2.line(img, (ox + int(x1 * PPM), oy + int(y1 * PPM)),
                 (ox + int(x2 * PPM), oy + int(y2 * PPM)), c, t)

    line(0, 0, L, 0); line(0, W, L, W)
    line(0, 0, 0, W, (255, 255, 255), 3); line(L, 0, L, W, (255, 255, 255), 3)
    line(L / 2, 0, L / 2, W)
    for x in (22, L - 22):
        line(x, 0, x, W)
    for x in (L / 2 - 10, L / 2 + 10):
        line(x, 0, x, W, (170, 200, 170), 1)
    return img


def summarise(rows, all_on, all_off, L, W) -> None:
    if not rows:
        print("\nno frames processed")
        return
    det = np.array([r[1] for r in rows]); on = np.array([r[2] for r in rows])
    off = np.array([r[3] for r in rows])
    red = np.array([r[4] for r in rows]); dark = np.array([r[5] for r in rows])
    P = np.vstack([a for a in all_on if len(a)])

    print("\n" + "=" * 64)
    print("WHAT THIS TELLS US")
    print("=" * 64)
    print(f"\n1. CROWD FILTER")
    print(f"   detections per frame        : {det.mean():.1f}")
    print(f"   kept (feet inside pitch)    : {on.mean():.1f}")
    print(f"   rejected (outside)          : {off.mean():.1f}  "
          f"= {100*off.sum()/det.sum():.0f}% of all detections")
    print(f"   -> the rejected ones are almost entirely spectators along the far")
    print(f"      touchline. No appearance model was used: they fail purely on")
    print(f"      where their feet land in metres.")

    print(f"\n2. IS THE COUNT RIGHT?")
    print(f"   kept per frame: min {on.min()}, median {np.median(on):.0f}, max {on.max()}")
    print(f"   Note this is an UPPER bound check, not an exact one: Veo's")
    print(f"   auto-follow crop shows only part of the pitch, so a frame")
    print(f"   normally contains FEWER than the {EXPECTED_PLAYERS} players on the field.")
    lo, hi = 12, EXPECTED_PLAYERS + 5
    verdict = "PLAUSIBLE" if lo <= np.median(on) <= hi else "SUSPICIOUS"
    print(f"   verdict: {verdict}  (plausible band {lo}-{hi})")
    if np.median(on) < lo:
        print("   too few: either detection is missing players, or the pitch")
        print("   bounds are too tight and are eating real ones.")
    elif np.median(on) > hi:
        print("   too many: more people than can be on the field, so crowd or")
        print("   substitutes are still getting through the margin.")

    print(f"\n3. ARE THE POSITIONS PLAUSIBLE?")
    print(f"   x (along pitch) : {P[:,0].min():6.1f} to {P[:,0].max():6.1f} m")
    print(f"   y (across)      : {P[:,1].min():6.1f} to {P[:,1].max():6.1f} m")
    print(f"   median y        : {np.median(P[:,1]):.1f} m  (pitch centre is {W/2:.0f})")
    frac_far = float((P[:, 1] < W * 0.25).mean())
    print(f"   fraction in the far quarter: {100*frac_far:.0f}%")
    print(f"   -> a real match spreads players across the width. If nearly all")
    print(f"      sit in one narrow band, the calibration is squashing them and")
    print(f"      the y axis is not trustworthy.")

    print(f"\n4. TEAM SPLIT  (crude red-vs-dark kit test)")
    print(f"   red {red.mean():.1f} / dark {dark.mean():.1f} per frame")
    bal = abs(red.sum() - dark.sum()) / max(1, red.sum() + dark.sum())
    print(f"   imbalance: {100*bal:.0f}%")
    print(f"   -> two teams of 15 should give a roughly even split. A large skew")
    print(f"      means either the colour test is wrong or one team's detections")
    print(f"      are being rejected.")

    print(f"\n5. WHAT THIS DOES NOT TELL US")
    print(f"   - nothing about tracking: these are independent per-frame")
    print(f"     positions with no identity linking them across time")
    print(f"   - nothing about rucks: that needs the positions over time")
    print(f"   - accuracy is unverified. Positions are self-consistent, but no")
    print(f"     ground truth has been measured against them.")
    print(f"\n   images: {OUT}")




def sweep(raw_pitch, L, W) -> None:
    """How many detections survive at each far-side margin?

    The far touchline is the only ambiguous boundary: spectators sit a few
    metres beyond it, and at 1.6 px/m that gap is under 8 pixels. This shows
    where tightening stops removing crowd and starts removing players.
    """
    if not raw_pitch:
        return
    P = np.vstack(raw_pitch)
    n = len(raw_pitch)
    print("\n" + "=" * 64)
    print("CHOOSING THE FAR-SIDE MARGIN")
    print("=" * 64)
    print(f"\n  {'margin':>8}{'kept/frame':>13}{'vs 30 players':>16}")
    for m in (6.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0):
        keep = ((P[:, 0] > -MARGIN_X) & (P[:, 0] < L + MARGIN_X) &
                (P[:, 1] > -m) & (P[:, 1] < W + MARGIN_Y_NEAR))
        per = keep.sum() / n
        print(f"  {m:>7.0f}m{per:>13.1f}{per - EXPECTED_PLAYERS:>+15.1f}")
    print("\n  Detections by distance beyond the far touchline:")
    for lo, hi in ((-10, -8), (-8, -6), (-6, -4), (-4, -2), (-2, 0), (0, 2), (2, 4), (4, 8)):
        c = int(((P[:, 1] >= lo) & (P[:, 1] < hi)).sum())
        print(f"    y {lo:>3} to {hi:>3} m : {c:>4}  {'#' * min(60, c)}")
    print("\n  A dense band at negative y that stops abruptly near 0 is the")
    print("  crowd. Players thin out gradually instead, so the cliff edge is")
    print("  where the touchline really is.")


if __name__ == "__main__":
    main()
