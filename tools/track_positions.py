"""Batch pass over the match: every sampled frame -> player positions in metres.

This is the data layer everything else sits on. Territory, rucks and any later
stat are computed from its output rather than from video.

Registration strategy: each frame is matched DIRECTLY against the temporally
nearest anchor frame (one of the mosaic frames, which already have a known
mosaic homography). No chaining, so no drift accumulates over 80 minutes --
an error in one frame cannot propagate to the next.

    frame --SIFT--> anchor --known--> mosaic --calibration--> metres

Registration runs on a downscaled image for speed, but keypoints are scaled
back to full resolution first, so the homography is full-res throughout.

Run:
    .venv/Scripts/python.exe tools/track_positions.py [--fps 2] [--start S] [--end S]

Resumable: re-running continues from the last checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))

VIDEO = Path("data/video/a-side-vs-msu-2025-09-13.mp4")
MOSAIC_META = Path("data/derived/mosaic/mosaic_meta.json")
CONFIG = Path("config/a-side-vs-msu-2025-09-13.json")
ANCHOR_DIR = Path("data/derived/mosaic_src")
OUT_DIR = Path("data/derived/positions")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "positions.csv"
DET_CSV = OUT_DIR / "detections.csv"     # one row per detected player
ANCHOR_CACHE = OUT_DIR / "anchor_features.npz"

REG_SCALE = 0.5            # register at half resolution
REG_FEATURES = 2000
MIN_INLIERS = 40
# Accept the first anchor immediately only if it registers strongly; otherwise
# try more and keep the best. A weakly-matched anchor can carry a bad mosaic
# homography of its own, and every frame registering against it inherits the
# error -- observed as one anchor (69 inliers vs 357 typical) dragging 35
# seconds of frames sideways until the entire crowd fell inside the pitch.
STRONG_INLIERS = 200
RANSAC_THRESH = 3.0
DET_CONF = 0.25
DET_IMGSZ = 1920
from rugbyviz.pitch import MARGIN_X, MARGIN_Y_FAR, MARGIN_Y_NEAR, on_pitch, feet_of  # noqa: E402
N_ANCHOR_TRIES = 3         # if the nearest anchor fails, try the next nearest
CHECKPOINT_EVERY = 100


def torso_chroma(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """(N,3) lighting-normalised colour of each detection's torso.

    Chromaticity -- each channel over total intensity -- divides out overall
    brightness, so the same jersey in sun and in shadow land in the same place.
    Absolute BGR does not, which is why hand-tuned thresholds on it returned
    83% "dark" against 17% "red" for two teams of fifteen.
    """
    out = []
    H, W = img.shape[:2]
    for x1, y1, x2, y2 in boxes.astype(int):
        h = y2 - y1
        ty1, ty2 = max(0, y1 + int(0.20 * h)), min(H, y1 + int(0.55 * h))
        tx1, tx2 = max(0, x1 + int(0.15 * (x2 - x1))), min(W, x2 - int(0.15 * (x2 - x1)))
        p = img[ty1:max(ty1 + 1, ty2), tx1:max(tx1 + 1, tx2)].reshape(-1, 3).astype(np.float32)
        if len(p) < 4:
            out.append([1 / 3, 1 / 3, 1 / 3]); continue
        med = np.median(p, axis=0)
        out.append(med / max(med.sum(), 1e-6))
    return np.array(out, np.float32)


def kit_colours(img: np.ndarray, boxes: np.ndarray, centres: np.ndarray) -> list[str]:
    """Nearest learned kit centroid for each detection.

    Centroids come from tools/fit_kit_colours.py, which clusters torso colours
    sampled across the whole match. Learned rather than hard-coded, so it
    adapts to the light and works for a different strip next season.

    Crude by design: a clustering aid, not a team classifier. A scrum hides
    most of both jerseys and this will do poorly there, which is exactly the
    situation the ruck/scrum work cares about -- so treat the labels as a hint
    and never as ground truth.
    """
    if not len(boxes):
        return []
    X = torso_chroma(img, boxes)
    d = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    return ["red" if i == 0 else "dark" for i in np.argmin(d, axis=1)]


def anchor_index(meta: dict) -> tuple[list[str], np.ndarray, dict]:
    """Anchor names sorted by their frame number, with their mosaic homographies."""
    homs = meta["homographies"]
    names = sorted(homs, key=lambda n: int("".join(c for c in n if c.isdigit())))
    # mosaic_src frames were sampled every 20 s starting at t=0
    times = np.array([(int("".join(c for c in n if c.isdigit())) - 1) * 20.0 for n in names])
    return names, times, homs


def build_anchor_features(names: list[str]) -> tuple[list, list]:
    if ANCHOR_CACHE.exists():
        z = np.load(ANCHOR_CACHE, allow_pickle=True)
        return list(z["kps"]), list(z["descs"])
    sift = cv2.SIFT_create(nfeatures=REG_FEATURES)
    kps, descs = [], []
    t0 = time.perf_counter()
    for i, n in enumerate(names):
        img = cv2.imread(str(ANCHOR_DIR / n), cv2.IMREAD_GRAYSCALE)
        small = cv2.resize(img, None, fx=REG_SCALE, fy=REG_SCALE)
        k, d = sift.detectAndCompute(small, None)
        # scale keypoints back to full resolution so homographies are full-res
        kps.append(np.float32([p.pt for p in k]) / REG_SCALE if k else np.empty((0, 2), np.float32))
        descs.append(d if d is not None else np.empty((0, 128), np.float32))
        if (i + 1) % 50 == 0:
            print(f"    anchor features {i+1}/{len(names)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    np.savez_compressed(ANCHOR_CACHE,
                        kps=np.array(kps, dtype=object), descs=np.array(descs, dtype=object))
    return kps, descs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=2.0, help="samples per second")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoint")
    args = ap.parse_args()

    meta = json.loads(MOSAIC_META.read_text())
    cfg = json.loads(CONFIG.read_text())
    H_m2p = np.array(cfg["mosaic_to_pitch"])
    L, W = cfg["pitch_length_m"], cfg["pitch_width_m"]
    if "kit_model" not in cfg:
        raise SystemExit("no kit_model in config - run tools/fit_kit_colours.py first")
    km = cfg["kit_model"]["centres"]
    kit_centres = np.float32([km["red"], km["dark"]])
    print(f"      kit centroids: red {km['red']}  dark {km['dark']}")

    names, times, homs = anchor_index(meta)
    print(f"[1/3] {len(names)} anchor frames spanning "
          f"{times.min():.0f}-{times.max():.0f}s")

    print("[2/3] anchor SIFT features")
    a_kps, a_descs = build_anchor_features(names)

    cap = cv2.VideoCapture(str(VIDEO))
    total_s = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    end = min(args.end if args.end else total_s, total_s)
    stamps = np.arange(args.start, end, 1.0 / args.fps)

    done = set()
    if OUT_CSV.exists() and not args.fresh:
        with OUT_CSV.open() as f:
            done = {round(float(r["t"]), 3) for r in csv.DictReader(f)}
        print(f"      resuming: {len(done)} timestamps already processed")
    todo = [t for t in stamps if round(float(t), 3) not in done]
    print(f"[3/3] {len(todo)} samples to process at {args.fps} fps")

    model = YOLO("yolo11x.pt")
    sift = cv2.SIFT_create(nfeatures=REG_FEATURES)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=40))

    new = not OUT_CSV.exists() or args.fresh
    fh = OUT_CSV.open("w" if new else "a", newline="")
    wr = csv.writer(fh)
    if new:
        wr.writerow(["t", "reg_ok", "anchor", "inliers", "reg_err_px",
                     "n_det", "n_on_pitch", "play_x", "play_y", "spread_x", "spread_y"])

    dnew = not DET_CSV.exists() or args.fresh
    dfh = DET_CSV.open("w" if dnew else "a", newline="")
    dwr = csv.writer(dfh)
    if dnew:
        dwr.writerow(["t", "x", "y", "kit", "box_h_px", "conf"])

    t0 = time.perf_counter()
    n_ok = n_fail = 0
    for i, t in enumerate(todo):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue

        small = cv2.resize(frame, None, fx=REG_SCALE, fy=REG_SCALE)
        k, d = sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), None)
        if d is None or len(d) < MIN_INLIERS:
            wr.writerow([f"{t:.3f}", 0, "", 0, "", 0, 0, "", "", "", ""]); n_fail += 1
            continue
        pts = np.float32([p.pt for p in k]) / REG_SCALE

        # try the nearest anchors in time until one registers
        order = np.argsort(np.abs(times - t))[:N_ANCHOR_TRIES]
        H_f2m, used, inl, err = None, "", 0, np.nan
        for ai in order:
            if len(a_descs[ai]) < 2:
                continue
            raw = flann.knnMatch(d, a_descs[ai], k=2)
            good = [m for m, n2 in (p for p in raw if len(p) == 2)
                    if m.distance < 0.75 * n2.distance]
            if len(good) < MIN_INLIERS:
                continue
            src = np.float64([pts[m.queryIdx] for m in good])
            dst = np.float64([a_kps[ai][m.trainIdx] for m in good])
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
            if H is None:
                continue
            m = mask.ravel().astype(bool)
            ninl = int(m.sum())
            if ninl < MIN_INLIERS or ninl <= inl:
                continue
            proj = cv2.perspectiveTransform(src[m].reshape(-1, 1, 2), H).reshape(-1, 2)
            H_f2m = np.array(homs[names[ai]]) @ H
            used, inl = names[ai], ninl
            err = float(np.median(np.linalg.norm(proj - dst[m], axis=1)))
            if inl >= STRONG_INLIERS:
                break       # good enough, skip the remaining anchors

        if H_f2m is None:
            wr.writerow([f"{t:.3f}", 0, "", 0, "", 0, 0, "", "", "", ""]); n_fail += 1
            continue
        n_ok += 1

        r = model.predict(frame, imgsz=DET_IMGSZ, conf=DET_CONF, classes=[0], verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        if len(boxes):
            P = cv2.perspectiveTransform(
                feet_of(boxes).reshape(-1, 1, 2).astype(np.float64),
                H_m2p @ H_f2m).reshape(-1, 2)
            keep = on_pitch(P, L, W)
            Q = P[keep]
            kits = kit_colours(frame, boxes[keep], kit_centres)
            for (px_, py_), kit, bx, cf in zip(Q, kits, boxes[keep], confs[keep]):
                dwr.writerow([f"{t:.3f}", f"{px_:.2f}", f"{py_:.2f}", kit,
                              f"{bx[3]-bx[1]:.0f}", f"{cf:.3f}"])
        else:
            Q = np.empty((0, 2))

        if len(Q) >= 3:
            # Median, not mean: robust to a stray detection on the sideline.
            px, py = float(np.median(Q[:, 0])), float(np.median(Q[:, 1]))
            sx = float(np.percentile(Q[:, 0], 75) - np.percentile(Q[:, 0], 25))
            sy = float(np.percentile(Q[:, 1], 75) - np.percentile(Q[:, 1], 25))
        else:
            px = py = sx = sy = None

        wr.writerow([f"{t:.3f}", 1, used, inl, f"{err:.3f}", len(boxes), len(Q),
                     "" if px is None else f"{px:.2f}", "" if py is None else f"{py:.2f}",
                     "" if sx is None else f"{sx:.2f}", "" if sy is None else f"{sy:.2f}"])

        if (i + 1) % CHECKPOINT_EVERY == 0:
            fh.flush(); dfh.flush()
            el = time.perf_counter() - t0
            rate = (i + 1) / el
            print(f"    {i+1}/{len(todo)}  t={t:7.1f}s  "
                  f"reg ok {100*n_ok/max(1,n_ok+n_fail):.0f}%  "
                  f"{rate:.1f} samples/s  eta {(len(todo)-i-1)/max(rate,1e-6)/60:.0f} min",
                  flush=True)

    fh.close(); dfh.close(); cap.release()
    print(f"\ndone: {n_ok} registered, {n_fail} failed "
          f"({100*n_ok/max(1,n_ok+n_fail):.1f}% ok)")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {DET_CSV}")


if __name__ == "__main__":
    main()
