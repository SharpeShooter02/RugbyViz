"""Learn this match's two kit colours from the footage instead of hard-coding them.

Hand-tuned HSV thresholds failed badly here: 83% of detections came back "dark"
against 17% "red" when two teams of 15 should be roughly even. Red jerseys in
shadow have low saturation and low value, so any fixed cut-off either swallows
them into "dark" or lets grass through as red. Lighting changes across a
96-minute afternoon match, so one threshold cannot serve the whole file.

Instead: sample torso patches from across the match, describe each in a
lighting-normalised colour space, and cluster. The two dominant clusters ARE the
two kits, whatever colour they happen to be, and the same code works next season
in different strip.

Feature is chromaticity -- each channel over total intensity -- which divides out
overall brightness, so a red jersey in shadow and the same jersey in sun land in
the same place. Absolute BGR does not.

Writes kit_model into config/<match>.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rugbyviz.pitch import feet_of, on_pitch  # noqa: E402

VIDEO = Path("data/video/a-side-vs-msu-2025-09-13.mp4")
CONFIG = Path("config/a-side-vs-msu-2025-09-13.json")
MOSAIC_META = Path("data/derived/mosaic/mosaic_meta.json")
N_FRAMES = 40
MIN_BOX_H = 28          # tiny boxes are mostly background


def torso_chroma(img: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """(N,3) lighting-normalised colour of each detection's torso."""
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


def main() -> None:
    cfg = json.loads(CONFIG.read_text())
    meta = json.loads(MOSAIC_META.read_text())
    H_m2p = np.array(cfg["mosaic_to_pitch"])
    L, W = cfg["pitch_length_m"], cfg["pitch_width_m"]
    homs = meta["homographies"]
    names = sorted(homs, key=lambda n: int("".join(c for c in n if c.isdigit())))
    picks = [names[int(i)] for i in np.linspace(0, len(names) - 1, N_FRAMES)]

    model = YOLO("yolo11x.pt")
    feats, heights = [], []
    for name in picks:
        img = cv2.imread(str(Path("data/derived/mosaic_src") / name))
        if img is None:
            continue
        r = model.predict(img, imgsz=1920, conf=0.3, classes=[0], verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy()
        if not len(b):
            continue
        P = cv2.perspectiveTransform(feet_of(b).reshape(-1, 1, 2).astype(np.float64),
                                     H_m2p @ np.array(homs[name])).reshape(-1, 2)
        b = b[on_pitch(P, L, W)]
        b = b[(b[:, 3] - b[:, 1]) >= MIN_BOX_H]
        if len(b):
            feats.append(torso_chroma(img, b))
            heights.append(b[:, 3] - b[:, 1])
    X = np.vstack(feats)
    print(f"sampled {len(X)} torsos from {len(feats)} frames")

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-5)
    best = None
    for k in (2, 3):
        compact, labels, centres = cv2.kmeans(X, k, None, crit, 12,
                                              cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=k)
        print(f"\nk={k}  compactness {compact:.4f}")
        for i, c in enumerate(centres):
            print(f"   cluster {i}: {100*counts[i]/len(X):5.1f}%  "
                  f"chroma B={c[0]:.3f} G={c[1]:.3f} R={c[2]:.3f}")
        if k == 2:
            best = centres

    # Name the clusters by which is reddest. Everything else is "other".
    order = np.argsort([-c[2] for c in best])       # most red first
    named = {"red": best[order[0]].tolist(), "dark": best[order[1]].tolist()}
    lab = np.argmin(((X[:, None, :] - best[None, :, :]) ** 2).sum(-1), axis=1)
    share = [float((lab == order[0]).mean()), float((lab == order[1]).mean())]
    print(f"\nassigned: red {100*share[0]:.1f}%   dark {100*share[1]:.1f}%")
    print("two teams of 15 should be roughly even; a large skew means the two")
    print("kits are not separable by colour alone in this footage")

    cfg["kit_model"] = {
        "space": "median BGR / sum (chromaticity)",
        "centres": named,
        "n_samples": int(len(X)),
        "balance": {"red": round(share[0], 3), "dark": round(share[1], 3)},
    }
    CONFIG.write_text(json.dumps(cfg, indent=2))
    print(f"\nwrote kit_model into {CONFIG}")


if __name__ == "__main__":
    main()
