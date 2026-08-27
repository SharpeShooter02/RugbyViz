"""Throughput + duplicate-box check at imgsz=1920, across model sizes."""
import cv2, numpy as np, time
from pathlib import Path
from ultralytics import YOLO

FRAMES = sorted(Path("data/derived/frames").glob("t*.jpg"))
IMGS = [cv2.imread(str(f)) for f in FRAMES]
TOTAL_FRAMES = 172673

def iou_mat(b):
    if len(b) < 2: return np.zeros((len(b), len(b)))
    x1 = np.maximum(b[:, None, 0], b[None, :, 0]); y1 = np.maximum(b[:, None, 1], b[None, :, 1])
    x2 = np.minimum(b[:, None, 2], b[None, :, 2]); y2 = np.minimum(b[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (a[:, None] + a[None, :] - inter + 1e-9)

print(f"{'model':<12}{'ms/frame':>10}{'det/frame':>11}{'dup pairs':>11}{'full match':>13}")
print("-" * 57)
for name in ["yolo11m.pt", "yolo11x.pt"]:
    m = YOLO(name)
    m.predict(IMGS[0], imgsz=1920, conf=0.25, classes=[0], verbose=False)  # warmup
    t0 = time.perf_counter(); dets, dups = [], 0
    for img in IMGS:
        r = m.predict(img, imgsz=1920, conf=0.25, classes=[0], verbose=False)[0]
        b = r.boxes.xyxy.cpu().numpy(); dets.append(len(b))
        M = iou_mat(b); np.fill_diagonal(M, 0)
        dups += int((np.triu(M) > 0.3).sum())
    el = (time.perf_counter() - t0) / len(IMGS)
    hrs = TOTAL_FRAMES * el / 3600
    print(f"{name:<12}{el*1000:>10.1f}{np.mean(dets):>11.1f}{dups/len(IMGS):>11.1f}{hrs:>11.1f} h")
