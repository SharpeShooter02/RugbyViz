"""Detection spike: can an off-the-shelf detector see far-touchline players?

Compares three inference strategies on the same frames:
  1. full frame @ 640px  (ultralytics default -- downscales 1920 -> 640)
  2. full frame @ 1920px (native resolution)
  3. tiled 3x2 @ 640px   (each tile upscaled, then merged with NMS)

Reports person counts split by how far up the frame the feet are, since
distance from camera maps to vertical position.
"""
import cv2, numpy as np, sys
from pathlib import Path
from ultralytics import YOLO

FRAMES = sorted(Path("data/derived/frames").glob("t*.jpg"))
OUT = Path("data/derived/detect_spike"); OUT.mkdir(parents=True, exist_ok=True)
MODEL = sys.argv[1] if len(sys.argv) > 1 else "yolo11x.pt"
CONF = 0.25
FAR_Y = 620          # feet above this line == far half of the pitch

model = YOLO(MODEL)

def nms(boxes, scores, thr=0.55):
    if not len(boxes): return []
    idx = cv2.dnn.NMSBoxes(
        [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes],
        [float(s) for s in scores], CONF, thr)
    return np.array(idx).flatten().tolist()

def full(img, imgsz):
    r = model.predict(img, imgsz=imgsz, conf=CONF, classes=[0], verbose=False)[0]
    b = r.boxes.xyxy.cpu().numpy()
    s = r.boxes.conf.cpu().numpy()
    return b, s

def tiled(img, nx=3, ny=2, overlap=0.2, imgsz=640):
    H, W = img.shape[:2]
    tw, th = int(W / nx), int(H / ny)
    ox, oy = int(tw * overlap), int(th * overlap)
    boxes, scores = [], []
    for iy in range(ny):
        for ix in range(nx):
            x0 = max(0, ix * tw - ox); y0 = max(0, iy * th - oy)
            x1 = min(W, (ix + 1) * tw + ox); y1 = min(H, (iy + 1) * th + oy)
            crop = img[y0:y1, x0:x1]
            r = model.predict(crop, imgsz=imgsz, conf=CONF, classes=[0], verbose=False)[0]
            for (bx1, by1, bx2, by2), sc in zip(r.boxes.xyxy.cpu().numpy(),
                                                r.boxes.conf.cpu().numpy()):
                boxes.append([bx1 + x0, by1 + y0, bx2 + x0, by2 + y0]); scores.append(sc)
    boxes, scores = np.array(boxes), np.array(scores)
    keep = nms(boxes, scores)
    return (boxes[keep], scores[keep]) if len(keep) else (np.empty((0, 4)), np.empty(0))

def stats(b):
    if not len(b): return 0, 0, 0, 0.0
    feet = b[:, 3]; h = b[:, 3] - b[:, 1]
    far = int((feet < FAR_Y).sum())
    return len(b), far, len(b) - far, float(np.median(h[feet < FAR_Y])) if far else 0.0

print(f"model={MODEL} conf={CONF} far/near split at y={FAR_Y}\n")
hdr = f"{'frame':<12}{'strategy':<18}{'total':>6}{'far':>6}{'near':>6}{'medH_far':>10}"
print(hdr); print("-" * len(hdr))
agg = {}
for f in FRAMES:
    img = cv2.imread(str(f))
    for name, fn in [("full@640", lambda i: full(i, 640)),
                     ("full@1920", lambda i: full(i, 1920)),
                     ("tiled3x2@640", tiled)]:
        b, s = fn(img)
        t, far, near, mh = stats(b)
        agg.setdefault(name, []).append((t, far))
        print(f"{f.stem:<12}{name:<18}{t:>6}{far:>6}{near:>6}{mh:>10.1f}")
        vis = img.copy()
        for (x1, y1, x2, y2) in b.astype(int):
            c = (0, 165, 255) if y2 < FAR_Y else (0, 255, 0)
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        cv2.line(vis, (0, FAR_Y), (img.shape[1], FAR_Y), (255, 0, 255), 1)
        cv2.putText(vis, f"{name}  total={t} far={far}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        cv2.imwrite(str(OUT / f"{f.stem}_{name.replace('@','_')}.jpg"), vis)
    print()

print("=== totals across all frames ===")
for name, v in agg.items():
    tot = sum(t for t, _ in v); far = sum(f for _, f in v)
    print(f"  {name:<16} total={tot:<5} far={far:<5} ({100*far/max(tot,1):.0f}% far)")
