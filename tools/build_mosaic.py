"""Stitch sampled frames into one static mosaic of the whole pitch.

Strategy: register every frame DIRECTLY to a single reference frame wherever
possible, and only fall back to chaining through a neighbour when direct
registration fails. Direct registration is what keeps drift bounded -- chaining
frame 1->2->3->...->288 would accumulate error with nothing to correct it.

Blending is a per-pixel MEDIAN across all contributing frames. Because players
and spectators move between samples but grass and trees do not, the median
throws the people away and leaves a clean empty pitch. That is exactly what we
want to click on for calibration.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

SRC = Path("data/derived/mosaic_src")
OUT = Path("data/derived/mosaic")
OUT.mkdir(parents=True, exist_ok=True)

MIN_INLIERS = 60        # below this, registration is not trustworthy
RANSAC_THRESH = 3.0
MAX_CANVAS_PX = 120e6   # guard against a runaway canvas

# Geometric sanity limits for an accepted homography. A homography fitted to
# inliers clustered in one small patch of the frame is well determined *there*
# and extrapolates absurdly to the frame corners -- 100x area blow-ups with
# 100+ inliers. RANSAC cannot detect this; it only ever sees the inliers.
MAX_AREA_RATIO = 4.0    # warped frame area vs original
MIN_AREA_RATIO = 0.25
MAX_ASPECT_GROWTH = 3.0  # warped width or height vs original
MIN_SPREAD = 0.25       # inliers must span >= this fraction of frame w and h


def validate_homography(H, inliers_src, w, h):
    """Reject homographies that are geometrically implausible.

    Returns (ok: bool, reason: str).
    """
    corners = np.float64([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    c = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    if not np.all(np.isfinite(c)):
        return False, "non-finite corners"

    # The quad must stay convex and correctly wound -- a homography that folds
    # the frame over itself is nonsense.
    cross = []
    for i in range(4):
        a, b, d = c[i], c[(i + 1) % 4], c[(i + 2) % 4]
        cross.append(np.sign((b - a)[0] * (d - b)[1] - (b - a)[1] * (d - b)[0]))
    if len(set(cross)) != 1:
        return False, "non-convex / folded"

    area = abs(cv2.contourArea(c.astype(np.float32))) / (w * h)
    if not (MIN_AREA_RATIO <= area <= MAX_AREA_RATIO):
        return False, f"area ratio {area:.2f}"

    ww = c[:, 0].max() - c[:, 0].min()
    hh = c[:, 1].max() - c[:, 1].min()
    if ww > MAX_ASPECT_GROWTH * w or hh > MAX_ASPECT_GROWTH * h:
        return False, f"warped size {ww:.0f}x{hh:.0f}"

    # Inliers clustered in a corner cannot constrain the whole frame.
    sx = (inliers_src[:, 0].max() - inliers_src[:, 0].min()) / w
    sy = (inliers_src[:, 1].max() - inliers_src[:, 1].min()) / h
    if sx < MIN_SPREAD or sy < MIN_SPREAD:
        return False, f"inlier spread {sx:.2f}x{sy:.2f}"

    return True, "ok"


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def compute_features(paths: list[Path]) -> tuple[list, list]:
    sift = cv2.SIFT_create(nfeatures=3000)
    kps, descs = [], []
    t0 = time.perf_counter()
    for i, p in enumerate(paths):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        k, d = sift.detectAndCompute(img, None)
        kps.append(np.float64([kp.pt for kp in k]) if k else np.empty((0, 2)))
        descs.append(d if d is not None else np.empty((0, 128), np.float32))
        if (i + 1) % 50 == 0:
            log(f"    features {i+1}/{len(paths)}  ({time.perf_counter()-t0:.0f}s)")
    return kps, descs


def match(d1: np.ndarray, d2: np.ndarray, k1: np.ndarray, k2: np.ndarray):
    """FLANN + Lowe ratio test. Returns matched point pairs."""
    if len(d1) < 2 or len(d2) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    raw = flann.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in raw if len(p) == 2) if m.distance < 0.75 * n.distance]
    if not good:
        return np.empty((0, 2)), np.empty((0, 2))
    return (np.float64([k1[m.queryIdx] for m in good]),
            np.float64([k2[m.trainIdx] for m in good]))


def register(d_from, d_to, k_from, k_to, w, h):
    """Homography mapping `from` frame coords into `to` frame coords."""
    src, dst = match(d_from, d_to, k_from, k_to)
    if len(src) < MIN_INLIERS:
        return None, 0, np.inf, "too few matches"
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
    if H is None:
        return None, 0, np.inf, "no homography"
    mask = mask.ravel().astype(bool)
    n = int(mask.sum())
    if n < MIN_INLIERS:
        return None, n, np.inf, f"only {n} inliers"
    ok, why = validate_homography(H, src[mask], w, h)
    if not ok:
        return None, n, np.inf, why
    proj = cv2.perspectiveTransform(src[mask].reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.median(np.linalg.norm(proj - dst[mask], axis=1)))
    return H, n, err, "ok"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    paths = sorted(SRC.glob("*.jpg"))
    if not paths:
        sys.exit(f"no frames in {SRC}")
    log(f"[1/5] {len(paths)} frames")
    h, w = cv2.imread(str(paths[0])).shape[:2]

    cache = OUT / "features.npz"
    if cache.exists():
        log("[2/5] SIFT features (cached)")
        z = np.load(cache, allow_pickle=True)
        kps, descs = list(z["kps"]), list(z["descs"])
    else:
        log("[2/5] SIFT features")
        kps, descs = compute_features(paths)
        np.savez_compressed(cache,
                            kps=np.array(kps, dtype=object),
                            descs=np.array(descs, dtype=object))

    # Reference selection matters a lot. The temporal midpoint is a poor proxy
    # for "the frame that overlaps everything else" -- a zoomed-in frame has a
    # narrow field of view and registers directly against very few others,
    # forcing the rest through error-compounding chains. Try several candidates
    # on a subsample and keep whichever wins the most direct registrations.
    n = len(paths)
    candidates = list(range(n // 10, n - n // 10, max(1, n // 12)))
    probe = list(range(0, n, 4))
    log(f"[3/5] choosing reference from {len(candidates)} candidates "
        f"(probing {len(probe)} frames each)")
    scores = []
    for c in candidates:
        wins = sum(1 for i in probe
                   if i != c and register(descs[i], descs[c], kps[i], kps[c], w, h)[0] is not None)
        scores.append((wins, c))
        log(f"    candidate {paths[c].name}: {wins}/{len(probe)-1} direct")
    ref = max(scores)[1]
    log(f"    -> reference frame {ref} ({paths[ref].name})")

    H_abs: dict[int, np.ndarray] = {ref: np.eye(3)}
    stats: dict[int, tuple[int, float, str]] = {ref: (0, 0.0, "reference")}
    rejects: dict[str, int] = {}

    # Pass 1 -- direct to reference.
    for i in range(len(paths)):
        if i == ref:
            continue
        H, n, err, why = register(descs[i], descs[ref], kps[i], kps[ref], w, h)
        if H is not None:
            H_abs[i] = H
            stats[i] = (n, err, "direct")
        else:
            key = why.split()[0] if why else "unknown"
            rejects[key] = rejects.get(key, 0) + 1
    log(f"    direct: {len(H_abs)-1}/{len(paths)-1}")
    if rejects:
        log(f"    rejected: {dict(sorted(rejects.items(), key=lambda kv: -kv[1]))}")

    # Pass 2 -- chain the stragglers through their nearest registered neighbour.
    # The COMPOSED homography is validated too: a chain of individually sane
    # steps can still compound into nonsense.
    for _ in range(6):
        missing = [i for i in range(len(paths)) if i not in H_abs]
        if not missing:
            break
        progress = False
        for i in missing:
            anchors = sorted(H_abs, key=lambda j: abs(j - i))
            for j in anchors[:10]:
                H, n, err, why = register(descs[i], descs[j], kps[i], kps[j], w, h)
                if H is None:
                    continue
                composed = H_abs[j] @ H
                corners = np.float64([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
                c = cv2.perspectiveTransform(corners, composed).reshape(-1, 2)
                ok, _ = validate_homography(composed, c, w, h)
                if not ok:
                    continue
                H_abs[i] = composed
                stats[i] = (n, err, f"chained via {j}")
                progress = True
                break
        if not progress:
            break

    unreg = [i for i in range(len(paths)) if i not in H_abs]
    log(f"    registered {len(H_abs)}/{len(paths)}   unregistered: {len(unreg)}")

    # Canvas bounds from projected frame corners.
    corners = np.float64([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    allc = np.vstack([cv2.perspectiveTransform(corners, H_abs[i]).reshape(-1, 2)
                      for i in sorted(H_abs)])
    lo = np.floor(allc.min(0)).astype(int)
    hi = np.ceil(allc.max(0)).astype(int)
    cw, ch = int(hi[0] - lo[0]), int(hi[1] - lo[1])
    log(f"[4/5] canvas {cw} x {ch}  ({cw*ch/1e6:.1f} MP)")
    if cw * ch > MAX_CANVAS_PX:
        sys.exit(f"canvas too large ({cw*ch/1e6:.0f} MP) -- registration likely diverged")

    offset = np.array([[1, 0, -lo[0]], [0, 1, -lo[1]], [0, 0, 1]], np.float64)

    # Median blend, in horizontal strips to bound memory.
    log("[5/5] median blend")
    STRIP = 256
    mosaic = np.zeros((ch, cw, 3), np.uint8)
    coverage = np.zeros((ch, cw), np.uint16)
    order = sorted(H_abs)

    warped_cache = []
    for i in order:
        img = cv2.imread(str(paths[i]))
        W = cv2.warpPerspective(img, offset @ H_abs[i], (cw, ch),
                                flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        M = cv2.warpPerspective(np.full((h, w), 255, np.uint8), offset @ H_abs[i],
                                (cw, ch), flags=cv2.INTER_NEAREST)
        warped_cache.append((W, M))
        coverage += (M > 0).astype(np.uint16)

    for y0 in range(0, ch, STRIP):
        y1 = min(ch, y0 + STRIP)
        stack, masks = [], []
        for W, M in warped_cache:
            sub = M[y0:y1]
            if not sub.any():
                continue
            stack.append(W[y0:y1].astype(np.float32))
            masks.append(sub > 0)
        if not stack:
            continue
        arr = np.stack(stack)
        msk = np.stack(masks)[..., None]
        arr = np.where(msk, arr, np.nan)
        with np.errstate(all="ignore"):
            med = np.nanmedian(arr, axis=0)
        mosaic[y0:y1] = np.nan_to_num(med).astype(np.uint8)
        log(f"    rows {y0}-{y1}")

    cv2.imwrite(str(OUT / "mosaic.jpg"), mosaic, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(OUT / "coverage.png"),
                cv2.applyColorMap(
                    (255 * coverage / max(1, coverage.max())).astype(np.uint8),
                    cv2.COLORMAP_VIRIDIS))

    meta = {
        "reference_frame": paths[ref].name,
        "canvas": [cw, ch],
        "offset": offset.tolist(),
        "n_frames": len(paths),
        "n_registered": len(H_abs),
        "unregistered": [paths[i].name for i in unreg],
        "homographies": {paths[i].name: (offset @ H_abs[i]).tolist() for i in sorted(H_abs)},
    }
    (OUT / "mosaic_meta.json").write_text(json.dumps(meta, indent=2))

    errs = [e for _, e, _ in stats.values() if np.isfinite(e) and e > 0]
    chained = sum(1 for _, _, how in stats.values() if how.startswith("chained"))
    log("")
    log(f"  registered   : {len(H_abs)}/{len(paths)}")
    log(f"  direct       : {len(H_abs) - chained - 1}")
    log(f"  chained      : {chained}")
    log(f"  median resid : {np.median(errs):.3f} px")
    log(f"  worst resid  : {np.max(errs):.3f} px")
    log(f"  mosaic       : {OUT/'mosaic.jpg'}  ({cw} x {ch})")


if __name__ == "__main__":
    main()
