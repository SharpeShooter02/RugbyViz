import cv2, numpy as np, sys, itertools

PAIRS = [("frames/pair_a0.jpg","frames/pair_a1.jpg"),
         ("frames/pair_b0.jpg","frames/pair_b1.jpg"),
         ("frames/pair_c0.jpg","frames/pair_c1.jpg")]

# background band: above the spectator/car line is static scenery
BAND = (0, 430)

sift = cv2.SIFT_create(nfeatures=4000)

def kp(img):
    m = np.zeros(img.shape[:2], np.uint8)
    m[BAND[0]:BAND[1], :] = 255
    return sift.detectAndCompute(img, m)

def resid(model, src, dst, kind):
    if kind == "H":
        proj = cv2.perspectiveTransform(src.reshape(-1,1,2), model).reshape(-1,2)
    else:
        proj = cv2.transform(src.reshape(-1,1,2), model).reshape(-1,2)
    return np.linalg.norm(proj - dst, axis=1)

for pa, pb in PAIRS:
    a = cv2.imread(pa); b = cv2.imread(pb)
    if a is None or b is None:
        print(f"skip {pa}"); continue
    ka, da = kp(a); kb, db = kp(b)
    bf = cv2.BFMatcher()
    raw = bf.knnMatch(da, db, k=2)
    good = [m for m,n in raw if m.distance < 0.75*n.distance]
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    print(f"\n=== {pa} -> {pb} : {len(good)} matches ===")
    if len(good) < 30:
        print("  too few matches"); continue

    H, hm = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    inl = hm.ravel().astype(bool)
    s, d = src[inl], dst[inl]
    print(f"  homography inliers: {inl.sum()}/{len(good)}")

    # translation only (2-DOF)
    t = np.median(d - s, axis=0)
    Mt = np.float32([[1,0,t[0]],[0,1,t[1]]])
    # similarity (4-DOF: rot+uniform scale+trans)
    Ms, _ = cv2.estimateAffinePartial2D(s, d, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    # full affine (6-DOF)
    Ma, _ = cv2.estimateAffine2D(s, d, method=cv2.RANSAC, ransacReprojThreshold=3.0)

    for name, M, kind in [("translation 2dof", Mt, "A"), ("similarity  4dof", Ms, "A"),
                          ("affine      6dof", Ma, "A"), ("homography  8dof", H, "H")]:
        if M is None: continue
        r = resid(M, s, d, kind)
        print(f"  {name}: median {np.median(r):6.2f} px   p90 {np.percentile(r,90):7.2f} px")
    sc = np.sqrt(Ms[0,0]**2 + Ms[0,1]**2) if Ms is not None else float('nan')
    print(f"  similarity scale={sc:.4f}  translation=({t[0]:.1f},{t[1]:.1f})")
