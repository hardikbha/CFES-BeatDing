"""Per-image face attributes for the multi-axis evaluation layer (CPU only; no scores are read here).

For every image: InsightFace buffalo_l SCRFD detection on a zero-PADDED copy of the crop (tight crops need padding,
not upscaling), the 5 landmarks, buffalo_l genderage (numeric apparent age + gender; the same estimator that produced
FairForge-2's `if_age`), and the individual typology angle (ITA) of face-skin pixels:

  ROI   landmark mode (face detected): two cheek squares (centre = eye + 0.60 IOD down the face axis, half-size
        0.18 IOD) + a forehead box (centre = eye midpoint - 0.55 IOD, half-size 0.35 x 0.12 IOD). Eyes, brows, nose
        tip, mouth and beard area are outside the ROI by construction.
        template mode (no detection): the face-template ellipse interior (cx .50, cy .48, ax .30, ay .38) minus the
        eye band (y in [.30, .50]) and the mouth box (y in [.62, .84], |x - .5| < .20).
  skin  ROI pixels inside a YCrCb skin-colour box (Cr 130-180, Cb 75-135), then the 10-90 % L* trimmed set. If fewer
        than 60 pixels survive the colour box, the whole ROI is used (flag skin_fallback = 1).
  light gray-world white balance estimated on the CONTEXT pixels of the crop (everything outside the face box
        expanded x1.1, or outside the ellipse x1.15 in template mode; channels in (8, 250)); fallback = whole crop
        when the context has < 10 % of the pixels. Gains are computed in linear RGB, clipped to [0.8, 1.25] and
        rescaled so the context luminance Y is unchanged (chromatic correction only).
  ITA   = atan((L* - 50) / b*) in degrees from the medians of L* and b* over the skin set (sRGB D65 -> CIELAB,
        implemented here, not via OpenCV). Three values are stored: ita_raw (no white balance), ita_gw (PRIMARY,
        declared before any score is read) and ita_gwexp (sensitivity: the gray-world gains are additionally scaled
        so the context mean luminance Y = 0.18, scale clipped to [0.5, 2.0], i.e. an exposure normalisation that
        assumes an average-grey surround). Invalid (NaN): b* <= 0, median L* < 15 (too dark), or a grayscale
        image (median chroma of the whole crop < 3).

  ~/.conda/envs/aiface/bin/python face_attrs.py --list paths.txt --out out.csv.gz --procs 16
"""
import argparse
import multiprocessing as mp
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

cv2.setNumThreads(1)
MODEL_DIR = os.path.expanduser("~/.insightface/models/buffalo_l")
_DET = _GA = None


def _init():
    global _DET, _GA
    import onnxruntime as ort
    from insightface.model_zoo.attribute import Attribute
    from insightface.model_zoo.retinaface import RetinaFace
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    pv = ["CPUExecutionProvider"]
    _DET = RetinaFace(f"{MODEL_DIR}/det_10g.onnx", ort.InferenceSession(f"{MODEL_DIR}/det_10g.onnx", so, providers=pv))
    _DET.prepare(-1, input_size=(320, 320), det_thresh=0.3)
    _GA = Attribute(f"{MODEL_DIR}/genderage.onnx", ort.InferenceSession(f"{MODEL_DIR}/genderage.onnx", so, providers=pv))
    _GA.prepare(-1)


# ------------------------------------------------------------------------------------------------ colour
def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


LUT = srgb_to_lin(np.arange(256) / 255.0).astype(np.float32)
_GRID = {}


def grid(H, W):
    if (H, W) not in _GRID:
        _GRID[(H, W)] = np.mgrid[0:H, 0:W].astype(np.float32)
    return _GRID[(H, W)]


M_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
WHITE = np.array([0.95047, 1.0, 1.08883])


def lin_to_lab(lin):
    xyz = lin @ M_XYZ.T / WHITE
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    return L, a, b


def ita_from(lin_pix):
    L, _, b = lin_to_lab(np.clip(lin_pix, 0, 1))
    lo, hi = np.percentile(L, [10, 90])
    k = (L >= lo) & (L <= hi)
    if k.sum() < 20:
        k = np.ones_like(L, bool)
    Lm, bm = float(np.median(L[k])), float(np.median(b[k]))
    ita = float(np.degrees(np.arctan((Lm - 50) / bm))) if bm > 0 and Lm >= 15 else float("nan")
    return ita, Lm, bm


# ------------------------------------------------------------------------------------------------ ROI
def roi_masks(H, W, kps):
    yy, xx = grid(H, W)
    if kps is not None:
        le, re = kps[0], kps[1]
        iod = float(np.linalg.norm(re - le))
        u = (re - le) / max(iod, 1e-6)
        v = np.array([-u[1], u[0]])                     # down the face (image y grows downwards)
        roi = np.zeros((H, W), bool)

        def box(c, hu, hv):
            du, dv = (xx - c[0]) * u[0] + (yy - c[1]) * u[1], (xx - c[0]) * v[0] + (yy - c[1]) * v[1]
            return (np.abs(du) <= hu) & (np.abs(dv) <= hv)
        for e in (le, re):
            roi |= box(e + 0.60 * iod * v, 0.18 * iod, 0.18 * iod)
        roi |= box((le + re) / 2 - 0.55 * iod * v, 0.35 * iod, 0.12 * iod)
        return roi, iod
    cx, cy, ax, ay = 0.50 * W, 0.48 * H, 0.30 * W, 0.38 * H
    ell = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 <= 1
    eye = (yy >= 0.30 * H) & (yy <= 0.50 * H)
    mouth = (yy >= 0.62 * H) & (yy <= 0.84 * H) & (np.abs(xx - 0.5 * W) < 0.20 * W)
    return ell & ~eye & ~mouth, float("nan")


def context_mask(H, W, bbox):
    yy, xx = grid(H, W)
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cx, cy, hw, hh = (x0 + x1) / 2, (y0 + y1) / 2, 0.55 * (x1 - x0), 0.55 * (y1 - y0)
        return ~((np.abs(xx - cx) <= hw) & (np.abs(yy - cy) <= hh))
    cx, cy, ax, ay = 0.50 * W, 0.48 * H, 0.30 * W * 1.15, 0.38 * H * 1.15
    return ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 > 1


# ------------------------------------------------------------------------------------------------ main per image
def process(path):
    out = {"path": path}
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        out["error"] = "unreadable"
        return out
    H, W = img.shape[:2]
    # detection on a zero-padded copy
    p = int(0.25 * max(H, W))
    pad = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    bboxes, kpss = _DET.detect(pad, input_size=(320, 320), max_num=0)
    bbox = kps = None
    if len(bboxes):
        c = np.array([W / 2 + p, H / 2 + p])
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        ctr = np.stack([(bboxes[:, 0] + bboxes[:, 2]) / 2, (bboxes[:, 1] + bboxes[:, 3]) / 2], 1)
        sc = area / area.max() - np.linalg.norm(ctr - c, axis=1) / max(H, W)
        j = int(np.argmax(sc))
        bb, kp = bboxes[j], kpss[j]

        class F(dict):
            __getattr__ = dict.get
        face = F(bbox=bb[:4], kps=kp, det_score=float(bb[4]))
        g, a = _GA.get(pad, face)
        out.update(det_score=float(bb[4]), if_age=int(a), if_male=int(g))
        bbox = bb[:4] - np.array([p, p, p, p])
        kps = kp - np.array([p, p])
        out.update(bx0=bbox[0], by0=bbox[1], bx1=bbox[2], by1=bbox[3])
    lin = LUT[img[:, :, ::-1]]
    # grayscale check
    flat = lin.reshape(-1, 3)
    sub = flat[:: max(1, len(flat) // 4000)]
    _, a_, b_ = lin_to_lab(sub)
    out["gray"] = int(np.median(np.hypot(a_, b_)) < 3)
    roi, iod = roi_masks(H, W, kps)
    out["roi_mode"] = "kps" if kps is not None else "template"
    out["iod"] = iod
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    skin = roi & (ycc[:, :, 1] >= 130) & (ycc[:, :, 1] <= 180) & (ycc[:, :, 2] >= 75) & (ycc[:, :, 2] <= 135)
    out["n_roi"], out["n_skin"] = int(roi.sum()), int(skin.sum())
    out["skin_fallback"] = int(skin.sum() < 60)
    sel = skin if skin.sum() >= 60 else roi
    if sel.sum() < 20:
        out["error"] = "empty_roi"
        return out
    # gray-world gains on the context
    ctx = context_mask(H, W, bbox)
    ok = ctx & (img.max(2) < 250) & (img.min(2) > 8)
    out["ctx_frac"] = float(ok.mean())
    src = lin[ok] if ok.mean() >= 0.10 else lin[(img.max(2) < 250) & (img.min(2) > 8)]
    if len(src) < 50:
        src = flat
    m = src.mean(0) + 1e-6
    gains = np.clip(m.mean() / m, 0.8, 1.25)
    Y = np.array([0.2126729, 0.7151522, 0.0721750])
    gains *= (Y @ m) / (Y @ (gains * m))
    out.update(gw_r=gains[0], gw_g=gains[1], gw_b=gains[2])
    expo = float(np.clip(0.18 / max(float(Y @ (gains * m)), 1e-6), 0.5, 2.0))
    out["expo"] = expo
    pix = lin[sel]
    ita_raw, L_raw, b_raw = ita_from(pix)
    ita_gw, L_gw, b_gw = ita_from(pix * gains)
    ita_ge, L_ge, b_ge = ita_from(pix * gains * expo)
    if out["gray"]:
        ita_raw = ita_gw = ita_ge = float("nan")
    out.update(ita_raw=ita_raw, L_raw=L_raw, b_raw=b_raw, ita_gw=ita_gw, L_gw=L_gw, b_gw=b_gw,
               ita_gwexp=ita_ge, L_gwexp=L_ge, b_gwexp=b_ge)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    paths = [l.strip() for l in open(a.list) if l.strip()]
    if a.limit:
        paths = paths[: a.limit]
    with mp.Pool(min(a.procs, 16), initializer=_init) as pool:
        rows = list(pool.imap(process, paths, chunksize=64))
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, float_format="%.5g")
    print(a.out, len(df), "det", float(df.get("det_score", pd.Series(dtype=float)).notna().mean()),
          "ita_gw valid", float(df.ita_gw.notna().mean()) if "ita_gw" in df else 0)


if __name__ == "__main__":
    main()
