"""Beat-Ding component A (decisions/2026-10-26-beat-ding-plan.md section 2A): label-preserving processing augmentation
for BOTH classes. With probability p (0.6), ONE op per image, drawn uniformly from:
  jpeg        quality q ~ U{30..90}                                   (cv2 JPEG, 4:2:0, re-decoded)
  bilateral   "beauty" smoothing, d in {5, 7, 9}, sigma_color, sigma_space ~ U[20, 80]   (cv2.bilateralFilter)
  gblur       Gaussian blur sigma ~ U[0.4, 1.6] px                     (cv2.GaussianBlur, kernel from sigma)
  downup      down-up scale s ~ U[0.4, 0.9], bicubic down and bicubic back up
  photometric gamma ~ U[0.6, 1.6], brightness factor ~ U[0.75, 1.25], contrast factor ~ U[0.75, 1.25] (about the grey
              mean), saturation factor ~ U[0.7, 1.3] (blend with the grey image), in this order
  unsharp     unsharp mask, Gaussian sigma ~ U[0.5, 1.5] px, amount ~ U[0.5, 1.5], no threshold (the plan names the op
              without parameters; these ranges are an implementation choice, fixed before any BD run, README)
then a horizontal flip with probability 0.5 (independent). Every op works on the uint8 RGB image AT THE MODEL INPUT
SIZE (after the dataset's resize to data.input_size, i.e. "pixels at 224" like cf.bg_blur_sigma) and returns uint8, so
the [0, 1] float range of the pipeline is unchanged.

Reproducibility: the parameters of one item are drawn from np.random.default_rng([SALT, seed, epoch, stream, index]),
independent of DataLoader workers and of the order in which items are fetched. One draw serves a (fake, own source real)
pair, so both images receive the same op with the same parameters (and the same flip). The CFES stability view of an
image is made from the already-augmented image inside the training step, so (image, view) pairs share the
augmentation too.
"""
import cv2
import numpy as np

cv2.setNumThreads(1)        # DataLoader workers: no thread oversubscription

OPS = ("jpeg", "bilateral", "gblur", "downup", "photometric", "unsharp")
SALT = 20261026
STREAMS = {"train": 0, "strat": 1, "pairs": 2}
DEFAULT = {"p": 0.6, "hflip": 0.5, "ops": list(OPS),
           "jpeg_q": [30, 90], "bilateral_d": [5, 7, 9], "bilateral_sigma": [20.0, 80.0], "gblur_sigma": [0.4, 1.6],
           "downup_scale": [0.4, 0.9], "gamma": [0.6, 1.6], "brightness": 0.25, "contrast": 0.25, "saturation": 0.30,
           "unsharp_sigma": [0.5, 1.5], "unsharp_amount": [0.5, 1.5]}
GREY = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def config(cfg):
    """data.aug block merged over DEFAULT (unknown keys rejected)."""
    cfg = dict(cfg or {})
    cfg.pop("enabled", None)
    bad = set(cfg) - set(DEFAULT)
    if bad:
        raise ValueError(f"unknown data.aug keys {sorted(bad)}")
    out = {**DEFAULT, **cfg}
    if set(out["ops"]) - set(OPS):
        raise ValueError(f"unknown aug ops {sorted(set(out['ops']) - set(OPS))}")
    return out


def item_rng(seed, epoch, stream, index):
    return np.random.default_rng([SALT, int(seed), int(epoch), int(STREAMS.get(stream, stream)), int(index)])


def draw(rng, c):
    """Parameters for one item (or one pair): {'op': name or None, 'flip': bool, ...op parameters}."""
    u = lambda lo, hi: float(rng.uniform(lo, hi))       # noqa: E731
    prm = {"op": None}
    if rng.random() < c["p"]:
        op = c["ops"][int(rng.integers(len(c["ops"])))]
        prm["op"] = op
        if op == "jpeg":
            prm["q"] = int(rng.integers(c["jpeg_q"][0], c["jpeg_q"][1] + 1))
        elif op == "bilateral":
            prm["d"] = int(c["bilateral_d"][int(rng.integers(len(c["bilateral_d"])))])
            prm["sigma_color"], prm["sigma_space"] = u(*c["bilateral_sigma"]), u(*c["bilateral_sigma"])
        elif op == "gblur":
            prm["sigma"] = u(*c["gblur_sigma"])
        elif op == "downup":
            prm["scale"] = u(*c["downup_scale"])
        elif op == "photometric":
            prm["gamma"] = u(*c["gamma"])
            prm["brightness"] = u(1 - c["brightness"], 1 + c["brightness"])
            prm["contrast"] = u(1 - c["contrast"], 1 + c["contrast"])
            prm["saturation"] = u(1 - c["saturation"], 1 + c["saturation"])
        elif op == "unsharp":
            prm["sigma"], prm["amount"] = u(*c["unsharp_sigma"]), u(*c["unsharp_amount"])
    prm["flip"] = bool(rng.random() < c["hflip"])
    return prm


def _u8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def apply(img, prm):
    """img: (H, W, 3) uint8 RGB -> (H, W, 3) uint8 RGB, C-contiguous."""
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"augmentation expects HxWx3 uint8 RGB, got {img.dtype} {img.shape}")
    op, out = prm["op"], img
    h, w = img.shape[:2]
    if op == "jpeg":
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), prm["q"]])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        out = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    elif op == "bilateral":
        out = cv2.bilateralFilter(img, prm["d"], prm["sigma_color"], prm["sigma_space"])
    elif op == "gblur":
        out = cv2.GaussianBlur(img, (0, 0), sigmaX=prm["sigma"], sigmaY=prm["sigma"])
    elif op == "downup":
        sw, sh = max(1, int(round(w * prm["scale"]))), max(1, int(round(h * prm["scale"])))
        out = cv2.resize(cv2.resize(img, (sw, sh), interpolation=cv2.INTER_CUBIC), (w, h), interpolation=cv2.INTER_CUBIC)
    elif op == "photometric":
        x = img.astype(np.float32) / 255.0
        x = np.power(x, prm["gamma"]) * prm["brightness"]
        m = float((x @ GREY).mean())
        x = (x - m) * prm["contrast"] + m
        g = (x @ GREY)[..., None]
        x = g + (x - g) * prm["saturation"]
        out = _u8(np.clip(x, 0.0, 1.0) * 255.0)
    elif op == "unsharp":
        x = img.astype(np.float32)
        b = cv2.GaussianBlur(x, (0, 0), sigmaX=prm["sigma"], sigmaY=prm["sigma"])
        out = _u8(x + prm["amount"] * (x - b))
    elif op is not None:
        raise ValueError(f"unknown op {op}")
    if prm["flip"]:
        out = out[:, ::-1]
    return np.ascontiguousarray(out)
