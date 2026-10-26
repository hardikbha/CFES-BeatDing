"""Per-image cue utilities from SAVED scores (numpy; no inference), with the training definition of
cfes/ces_losses.utilities and the training replacement rule of cfes/ces_losses.replacement (additive head):

  s = sum_k s_k + b,  s_k = w_k^T c_k;  removing cue k: s^(-k) = s - s_k + w_k^T R_k
  R_k (no pairs at evaluation): reals -> mu[fake, k] (EMA class mean); fakes -> mu[real, g, k], the EMA real-cue mean
  of the image's OWN group (ces.fake_baseline group; class mean mu[real, k] when that group has no estimate or
  fake_baseline = class)
  u_ik = m(s_i, y_i) - m(s_i^(-k), y_i),  m(s, y) = (2y - 1) kappa asinh(s / kappa)  (ces.margin linear: (2y - 1) s)
The EMA means are the checkpoint's CES state (buffers cls_mean (2,K,d), grp_mean (2,G,K,d), grp_init (2,G)), so
beta = w_k^T mu is a scalar per (class[, group], cue). Test sets on the six-group scheme are mapped to the training
(eight-group) indices BY GROUP NAME (cfes.data.SCHEMES) for the group baseline.
"""
import numpy as np
import torch

from cfes.data import SCHEMES


def load_ref(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = ck["model"]
    if "head.w" not in m:
        raise SystemExit(f"{ckpt_path}: cue statistics need the additive head (head.w)")
    w, b = m["head.w"].float().numpy(), float(m["head.b"])
    buf = ck["state"]["ces_state"]["buffers"]
    cls_mean, grp_mean = buf["cls_mean"].float().numpy(), buf["grp_mean"].float().numpy()
    return {"w": w, "b": b, "beta_c": np.einsum("ckd,kd->ck", cls_mean, w),
            "beta_g": np.einsum("cgkd,kd->cgk", grp_mean, w), "grp_init": buf["grp_init"].numpy().astype(bool),
            "cls_init": buf["cls_init"].numpy().astype(bool)}


def to_train_groups(g, set_scheme, train_scheme="eight"):
    names_t = SCHEMES[train_scheme]
    mp = np.array([names_t.index(n) if n in names_t else -1 for n in SCHEMES[set_scheme]])
    return mp[np.asarray(g, dtype=int)]


def margin(s, y, ces):
    yp = 2.0 * np.asarray(y, dtype=np.float64) - 1.0
    if ces.get("margin", "asinh") == "linear":
        return yp * s
    kap = float(ces.get("kappa", 1.0))
    return yp * kap * np.arcsinh(s / kap)


def baselines(y, g_train, ref, fake_baseline="group"):
    """(N,K) beta_ik = w_k^T R_ik under the training replacement rule (no pairs)."""
    y = np.asarray(y, dtype=int)
    beta = np.where(y[:, None] == 0, ref["beta_c"][1][None], ref["beta_c"][0][None]).copy()
    if fake_baseline == "group":
        G = ref["grp_init"].shape[1]
        gt = np.asarray(g_train, dtype=int)
        ok = (y == 1) & (gt >= 0) & (gt < G)
        ok &= ref["grp_init"][0, np.clip(gt, 0, G - 1)]
        beta[ok] = ref["beta_g"][0, gt[ok]]
    elif fake_baseline != "class":
        raise ValueError(fake_baseline)
    return beta


def utilities(s, sk, y, g_train, ref, ces):
    s, sk = np.asarray(s, dtype=np.float64), np.asarray(sk, dtype=np.float64)
    beta = baselines(y, g_train, ref, ces.get("fake_baseline", "group"))
    s_minus = s[:, None] - sk + beta
    yy = np.asarray(y, dtype=np.float64)
    return margin(s, yy, ces)[:, None] - margin(s_minus, yy[:, None], ces)
