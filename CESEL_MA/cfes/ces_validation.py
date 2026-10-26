"""Which cues are "validated useful", and their L_sens targets.

Declared rule (decide_validated; inputs = utilities u (N,K) of the STARTING model on the validation split, computed
with the same margin and replacement as training: no pairs on validation, so every image uses the opposite-class
EMA baseline stored in the starting checkpoint's CES state):
  ubar_k   = 0.5 * (mean_{reals} u_k + mean_{fakes} u_k)                (class-balanced mean utility)
  CI_k     = percentile bootstrap (B = 1000, class-stratified resampling of images, seed 0) of ubar_k
  valid_k  = lower(CI_k, 2.5%) > min_util (default 0): removing cue k reliably lowers the correct-class margin
  tau_k    = clamp(median_i u_{i,k}, tau_min, tau_max) for valid cues (defaults 0.1, 2.0): each validated cue is
             asked to reach, on every image, at least the contribution it already has on the median image.
The JSON written by scripts/ces_validate_cues.py holds {cues, valid, target, rule, stats}. Without a JSON
(ces.validated_json: null) every cue is valid with target 0.5 (asinh units), the declared default.
"""
import json

import numpy as np
import torch

from .ces_model import CUES


def decide_validated(u, y, cfg=None, seed=0):
    cfg = cfg or {}
    B = int(cfg.get("bootstrap", 1000))
    lo_q = float(cfg.get("alpha", 0.05)) / 2
    min_util = float(cfg.get("min_util", 0.0))
    t_min, t_max = float(cfg.get("tau_min", 0.1)), float(cfg.get("tau_max", 2.0))
    u, y = np.asarray(u, dtype=np.float64), np.asarray(y)
    rng = np.random.default_rng(seed)
    ir, i_f = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    if len(ir) == 0 or len(i_f) == 0:
        raise ValueError("decide_validated needs both classes")
    point = 0.5 * (u[ir].mean(0) + u[i_f].mean(0))
    boots = np.empty((B, u.shape[1]))
    for b in range(B):
        br, bf = rng.choice(ir, len(ir)), rng.choice(i_f, len(i_f))
        boots[b] = 0.5 * (u[br].mean(0) + u[bf].mean(0))
    lo, hi = np.quantile(boots, lo_q, axis=0), np.quantile(boots, 1 - lo_q, axis=0)
    med = np.median(u, 0)
    out = {"cues": list(CUES), "valid": {}, "target": {}, "stats": {},
           "rule": {"valid": f"bootstrap lower {lo_q:.3f} quantile of class-balanced mean utility > {min_util}",
                    "target": f"clamp(median utility, {t_min}, {t_max})", "bootstrap": B, "seed": seed}}
    for k, name in enumerate(CUES):
        v = bool(lo[k] > min_util)
        out["valid"][name] = v
        out["target"][name] = float(np.clip(med[k], t_min, t_max)) if v else None
        out["stats"][name] = {"ubar": float(point[k]), "ci": [float(lo[k]), float(hi[k])], "median": float(med[k]),
                              "frac_pos": float((u[:, k] > 0).mean()), "mean_real": float(u[ir, k].mean()),
                              "mean_fake": float(u[i_f, k].mean())}
    return out


def load_validated(cfg):
    """(valid (K,) bool tensor, target (K,) float tensor, source string)."""
    c = cfg.get("ces", {})
    path = c.get("validated_json")
    if not path:
        t = float(c.get("default_target", 0.5))
        return torch.ones(len(CUES), dtype=torch.bool), torch.full((len(CUES),), t), "default_all_valid"
    d = json.load(open(path))
    if list(d["cues"]) != list(CUES):
        raise ValueError(f"{path}: cue order {d['cues']} != {list(CUES)}")
    valid = torch.tensor([bool(d["valid"][k]) for k in CUES])
    target = torch.tensor([float(d["target"][k]) if d["target"][k] is not None else 0.0 for k in CUES])
    return valid, target, path


# ---------------------------------------------------------------------------------------------------------------------
# Per-family validation (Amendment 2, user request 25 Oct; FairForge-2 ONLY, flag ces.validation.per_family; FF++ stays
# global as pre-registered). For each manipulation family m (condition code m >= 1, data.families order):
#   n_m = number of validation FAKES of family m; if n_m < family_min_n (default 200) -> the global decision is used;
#   else, the same rule on the family's fakes: valid_mk = bootstrap (B, seed 0, images resampled) lower 2.5% bound of
#   mean_{fakes of m} u_k > min_util; tau_mk = clamp(median_{fakes of m} u_k, tau_min, tau_max).
# (Fakes only: the family mask is applied only to that family's fakes; reals keep the global mask.) L_sens then uses,
# per image, the mask / target of its condition (reals and unmapped fakes: global).
def decide_validated_per_family(u, y, cond, fam_names, cfg=None, global_res=None, seed=0):
    cfg = cfg or {}
    B = int(cfg.get("bootstrap", 1000))
    lo_q = float(cfg.get("alpha", 0.05)) / 2
    min_util = float(cfg.get("min_util", 0.0))
    t_min, t_max = float(cfg.get("tau_min", 0.1)), float(cfg.get("tau_max", 2.0))
    n_min = int(cfg.get("family_min_n", 200))
    u, y, cond = np.asarray(u, dtype=np.float64), np.asarray(y), np.asarray(cond)
    rng = np.random.default_rng(seed)
    out = {"families": list(fam_names), "family_min_n": n_min, "valid": {}, "target": {}, "source": {}, "n": {},
           "stats": {}, "rule": "per family (fakes of the family only): bootstrap lower quantile of mean utility > "
                                 f"{min_util}; target clamp(median, {t_min}, {t_max}); n < {n_min} -> global"}
    for m, fam in enumerate(fam_names, start=1):
        idx = np.flatnonzero((y == 1) & (cond == m))
        out["n"][fam] = int(len(idx))
        if len(idx) < n_min:
            out["source"][fam] = "global_fallback"
            out["valid"][fam] = dict(global_res["valid"])
            out["target"][fam] = dict(global_res["target"])
            continue
        boots = np.empty((B, u.shape[1]))
        for b in range(B):
            boots[b] = u[rng.choice(idx, len(idx))].mean(0)
        lo = np.quantile(boots, lo_q, axis=0)
        med = np.median(u[idx], 0)
        out["source"][fam] = "family"
        out["valid"][fam], out["target"][fam], out["stats"][fam] = {}, {}, {}
        for k, name in enumerate(CUES):
            v = bool(lo[k] > min_util)
            out["valid"][fam][name] = v
            out["target"][fam][name] = float(np.clip(med[k], t_min, t_max)) if v else None
            out["stats"][fam][name] = {"mean": float(u[idx, k].mean()), "ci_lo": float(lo[k]), "median": float(med[k])}
    return out


def load_validated_family(cfg, n_cond):
    """(valid (n_cond,K) bool, target (n_cond,K) float) per condition code, or None when the flag is off / the JSON
    has no per-family block. Row 0 (reals) and conditions without a family entry use the global decision."""
    c = cfg.get("ces", {})
    if not c.get("validation", {}).get("per_family", False):
        return None
    path = c.get("validated_json")
    valid, target, _ = load_validated(cfg)
    if not path:
        return None
    d = json.load(open(path))
    pf = d.get("per_family")
    if not pf:
        raise ValueError(f"{path}: ces.validation.per_family is on but the JSON has no per_family block")
    fams = list(cfg["data"].get("families") or [])
    if list(pf["families"]) != fams:
        raise ValueError(f"{path}: families {pf['families']} != data.families {fams}")
    V = valid[None].repeat(n_cond, 1).clone()
    T = target[None].repeat(n_cond, 1).clone()
    for m, fam in enumerate(fams, start=1):
        if m < n_cond:
            V[m] = torch.tensor([bool(pf["valid"][fam][k]) for k in CUES])
            T[m] = torch.tensor([float(pf["target"][fam][k]) if pf["target"][fam][k] is not None else 0.0 for k in CUES])
    return V, T
