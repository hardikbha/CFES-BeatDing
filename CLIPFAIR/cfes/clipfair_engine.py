"""One training step of a CLIPFAIR baseline on EXACTLY the images CESel B0 forwards in that step.

Data: cfes.ces_engine.train_loader / pair_loader / strat_loader with the B0 config (same seed -> the same image stream as
B0): 64 balanced images (32 real + 32 fake), 8 (fake, own-source real) pairs = 16 images, 96 stratified images
(4 per label x group x manipulation-family cell). Pool = [batch; pair fakes; pair reals; stratified] = 176 images,
one backbone pass (the same exposure as B0's det + pair_det + strat_det).

Per method (weights in configs/clipfair_ffpp.yaml, from the official code):
  daw   : DAW-FDD over the pool (per-group CVaR_0.9 of per-sample BCE, CVaR_0.5 across groups)      + orth 0.1
  dag   : DAG-FDD over the pool (CVaR_0.5 of per-sample BCE)                                       + orth 0.1
  pgfdd : CE(sha) + 0.1 CE(spe) + 0.3 L_rec + 0.05 L_con + 0.1 LDAM(fair) + bi-level CVaR(fused)    + orth 0.1
          (sha / spe / fair / fused over the pool; rec / con over the [real; fake] halves of batch + pairs, 40 + 40);
          SAM (rho 0.05) wraps the whole loss in scripts/train_clipfair.py
  ding  : CE over the pool + 0.005 * Sinkhorn(pool logits, one random group's pool logits) inside no_grad (released
          GDA: no gradient), head only trainable (GDA.py:121-124), SFD channels zeroed
orth = the FairReL rank-1 orthogonality regulariser of the adapters, weight 0.1, as in B0 / CFES (backbone recipe).
"""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import clipfair_losses as FL
from .data import ffpp_method


def _w(cfg, m, k):
    return float(cfg["clipfair"][m]["weights"].get(k, 0.0))


def active_terms(cfg):
    m = cfg["clipfair"]["method"]
    keys = {"daw": ["daw"], "dag": ["dag"], "pgfdd": ["sha", "spe", "rec", "con", "fair", "fuse"],
            "ding": ["ce", "gda_fair"]}[m]
    on = {k: _w(cfg, m, k) for k in keys}
    if cfg["model"].get("backend", "clip") in ("clip", "stub") and not (m == "ding" and
                                                                         not cfg["clipfair"]["ding"].get("train_backbone")):
        on["orth"] = float(cfg["loss"]["orth"]["weight"])
    return {k: v for k, v in on.items() if v > 0}


class SpeLookup:
    """PG-FDD specific-forgery label (6-way: 0 real, 1 Deepfakes, 2 Face2Face, 3 FaceSwap, 4 NeuralTextures,
    5 FaceShifter = the spe_label column of Lin's train.csv, NaN for reals -> 0). For images without the column,
    the FF++ method in the file name is mapped with the same table."""
    TABLE = {"real": 0, "Deepfakes": 1, "Face2Face": 2, "FaceSwap": 3, "NeuralTextures": 4, "FaceShifter": 5}

    def __init__(self, train_df, pair_ds=None):
        if "spe_label" in train_df.columns:
            spe = train_df["spe_label"].fillna(0).astype(int).values
        else:
            spe = train_df["method"].map(lambda m: self.TABLE.get(m, 0)).values
        spe = np.where(train_df["label"].values == 0, 0, spe)
        self.by_index = torch.as_tensor(spe, dtype=torch.long)
        self.pair_fake = None
        if pair_ds is not None:
            mp = dict(zip(train_df["img_path"], spe))
            self.pair_fake = torch.as_tensor([int(mp.get(p, self.TABLE.get(ffpp_method(p), 0)))
                                              for p in pair_ds.df["fake_path"]], dtype=torch.long)


def assemble(batch, pb, sb, device, spe):
    """Pool tensors in the CES order. Returns dict(x, y, g, spe, n, npair, nst, real_idx, fake_idx)."""
    x = [batch["image"]]
    y, g = [batch["label"]], [batch["group"]]
    s = [spe.by_index[batch["index"]]]
    n = x[0].shape[0]
    npair = 0
    if pb is not None:
        npair = pb["fake"].shape[0]
        x += [pb["fake"], pb["real"]]
        y += [torch.ones(npair, dtype=torch.long), torch.zeros(npair, dtype=torch.long)]
        g += [pb["fake_group"], pb["real_group"]]
        s += [spe.pair_fake[pb["index"]] if spe.pair_fake is not None else torch.ones(npair, dtype=torch.long),
              torch.zeros(npair, dtype=torch.long)]
    nst = 0
    if sb is not None:
        nst = sb["image"].shape[0]
        x.append(sb["image"]); y.append(sb["label"]); g.append(sb["group"]); s.append(spe.by_index[sb["index"]])
    out = dict(x=torch.cat(x).to(device, non_blocking=True), y=torch.cat(y).long().to(device),
               g=torch.cat(g).long().to(device), spe=torch.cat(s).long().to(device), n=n, npair=npair, nst=nst)
    # [real; fake] halves for PG-FDD's reconstruction / contrastive terms: batch + pair images (40 + 40 at defaults)
    yy = out["y"][:n + 2 * npair]
    r = torch.nonzero(yy == 0).flatten()
    f = torch.nonzero(yy == 1).flatten()
    k = min(len(r), len(f))
    out["real_idx"], out["fake_idx"] = r[:k], f[:k]
    if not torch.isfinite(out["x"]).all():
        raise FloatingPointError("non-finite input pixels")
    return out


def compute_losses(model, pool, cfg, gen, device, terms):
    """Unweighted terms {name: loss} and stats for one forward pass at the current parameters."""
    m = cfg["clipfair"]["method"]
    c = cfg["clipfair"][m]
    amp = bool(cfg["optim"]["amp"]) and device.type == "cuda"
    x, y, g = pool["x"], pool["y"], pool["g"]
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        out = model(x)
    vals, stats = {}, {}
    S = out["score"].float()
    if m == "daw":
        vals["daw"], st = FL.daw_fdd_loss(S, y, g, c["inner_alpha"], c["outer_alpha"])
        stats.update(st)
    elif m == "dag":
        vals["dag"], st = FL.dag_fdd_loss(S, y, c["alpha"])
        stats.update(st)
    elif m == "ding":
        vals["ce"] = F.cross_entropy(out["logits2"].float(), y)                      # XceptionDetector CE
        fair, gi = FL.gda_fair_loss(out["logits2"].float(), g, gen, c["sinkhorn_blur"], bool(c.get("gda_grad", False)))
        vals["gda_fair"] = fair
        stats.update(gda_fair_value=float(fair), gda_group=gi, gda_fair_requires_grad=float(fair.requires_grad))
    elif m == "pgfdd":
        rng = random.Random(int(torch.randint(2 ** 62, (1,), generator=gen)))    # contrastive shuffles
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            rec_1, rec_2, srec_1, srec_2 = model.reconstruct(out, pool["real_idx"], pool["fake_idx"])
        img = x * 2.0 - 1.0                                  # official Normalize([0.5]*3, [0.5]*3): [-1, 1] targets
        real_img, fake_img = img[pool["real_idx"]].float(), img[pool["fake_idx"]].float()
        l1 = lambda a, b: F.l1_loss(b.float(), a)           # loss/l1_loss.py: nn.L1Loss()(inputs, targets)
        vals["rec"] = l1(fake_img, srec_1) + l1(real_img, srec_2) + l1(fake_img, rec_2) + l1(real_img, rec_1)  # :227-237
        vals["sha"] = F.cross_entropy(out["cls"].float(), y)                         # :220
        vals["spe"] = F.cross_entropy(out["cls_spe"].float(), pool["spe"])           # :224
        sub = torch.cat([pool["real_idx"], pool["fake_idx"]])
        vals["con"] = FL.contrastive_loss(out["feat"][sub], out["feat_spe"][sub], pool["spe"][sub], rng,
                                          c["con_margin"])                            # :240-243
        vals["fair"] = FL.ldam_loss(out["cls_fair"], g, c["cls_num_list"], c["ldam_max_m"], c["ldam_s"])  # :246
        vals["fuse"], st = FL.bilevel_fuse_loss(out["cls_fused"], y, g, c["inner_alpha"], c["outer_alpha"],
                                                c["tau1"], c["tau2"])                 # :248-268
        stats.update(st)
        stats["n_rec_pairs"] = int(len(pool["real_idx"]))
    else:
        raise ValueError(m)
    if "orth" in terms:
        vals["orth"] = model.orthogonality_loss()
    with torch.no_grad():
        stats["train_acc"] = float(((S > 0).long() == y).float().mean())
        stats["score_absmean"] = float(S.abs().mean())
    for k in terms:
        if k not in vals:
            raise RuntimeError(f"term {k} not computed")
        if not torch.isfinite(vals[k]):
            raise FloatingPointError(f"non-finite loss {k}: {float(vals[k])}")
    return vals, stats


def weighted_total(vals, terms):
    return sum(terms[k] * v for k, v in vals.items() if k in terms)


# ------------------------------------------------------------------------------------------------ SAM (PG-FDD)
def sam_first_step(params, rho, adaptive=False):
    """FG sam.py:15-27 (first_step) on the parameters that hold a gradient; returns the saved weights."""
    grads = [p for p in params if p.grad is not None]
    norm = torch.norm(torch.stack([((torch.abs(p) if adaptive else 1.0) * p.grad).norm(p=2) for p in grads]), p=2)
    scale = rho / (norm + 1e-12)
    old = {}
    with torch.no_grad():
        for p in grads:
            old[p] = p.data.clone()
            e_w = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale.to(p)
            p.add_(e_w)
    return old, float(norm)


def sam_restore(old):
    """FG sam.py:29-34 (second_step, before the base optimiser step): back to w."""
    with torch.no_grad():
        for p, w in old.items():
            p.data = w


def disable_running_stats(model):
    """FG utils/bypass_bn.py:6-12 (BN momentum 0 during the perturbed pass; only the PG-FDD necks have BN)."""
    for mod in model.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            mod.backup_momentum = mod.momentum
            mod.momentum = 0


def enable_running_stats(model):
    """FG utils/bypass_bn.py:14-19."""
    for mod in model.modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) and hasattr(mod, "backup_momentum"):
            mod.momentum = mod.backup_momentum
