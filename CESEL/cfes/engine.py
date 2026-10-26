"""Shared pieces of training, validation and evaluation, so validate.py tests the exact code path train.py runs."""
import numpy as np
import torch
from torch.utils.data import DataLoader

from . import losses as L
from .counterfactuals import BACKGROUND_KINDS, FaceGeometry, sample_views, shared_intervention
from .data import BalancedRealFakeSampler, ImageTable, PairTable, StratifiedGroupBatchSampler, read_split
from .model import CFESDetector, convert_alpha_state, topk_mean
from .model_xception import CFESXception
from .model_resnet import CFESResNet
from . import matchcheck as MC
from . import mfer as M

TERMS = ("det", "stab", "sens", "loc", "align", "orth")


def active_terms(cfg, have_pairs):
    """Which loss terms run, after checking each term's prerequisites. Returns {name: weight}."""
    lc = cfg["loss"]
    on = {"det": lc["det"]["weight"], "orth": lc["orth"]["weight"]}
    if cfg["cf"]["enabled"]:
        on["stab"] = lc["stab"]["weight"]
        on["view_det"] = lc.get("view_det", {}).get("weight", 0.0)   # review item 1G: BCE on the views, no consistency
    if have_pairs:
        on["sens"] = lc["sens"]["weight"]
        on["loc"] = lc["loc"]["weight"]
        on["pair_det"] = lc.get("pair_det", {}).get("weight", 0.0)   # detection BCE on the pair images themselves
    if cfg.get("strat", {}).get("enabled"):
        on["strat_det"] = lc.get("strat_det", {}).get("weight", 0.0)  # detection BCE on the group-stratified batch
    mf = cfg.get("mfer", {})
    if mf.get("enabled"):
        if not mf.get("csv") or not (mf.get("validated") or mf.get("smoke")):
            raise RuntimeError("MFER training blocked: no validated cross-group matched-set file "
                               "(set mfer.csv and, only after the matched-set audit, mfer.validated: true; "
                               "code-path fixtures run only with train.py --smoke)")
        on["mfer_margin"] = lc["mfer_margin"]["weight"]
        on["mfer_cons"] = lc["mfer_cons"]["weight"]
        on["mfer_det"] = lc.get("mfer_det", {}).get("weight", 0.0)     # BCE on the valid matched images
    if lc.get("risk", {}).get("weight", 0.0) > 0:
        if not cfg["data"].get("demographics_reliable", False):
            raise ValueError("loss.risk.weight > 0 but data.demographics_reliable is false for this dataset")
        if not cfg.get("strat", {}).get("enabled"):
            raise ValueError("loss.risk needs the group-stratified batch (strat.enabled)")
        on["risk"] = lc["risk"]["weight"]
    if lc.get("fprgap", {}).get("weight", 0.0) > 0:          # arm E: soft group-FPR-gap penalty (losses.fpr_gap_loss)
        if not cfg["data"].get("demographics_reliable", False):
            raise ValueError("loss.fprgap.weight > 0 but data.demographics_reliable is false for this dataset")
        if not cfg.get("strat", {}).get("enabled"):
            raise ValueError("loss.fprgap needs the group-stratified batch (strat.enabled)")
        on["fprgap"] = lc["fprgap"]["weight"]
    if lc.get("rel_sup", {}).get("weight", 0.0) > 0:         # C-FER R4: cross-fitted reliability supervision
        if not (cfg["model"].get("reliability") or {}).get("enabled"):
            raise ValueError("loss.rel_sup needs model.reliability.enabled (the C-FER reliability head)")
        if not lc["rel_sup"].get("targets"):
            raise ValueError("loss.rel_sup needs loss.rel_sup.targets (out-of-fold targets csv, scripts/cfer_oof.py)")
        on["rel_sup"] = lc["rel_sup"]["weight"]
    if lc.get("dalign", {}).get("weight", 0.0) > 0:
        if not cfg["data"].get("demographics_reliable", False):
            raise ValueError("loss.dalign.weight > 0 but data.demographics_reliable is false for this dataset")
        if not cfg.get("strat", {}).get("enabled"):
            raise ValueError("loss.dalign needs the group-stratified batch (strat.enabled)")
        on["dalign"] = lc["dalign"]["weight"]
    if lc.get("scale", {}).get("weight", 0.0) > 0:
        on["scale"] = lc["scale"]["weight"]     # logit-scale control on all detection-labelled scores
    if lc["align"]["weight"] > 0:
        if not cfg["data"].get("demographics_reliable", False):
            raise ValueError("loss.align.weight > 0 but data.demographics_reliable is false for this dataset")
        if lc["align"].get("require_strat", False) and not cfg.get("strat", {}).get("enabled"):
            raise ValueError("loss.align.require_strat: the alignment term needs the group-stratified batch (strat.enabled)")
        on["align"] = lc["align"]["weight"]
    if cfg["model"].get("backend", "clip") != "clip":
        on.pop("orth", None)     # rank-1 orthogonality exists only for the CLIP SVD adapters
    return {k: float(v) for k, v in on.items() if float(v) > 0}


def build_model(cfg, device):
    if cfg["model"].get("backend", "clip") == "xception":
        model = CFESXception(cfg["model"], cfg["paths"]["ding_repo"])
    elif cfg["model"].get("backend", "clip") == "resnet":
        model = CFESResNet(cfg["model"])
    elif cfg["model"].get("backend", "clip") == "fsvfm":
        from .model_fsvfm import CFESFSVFM      # FS-VFM ViT-S/16, full fine-tune (agents/LIGHT_BACKBONE)
        model = CFESFSVFM(cfg["model"])
    else:
        model = CFESDetector(cfg["model"], cfg["paths"]["fairrel_repo"])
        if cfg["model"].get("init_from"):
            load_fairrel_checkpoint(model, cfg["model"]["init_from"])
        if model.rel_enabled:                                  # C-FER: the reliability view's face template
            model.set_reliability_face(cfg["cf"]["face_mask"])
    if cfg["model"].get("init_ckpt"):
        # phase 2 starts from this model's own phase-1 checkpoint (CFES format, all keys must match)
        ck = torch.load(cfg["model"]["init_ckpt"], map_location="cpu", weights_only=False)
        if getattr(model, "rel_head", None) is not None:
            # C-FER: the reliability head is new in phase 2 and keeps its init; every other key must match
            missing, unexpected = model.load_state_dict(convert_alpha_state(ck["model"], model), strict=False)
            bad = [k for k in missing if not k.startswith("rel_head.")]
            if bad or unexpected:
                raise RuntimeError(f"init_ckpt: missing {bad[:5]} unexpected {unexpected[:5]}")
            reliability_start_from_r2(model)
        else:
            model.load_state_dict(convert_alpha_state(ck["model"], model), strict=True)
    return model.to(device)


@torch.no_grad()
def reliability_start_from_r2(model):
    """C-FER Amendment 1 (B2): R3 / R4 start EXACTLY at R2's starting score. After the phase-1 load (tanh conversion),
    R2's start is s + a_D t (a_D = fusion_weight > 0). With the reliability head's zero-initialised outputs (r_global
    logit 0, uniform patch weights) and the gate map at identity, alpha = sigmoid(gate_b) mapped into [lo, hi]; setting
    gate_b = logit((a0 - lo) / (hi - lo)) with a0 = 1 / (1 + a_D) and scaling cls_head and the last patch-head linear
    by (1 + a_D) gives a0 (1 + a_D) s + (1 - a0)(1 + a_D) t = s + a_D t. Local normalisation is scale invariant (its
    factor is 1 before the first step). A later checkpoint load (resume, evaluation) overwrites all of this."""
    import math
    from .model import fusion_weight, last_patch_linear
    aD = float(fusion_weight(model))
    if not aD > 0:
        raise ValueError(f"C-FER exact start needs a positive phase-2 fusion weight a_D, got {aD}")
    lo, hi = model.rel_gate
    a0 = 1.0 / (1.0 + aD)
    if not lo < a0 < hi:
        raise ValueError(f"C-FER start gate a0 = {a0:.4f} outside the bounds [{lo}, {hi}]")
    u = (a0 - lo) / (hi - lo)
    model.rel_head.gate_w.fill_(1.0)
    model.rel_head.gate_b.fill_(math.log(u / (1.0 - u)))
    lin = dict(model.named_modules())[last_patch_linear(model)]
    for mod in (model.cls_head, lin):
        mod.weight.mul_(1.0 + aD)
        mod.bias.mul_(1.0 + aD)
    model.rel_start_conversion = {"a_D": aD, "alpha0": a0, "gate_b": float(model.rel_head.gate_b),
                                  "scale_cls_and_last_patch_linear": 1.0 + aD, "last_patch_linear": last_patch_linear(model)}


def reliability_targets(cfg, train_df):
    """C-FER R4: out-of-fold correctness targets (scripts/cfer_oof.py) aligned to train_df rows (float tensor). Every
    training image must have exactly one target in [0, 1]."""
    import pandas as pd
    t = pd.read_csv(cfg["loss"]["rel_sup"]["targets"])
    if t.img_path.duplicated().any():
        raise ValueError("duplicate img_path in reliability targets")
    m = train_df[["img_path"]].merge(t[["img_path", "target"]], on="img_path", how="left")
    if m.target.isna().any() or not m.target.between(0, 1).all():
        raise ValueError(f"reliability targets: {int(m.target.isna().sum())} training images without a target in [0,1]")
    return torch.tensor(m.target.values, dtype=torch.float32)


def load_fairrel_checkpoint(model, path):
    """Warm start from a FairReL-code checkpoint (backbone.* + classifier.*), e.g. the plain-CLIP control
    FAIRREL_REPRO/runs/CTRL_s7/best.pt. classifier.* maps to cls_head.*; the patch head keeps its init."""
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    sd = {("cls_head." + k[len("classifier."):]) if k.startswith("classifier.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not (k.startswith("patch_head.") or k in ("alpha", "alpha_raw"))]
    if bad or unexpected:
        raise RuntimeError(f"init_from {path}: missing {bad[:5]} unexpected {unexpected[:5]}")


def build_geometry(cfg, model, device):
    return FaceGeometry.from_cfg(cfg["cf"], cfg["data"]["input_size"], model.grid, device)


def train_loader(cfg, seed):
    d = cfg["data"]
    df = read_split({**d["train"], "max_items": cfg["debug"].get("max_train_items")})
    if d.get("train_aug"):
        if d["train_aug"] != "fsvfm":
            raise ValueError(f"unknown data.train_aug {d['train_aug']}")
        from .model_fsvfm import FSVFMTrainTable    # FS-VFM fine-tuning augmentation (opt-in, LIGHT_BACKBONE)
        ds = FSVFMTrainTable(df, d["input_size"], cfg.get("aug", {}))
    else:
        ds = ImageTable(df, d["input_size"], d.get("resample", "bicubic"))
    steps = cfg["debug"].get("max_steps_per_epoch") or d.get("steps_per_epoch")
    sampler = BalancedRealFakeSampler(df["label"].values, d["batch_size"], steps=steps, seed=seed)
    return df, sampler, DataLoader(ds, batch_size=d["batch_size"], sampler=sampler, num_workers=d["num_workers"],
                                   pin_memory=True, drop_last=True, persistent_workers=d["num_workers"] > 0)


def pair_loader(cfg, geom_cpu, seed):
    p = cfg["pairs"]
    if not p.get("enabled") or not p.get("csv"):
        return None
    ds = PairTable(p["csv"], cfg["data"]["input_size"], geom_cpu.grid, geom_cpu.core, p["diff_tau"],
                   align_tau=p.get("align_tau"), max_items=cfg["debug"].get("max_pairs"), seed=seed,
                   resample=cfg["data"].get("resample", "bicubic"))
    if len(ds) == 0:
        return None
    g, persistent = None, cfg["data"]["num_workers"] > 0
    if cfg.get("train", {}).get("decoupled_rng"):
        # re-seeded per epoch by train.py; workers are NOT persistent, because a persistent loader draws its worker
        # seed from the generator only on the first iterator, which made resumed runs shuffle pairs differently
        g, persistent = torch.Generator().manual_seed(int(seed)), False
    return DataLoader(ds, batch_size=p["per_step"], shuffle=True, num_workers=cfg["data"]["num_workers"],
                      drop_last=True, pin_memory=True, persistent_workers=persistent, generator=g)


def mfer_loader(cfg, seed):
    """Returns (loader, audit report) for a matched-set file whose validation sidecar allows this use, or
    (None, None) when MFER is off. The loader's generator is re-seeded per epoch by train.py (mfer_epoch_seed), so
    the matched-set order is reproducible from any epoch boundary."""
    mf = cfg.get("mfer", {})
    if not mf.get("enabled"):
        return None, None
    import pandas as pd
    side = MC.require_usable(mf["csv"], smoke=bool(mf.get("smoke")))
    df, rep = M.read_matched_sets(pd.read_csv(mf["csv"]), mf["min_groups"], mf["max_groups"])
    rep.update(sidecar_status=side["status"], csv_sha256=side["csv_sha256"])
    if rep["sets"] == 0:
        raise RuntimeError(f"MFER training blocked: matched-set file has no valid set after checks: {rep}")
    if rep["sets"] < mf["sets_per_step"]:
        raise RuntimeError(f"MFER would yield zero batches: {rep['sets']} valid matched sets but mfer.sets_per_step="
                           f"{mf['sets_per_step']} with drop_last=True. Lower mfer.sets_per_step to <= {rep['sets']} "
                           "or add matched sets.")
    ds = M.MatchedSetTable(df, cfg["data"]["input_size"], mf["max_groups"], cfg["data"].get("resample", "bicubic"),
                           oversize=mf.get("oversize", "error"))
    g = torch.Generator().manual_seed(mfer_epoch_seed(seed, 0))
    dl = DataLoader(ds, batch_size=mf["sets_per_step"], shuffle=True, generator=g, drop_last=True,
                    num_workers=cfg["data"]["num_workers"], pin_memory=True)
    if len(dl) == 0:
        raise RuntimeError("MFER DataLoader has zero batches (check mfer.sets_per_step vs the number of sets)")
    rep["batches_per_pass"] = len(dl)
    return dl, rep


def mfer_epoch_seed(seed, epoch):
    return int(seed) * 100003 + int(epoch)


def strat_loader(cfg, train_df, steps, seed):
    """Group-stratified auxiliary batches (k per (label, group) cell per step), or None when strat is off."""
    st = cfg.get("strat", {})
    if not st.get("enabled"):
        return None
    d = cfg["data"]
    ds = ImageTable(train_df, d["input_size"], d.get("resample", "bicubic"))
    bs = StratifiedGroupBatchSampler(train_df["label"].values, train_df["group"].values, st["per_cell"], steps, seed)
    return DataLoader(ds, batch_sampler=bs, num_workers=d["num_workers"], pin_memory=True,
                      persistent_workers=d["num_workers"] > 0)


def eval_loader(spec, cfg, max_items=None):
    df = read_split({**spec, "max_items": max_items})
    d = cfg["data"]
    ds = ImageTable(df, d.get("eval_size", d["input_size"]), d.get("resample", "bicubic"))
    return df, DataLoader(ds, batch_size=cfg["eval"]["batch_size"], shuffle=False,
                          num_workers=cfg["data"]["num_workers"], pin_memory=True)


def fprgap_quantile(cfg):
    """Running-threshold state of arm E's penalty (one per training process; train.py keeps it in last.pt)."""
    c = cfg["loss"]["fprgap"]
    return L.RunningQuantile(c.get("q", 0.9), c.get("window", 4096))


def compute_losses(model, batch, pair_batch, cfg, geom, gen, device, terms, mfer_batch=None, strat_batch=None,
                   aux=None):
    """One forward over [x, x_cf, fake_pairs, real_pairs, mfer fakes, mfer reals];
    returns ({term: unweighted loss}, stats). aux: cross-step state ({"fprgap_rq": RunningQuantile}); a caller that
    passes none gets a fresh threshold window per call."""
    x = batch["image"].to(device, non_blocking=True)
    y = batch["label"].to(device)
    g = batch["group"].to(device)
    if any(t in terms for t in M.TERMS) and mfer_batch is None:
        raise RuntimeError("MFER loss enabled but no matched-set batch was supplied this step")
    soft = None
    if cfg.get("aug", {}).get("mixup_bce"):
        # FS-VFM recipe (opt-in): Mixup/CutMix + label smoothing on the clean batch, BCE on the soft fake target.
        # Only the detection term is allowed with it (the other terms assume unmixed images and hard labels).
        if set(terms) - {"det"}:
            raise ValueError(f"aug.mixup_bce supports only the det term, got {sorted(terms)}")
        from .model_fsvfm import MixupBCE
        x, soft = MixupBCE(cfg["aug"])(x, y, gen)
    chunks, n = [x], x.shape[0]
    kind = None
    vgen = pgen = gen
    if cfg.get("train", {}).get("decoupled_rng"):
        # exactly two draws from `gen` per step in every arm: separate streams for views and pair interventions, so
        # arms with and without views receive identical pair images (exposure matching)
        seeds = torch.randint(2 ** 62, (2,), generator=gen)
        vgen = torch.Generator().manual_seed(int(seeds[0]))
        pgen = torch.Generator().manual_seed(int(seeds[1]))
    views = "stab" in terms or "view_det" in terms
    if views:
        x_cf, kind = sample_views(x, y, cfg["cf"], geom, vgen)
        chunks.append(x_cf)
    npair = 0
    if pair_batch is not None and ("sens" in terms or "loc" in terms or "pair_det" in terms):
        f = pair_batch["fake"].to(device, non_blocking=True)
        r = pair_batch["real"].to(device, non_blocking=True)
        if cfg["pairs"]["shared_intervention"]:
            f, r, _ = shared_intervention(f, r, cfg["cf"], geom, pgen, donor_pool=x[y == 0])
        chunks += [f, r]
        npair = f.shape[0]
    nst = 0
    if "strat_det" in terms or (strat_batch is not None and ("align" in terms or "dalign" in terms or "risk" in terms
                                                              or "fprgap" in terms)):
        if strat_batch is None:
            raise RuntimeError("strat_det / align enabled but no stratified batch was supplied this step")
        xs = strat_batch["image"].to(device, non_blocking=True)
        ys, gs = strat_batch["label"].to(device), strat_batch["group"].to(device)
        chunks.append(xs)
        nst = xs.shape[0]
    nm = 0
    if mfer_batch is not None and any(t in terms for t in M.TERMS):
        xm = M.flatten_for_forward(mfer_batch)          # valid slots only; padded slots never reach the model
        chunks.append(xm.to(device, non_blocking=True))
        nm = xm.shape[0]
    amp = cfg["optim"]["amp"]
    xin = torch.cat(chunks)
    if not torch.isfinite(xin).all():
        bad = [i for i, c in enumerate(chunks) if not torch.isfinite(c).all()]
        raise FloatingPointError(f"non-finite input pixels in chunk(s) {bad} (order: batch, [views], [pairs], [matched])")
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bool(amp) and device.type == "cuda"):
        if getattr(model, "rel_enabled", False):
            out = reliability_forward(model, xin, n if views else 0)
        else:
            out = model(xin)

    def part(a, b):
        return {k: (v[a:b] if v is not None else None) for k, v in out.items()}

    clean = part(0, n)
    vals, stats = {}, {}
    if soft is not None:
        vals["det"] = torch.nn.functional.binary_cross_entropy_with_logits(clean["score"], soft)
    else:
        vals["det"], _ = L.detection_loss(clean["score"], y, cfg["loss"]["det"]["mode"])
    off = n
    if views:
        view = part(off, off + n)
        off += n
        if "stab" in terms:
            vals["stab"], s = L.stability_loss(clean, view, kind, geom, cfg["loss"]["stab"], BACKGROUND_KINDS)
            stats.update(s)
        if "view_det" in terms:
            vals["view_det"], _ = L.detection_loss(view["score"], y, cfg["loss"]["det"]["mode"])
    if npair:
        of, orr = part(off, off + npair), part(off + npair, off + 2 * npair)
        frac = pair_batch["frac"].to(device)
        aligned = pair_batch["aligned"].to(device)
        if "sens" in terms:
            vals["sens"], s = L.sensitivity_loss(of, orr, frac, aligned, geom, cfg["loss"]["sens"])
            stats.update(s)
        if "loc" in terms:
            vals["loc"], s = L.localization_loss(of["patch"], orr["patch"], frac, aligned, cfg["loss"]["sens"])
            stats.update(s)
        if "pair_det" in terms:
            vals["pair_det"], _ = L.detection_loss(torch.cat([of["score"], orr["score"]]),
                                                   torch.cat([torch.ones(npair), torch.zeros(npair)]).long().to(device),
                                                   cfg["loss"]["det"]["mode"])
        off += 2 * npair
    st_out = None
    if nst:
        st_out = part(off, off + nst)
        off += nst
        if "strat_det" in terms:
            vals["strat_det"], _ = L.detection_loss(st_out["score"], ys, cfg["loss"]["det"]["mode"])
    if nm:
        start = sum(c.shape[0] for c in chunks[:-1])
        msk = mfer_batch["mask"].to(device)
        grp = mfer_batch["group"].to(device)
        sm = out["score"][start:start + nm]
        sf, sr = M.unflatten_scores(sm, msk)
        dl = M.deltas(sf, sr)
        if "mfer_det" in terms:
            vals["mfer_det"], s = M.detection_loss(sm, cfg["loss"]["det"]["mode"])
            stats.update(s)
        if "mfer_margin" in terms:
            vals["mfer_margin"], s = M.margin_loss(dl, msk, cfg["loss"]["mfer_margin"]["margin"])
            stats.update(s)
        if "mfer_cons" in terms:
            vals["mfer_cons"], s = M.consistency_loss(dl, msk, grp)
            stats.update(s)
    if "align" in terms:
        if st_out is not None:     # ordinary batch + group-stratified batch: every subgroup present each step
            a_s, a_y, a_g = torch.cat([clean["score"], st_out["score"]]), torch.cat([y, ys]), torch.cat([g, gs])
        else:
            a_s, a_y, a_g = clean["score"], y, g
        vals["align"], s = L.alignment_loss(a_s, a_y, a_g, cfg["loss"]["align"])
        stats.update(s)
    if "scale" in terms:
        sc = [clean["score"]]
        if npair:
            sc += [of["score"], orr["score"]]
        if st_out is not None:
            sc.append(st_out["score"])
        vals["scale"], s = L.scale_loss(torch.cat(sc), cfg["loss"]["scale"])
        stats.update(s)
    if "risk" in terms:
        r_s, r_y, r_g = [clean["score"], st_out["score"]], [y, ys], [g, gs]
        if views and cfg["loss"]["risk"].get("include_views", True):      # reals under nuisance views carry more errors
            r_s.append(view["score"]); r_y.append(y); r_g.append(g)
        r_s, r_y, r_g = torch.cat(r_s), torch.cat(r_y), torch.cat(r_g)
        vals["risk"], s = L.real_group_risk_loss(r_s, r_y, r_g, cfg["loss"]["risk"])
        stats.update(s)
    if "fprgap" in terms:
        # arm E: real images of the ordinary + stratified batch and their nuisance views (same pool as the risk term)
        f_s, f_y, f_g = [clean["score"], st_out["score"]], [y, ys], [g, gs]
        if views and cfg["loss"]["fprgap"].get("include_views", True):
            f_s.append(view["score"]); f_y.append(y); f_g.append(g)
        rq = (aux or {}).get("fprgap_rq") or fprgap_quantile(cfg)
        vals["fprgap"], s = L.fpr_gap_loss(torch.cat(f_s), torch.cat(f_y), torch.cat(f_g), cfg["loss"]["fprgap"], rq)
        stats.update(s)
    if "rel_sup" in terms:
        # C-FER R4: BCE of r_global against the out-of-fold correctness target, ordinary + stratified clean images
        tg = (aux or {}).get("rel_targets")
        if tg is None:
            raise RuntimeError("rel_sup enabled but no reliability targets were supplied (aux['rel_targets'])")
        r_l, r_t = [clean["rel_global"]], [tg[batch["index"].cpu()]]
        if st_out is not None:
            r_l.append(st_out["rel_global"]); r_t.append(tg[strat_batch["index"].cpu()])
        vals["rel_sup"], s = L.reliability_supervision_loss(torch.cat(r_l), torch.cat(r_t).to(device))
        stats.update(s)
    if "dalign" in terms:
        a_s, a_y, a_g = torch.cat([clean["score"], st_out["score"]]), torch.cat([y, ys]), torch.cat([g, gs])
        vals["dalign"], s = L.dist_align_loss(a_s, a_y, a_g, cfg["loss"]["dalign"])
        stats.update(s)
    if "orth" in terms:
        vals["orth"] = model.orthogonality_loss()
    with torch.no_grad():
        stats["train_acc"] = float(((clean["score"] > 0).long() == y).float().mean())
        a = model.fusion_alpha()
        if a is not None:
            stats["alpha"] = float(a)
        stats["s_global_absmean"] = float(clean["s_cls"].abs().mean())
        stats["s_global_mean_real"] = float(clean["s_cls"][y == 0].mean()) if (y == 0).any() else 0.0
        stats["s_global_mean_fake"] = float(clean["s_cls"][y == 1].mean()) if (y == 1).any() else 0.0
        if clean["patch"] is not None:
            loc = topk_mean(clean["patch"], model.k)
            stats["s_local_absmean"] = float(loc.abs().mean())
            stats["s_local_mean"] = float(loc.mean())
            stats["patch_logit_min"] = float(clean["patch"].min())
            stats["patch_logit_max"] = float(clean["patch"].max())
            if a is not None:
                stats["alpha_x_local_mean"] = float(a) * float(loc.mean())
        if clean.get("gate") is not None:          # C-FER reliability head (absent in every other arm)
            r = torch.sigmoid(clean["rel_global"])
            stats.update(rel_gate_mean=float(clean["gate"].mean()), rel_gate_min=float(clean["gate"].min()),
                         rel_gate_max=float(clean["gate"].max()), rel_r_mean=float(r.mean()),
                         rel_r_real=float(r[y == 0].mean()) if (y == 0).any() else 0.0,
                         rel_r_fake=float(r[y == 1].mean()) if (y == 1).any() else 0.0,
                         rel_wpatch_max=float(torch.softmax(clean["rel_patch"].flatten(1), 1).max(1).values.mean()),
                         rel_inst_mean=float(clean["inst"].mean()), rel_gate_w=float(model.rel_head.gate_w),
                         rel_gate_b=float(model.rel_head.gate_b))
    check_terms(vals, terms)
    return vals, stats


def reliability_forward(model, xin, n_views):
    """C-FER R3 training forward. One backbone pass over every chunk, then the fusion with z_view =
    - the stability view's global logit for a clean image (the view already drawn for that image, chunk 2), and the
      clean image's global logit for its view (instability is symmetric), when views are drawn (n_views = batch size);
    - the deterministic evaluation view's logit (extra no-grad pass) for every other image (pairs, stratified batch,
      matched sets, or everything when no views are drawn)."""
    parts = model.forward_parts(xin)
    s = parts["s_cls"].detach()
    zv = torch.empty_like(s)
    k = 2 * n_views
    if n_views:
        zv[:n_views], zv[n_views:k] = s[n_views:k], s[:n_views]
    if k < xin.shape[0]:
        zv[k:] = model.view_logit(xin[k:])
    return model.fuse(parts, zv)


def check_terms(vals, terms):
    """Every loss with a nonzero weight must have been computed this step and be finite."""
    missing = [k for k in terms if k not in vals]
    if missing:
        raise RuntimeError(f"enabled loss(es) {missing} (weights {[terms[k] for k in missing]}) were not computed this "
                           "step: their inputs (pairs / matched sets / views) were not supplied")
    bad = {k: float(v) for k, v in vals.items() if k in terms and not torch.isfinite(v)}
    if bad:
        raise FloatingPointError(f"non-finite loss term(s): {bad}")


def weighted_total(vals, terms):
    return sum(terms[k] * v for k, v in vals.items() if k in terms)


@torch.no_grad()
def predict(model, loader, device, amp=True, max_batches=None, return_logits=False, prob64=False):
    """prob64: probabilities as float64 sigmoid of the float32 logits. float32 sigmoid returns exactly 0 or 1 for
    |logit| > ~17, which creates ties and coarsens AUC on saturated detectors; the 0.5 threshold is unaffected."""
    model.eval()
    probs, logits = [], []
    for i, b in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp and device.type == "cuda"):
            s = model(b["image"].to(device, non_blocking=True))["score"]
        probs.append(torch.sigmoid(s.float()).cpu().numpy())
        logits.append(s.float().cpu().numpy())
    model.train()
    p = np.concatenate(probs) if probs else np.zeros(0)
    z = np.concatenate(logits) if logits else np.zeros(0)
    if prob64:
        p = 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
    return (p, z) if return_logits else p
