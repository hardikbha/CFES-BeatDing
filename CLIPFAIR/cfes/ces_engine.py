"""CES training step: one backbone pass over [batch, (views), (pair fakes, pair reals), (stratified), (matched)],
then every CES quantity at head level.

  L_total = L_det + l_sens L_sens + l_inv L_inv + l_floor L_floor  [+ optional CFES-D terms, all off by default]
  L_det   = det (balanced BCE, ordinary batch) + pair_det (pair images) + strat_det (stratified batch): the same
            detection exposure as CFES-D (configs/rev4/clip_D.yaml); B0 = these three terms only.
The CES pool = every clean labelled image of the step (ordinary + pair fakes + pair reals + stratified). Pair fakes
take their source's cues as the removal baseline (pair_src); other fakes the EMA real-cue mean of their own group
(class mean until the group has an estimate; ces.fake_baseline), reals the EMA fake class mean.
State (ces_state.CESState) is updated AFTER the losses of the step (it is initialised from the first batch before
use), so a step's baselines, subspaces and probe never contain that step's own images except at step 0.
"""
import torch
from torch.utils.data import DataLoader

from . import ces_losses as CL
from . import engine as E
from . import losses as L
from . import mfer as M
from .ces_data import (CESPairTable, CondImageTable, StratifiedCondBatchSampler, condition_codes)
from .ces_model import CUES, CESDetector, load_cfes_backbone
from .ces_state import CESState
from .counterfactuals import BACKGROUND_KINDS, FaceGeometry, sample_views, shared_intervention
from .data import BalancedRealFakeSampler, StratifiedGroupBatchSampler, read_split

N_GROUPS = {"eight": 8, "six": 6}


def _w(cfg, k):
    return float(cfg["loss"].get(k, {}).get("weight", 0.0))


def active_terms(cfg, have_pairs, have_mfer=False):
    """{term: weight} after checking prerequisites (same contract as cfes.engine.active_terms)."""
    demo = bool(cfg["data"].get("demographics_reliable", False))
    strat = bool(cfg.get("strat", {}).get("enabled"))
    on = {"det": _w(cfg, "det"), "sens_ces": _w(cfg, "sens_ces"), "floor": _w(cfg, "floor"), "inv": _w(cfg, "inv")}
    if have_pairs:
        on["pair_det"] = _w(cfg, "pair_det")
    if strat:
        on["strat_det"] = _w(cfg, "strat_det")
    for k in ("floor", "inv", "risk", "fprgap"):
        if _w(cfg, k) > 0 and not demo:
            raise ValueError(f"loss.{k}.weight > 0 but data.demographics_reliable is false")
    if on["inv"] > 0 and cfg["loss"]["inv"].get("source", "feature") == "matched" and not have_mfer:
        raise ValueError("loss.inv.source = matched needs a validated matched-set file (mfer.enabled, mfer.csv)")
    if _w(cfg, "stab") > 0:                                   # CFES-D options (B0 / ablation arms)
        if not cfg["cf"].get("enabled"):
            raise ValueError("loss.stab needs cf.enabled (nuisance views)")
        on["stab"] = _w(cfg, "stab")
    if _w(cfg, "sens") > 0:
        if not have_pairs:
            raise ValueError("loss.sens needs pairs")
        on["sens"] = _w(cfg, "sens")
    for k in ("risk", "fprgap"):
        if _w(cfg, k) > 0:
            if not strat:
                raise ValueError(f"loss.{k} needs the stratified batch (strat.enabled)")
            on[k] = _w(cfg, k)
    if cfg["model"].get("backend", "clip") == "clip":
        on["orth"] = _w(cfg, "orth")
    return {k: float(v) for k, v in on.items() if float(v) > 0}


def build_model(cfg, device):
    """CESDetector; model.init_backbone_ckpt = CFES checkpoint (backbone only), model.init_ckpt = CES checkpoint
    (whole model, strict). Returns (model, ces_state dict from init_ckpt or None)."""
    model = CESDetector(cfg)
    m, st = cfg["model"], None
    if m.get("init_backbone_ckpt"):
        model.backbone_init_keys = load_cfes_backbone(model, m["init_backbone_ckpt"])
    if m.get("init_ckpt"):
        ck = torch.load(m["init_ckpt"], map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=bool(ck.get("full", True)))
        st = (ck.get("state") or {}).get("ces_state")
    return model.to(device), st


def build_state(cfg, model, device, init=None):
    st = CESState(len(CUES), model.dim, N_GROUPS[cfg["data"]["train"].get("scheme", "eight")], cfg).to(device)
    if init is not None and cfg.get("ces", {}).get("load_state", True):
        st.load_full_state(init)
    return st


def face_geometry(cfg, model, device):
    """CFES nuisance-view geometry (only used by the optional CFES-D stability / sensitivity terms)."""
    return FaceGeometry.from_cfg(cfg["cf"], cfg["data"]["input_size"], model.grid, device)


def train_loader(cfg, seed):
    d = cfg["data"]
    df = read_split({**d["train"], "max_items": cfg["debug"].get("max_train_items")})
    cond = condition_codes(df, cfg)
    ds = CondImageTable(df, d["input_size"], cond, d.get("resample", "bicubic"))
    steps = cfg["debug"].get("max_steps_per_epoch") or d.get("steps_per_epoch")
    sampler = BalancedRealFakeSampler(df["label"].values, d["batch_size"], steps=steps, seed=seed)
    dl = DataLoader(ds, batch_size=d["batch_size"], sampler=sampler, num_workers=d["num_workers"], pin_memory=True,
                    drop_last=True, persistent_workers=d["num_workers"] > 0)
    return df, cond, sampler, dl


def pair_loader(cfg, train_df, cond, seed):
    p = cfg["pairs"]
    if not p.get("enabled") or not p.get("csv"):
        return None
    ds = CESPairTable(p["csv"], train_df, cond, cfg["data"]["input_size"], cfg["debug"].get("max_pairs"), seed,
                      cfg["data"].get("resample", "bicubic"), p.get("exclude_families") or ())
    if len(ds) < p["per_step"]:
        return None
    g, persistent = None, cfg["data"]["num_workers"] > 0
    if cfg.get("train", {}).get("decoupled_rng"):
        g, persistent = torch.Generator().manual_seed(int(seed)), False
    return DataLoader(ds, batch_size=p["per_step"], shuffle=True, num_workers=cfg["data"]["num_workers"],
                      drop_last=True, pin_memory=True, persistent_workers=persistent, generator=g)


def strat_loader(cfg, train_df, cond, steps, seed):
    st = cfg.get("strat", {})
    if not st.get("enabled"):
        return None
    d = cfg["data"]
    ds = CondImageTable(train_df, d["input_size"], cond, d.get("resample", "bicubic"))
    if st.get("by_cond", False):
        bs = StratifiedCondBatchSampler(train_df["label"].values, train_df["group"].values, cond, st["per_cell"], steps,
                                        seed)
    else:
        bs = StratifiedGroupBatchSampler(train_df["label"].values, train_df["group"].values, st["per_cell"], steps, seed)
    return DataLoader(ds, batch_sampler=bs, num_workers=d["num_workers"], pin_memory=True,
                      persistent_workers=d["num_workers"] > 0)


def sens_targets(cfg, valid, target):
    """Cue masks for L_sens and L_floor (floor.cues: validated | all)."""
    fc = cfg["loss"].get("floor", {}).get("cues", "validated")
    return valid, target, (valid if fc == "validated" else torch.ones_like(valid))


def compute_ces_losses(model, state, batch, pair_batch, cfg, gen, device, terms, strat_batch=None, mfer_batch=None,
                       fgeom=None, aux=None, valid=None, target=None, fam=None):
    """Returns ({term: unweighted loss}, stats). valid / target: (K,) validated-cue mask and L_sens targets.
    fam: optional (V (n_cond,K), T (n_cond,K)) per-condition masks / targets for L_sens (per-family validation,
    Amendment 2, FF2 only); L_floor keeps the global mask."""
    ces, lc = cfg.get("ces", {}), cfg["loss"]
    x = batch["image"].to(device, non_blocking=True)
    y, g, cond = batch["label"].to(device), batch["group"].to(device), batch["cond"].to(device)
    n = x.shape[0]
    chunks = [x]
    vgen = pgen = gen
    if cfg.get("train", {}).get("decoupled_rng"):
        seeds = torch.randint(2 ** 62, (2,), generator=gen)
        vgen = torch.Generator().manual_seed(int(seeds[0]))
        pgen = torch.Generator().manual_seed(int(seeds[1]))
    # CES draws (random-cue control, floor group shuffle, inv partners) use their own streams, drawn in EVERY arm, so a
    # control changes only its own quantity: the other arms' partners and views are identical draw for draw
    rgen, fgen, igen = (torch.Generator().manual_seed(int(v)) for v in torch.randint(2 ** 62, (3,), generator=gen))
    views = "stab" in terms
    kind = None
    if views:
        x_cf, kind = sample_views(x, y, cfg["cf"], fgeom, vgen)
        chunks.append(x_cf)
    npair = 0
    if pair_batch is not None:                      # always forwarded when pairs exist: identical exposure in all arms
        f = pair_batch["fake"].to(device, non_blocking=True)
        r = pair_batch["real"].to(device, non_blocking=True)
        if cfg["pairs"].get("shared_intervention", False):
            f, r, _ = shared_intervention(f, r, cfg["cf"], fgeom, pgen, donor_pool=x[y == 0])
        chunks += [f, r]
        npair = f.shape[0]
    nst = 0
    if strat_batch is not None:
        chunks.append(strat_batch["image"].to(device, non_blocking=True))
        nst = chunks[-1].shape[0]
    nm = 0
    if mfer_batch is not None:
        chunks.append(M.flatten_for_forward(mfer_batch).to(device, non_blocking=True))
        nm = chunks[-1].shape[0]
    xin = torch.cat(chunks)
    if not torch.isfinite(xin).all():
        raise FloatingPointError("non-finite input pixels")
    amp = bool(cfg["optim"]["amp"]) and device.type == "cuda"
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
        out = model(xin)
    S, Call = out["score"], out["cues"]

    # ---- slices and the CES pool -------------------------------------------------------------------------------
    off = n
    s_view = None
    if views:
        s_view = S[off:off + n]
        off += n
    pool_C, pool_s, pool_y, pool_g, pool_c = [Call[:n]], [S[:n]], [y], [g], [cond]
    pair_src = [torch.full((n,), -1, dtype=torch.long, device=device)]
    s_pf = s_pr = None
    if npair:
        s_pf, s_pr = S[off:off + npair], S[off + npair:off + 2 * npair]
        pool_C += [Call[off:off + npair], Call[off + npair:off + 2 * npair]]
        pool_s += [s_pf, s_pr]
        pool_y += [torch.ones(npair, dtype=torch.long, device=device), torch.zeros(npair, dtype=torch.long, device=device)]
        pool_g += [pair_batch["fake_group"].to(device), pair_batch["real_group"].to(device)]
        pool_c += [pair_batch["fake_cond"].to(device), pair_batch["real_cond"].to(device)]
        base = n                                                     # pool index of the first pair fake
        pair_src += [torch.arange(npair, device=device) + base + npair, torch.full((npair,), -1, dtype=torch.long,
                                                                                  device=device)]
        off += 2 * npair
    s_st = ys = gs = None
    if nst:
        s_st = S[off:off + nst]
        ys, gs = strat_batch["label"].to(device), strat_batch["group"].to(device)
        pool_C.append(Call[off:off + nst]); pool_s.append(s_st); pool_y.append(ys); pool_g.append(gs)
        pool_c.append(strat_batch["cond"].to(device)); pair_src.append(torch.full((nst,), -1, dtype=torch.long,
                                                                                device=device))
        off += nst
    s_m = S[off:off + nm] if nm else None
    Cp, sp, yp, gp, cp = (torch.cat(v) for v in (pool_C, pool_s, pool_y, pool_g, pool_c))
    psrc = torch.cat(pair_src)

    vals, stats = {}, {}
    vals["det"], _ = L.detection_loss(S[:n], y, lc["det"].get("mode", "balanced"))
    if "pair_det" in terms and npair:
        vals["pair_det"], _ = L.detection_loss(torch.cat([s_pf, s_pr]), torch.cat([pool_y[1], pool_y[2]]),
                                               lc["det"].get("mode", "balanced"))
    if "strat_det" in terms and nst:
        vals["strat_det"], _ = L.detection_loss(s_st, ys, lc["det"].get("mode", "balanced"))

    # ---- utilities, L_sens, L_floor ----------------------------------------------------------------------------
    if valid is None:
        valid, target = torch.ones(len(CUES), dtype=torch.bool), torch.full((len(CUES),), 0.5)
    valid, target, floor_mask = sens_targets(cfg, valid, target)
    if not state.ready():
        state.update(Cp, yp, gp)                                     # step 0: baselines from the first batch
    R = CL.replacement(Cp, yp, state, psrc, ces.get("replace", "counterfactual"), rgen, g=gp,
                       fake_baseline=ces.get("fake_baseline", "group"))
    U, _ = CL.utilities(model, Cp, sp, yp, R, ces)
    if "sens_ces" in terms:
        if fam is not None:
            vals["sens_ces"], s = CL.sens_loss(U, fam[0].to(device)[cp], fam[1].to(device)[cp])
        else:
            vals["sens_ces"], s = CL.sens_loss(U, valid, target)
        stats.update(s)
    if "floor" in terms:
        vals["floor"], s = CL.floor_loss(U, yp, gp, cp, floor_mask, lc["floor"], fgen)
        stats.update(s)

    # ---- L_inv ---------------------------------------------------------------------------------------------------
    ic = lc.get("inv", {})
    if "inv" in terms:
        if ic.get("source", "feature") == "matched":
            msk = mfer_batch["mask"].to(device)
            sf, sr = M.unflatten_scores(s_m, msk)
            vals["inv"], s = CL.matched_inv_loss(sf, sr, msk, ic)
            stats.update(s)
        else:
            partner, has = CL.demographic_partner(yp, gp, igen)
            ready = int(state.steps) >= int(ic.get("warmup_steps", 200)) and bool(state.U_ok.any())
            stats["inv_ready"] = float(ready)
            if ready:
                Cs = CL.demographic_swap(Cp, state, partner, has, bool(ic.get("detach_donor", True)))
                s_sw = model.score_from(Cs)
                ok = CL.swap_validity(state, Cs, gp, partner, has)
                use = ok if ic.get("validity_mask", True) else has
                vals["inv"], s = CL.inv_loss(sp, s_sw, use, ic)
                stats.update(s)
                stats["inv_valid_rate"] = float(ok[has].float().mean()) if has.any() else 0.0
                for gg in range(state.G):
                    sel = has & (gp == gg)
                    if sel.any():
                        stats[f"inv_valid_g{gg}"] = float(ok[sel].float().mean())
            else:
                vals["inv"] = Cp.sum() * 0.0

    # ---- optional CFES-D terms -------------------------------------------------------------------------------------
    if "stab" in terms:
        vals["stab"], s = L.stability_loss({"score": S[:n], "patch": None}, {"score": s_view, "patch": None}, kind,
                                           fgeom, lc["stab"], BACKGROUND_KINDS)
        stats.update(s)
    if "sens" in terms:
        fr = pair_batch["frac"].to(device)
        al = pair_batch["aligned"].to(device)
        vals["sens"], s = L.sensitivity_loss({"score": s_pf, "patch": None}, {"score": s_pr, "patch": None}, fr, al,
                                             fgeom, lc["sens"])
        stats.update(s)
    for k in ("risk", "fprgap"):
        if k not in terms:
            continue
        r_s, r_y, r_g = [S[:n], s_st], [y, ys], [g, gs]
        if views and lc[k].get("include_views", True):
            r_s.append(s_view); r_y.append(y); r_g.append(g)
        r_s, r_y, r_g = torch.cat(r_s), torch.cat(r_y), torch.cat(r_g)
        if k == "risk":
            vals[k], s = L.real_group_risk_loss(r_s, r_y, r_g, lc["risk"])
        else:
            rq = (aux or {}).get("fprgap_rq") or E.fprgap_quantile(cfg)
            vals[k], s = L.fpr_gap_loss(r_s, r_y, r_g, lc["fprgap"], rq)
        stats.update(s)
    if "orth" in terms:
        vals["orth"] = model.orthogonality_loss()

    # ---- state update (no grad) and diagnostics --------------------------------------------------------------------
    with torch.no_grad():
        lg = state.probe_logits(Cp)
        stats["probe_acc"] = float((lg.argmax(1) == gp).float().mean())
        stats["train_acc"] = float(((S[:n] > 0).long() == y).float().mean())
        stats["score_absmean"] = float(S[:n].abs().mean())
        for k, name in enumerate(CUES):
            for c, cname in ((0, "real"), (1, "fake")):
                sel = yp == c
                if sel.any():
                    stats[f"u_{name}_{cname}"] = float(U[sel, k].mean())
        if out.get("s_k") is not None:
            for k, name in enumerate(CUES):
                stats[f"s_{name}_absmean"] = float(out["s_k"][:n, k].abs().mean())
        stats["U_ok"] = int(state.U_ok.sum())
    if model.training:
        state.update(Cp, yp, gp)
        stats["probe_loss"] = state.probe_step(Cp, gp)
    E.check_terms(vals, terms)
    return vals, stats
