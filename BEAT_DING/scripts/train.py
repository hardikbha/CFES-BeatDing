"""Train CFES (or any ablation of it) on one dataset.

  python scripts/train.py --config configs/fairforge2.yaml --out runs/ff2_cfes_s7 --set seed=7
  python scripts/train.py --config configs/fairforge2.yaml --out runs/ff2_cfes_s7 --resume     # continue
  python scripts/train.py --config configs/debug.yaml --out runs/debug                         # tiny smoke run
  python scripts/train.py --config configs/mfer_min_xception.yaml --out runs_mfer/smoke_x --smoke \
         --set mfer.csv=data/UNVALIDATED_fixture_ffpp_reenact_reciprocal.csv            # code-path fixture only

Model selection uses ONE fixed rule on the validation split (cfg.select.metric); test sets are never read here.
--smoke: allows a matched-set file whose sidecar status is smoke_only; the output directory must be named smoke_*.
Every run writes provenance.jsonl (code hash, manifest and checkpoint sha256, resolved config hash, seed, command).
Resume: checkpoints are written at epoch ends and restore model, optimiser, scheduler and the global RNGs; the
balanced sampler and the matched-set order are re-seeded per epoch. Resumption is therefore exact for configs
without counterfactual views or pairs (e.g. det-only, minimal MFER). With views or pairs it is not bit-exact: the
view generator is seeded once per process start and the pair loader's shuffle continues across epochs, unless
train.decoupled_rng is true (study configs): then views, pair interventions and pair order restart from (seed, epoch).
"""
import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cfes import engine  # noqa: E402
from cfes.config import dump_config, load_config  # noqa: E402
from cfes.data import cycle  # noqa: E402
from cfes.metrics import evaluate_scores, rank_prob, threshold_at_tpr  # noqa: E402
from cfes.matchcheck import sha256_file  # noqa: E402
from cfes.utils import JsonlLogger, code_hash, load_checkpoint, save_checkpoint, set_seed  # noqa: E402

GROUPS = (("patch_head", ("patch_head.",)), ("global_head", ("cls_head.", "net.last_linear.", "net.fc.")),
          ("fusion", ("alpha", "alpha_raw")), ("reliability", ("rel_head.",)),
          ("multilayer", ("ml_heads.", "ml_gate_raw")), ("backbone", ("",)))


def grad_norms(model):
    """L2 gradient norm per parameter group (before clipping); groups are matched by name prefix, first match wins."""
    acc = {name: 0.0 for name, _ in GROUPS}
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        for name, prefixes in GROUPS:
            if any(n == pre or (pre and n.startswith(pre)) or pre == "" for pre in prefixes):
                acc[name] += float(p.grad.detach().float().pow(2).sum())
                break
    return {f"gn_{k}": v ** 0.5 for k, v in acc.items()}


def provenance(cfg, a, out):
    def h(p):
        return {"path": p, "sha256": sha256_file(p)} if p and os.path.exists(p) else {"path": p, "sha256": None}

    def csvs(spec):
        c = spec["csv"]
        return [h(x) for x in (c if isinstance(c, list) else [c])]
    m = cfg["model"]
    rec = {"code_hash": code_hash(ROOT), "argv": sys.argv, "seed": cfg["seed"], "smoke": bool(a.smoke),
           "resume": bool(a.resume), "host": platform.node(), "python": platform.python_version(),
           "torch": torch.__version__,
           "config_sha256": hashlib.sha256(open(os.path.join(out, "config.yaml"), "rb").read()).hexdigest(),
           "train_manifest": csvs(cfg["data"]["train"]), "val_manifest": csvs(cfg["data"]["val"]),
           "pairs_csv": h(cfg["pairs"]["csv"]) if cfg["pairs"].get("enabled") else None,
           "init_ckpt": h(m.get("init_ckpt")), "init_from": h(m.get("init_from")),
           "xception_init_from": h(m.get("xception", {}).get("init_from")) if m.get("backend") == "xception" else None,
           "resnet_weights": h(m.get("resnet", {}).get("weights")) if m.get("backend") == "resnet" else None}
    if m.get("backend") == "fsvfm":
        rec["fsvfm_weights"] = h(m.get("fsvfm", {}).get("weights"))
    mf = cfg.get("mfer", {})
    if mf.get("enabled"):
        side = mf["csv"] + ".validation.json"
        rec["mfer_csv"] = h(mf["csv"])
        rec["mfer_sidecar_status"] = json.load(open(side))["status"] if os.path.exists(side) else None
    return rec


def oppoint_stats(y, z, g, scheme):
    """Review 4 item 5: operating point from ONE common intercept fitted on validation logits (class-balanced log loss,
    scripts/calibrate.py); pooled and per-intersectional-group FPR / TPR at z + b >= 0 (groups with both classes)."""
    from calibrate import fit_intercept
    from cfes.metrics import SCHEMES
    b = fit_intercept(z, y)[0]
    d = (z + b) >= 0
    fpr, tpr = {}, {}
    for k, name in enumerate(SCHEMES[scheme]):
        r, f = (g == k) & (y == 0), (g == k) & (y == 1)
        if r.any() and f.any():
            fpr[name], tpr[name] = float(100 * d[r].mean()), float(100 * d[f].mean())
    return {"intercept": float(b), "pooled_FPR": float(100 * d[y == 0].mean()), "pooled_TPR": float(100 * d[y == 1].mean()),
            "group_FPR": fpr, "group_TPR": tpr, "worst_FPR": max(fpr.values()),
            "fpr_gap": max(fpr.values()) - min(fpr.values()), "tpr_gap": max(tpr.values()) - min(tpr.values())}


def validate(model, cfg, device, val_df, val_loader):
    """Codex review F3 (25 Oct; as agents/CESEL/scripts/train.py): with eval.rank_metrics the validation AUC, group AUCs
    and ding@0.5 block are computed from rank_prob(logit) (exact, no sigmoid saturation: float64 sigmoid ties every logit
    above ~37); 0.5 decisions are identical (z >= 0). The validation-TPR threshold and the operating point (common
    intercept, pooled / group FPR and TPR) are on the LOGIT scale. Before this fix validation used sigmoid probabilities."""
    p, z = engine.predict(model, val_loader, device, amp=bool(cfg["optim"]["amp"]), return_logits=True, prob64=bool(cfg["eval"].get("prob_float64")))
    y, g = val_df["label"].values[:len(p)], val_df["group"].values[:len(p)]
    z64 = np.asarray(z, dtype=np.float64)
    rank = bool(cfg["eval"].get("rank_metrics"))
    res = evaluate_scores(y, rank_prob(z64) if rank else p, g, cfg["data"]["val"].get("scheme", "eight"))
    res["metric_scores"] = "rank_prob(logit)" if rank else "prob"
    res["threshold_tpr"] = threshold_at_tpr(y, z64, cfg["eval"]["val_tpr"])
    res["threshold_tpr_units"] = "logit"
    res["val_stats"] = {"fpr@0.5": float(100 * (p[y == 0] >= 0.5).mean()), "tpr@0.5": float(100 * (p[y == 1] >= 0.5).mean()),
                        "logit_mean_real": float(z[y == 0].mean()), "logit_mean_fake": float(z[y == 1].mean()),
                        "logit_absmean": float(np.abs(z).mean()), "logit_std": float(z.std())}
    res["_probs"] = p
    res["oppoint"] = oppoint_stats(y, z, g, cfg["data"]["val"].get("scheme", "eight"))
    if cfg.get("bd", {}).get("log_val_ding", True) and len(np.unique(y)) == 2:
        # Beat-Ding diagnostic only (never a selection input): val Ding-score at the matched val TPR (plan section 4)
        from cfes import bd_metrics as BM
        try:
            vd = BM.val_ding_score(z64, y, g)
            res["val_ding"] = {"score": vd["score"], "wins": vd["wins"], "t": vd["t"], "val_FPR": vd["val_FPR"],
                               "cols": vd["cols"]}
        except (OSError, ValueError, IndexError) as e:      # tiny debug val sets may lack the two classes in a group
            res["val_ding"] = {"error": repr(e)}
    return res


OPP = {"min_tpr": 80.0, "max_fpr": 20.0, "tpr_drop": 2.0, "gap_weight": 0.5}     # STUDY_PLAN Amendment 9


def selection_value(res, metric, ref=None):
    if metric == "val_oppoint":
        o = res["oppoint"]
        ok = o["pooled_TPR"] >= OPP["min_tpr"] and o["pooled_FPR"] <= OPP["max_fpr"]
        if ref is not None:
            ok = ok and o["worst_FPR"] <= ref["worst_FPR"] and all(
                o["group_TPR"].get(k, 0.0) >= v - OPP["tpr_drop"] for k, v in ref["group_TPR"].items())
        return (res["auc"] - OPP["gap_weight"] * o["fpr_gap"]) if ok else -1e9      # res["auc"] is in %
    if metric == "val_auc":
        return res["auc"]
    if metric == "val_auc_minus_inter_ffpr":
        return res["auc"] - res["ding@0.5"]["intersection"].get("F_FPR", 0.0)
    if metric == "val_inter_es_auc":
        return res["ding@0.5"]["intersection"]["es_AUC"]
    if metric == "val_auc_plus_inter_es_auc":
        # Beat-Ding plan section 4, within-run checkpoint selection: val AUC + val intersectional es-AUC (both in %, both
        # threshold-free; rank_prob(logit) metrics under eval.rank_metrics, Codex F3)
        return res["auc"] + res["ding@0.5"]["intersection"]["es_AUC"]
    raise ValueError(metric)


def term_grad_norms(model, vals, terms):
    """Gradient-norm contribution of each weighted loss term w.r.t. the trainable parameters (diagnostic; needs the
    graph, so it is computed before the main backward with retain_graph)."""
    params = model.trainable_parameters()
    out = {}
    for k, v in vals.items():
        if k not in terms:
            continue
        gs = torch.autograd.grad(terms[k] * v, params, retain_graph=True, allow_unused=True)
        out[f"tg_{k}"] = float(torch.sqrt(sum((x.float() ** 2).sum() for x in gs if x is not None)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. loss.stab.weight=0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="code-path run; allows smoke_only matched-set fixtures")
    a = ap.parse_args()

    if a.smoke:
        if not os.path.basename(os.path.normpath(a.out)).startswith("smoke_"):
            raise SystemExit("--smoke runs must write to a directory named smoke_*")
        a.set = list(a.set) + ["mfer.smoke=true"]
    cfg = load_config(a.config, a.set)
    if cfg.get("mfer", {}).get("smoke") and not a.smoke:
        raise SystemExit("mfer.smoke may only be set through --smoke")
    os.makedirs(a.out, exist_ok=True)
    if not a.resume and os.path.exists(os.path.join(a.out, "last.pt")):
        raise SystemExit(f"{a.out} already has last.pt; pass --resume or use a new --out")
    dump_config(cfg, os.path.join(a.out, "config.yaml"))
    JsonlLogger(os.path.join(a.out, "provenance.jsonl")).log(**provenance(cfg, a, a.out))
    log = JsonlLogger(os.path.join(a.out, "train_log.jsonl"))
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = engine.build_model(cfg, device)
    geom = engine.build_geometry(cfg, model, device)
    geom_cpu = engine.build_geometry(cfg, model, "cpu")
    train_df, sampler, loader = engine.train_loader(cfg, cfg["seed"])
    pairs = engine.pair_loader(cfg, geom_cpu, cfg["seed"])
    terms = engine.active_terms(cfg, have_pairs=pairs is not None)
    mfer_dl, mfer_rep = engine.mfer_loader(cfg, cfg["seed"])
    strat_dl = engine.strat_loader(cfg, train_df, len(loader), cfg["seed"])
    val_df, val_loader = engine.eval_loader(cfg["data"]["val"], cfg, cfg["debug"].get("max_eval_items"))

    o = cfg["optim"]
    if o.get("name", "adamw") == "sgd":
        opt = torch.optim.SGD(model.trainable_parameters(), lr=o["lr"], momentum=o.get("momentum", 0.9),
                              weight_decay=o["weight_decay"])
    elif o.get("layer_decay") is not None:
        # layer-wise lr decay (FS-VFM/MAE fine-tuning recipe; only configs that set optim.layer_decay)
        groups = model.optim_param_groups(o["weight_decay"], o["layer_decay"])
        for g in groups:
            g["lr"] = o["lr"] * g["lr_scale"]
        opt = torch.optim.AdamW(groups, lr=o["lr"])
    else:
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    steps_total = o["epochs"] * len(loader)
    # warmup is capped at 10% of training unless optim.warmup_max_frac says otherwise (FS-VFM recipe: 5 of 10 epochs)
    cap = steps_total // 10 if "warmup_max_frac" not in o else int(round(steps_total * o["warmup_max_frac"]))
    warm = max(1, min(o["warmup_steps"], max(1, cap)))
    r_min = float(o.get("min_lr_ratio", 0.0))       # cosine floor as a fraction of the peak lr (FS-VFM: 1e-6 / lr)

    def cos_factor(s):
        c = 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps_total - warm)))
        return r_min + (1 - r_min) * c if r_min > 0 else c
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        (cos_factor(s) if o["schedule"] == "cosine" else 1.0))

    state = {"epoch": 0, "step": 0, "best": -float("inf"), "best_epoch": -1}
    aux = {"fprgap_rq": engine.fprgap_quantile(cfg)} if "fprgap" in terms else {}     # arm E running threshold
    if "rel_sup" in terms:          # C-FER R4: out-of-fold reliability targets aligned to train_df rows
        aux["rel_targets"] = engine.reliability_targets(cfg, train_df)
    if a.resume:
        state = load_checkpoint(os.path.join(a.out, "last.pt"), model, opt, sched, None, restore_rng=True)
        if "fprgap_buf" in state:
            buf = state.pop("fprgap_buf")
            if "fprgap_rq" in aux:
                aux["fprgap_rq"].load(buf)
        print(f"resumed at epoch {state['epoch']} step {state['step']}")

    log.log(event="start", code_hash=code_hash(ROOT), device=str(device), terms=terms,
            params=model.parameter_report(), n_train=len(train_df), n_val=len(val_df),
            n_pairs=len(pairs.dataset) if pairs else 0, steps_per_epoch=len(loader), resume=a.resume,
            mfer_matched_sets=mfer_rep, alpha_conversion=getattr(model, "alpha_conversion", None),
            rel_start_conversion=getattr(model, "rel_start_conversion", None),
            strat_cells=strat_dl.batch_sampler.describe() if strat_dl else None)
    print("terms", terms, "params", model.parameter_report(), flush=True)

    if cfg["select"].get("include_init") and not a.resume:
        # the starting checkpoint is an eligible candidate of the selection rule (epoch -1)
        res = validate(model, cfg, device, val_df, val_loader)
        val = selection_value(res, cfg["select"]["metric"])
        state["select_ref"] = res["oppoint"]
        state.update(best=val, best_epoch=-1, best_threshold_tpr=res["threshold_tpr"])
        save_checkpoint(os.path.join(a.out, "best.pt"), model, None, None, None, dict(state), full=cfg["ckpt"]["full"])
        log.log(event="epoch", epoch=-1, seconds=0.0, select_value=val, improved=True, val_auc=res["auc"],
                val_ding=res["ding@0.5"], val_stats=res["val_stats"], val_oppoint=res["oppoint"], threshold_tpr=res["threshold_tpr"],
                val_ding_matched=res.get("val_ding"))
        print(f"epoch -1 (start) val_auc {res['auc']:.2f} select {val:.3f}", flush=True)
    gen = torch.Generator().manual_seed(cfg["seed"] * 7919 + state["epoch"])
    pair_iter = cycle(pairs) if pairs else None
    stop_at = cfg["debug"].get("stop_after_epochs")
    model.train()
    for epoch in range(state["epoch"], o["epochs"]):
        if stop_at is not None and epoch >= stop_at:
            print(f"debug.stop_after_epochs={stop_at}: stopping (simulated interruption)", flush=True)
            return
        sampler.set_epoch(epoch)
        if cfg.get("train", {}).get("decoupled_rng"):
            # every random stream restarts from (seed, epoch): resuming at an epoch boundary is exact
            gen.manual_seed(cfg["seed"] * 7919 + epoch)
            if pairs is not None:
                pairs.generator.manual_seed(cfg["seed"] * 104729 + epoch)
                if hasattr(pairs.sampler, "set_epoch"):        # Beat-Ding A: epoch-tagged pair keys
                    pairs.sampler.set_epoch(epoch)
                pair_iter = cycle(pairs)
        strat_iter = None
        if strat_dl is not None:
            strat_dl.batch_sampler.set_epoch(epoch)
            strat_iter = iter(strat_dl)
        mfer_iter = None
        if mfer_dl is not None:
            mfer_dl.generator.manual_seed(engine.mfer_epoch_seed(cfg["seed"], epoch))
            mfer_iter = cycle(mfer_dl)
        t0, agg = time.time(), {}
        t_step = time.time()
        for it, batch in enumerate(loader):
            pb = next(pair_iter) if pair_iter else None
            mb = next(mfer_iter) if mfer_iter else None
            sb = next(strat_iter) if strat_iter else None
            vals, stats = engine.compute_losses(model, batch, pb, cfg, geom, gen, device, terms, mfer_batch=mb,
                                                strat_batch=sb, aux=aux)
            total = engine.weighted_total(vals, terms)
            opt.zero_grad(set_to_none=True)
            log_now = (state["step"] + 1) % cfg["log_every"] == 0
            tgs = term_grad_norms(model, vals, terms) if (log_now and cfg["debug"].get("term_grad_norms")) else {}
            total.backward()
            gns = grad_norms(model) if log_now else {}
            gns.update(tgs)
            gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), o["grad_clip"])
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {state['step']}: {vals}")
            opt.step()
            sched.step()
            state["step"] += 1
            rec = {k: float(v) for k, v in vals.items()}
            rec.update(stats, total=float(total), grad_norm=float(gn), lr=sched.get_last_lr()[0], **gns)
            rec["sec_step"] = time.time() - t_step          # wall time of this step incl. data wait (smoke timing)
            t_step = time.time()
            for k, v in rec.items():
                agg.setdefault(k, []).append(v)
            if state["step"] % cfg["log_every"] == 0:
                log.log(event="step", epoch=epoch, step=state["step"], **rec)
                print(f"ep {epoch} it {it + 1}/{len(loader)} " +
                      " ".join(f"{k}={rec[k]:.4f}" for k in ["total", *vals.keys()]), flush=True)
        res = validate(model, cfg, device, val_df, val_loader)
        if cfg["debug"].get("save_val_probs"):
            np.save(os.path.join(a.out, f"val_probs_epoch{epoch}.npy"), res["_probs"])
        val = selection_value(res, cfg["select"]["metric"],         # ref_checks false: TTD only (cfer spec Amendment 4)
                              state.get("select_ref") if cfg["select"].get("ref_checks", True) else None)
        state["epoch"] = epoch + 1
        improved = val > state["best"]
        if improved:
            state.update(best=val, best_epoch=epoch, best_threshold_tpr=res["threshold_tpr"])
            save_checkpoint(os.path.join(a.out, "best.pt"), model, None, None, None, dict(state),
                            full=cfg["ckpt"]["full"])
        st_last = dict(state, fprgap_buf=aux["fprgap_rq"].state()) if "fprgap_rq" in aux else dict(state)
        save_checkpoint(os.path.join(a.out, "last.pt"), model, opt, sched, None, st_last, full=cfg["ckpt"]["full"])
        peak = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else 0.0
        log.log(event="epoch", epoch=epoch, seconds=time.time() - t0, peak_mem_gb=peak, select_value=val, improved=improved,
                val_auc=res["auc"], val_ding=res["ding@0.5"], val_stats=res["val_stats"], val_oppoint=res["oppoint"], threshold_tpr=res["threshold_tpr"],
                val_ding_matched=res.get("val_ding"), train_means={k: float(np.mean(v)) for k, v in agg.items()},
                peak_reserved_gb=torch.cuda.max_memory_reserved() / 2 ** 30 if device.type == "cuda" else 0.0)
        tm = {k: float(np.mean(v)) for k, v in agg.items() if k in vals}
        print(f"epoch {epoch} val_auc {res['auc']:.2f} select {val:.3f} best_epoch {state['best_epoch']} "
              f"sec {time.time() - t0:.0f} peak_mem_gb {peak:.1f} train " +
              " ".join(f"{k}={v:.3f}" for k, v in tm.items()), flush=True)
    with open(os.path.join(a.out, "TRAINED"), "w") as f:
        json.dump(state, f, default=float)


if __name__ == "__main__":
    main()
