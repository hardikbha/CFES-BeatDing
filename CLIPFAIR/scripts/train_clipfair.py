"""Train one CLIPFAIR baseline (Amendment 3). Adapted from scripts/train_ces.py (the CESel trainer, copied unchanged
next to this file): same data stream as B0 (ces_engine loaders, same seed), same optimiser recipe (AdamW, adapters lr
1e-4, new modules lr 1e-3, wd 4e-3, warmup 200, cosine, grad clip 1.0, bf16), same validation (train.validate), same
Amendment-9 selection rule (start checkpoint eligible), same checkpoint format (full state dict). PG-FDD wraps every
step in SAM (two forward/backward passes, rho 0.05), as its official train.py:223-242.

  python scripts/train_clipfair.py --config configs/ffpp/daw.yaml --out runs/ffpp_daw_s7 --set seed=7 \
         model.init_backbone_ckpt=<CFES runs/clip_s7_p1/last.pt>
  (ding: first scripts/ding_sfd.py, then --set clipfair.ding.channels_json=<sfd_channels.json>)
Test sets are never read here.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import train as T  # noqa: E402  (CFES trainer: validate, selection_value, provenance)
from cfes import ces_engine as CE  # noqa: E402
from cfes import clipfair_engine as FE  # noqa: E402
from cfes import engine as E  # noqa: E402
from cfes.clipfair_models import build_model  # noqa: E402
from cfes.config import dump_config, load_config  # noqa: E402
from cfes.data import cycle  # noqa: E402
from cfes.utils import JsonlLogger, code_hash, load_checkpoint, save_checkpoint, set_seed  # noqa: E402


def grad_norms(model):
    acc = {"backbone": 0.0, "new": 0.0}
    for n, p in model.named_parameters():
        if p.grad is not None:
            acc["backbone" if n.startswith("backbone.") else "new"] += float(p.grad.detach().float().pow(2).sum())
    return {f"gn_{k}": v ** 0.5 for k, v in acc.items()}


def select_epoch(model, cfg, out, state, epoch, res):
    """Amendment 9 selection plus Amendment 7's trained-only pooled candidate."""
    metric = cfg["select"]["metric"]
    val = T.selection_value(res, metric,
                            state.get("select_ref") if cfg["select"].get("ref_checks", True) else None)
    if epoch >= 0 and metric == "val_oppoint":
        # Eligibility is independent of whether this epoch beats the start's score.
        if val > -1e9:
            state["trained_eligible_epochs"] = state.get("trained_eligible_epochs", 0) + 1
        pooled = T.selection_value(res, metric, ref=None)
        if pooled > -1e9 and pooled > state.get("best_pooled", -float("inf")):
            state.update(best_pooled=pooled, best_pooled_epoch=epoch,
                         best_pooled_threshold_tpr=res["threshold_tpr"])
            pooled_state = dict(state, best=pooled, best_epoch=epoch,
                                best_threshold_tpr=res["threshold_tpr"])
            save_checkpoint(os.path.join(out, "best_pooled.pt"), model, None, None, None,
                            pooled_state, full=cfg["ckpt"]["full"])
    improved = val > state["best"]
    if improved:
        state.update(best=val, best_epoch=epoch, best_threshold_tpr=res["threshold_tpr"])
        save_checkpoint(os.path.join(out, "best.pt"), model, None, None, None, state, full=cfg["ckpt"]["full"])
    return val, improved


def finalize_selection(out, state, metric):
    """Keep best.pt as the evaluation entry point; disclose fallback in TRAINED."""
    state["selection_fallback"] = False
    if metric == "val_oppoint" and not state.get("trained_eligible_epochs", 0):
        if state.get("best_pooled_epoch", -1) >= 0:
            shutil.copyfile(os.path.join(out, "best_pooled.pt"), os.path.join(out, "best.pt"))
            state.update(selection_fallback=True, best=state["best_pooled"],
                         best_epoch=state["best_pooled_epoch"],
                         best_threshold_tpr=state["best_pooled_threshold_tpr"])
        else:
            state["selection_fallback"] = "none_eligible"
    with open(os.path.join(out, "TRAINED"), "w") as f:
        json.dump(state, f, default=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    method = cfg["clipfair"]["method"]
    if method == "ding" and not cfg["clipfair"]["ding"].get("channels_json"):
        raise SystemExit("ding needs clipfair.ding.channels_json (scripts/ding_sfd.py output)")
    if method == "ding" and cfg["clipfair"]["ding"].get("gda_grad") and not os.environ.get("CLIPFAIR_ALLOW_GDA_GRAD"):
        raise SystemExit("gda_grad=true is the gradient-enabled variant; not run unless the decision file is amended "
                         "(set CLIPFAIR_ALLOW_GDA_GRAD=1 only then)")
    os.makedirs(a.out, exist_ok=True)
    if not a.resume and os.path.exists(os.path.join(a.out, "last.pt")):
        raise SystemExit(f"{a.out} already has last.pt; pass --resume or use a new --out")
    dump_config(cfg, os.path.join(a.out, "config.yaml"))
    JsonlLogger(os.path.join(a.out, "provenance.jsonl")).log(**T.provenance(cfg, a, a.out))
    log = JsonlLogger(os.path.join(a.out, "train_log.jsonl"))
    set_seed(cfg["seed"])
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = build_model(cfg, device)
    train_df, cond, sampler, loader = CE.train_loader(cfg, cfg["seed"])
    pairs = CE.pair_loader(cfg, train_df, cond, cfg["seed"])
    strat_dl = CE.strat_loader(cfg, train_df, cond, len(loader), cfg["seed"])
    spe = FE.SpeLookup(train_df, pairs.dataset if pairs is not None else None)
    terms = FE.active_terms(cfg)
    val_df, val_loader = E.eval_loader(cfg["data"]["val"], cfg, cfg["debug"].get("max_eval_items"))
    sam = cfg["clipfair"]["pgfdd"]["sam"] if method == "pgfdd" else None

    o = cfg["optim"]
    opt = torch.optim.AdamW(model.param_groups(o["lr"], o.get("lr_new", o["lr"]), o["weight_decay"]))
    steps_total = o["epochs"] * len(loader)
    warm = max(1, min(o["warmup_steps"], max(1, steps_total // 10)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              (0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps_total - warm)))
                                               if o["schedule"] == "cosine" else 1.0))
    state = {"epoch": 0, "step": 0, "best": -float("inf"), "best_epoch": -1,
             "trained_eligible_epochs": 0, "best_pooled": -float("inf"), "best_pooled_epoch": -1}
    if a.resume:
        state = load_checkpoint(os.path.join(a.out, "last.pt"), model, opt, sched, None, restore_rng=True)
        if cfg["select"]["metric"] == "val_oppoint" and "trained_eligible_epochs" not in state:
            raise SystemExit("Pre-Amendment-7 checkpoint lacks pooled history; restart in a new --out")
    log.log(event="start", method=method, code_hash=code_hash(ROOT), device=str(device), terms=terms,
            params=model.parameter_report(), init=getattr(model, "init_info", None), n_train=len(train_df),
            n_val=len(val_df), steps_per_epoch=len(loader), pairs=pairs.dataset.report if pairs else None,
            strat_cells=strat_dl.batch_sampler.describe() if strat_dl else None, sam=sam,
            clipfair=cfg["clipfair"][method])
    print("method", method, "terms", terms, "params", model.parameter_report(), flush=True)

    if cfg["select"].get("include_init") and not a.resume:
        res = T.validate(model, cfg, device, val_df, val_loader)
        val = T.selection_value(res, cfg["select"]["metric"])
        state["select_ref"] = res["oppoint"]
        state.update(best=val, best_epoch=-1, best_threshold_tpr=res["threshold_tpr"])
        save_checkpoint(os.path.join(a.out, "best.pt"), model, None, None, None, state, full=cfg["ckpt"]["full"])
        log.log(event="epoch", epoch=-1, select_value=val, val_auc=res["auc"], val_oppoint=res["oppoint"])
    gen = torch.Generator().manual_seed(cfg["seed"] * 7919 + state["epoch"])
    pair_iter = cycle(pairs) if pairs else None
    params = model.trainable_parameters()
    model.train()
    for epoch in range(state["epoch"], o["epochs"]):
        sampler.set_epoch(epoch)
        if cfg.get("train", {}).get("decoupled_rng"):
            gen.manual_seed(cfg["seed"] * 7919 + epoch)
            if pairs is not None:
                pairs.generator.manual_seed(cfg["seed"] * 104729 + epoch)
                pair_iter = cycle(pairs)
        strat_iter = None
        if strat_dl is not None:
            strat_dl.batch_sampler.set_epoch(epoch)
            strat_iter = iter(strat_dl)
        t0, agg = time.time(), {}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t_it = time.time()
        for it, batch in enumerate(loader):
            pb = next(pair_iter) if pair_iter else None
            sb = next(strat_iter) if strat_iter else None
            pool = FE.assemble(batch, pb, sb, device, spe)
            if sam is not None:
                FE.enable_running_stats(model)                       # FG train.py:225
            vals, stats = FE.compute_losses(model, pool, cfg, gen, device, terms)
            total = FE.weighted_total(vals, terms)
            opt.zero_grad(set_to_none=True)
            total.backward()
            log_now = (state["step"] + 1) % cfg["log_every"] == 0
            if sam is not None:                                      # FG train.py:223-242 + sam.py
                old, gnorm1 = FE.sam_first_step(params, float(sam["rho"]), bool(sam.get("adaptive", False)))
                opt.zero_grad(set_to_none=True)
                FE.disable_running_stats(model)                      # FG train.py:235
                vals2, _ = FE.compute_losses(model, pool, cfg, gen, device, terms)
                total2 = FE.weighted_total(vals2, terms)
                total2.backward()
                FE.sam_restore(old)
                FE.enable_running_stats(model)
                stats.update(sam_grad_norm_w=gnorm1, total_perturbed=float(total2))
            gns = grad_norms(model) if log_now else {}
            gn = torch.nn.utils.clip_grad_norm_(params, o["grad_clip"])
            opt.step()
            sched.step()
            state["step"] += 1
            rec = {k: float(v) for k, v in vals.items()}
            rec.update(stats, total=float(total), grad_norm=float(gn), lr=sched.get_last_lr()[0], **gns)
            rec["sec_step"] = time.time() - t_it
            t_it = time.time()
            for k, v in rec.items():
                if isinstance(v, (int, float)):
                    agg.setdefault(k, []).append(v)
            if log_now:
                log.log(event="step", epoch=epoch, step=state["step"], **rec)
                print(f"ep {epoch} it {it + 1}/{len(loader)} " +
                      " ".join(f"{k}={rec[k]:.4f}" for k in ["total", *vals.keys()]), flush=True)
        t_train = time.time() - t0
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0
        res = T.validate(model, cfg, device, val_df, val_loader)
        state["epoch"] = epoch + 1
        val, improved = select_epoch(model, cfg, a.out, state, epoch, res)
        save_checkpoint(os.path.join(a.out, "last.pt"), model, opt, sched, None, state, full=cfg["ckpt"]["full"])
        log.log(event="epoch", epoch=epoch, seconds=time.time() - t0, train_seconds=t_train, peak_mem_gb=peak,
                peak_reserved_gb=(torch.cuda.max_memory_reserved(device) / 2 ** 30 if device.type == "cuda" else 0.0),
                select_value=val, improved=improved, val_auc=res["auc"], val_ding=res["ding@0.5"],
                val_stats=res["val_stats"], val_oppoint=res["oppoint"],
                train_means={k: float(np.mean(v)) for k, v in agg.items()})
        print(f"epoch {epoch} val_auc {res['auc']:.2f} select {val:.3f} best_epoch {state['best_epoch']} "
              f"sec {time.time() - t0:.0f}", flush=True)
    finalize_selection(a.out, state, cfg["select"]["metric"])


if __name__ == "__main__":
    main()
