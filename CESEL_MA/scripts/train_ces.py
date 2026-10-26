"""Train a CES arm (Counterfactual Evidence Selectivity). Adapted from scripts/train.py (the CFES trainer, copied
unchanged next to this file); validation, selection rule, provenance and checkpoint format are the CFES ones, with the
CES state (EMA baselines, demographic subspaces, online probe) stored in the checkpoint's state["ces_state"].

  python scripts/train_ces.py --config configs/ffpp/B0.yaml --out runs/ffpp_B0_s7 --set seed=7 \
         model.init_backbone_ckpt=<CFES phase-1 last.pt of seed 7>
  python scripts/ces_validate_cues.py --config configs/ffpp/B0.yaml --ckpt runs/ffpp_B0_s7/best.pt \
         --out runs/ffpp_B0_s7/validated_cues.json
  python scripts/train_ces.py --config configs/ffpp/full.yaml --out runs/ffpp_full_s7 --set seed=7 \
         model.init_ckpt=runs/ffpp_B0_s7/best.pt ces.validated_json=runs/ffpp_B0_s7/validated_cues.json

Model selection uses ONE fixed rule on the validation split (cfg.select.metric, the CFES Amendment 9 rule by default);
test sets are never read here. NOTHING has been run with this script on GPU (CPU smoke only, tests/ces_checks.py).
"""
import argparse
import json
import math
import os
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
from cfes import engine as E  # noqa: E402
from cfes.ces_model import CUES  # noqa: E402
from cfes.ces_validation import load_validated, load_validated_family  # noqa: E402
from cfes.config import dump_config, load_config  # noqa: E402
from cfes import public_guard  # noqa: E402
from cfes import ma_labels as MAL  # noqa: E402
from cfes.data import cycle  # noqa: E402
from cfes.utils import JsonlLogger, code_hash, load_checkpoint, save_checkpoint, set_seed  # noqa: E402

GROUPS = [("backbone", "backbone.")] + [(f"branch_{k}", f"branches.{k}.") for k in CUES] + \
         [("cue_norm", "cue_norm."), ("head", "head.")]


def grad_norms(model):
    acc = {k: 0.0 for k, _ in GROUPS}
    for n, p in model.named_parameters():
        if p.grad is not None:
            for k, pre in GROUPS:
                if n.startswith(pre):
                    acc[k] += float(p.grad.detach().float().pow(2).sum())
                    break
    return {f"gn_{k}": v ** 0.5 for k, v in acc.items()}


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
    guard = public_guard.check(cfg)          # FF2-public provenance guard (Codex F5); None for other configs
    os.makedirs(a.out, exist_ok=True)
    public_guard.write_lineage(a.out, guard)
    if not a.resume and os.path.exists(os.path.join(a.out, "last.pt")):
        raise SystemExit(f"{a.out} already has last.pt; pass --resume or use a new --out")
    dump_config(cfg, os.path.join(a.out, "config.yaml"))
    JsonlLogger(os.path.join(a.out, "provenance.jsonl")).log(**T.provenance(cfg, a, a.out))
    log = JsonlLogger(os.path.join(a.out, "train_log.jsonl"))
    set_seed(cfg["seed"])
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model, st_init = CE.build_model(cfg, device)
    state_ces = CE.build_state(cfg, model, device, st_init)
    fgeom = CE.face_geometry(cfg, model, device)
    train_df, cond, sampler, loader = CE.train_loader(cfg, cfg["seed"])
    pairs = CE.pair_loader(cfg, train_df, cond, cfg["seed"])
    mfer_dl, mfer_rep = E.mfer_loader(cfg, cfg["seed"]) if cfg.get("mfer", {}).get("enabled") else (None, None)
    terms = CE.active_terms(cfg, have_pairs=pairs is not None, have_mfer=mfer_dl is not None)
    strat_dl = CE.strat_loader(cfg, train_df, cond, len(loader), cfg["seed"])
    val_df, val_loader = E.eval_loader(cfg["data"]["val"], cfg, cfg["debug"].get("max_eval_items"))
    val_df, ma_val_rep = CE.attach_ma(val_df, cfg, "val")      # CESel-MA: axes for the selection gap (row order kept)
    ma_ctx = CE.ma_context(cfg, train_df)
    valid, target, vsrc = load_validated(cfg)
    fam = load_validated_family(cfg, len(cfg["data"].get("families") or []) + 2)   # None unless the FF2 flag is on

    o = cfg["optim"]
    opt = torch.optim.AdamW(model.param_groups(o["lr"], o.get("lr_new", o["lr"]), o["weight_decay"]))
    steps_total = o["epochs"] * len(loader)
    warm = max(1, min(o["warmup_steps"], max(1, steps_total // 10)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                              (0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps_total - warm)))
                                               if o["schedule"] == "cosine" else 1.0))
    state = {"epoch": 0, "step": 0, "best": -float("inf"), "best_epoch": -1}
    aux = {"fprgap_rq": E.fprgap_quantile(cfg)} if "fprgap" in terms else {}
    if "freq_aux" in terms:
        aux["freq_warm_steps"] = CE.freq_warmup_steps(cfg, len(loader))
    if a.resume:
        state = load_checkpoint(os.path.join(a.out, "last.pt"), model, opt, sched, None, restore_rng=True)
        state_ces.load_full_state(state.pop("ces_state"))
        if "fprgap_buf" in state:
            buf = state.pop("fprgap_buf")
            if "fprgap_rq" in aux:
                aux["fprgap_rq"].load(buf)

    def ck_state(extra=None):
        d = dict(state, ces_state=state_ces.full_state())
        d.update(extra or {})
        return d
    log.log(event="start", code_hash=code_hash(ROOT), device=str(device), terms=terms, params=model.parameter_report(),
            n_train=len(train_df), n_val=len(val_df), steps_per_epoch=len(loader),
            pairs=pairs.dataset.report if pairs else None, cue_geometry={k: v for k, v in model.geom.describe().items()},
            validated={"source": vsrc, "valid": valid.tolist(), "target": target.tolist(),
                       "per_family": None if fam is None else {"valid": fam[0].tolist(), "target": fam[1].tolist()}},
            strat_cells=strat_dl.batch_sampler.describe() if strat_dl else None, mfer=mfer_rep,
            backbone_init_keys=getattr(model, "backbone_init_keys", None),
            ma=None if ma_ctx is None else {"ita_range": ma_ctx["ita_range"], "h": ma_ctx["h"],
                                            "anchors": ma_ctx["anchors"].tolist(), "val_labels": ma_val_rep,
                                            "train_labels": MAL.attach(train_df[["img_path", "group"]], cfg, "train")[1],
                                            "freq_warm_steps": aux.get("freq_warm_steps")})
    print("terms", terms, "params", model.parameter_report(), flush=True)

    if cfg["select"].get("include_init") and not a.resume:
        res = T.validate(model, cfg, device, val_df, val_loader)
        val = T.selection_value(res, cfg["select"]["metric"])
        state["select_ref"] = res["oppoint"]
        state.update(best=val, best_epoch=-1, best_threshold_tpr=res["threshold_tpr"])
        save_checkpoint(os.path.join(a.out, "best.pt"), model, None, None, None, ck_state(), full=cfg["ckpt"]["full"])
        log.log(event="epoch", epoch=-1, select_value=val, val_auc=res["auc"], val_oppoint=res["oppoint"],
                val_oppoint_ma=res.get("oppoint_ma"))
    gen = torch.Generator().manual_seed(cfg["seed"] * 7919 + state["epoch"])
    pair_iter = cycle(pairs) if pairs else None
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
        mfer_iter = None
        if mfer_dl is not None:
            mfer_dl.generator.manual_seed(E.mfer_epoch_seed(cfg["seed"], epoch))
            mfer_iter = cycle(mfer_dl)
        t0, agg = time.time(), {}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t_it = time.time()
        for it, batch in enumerate(loader):
            pb = next(pair_iter) if pair_iter else None
            sb = next(strat_iter) if strat_iter else None
            mb = next(mfer_iter) if mfer_iter else None
            vals, stats = CE.compute_ces_losses(model, state_ces, batch, pb, cfg, gen, device, terms, strat_batch=sb,
                                                mfer_batch=mb, fgeom=fgeom, aux=aux, valid=valid, target=target,
                                                fam=fam, step=state["step"], ma_ctx=ma_ctx)
            total = E.weighted_total(vals, terms)
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {state['step']}: {vals}")
            opt.zero_grad(set_to_none=True)
            total.backward()
            log_now = (state["step"] + 1) % cfg["log_every"] == 0
            gns = grad_norms(model) if log_now else {}
            gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), o["grad_clip"])
            opt.step()
            sched.step()
            state["step"] += 1
            rec = {k: float(v) for k, v in vals.items()}
            rec.update(stats, total=float(total), grad_norm=float(gn), lr=sched.get_last_lr()[0], **gns)
            rec["sec_step"] = time.time() - t_it          # wall time of this step incl. data wait (float() syncs)
            t_it = time.time()
            for k, v in rec.items():
                agg.setdefault(k, []).append(v)
            if log_now:
                log.log(event="step", epoch=epoch, step=state["step"], **rec)
                print(f"ep {epoch} it {it + 1}/{len(loader)} " +
                      " ".join(f"{k}={rec[k]:.4f}" for k in ["total", *vals.keys()]), flush=True)
        t_train = time.time() - t0
        peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else 0.0
        res = T.validate(model, cfg, device, val_df, val_loader)
        val = T.selection_value(res, cfg["select"]["metric"],
                                state.get("select_ref") if cfg["select"].get("ref_checks", True) else None)
        state["epoch"] = epoch + 1
        improved = val > state["best"]
        if improved:
            state.update(best=val, best_epoch=epoch, best_threshold_tpr=res["threshold_tpr"])
            save_checkpoint(os.path.join(a.out, "best.pt"), model, None, None, None, ck_state(),
                            full=cfg["ckpt"]["full"])
        extra = {"fprgap_buf": aux["fprgap_rq"].state()} if "fprgap_rq" in aux else {}
        save_checkpoint(os.path.join(a.out, "last.pt"), model, opt, sched, None, ck_state(extra),
                        full=cfg["ckpt"]["full"])
        log.log(event="epoch", epoch=epoch, seconds=time.time() - t0, train_seconds=t_train, peak_mem_gb=peak,
                peak_reserved_gb=(torch.cuda.max_memory_reserved(device) / 2 ** 30 if device.type == "cuda" else 0.0),
                select_value=val, improved=improved, val_oppoint_ma=res.get("oppoint_ma"),
                val_auc=res["auc"], val_ding=res["ding@0.5"], val_stats=res["val_stats"], val_oppoint=res["oppoint"],
                train_means={k: float(np.mean(v)) for k, v in agg.items()})
        print(f"epoch {epoch} val_auc {res['auc']:.2f} select {val:.3f} best_epoch {state['best_epoch']} "
              f"sec {time.time() - t0:.0f}", flush=True)
    with open(os.path.join(a.out, "TRAINED"), "w") as f:
        json.dump(state, f, default=float)
    public_guard.write_p1(a.out, guard)


if __name__ == "__main__":
    main()
