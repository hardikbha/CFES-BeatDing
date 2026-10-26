"""Train FairReL on FF++ (c23) with demographic subgroup labels.

Example:
    python train.py --config configs/fairrel_clip_vitl14.yaml \
        --train-csv data/ffpp_train.csv --val-csv data/ffpp_val.csv \
        --data-root /path/to/images --output-dir outputs/fairrel

Objective (Eq. 9):  L = L_CE + lambda_orth * L_orth + lambda_G * L_GCWD
                        + lambda_S * L_SLMA
Only the SVD residual factors and the classifier are updated.

Model selection follows the protocol in the supplementary material: the
checkpoint maximising (val AUC - val F_FPR) on the FF++ validation split at
the TPR = 0.90 operating point.  Target-domain data is never used.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from fairrel.data import DeepfakeDataset, GroupBalancedBatchSampler, SUBGROUPS
from fairrel.losses import GCWDLoss, SLMALoss
from fairrel.metrics import fairness_report, format_report, threshold_at_tpr
from fairrel.models import FairReLDetector
from fairrel.utils import AverageMeter, load_config, save_json, set_seed


# ---------------- FAIRREL_REPRO gradient-norm instrumentation ----------------
# Every GRADLOG_EVERY optimiser steps, the gradient of each WEIGHTED loss term (lambda * term, i.e. the term's actual
# contribution to the update) w.r.t. the trainable parameters is computed with torch.autograd.grad(retain_graph=True).
# autograd.grad does not touch .grad, consumes no RNG and does not change any module state, so the optimiser update
# that follows is the official one.  Logged per term: L2 norm over all trainable params, over the backbone SVD
# factors only, over the classifier only; the cosine with the CE gradient on the backbone factors; raw (unweighted)
# loss values; and GCWD's per-band discrepancies D_l.  Output: <output_dir>/gradlog.jsonl.
import json as _json  # noqa: E402
import math as _math  # noqa: E402

GRADLOG_EVERY = int(os.environ.get("GRADLOG_EVERY", "25"))


def _flat(gs, ps, keep):
    out = []
    for gr, p in zip(gs, ps):
        if keep(p):
            out.append(torch.zeros_like(p).flatten() if gr is None else gr.detach().float().flatten())
    return torch.cat(out) if out else torch.zeros(1, device=ps[0].device)


def _gradlog(step, epoch, params, model, lcfg, terms, gcwd, feat_map, g, out_dir):
    cls_ids = {id(p) for p in model.classifier.parameters()}
    is_bb = lambda p: id(p) not in cls_ids          # noqa: E731
    is_cls = lambda p: id(p) in cls_ids             # noqa: E731
    lam = {"ce": 1.0, "orth": lcfg["lambda_orth"], "gcwd": lcfg["lambda_gcwd"], "slma": lcfg["lambda_slma"]}
    rec = {"step": step, "epoch": epoch, "lambda": lam}
    vec_bb = {}
    for name, t in terms.items():
        rec[f"{name}_value"] = float(t.detach().float())
        if not t.requires_grad:
            rec[f"{name}_gnorm_all"] = rec[f"{name}_gnorm_backbone"] = rec[f"{name}_gnorm_cls"] = 0.0
            rec[f"{name}_raw_gnorm_backbone"] = 0.0
            continue
        gs = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)   # UNWEIGHTED term gradient
        vb = _flat(gs, params, is_bb)
        rec[f"{name}_raw_gnorm_backbone"] = float(vb.norm())
        w = float(lam[name])
        rec[f"{name}_gnorm_all"] = w * float(_flat(gs, params, lambda p: True).norm())
        rec[f"{name}_gnorm_backbone"] = w * float(vb.norm())
        rec[f"{name}_gnorm_cls"] = w * float(_flat(gs, params, is_cls).norm())
        vec_bb[name] = vb
    if "ce" in vec_bb:
        for name, v in vec_bb.items():
            if name != "ce":
                den = float(v.norm()) * float(vec_bb["ce"].norm())
                rec[f"cos_{name}_ce_backbone"] = float(v @ vec_bb["ce"]) / den if den > 0 else float("nan")
    with torch.no_grad():   # GCWD per-band discrepancies D_l (same computation as GCWDLoss.forward)
        en = gcwd.band_energies(feat_map.detach().float())
        gm = [en[g == k].mean(0) for k in range(gcwd.num_groups) if (g == k).any()]
        if len(gm) >= 2:
            E = torch.stack(gm)
            D = (E - E.mean(0, keepdim=True)).pow(2).sum(0)
            rec["gcwd_D"] = [float(x) for x in D]
            rec["gcwd_w"] = [float(x) for x in torch.softmax(D / gcwd.temperature, 0)]
            rec["gcwd_band_energy_mean"] = [float(x) for x in en.mean(0)]
    with open(os.path.join(out_dir, "gradlog.jsonl"), "a") as f:
        f.write(_json.dumps(rec) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Train FairReL")
    p.add_argument("--config", default="configs/fairrel_clip_vitl14.yaml")
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--data-root", default="")
    p.add_argument("--output-dir", default="outputs/fairrel")
    p.add_argument("--seed", type=int, default=None, help="overrides config seed")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, tpr_anchor):
    model.eval()
    scores, labels, groups = [], [], []
    for img, y, g in tqdm(loader, desc="val", leave=False):
        p = model.predict(img.to(device, non_blocking=True))
        scores.extend(p.float().cpu().tolist())
        labels.extend(y.tolist())
        groups.extend(g.tolist())
    thr = threshold_at_tpr(scores, labels, tpr_anchor)
    return fairness_report(scores, labels, groups, thr)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg["train"]["seed"]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    tcfg, mcfg, lcfg = cfg["train"], cfg["model"], cfg["loss"]

    train_set = DeepfakeDataset(args.train_csv, args.data_root,
                                tcfg["input_size"], train=True)
    val_set = DeepfakeDataset(args.val_csv, args.data_root,
                              tcfg["input_size"], train=False)
    sampler = GroupBalancedBatchSampler(train_set.groups, tcfg["batch_size"], seed=seed)
    train_loader = DataLoader(train_set, batch_sampler=sampler,
                              num_workers=tcfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=tcfg["batch_size"], shuffle=False,
                            num_workers=tcfg["num_workers"], pin_memory=True)

    model = FairReLDetector(mcfg["backbone"], mcfg["pretrained"], mcfg["svd_rank"],
                            mcfg.get("pooling", "cls")).to(device)
    gcwd = GCWDLoss(num_groups=len(SUBGROUPS), levels=lcfg["gcwd"]["levels"],
                    temperature=lcfg["gcwd"]["temperature"],
                    normalize_energy=lcfg["gcwd"]["normalize_energy"]).to(device)
    slma = SLMALoss(num_groups=len(SUBGROUPS), feat_dim=model.backbone.feat_dim,
                    ema_decay=lcfg["slma"]["ema_decay"],
                    min_count=lcfg["slma"]["min_count"]).to(device)

    params = model.trainable_parameters()
    n_trainable = sum(p.numel() for p in params)
    print(f"trainable parameters: {n_trainable:,} "
          f"(SVD residual factors + classifier only)")
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])

    use_amp = tcfg["amp"] and device.type == "cuda"
    best_score, history = float("-inf"), []
    gstep = 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        meters = {k: AverageMeter() for k in ("loss", "ce", "orth", "gcwd", "slma")}
        for img, y, g in tqdm(train_loader, desc=f"epoch {epoch}"):
            img = img.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            g = g.to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                feat_map, _, logit = model(img)              # standard path
                z_delta = model.forward_residual(img)        # residual-only path
                loss_ce = F.binary_cross_entropy_with_logits(logit.float(), y.float())
                loss_gcwd = gcwd(feat_map.float(), g)
                loss_slma = slma(z_delta.float(), y, g)
            loss_orth = model.orthogonality_loss()           # fp32 factors
            loss = (loss_ce + lcfg["lambda_orth"] * loss_orth
                    + lcfg["lambda_gcwd"] * loss_gcwd
                    + lcfg["lambda_slma"] * loss_slma)

            # ---- FAIRREL_REPRO instrumentation (diagnostic copy only; the update below is unchanged) ----
            if GRADLOG_EVERY > 0 and gstep % GRADLOG_EVERY == 0:
                _gradlog(gstep, epoch, params, model, lcfg,
                         {"ce": loss_ce, "orth": loss_orth, "gcwd": loss_gcwd, "slma": loss_slma},
                         gcwd, feat_map, g, args.output_dir)
            gstep += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            n = img.size(0)
            for k, v in (("loss", loss), ("ce", loss_ce), ("orth", loss_orth),
                         ("gcwd", loss_gcwd), ("slma", loss_slma)):
                meters[k].update(v.item(), n)

        slma_fallback = slma.fallback_fraction()
        slma.reset_stats()
        report = evaluate(model, val_loader, device, cfg["eval"]["tpr_anchor"])
        score = report["auc"] - report["f_fpr"]
        print(f"epoch {epoch}: " + " ".join(f"{k}={m.avg:.4f}" for k, m in meters.items())
              + f" slma_ema_fallback={100.0 * slma_fallback:.2f}%")
        print(format_report("FF++ val", report)
              + f" | realised TPR {report['tpr_overall']:.1f}"
              + f" | selection score {score:.2f}")
        history.append({"epoch": epoch, "val": report, "score": score,
                        "slma_ema_fallback": slma_fallback,
                        **{k: m.avg for k, m in meters.items()}})
        save_json(history, os.path.join(args.output_dir, "history.json"))

        state = {"model": model.state_dict(), "config": cfg, "epoch": epoch,
                 "val_report": report, "seed": seed}
        torch.save(state, os.path.join(args.output_dir, "last.pt"))
        if score > best_score:
            best_score = score
            torch.save(state, os.path.join(args.output_dir, "best.pt"))
            print(f"  new best checkpoint (score {score:.2f})")

    print(f"done. best selection score: {best_score:.2f}")


if __name__ == "__main__":
    main()
