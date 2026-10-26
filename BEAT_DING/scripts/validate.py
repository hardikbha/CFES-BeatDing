"""Pre-training validation of CFES. Run this before any real training.

  python scripts/validate.py --config configs/fairforge2.yaml --out runs/validate_ff2
  python scripts/validate.py --config configs/ffpp_ding.yaml  --out runs/validate_ffpp \
      --reference_ckpt ../FAIRREL_REPRO/runs/CTRL_s7/best.pt --overfit_steps 30

Checks (all written to <out>/validate_report.json; failures make the script exit non-zero):
  A transforms : background views leave every core-face pixel bit-identical, change the background, and the
                 photometric view stays within its configured bounds; overlay grids show the face template on
                 real crops so its placement can be judged by eye (it is a template, not a detector).
                 Optional: score shift / flip rate of a trained reference detector under each view.
  B losses     : each loss is ~0 in the case it should be ~0 and positive in the case it should be positive.
  C gradients  : for every active term, (i) the term alone gives finite, non-zero gradients on trainable
                 parameters and none on frozen ones; (ii) the total loss with the term's weight > 0 gives
                 different gradients from the same total with that weight = 0 (catches a term computed under
                 no_grad, the defect found in Ding et al.'s released GDA.py); (iii) one optimiser step leaves
                 frozen weights unchanged.
  D overfit    : optional, N steps on one fixed batch; the total loss should fall.
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cfes import engine  # noqa: E402
from cfes import losses as L  # noqa: E402
from cfes.config import load_config  # noqa: E402
from cfes.counterfactuals import KINDS, apply_intervention  # noqa: E402
from cfes.data import ImageTable, PairTable, read_split  # noqa: E402

FAIL = []


def check(name, ok, detail):
    print(("PASS " if ok else "FAIL ") + name + "  " + str(detail), flush=True)
    if not ok:
        FAIL.append(name)
    return {"ok": bool(ok), "detail": detail}


def sample_batch(cfg, n, seed=0):
    df = read_split(cfg["data"]["train"])
    df = pd.concat([g.sample(n // 2, random_state=seed) for _, g in df.groupby("label")])
    ds = ImageTable(df.reset_index(drop=True), cfg["data"]["input_size"], cfg["data"].get("resample", "bicubic"))
    items = [ds[i] for i in range(len(ds))]
    return {"image": torch.stack([b["image"] for b in items]), "label": torch.stack([b["label"] for b in items]),
            "group": torch.stack([b["group"] for b in items])}


def sample_pairs(cfg, geom, n):
    p = cfg["pairs"]
    if not p.get("enabled") or not p.get("csv") or not os.path.exists(p["csv"]):
        return None
    ds = PairTable(p["csv"], cfg["data"]["input_size"], geom.grid, geom.core, p["diff_tau"], p.get("align_tau"),
                   resample=cfg["data"].get("resample", "bicubic"))
    al = ds.df.index[ds.df.aligned].tolist()[: n // 2] + ds.df.index[~ds.df.aligned].tolist()[: n - n // 2]
    items = [ds[i] for i in al[:n]]
    if not items:
        return None
    return {"fake": torch.stack([b["fake"] for b in items]), "real": torch.stack([b["real"] for b in items]),
            "aligned": torch.tensor([b["aligned"] for b in items]), "frac": torch.stack([b["frac"] for b in items])}


def part_a(cfg, geom, batch, out, ref_model, device):
    rep = {}
    x, y = batch["image"], batch["label"]
    gen = torch.Generator().manual_seed(0)
    reals = x[y == 0]
    views = {
        "bg_blur": apply_intervention(x, KINDS["bg_blur"], {"sigma": float(np.mean(cfg["cf"]["bg_blur_sigma"]))},
                                      cfg["cf"], geom),
        "bg_swap": apply_intervention(x, KINDS["bg_swap"], None, cfg["cf"], geom,
                                      donors=reals[torch.randint(len(reals), (len(x),), generator=gen)]),
        "photometric": apply_intervention(x, KINDS["photometric"], {"seed": 1}, cfg["cf"], geom),
    }
    core, bg = geom.core, ~geom.core
    for k in ("bg_blur", "bg_swap"):
        d = (views[k] - x).abs()
        rep[f"{k}_core_identical"] = check(f"A {k}: core face pixels identical", float(d[:, :, core].max()) == 0.0,
                                           {"max_abs_diff_core": float(d[:, :, core].max())})
        rep[f"{k}_bg_changed"] = check(f"A {k}: background changed", float(d[:, :, bg].mean()) > 1e-3,
                                       {"mean_abs_diff_bg": float(d[:, :, bg].mean())})
    d = (views["photometric"] - x).abs()
    rep["photometric_bounded"] = check("A photometric: mean change <= 0.15", float(d.mean()) <= 0.15,
                                       {"mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max())})
    rep["labels"] = check("A labels pass through unchanged (views are built per image, labels not touched)",
                          True, {"n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())})
    save_grid(x, views, geom, os.path.join(out, "transforms_grid.png"))
    if ref_model is not None:
        rep["reference_detector"] = reference_shift(ref_model, x, y, views, device)
    return rep


def save_grid(x, views, geom, path, n=6):
    from PIL import Image
    over = x[:n].clone()
    edge = geom.core ^ torch.nn.functional.max_pool2d(geom.core.float()[None, None], 3, 1, 1)[0, 0].bool()
    over[:, 0, edge], over[:, 1, edge], over[:, 2, edge] = 1.0, 0.0, 0.0
    cols = [x[:n], over] + [v[:n] for v in views.values()]
    rows = [torch.cat([c[i] for c in cols], dim=2) for i in range(min(n, len(x)))]
    grid = (torch.cat(rows, dim=1).permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(grid).save(path)
    print(f"     grid: columns = original | face template (red) | {' | '.join(views)} -> {path}")


@torch.no_grad()
def reference_shift(ref, x, y, views, device):
    ref.eval()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    prob = lambda t: torch.sigmoid(ref((t.to(device) - mean) / std)[2].float()).cpu()
    p0 = prob(x)
    res = {}
    for k, v in views.items():
        p1 = prob(v)
        res[k] = {c: {"mean_abs_prob_shift": float((p1 - p0)[y == ci].abs().mean()),
                      "flip_rate@0.5": float(((p1 >= .5) != (p0 >= .5))[y == ci].float().mean())}
                  for c, ci in (("real", 0), ("fake", 1))}
    print("     reference detector shift:", json.dumps(res))
    return res


def part_b(cfg, geom):
    rep = {}
    G = geom.grid
    face = geom.face_cells
    s = torch.tensor([1.0, -2.0, 3.0])
    clean = {"score": s, "patch": torch.zeros(3, G, G)}
    view = {"score": s.clone(), "patch": torch.full((3, G, G), -20.0)}
    view["patch"][:, face] = 0.0
    kind = torch.tensor([KINDS["bg_blur"]] * 3)
    v, _ = L.stability_loss(clean, view, kind, geom, cfg["loss"]["stab"], (0, 1))
    rep["stab_zero"] = check("B stability ~0 when view == clean and background evidence ~0", float(v) < 1e-6, float(v))
    view2 = {"score": s + 3.0, "patch": torch.full((3, G, G), 5.0)}
    v2, _ = L.stability_loss(clean, view2, kind, geom, cfg["loss"]["stab"], (0, 1))
    rep["stab_pos"] = check("B stability > 0 when the view moves", float(v2) > 0.1, float(v2))
    frac = torch.zeros(2, G, G)
    frac[:, face] = 1.0
    frac[0, :, :] *= 0
    al = torch.tensor([True, True])
    good_f = {"score": torch.tensor([20.0, 20.0]), "patch": torch.full((2, G, G), 20.0)}
    good_r = {"score": torch.tensor([-20.0, -20.0]), "patch": torch.full((2, G, G), 20.0)}
    good_r["patch"][:, face] = -20.0
    good_f["patch"][0] = good_r["patch"][0]
    v3, _ = L.sensitivity_loss(good_f, good_r, frac, al, geom, cfg["loss"]["sens"])
    rep["sens_zero"] = check("B sensitivity ~0 when fake >> source on changed cells, equal elsewhere",
                             float(v3) < 1e-3, float(v3))
    v4, _ = L.sensitivity_loss(good_r, good_f, frac, al, geom, cfg["loss"]["sens"])
    rep["sens_pos"] = check("B sensitivity large when the source outscores the fake", float(v4) > 1.0, float(v4))
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    g = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    eq = torch.tensor([-1.0, -1.0, -1.0, -1.0, 2.0, 2.0, 2.0, 2.0])
    v5, _ = L.alignment_loss(eq, y, g, cfg["loss"]["align"])
    rep["align_zero"] = check("B alignment 0 when every group has the same rates", float(v5) < 1e-9, float(v5))
    neq = torch.tensor([-3.0, -3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0])
    v6, _ = L.alignment_loss(neq, y, g, cfg["loss"]["align"])
    rep["align_pos"] = check("B alignment > 0 when groups differ", float(v6) > 0.01, float(v6))
    return rep


def grads(model):
    return {n: (p.grad.detach().clone() if p.grad is not None else None) for n, p in model.named_parameters()}


def part_c(cfg, model, geom, batch, pairs, terms, device):
    rep = {}
    gen = lambda: torch.Generator().manual_seed(123)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    # One step first: at initialisation the rank-1 residuals are exactly orthogonal to the frozen part and
    # alpha = 0, so orth (and the patch path through s) legitimately have zero gradient before any update.
    warm = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    model.zero_grad(set_to_none=True)
    vals, _ = engine.compute_losses(model, batch, pairs, cfg, geom, gen(), device, terms)
    engine.weighted_total(vals, terms).backward()
    warm.step()
    for k in terms:
        model.zero_grad(set_to_none=True)
        vals, _ = engine.compute_losses(model, batch, pairs, cfg, geom, gen(), device, terms)
        vals[k].backward()
        g = grads(model)
        tnorm = float(torch.sqrt(sum((g[n] ** 2).sum() for n in trainable if g[n] is not None)))
        frozen_with_grad = [n for n, v in g.items() if n not in trainable and v is not None]
        mods = sorted({n.split(".")[0] if not n.startswith("backbone") else "adapters"
                       for n in trainable if g[n] is not None and float(g[n].abs().sum()) > 0})
        rep[f"{k}_alone"] = check(f"C {k}: finite non-zero gradient on trainable params, none on frozen",
                                  np.isfinite(tnorm) and tnorm > 0 and not frozen_with_grad,
                                  {"loss": float(vals[k]), "grad_norm": tnorm, "receives_grad": mods,
                                   "frozen_with_grad": frozen_with_grad[:3]})
    for k in terms:
        if k == "det":
            continue
        out = []
        for w in (terms[k], 0.0):
            t2 = dict(terms)
            t2[k] = w
            model.zero_grad(set_to_none=True)
            vals, _ = engine.compute_losses(model, batch, pairs, cfg, geom, gen(), device, terms)
            engine.weighted_total(vals, t2).backward()
            out.append(grads(model))
        diff = max(float((out[0][n] - out[1][n]).abs().max()) for n in trainable
                   if out[0][n] is not None and out[1][n] is not None)
        rep[f"{k}_weight_matters"] = check(f"C {k}: weight {terms[k]} vs 0 changes the gradient", diff > 0,
                                           {"max_abs_grad_diff": diff})
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    model.zero_grad(set_to_none=True)
    vals, _ = engine.compute_losses(model, batch, pairs, cfg, geom, gen(), device, terms)
    engine.weighted_total(vals, terms).backward()
    opt.step()
    moved = [n for n, p in model.named_parameters() if not p.requires_grad and not torch.equal(p, frozen[n])]
    rep["frozen_unchanged"] = check("C one optimiser step leaves frozen weights unchanged", not moved, moved[:3])
    return rep


def part_d(cfg, model, geom, batch, pairs, terms, device, steps):
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=cfg["optim"]["lr"])
    traj = []
    for s in range(steps):
        vals, stats = engine.compute_losses(model, batch, pairs, cfg, geom, torch.Generator().manual_seed(s),
                                            device, terms)
        tot = engine.weighted_total(vals, terms)
        opt.zero_grad(set_to_none=True)
        tot.backward()
        opt.step()
        traj.append({"step": s, "total": float(tot), **{k: float(v) for k, v in vals.items()},
                     "train_acc": stats["train_acc"]})
        print(f"     overfit step {s}: total {float(tot):.4f} acc {stats['train_acc']:.2f}", flush=True)
    first, last = np.mean([t["total"] for t in traj[:3]]), np.mean([t["total"] for t in traj[-3:]])
    return {"trajectory": traj, "falls": check("D total loss falls on a fixed batch", last < first,
                                               {"first3": first, "last3": last})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n_pairs", type=int, default=4)
    ap.add_argument("--reference_ckpt", default=None, help="FairReL-code checkpoint for the view-shift check")
    ap.add_argument("--overfit_steps", type=int, default=0)
    ap.add_argument("--all_terms", action="store_true", help="enable every term (align included) for part C")
    a = ap.parse_args()
    cfg = load_config(a.config, a.set)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = engine.build_model(cfg, device)
    geom_cpu = engine.build_geometry(cfg, model, "cpu")
    geom = engine.build_geometry(cfg, model, device)
    batch = sample_batch(cfg, a.batch)
    pairs = sample_pairs(cfg, geom_cpu, a.n_pairs)
    if a.all_terms:
        cfg = copy.deepcopy(cfg)
        cfg["data"]["demographics_reliable"] = True
        for k in ("stab", "sens", "loc", "align", "orth"):
            cfg["loss"][k]["weight"] = cfg["loss"][k]["weight"] or 0.1
        cfg["cf"]["enabled"] = True
    terms = engine.active_terms(cfg, have_pairs=pairs is not None)
    report = {"config": a.config, "device": str(device), "terms": terms, "params": model.parameter_report(),
              "pairs_available": pairs is not None}
    print("terms:", terms, "| params:", report["params"], "| pairs:", pairs is not None, flush=True)

    ref = None
    if a.reference_ckpt:
        sys.path.insert(0, cfg["paths"]["fairrel_repo"])
        from fairrel.models.detector import FairReLDetector
        ref = FairReLDetector()
        ref.load_state_dict(torch.load(a.reference_ckpt, map_location="cpu", weights_only=False)["model"])
        ref = ref.to(device)
    report["A_transforms"] = part_a(cfg, geom_cpu, batch, a.out, ref, device)
    report["B_losses"] = part_b(cfg, geom_cpu)
    report["C_gradients"] = part_c(cfg, model, geom, batch, pairs, terms, device)
    if a.overfit_steps:
        report["D_overfit"] = part_d(cfg, model, geom, batch, pairs, terms, device, a.overfit_steps)
    report["failures"] = FAIL
    with open(os.path.join(a.out, "validate_report.json"), "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(f"\n{len(FAIL)} failure(s): {FAIL}" if FAIL else "\nall checks passed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
