"""Beat-Ding CPU checks (decisions/2026-10-26-beat-ding-plan.md section 2 + Amendments 1-2). Prints PASS / FAIL per check
and ALL_PASS at the end.

  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 /home/sachin.chaudhary/.conda/envs/pulid/bin/python tests/bd_checks.py \
      [--scratch DIR] [--skip_real] [--skip_e2e] [--only name,...]

A  augmentation: every op deterministic / uint8 / changes the image; draw statistics and parameter ranges; per-item RNG
   (seed, epoch, stream, index) determinism; persistent DataLoader workers see the epoch (EpochTagged); a (fake, own
   source) pair gets the identical op + parameters + flip (identical outputs for an identical pair, all ops); the CFES
   stability view of an augmented image keeps its augmented face pixels; labels / groups unchanged; image range.
B  ESA: exact hand-computed value and gradient, 0 for identical group distributions, > 0 and the right sign for a shifted
   group, n < 4 groups skipped (no cell, no gradient), stop-gradient pool, levels.
C  multi-layer evidence (stub = same classes as the FairReL CLIP wrapper; real CLIP ViT-L/14 + phase-1 checkpoint; real
   ResNet-50 multi-stage): hooks capture the right blocks / stages and shapes, gate init 0 gives EXACTLY the base score at
   the start (< 1e-4, eval and first train-mode forward), gradients (gate at 0: gate only; gate open: heads + adapters),
   phase-1 checkpoint load through engine.build_model.
D  group-AUC floor: sign of value and gradients, zero for groups above the pool and identical groups.
   Also: every config loads with the intended component flags; selection metric; Codex F3 / F4 / F8 in this copy; Ding
   published Tables 1-3 values against refs/Ding_CVPR2026.txt; winner / promotion rule; and a stub-backbone CPU dry run
   of the whole task chain through run/bd_run.sh (smoke, arms, screen, winner seeds + evals, final, ResNet seeds, final).
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "tests")]
os.chdir(ROOT)
from cfes import bd_aug, engine  # noqa: E402
from cfes import bd_metrics as BM  # noqa: E402
from cfes import losses as L  # noqa: E402
from cfes.config import load_config  # noqa: E402
from cfes.counterfactuals import BACKGROUND_KINDS, sample_views  # noqa: E402
from cfes.data import (AugImageTable, BalancedRealFakeSampler, EpochTagged, ImageTable, PairTable,  # noqa: E402
                       cluster_key, load_image, load_image_u8, read_split, to_tensor)
import bd_fixture as FX  # noqa: E402

CFES = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CFES"
PY, PYD = "/home/sachin.chaudhary/.conda/envs/pulid/bin/python", "/home/sachin.chaudhary/.conda/envs/aiface/bin/python"
RESULTS, REPORT, ARGS, CASES = [], {}, None, []


def case(fn):
    CASES.append(fn)
    return fn


def run_cases():
    for fn in CASES:
        if ARGS.only and fn.__name__ not in ARGS.only:
            continue
        t0 = time.time()
        try:
            fn()
            RESULTS.append((fn.__name__, "PASS", f"{time.time() - t0:.1f}s"))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((fn.__name__, "FAIL", f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"))
        print(f"{RESULTS[-1][1]}  {fn.__name__}  {RESULTS[-1][2] if RESULTS[-1][1] == 'PASS' else ''}", flush=True)
        if RESULTS[-1][1] == "FAIL":
            print("      " + RESULTS[-1][2].replace("\n", "\n      "), flush=True)


def script(name):
    """Import scripts/<name>.py by PATH (the FairReL repo, prepended to sys.path when a CLIP model is built, has its own
    train.py that would shadow ours)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"bd_{name}", os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stub_cfg(arm, fx, extra=()):
    return load_config(f"configs/bd/{arm}.yaml", ["model.init_ckpt=null", *FX.clip_overrides(fx), "data.num_workers=0",
                                                  *extra])


def build(cfg, seed=0):
    torch.manual_seed(seed)
    return engine.build_model(cfg, torch.device("cpu"))


def fx():
    if "fx" not in REPORT:
        REPORT["fx"] = FX.make(os.path.join(ARGS.scratch, "fixture"))
    return REPORT["fx"]


def real_image():
    df = read_split({"csv": BM.R + "/agents/DINGREPRO_LIN/data/ff++_ding/train.csv", "scheme": "eight"})
    return df, df.img_path.iloc[0]


# ================================================================================================ A. augmentation
@case
def a1_ops_deterministic_uint8_and_change():
    _, p = real_image()
    img = load_image_u8(p, 224)
    rng = np.random.default_rng(1)
    c = bd_aug.config({})
    rep = {}
    for op in bd_aug.OPS:
        for trial in range(3):
            prm = dict(bd_aug.draw(np.random.default_rng([trial, 7]), dict(c, p=1.0, ops=[op], hflip=0.0)))
            assert prm["op"] == op and prm["flip"] is False, prm
            a, b = bd_aug.apply(img, prm), bd_aug.apply(img.copy(), prm)
            assert a.dtype == np.uint8 and a.shape == img.shape and a.flags["C_CONTIGUOUS"], (op, a.dtype, a.shape)
            assert np.array_equal(a, b), f"{op}: not deterministic"
            d = float(np.abs(a.astype(int) - img.astype(int)).mean())
            assert d > 0.05, f"{op}: image unchanged (mean abs diff {d})"
            rep.setdefault(op, []).append(round(d, 2))
    f = bd_aug.apply(img, {"op": None, "flip": True})
    assert np.array_equal(f, img[:, ::-1]) and np.array_equal(bd_aug.apply(img, {"op": None, "flip": False}), img)
    REPORT["aug_mean_abs_change_per_op"] = rep
    del rng


@case
def a2_draw_statistics_and_ranges():
    c = bd_aug.config({})
    P = [bd_aug.draw(bd_aug.item_rng(7, 0, "train", i), c) for i in range(4000)]
    f_op = np.mean([p["op"] is not None for p in P])
    f_flip = np.mean([p["flip"] for p in P])
    per = {op: np.mean([p["op"] == op for p in P if p["op"] is not None]) for op in bd_aug.OPS}
    assert abs(f_op - 0.6) < 0.03 and abs(f_flip - 0.5) < 0.03, (f_op, f_flip)
    assert all(abs(v - 1 / 6) < 0.03 for v in per.values()), per
    for p in P:
        op = p["op"]
        if op == "jpeg":
            assert 30 <= p["q"] <= 90 and isinstance(p["q"], int)
        elif op == "bilateral":
            assert p["d"] in (5, 7, 9) and 20 <= p["sigma_color"] <= 80 and 20 <= p["sigma_space"] <= 80
        elif op == "gblur":
            assert 0.4 <= p["sigma"] <= 1.6
        elif op == "downup":
            assert 0.4 <= p["scale"] <= 0.9
        elif op == "photometric":
            assert 0.6 <= p["gamma"] <= 1.6 and 0.75 <= p["brightness"] <= 1.25 and 0.75 <= p["contrast"] <= 1.25
            assert 0.7 <= p["saturation"] <= 1.3
        elif op == "unsharp":
            assert 0.5 <= p["sigma"] <= 1.5 and 0.5 <= p["amount"] <= 1.5
    assert {p["q"] for p in P if p["op"] == "jpeg"} >= {30, 90}, "jpeg quality range not covered"
    REPORT["aug_draw_stats"] = {"frac_op": f_op, "frac_flip": f_flip, "per_op": per}
    try:
        bd_aug.config({"bogus": 1})
        raise AssertionError("unknown aug key accepted")
    except ValueError:
        pass


@case
def a3_table_determinism_labels_range():
    f = fx()
    df = read_split({"csv": f["train"], "scheme": "eight"})
    c = load_config("configs/bd/BD1.yaml")["data"]["aug"]
    plain, aug = ImageTable(df, 64), AugImageTable(df, 64, "bicubic", c, seed=7, stream="train")
    diff_epoch, ops = 0, set()
    for i in range(len(df)):
        a, b = aug[(0, i)], aug[(0, i)]
        assert torch.equal(a["image"], b["image"]), i
        assert a["image"].dtype == torch.float32 and a["image"].shape == (3, 64, 64)
        assert float(a["image"].min()) >= 0.0 and float(a["image"].max()) <= 1.0
        q = plain[i]
        assert int(a["label"]) == int(q["label"]) and int(a["group"]) == int(q["group"]) and a["index"] == i
        diff_epoch += int(not torch.equal(aug[(1, i)]["image"], a["image"]))
        ops.add(a["aug_op"])
        if a["aug_op"] == -1 and not a["aug_flip"]:
            assert torch.equal(a["image"], q["image"]), "no-op item differs from the plain pipeline"
    assert diff_epoch >= len(df) // 3, f"only {diff_epoch} items change between epochs"
    other = AugImageTable(df, 64, "bicubic", c, seed=7, stream="strat")
    assert sum(not torch.equal(other[(0, i)]["image"], aug[(0, i)]["image"]) for i in range(len(df))) >= len(df) // 3
    REPORT["aug_table"] = {"n": len(df), "changed_between_epochs": diff_epoch, "ops_seen": sorted(ops)}


@case
def a4_persistent_workers_see_the_epoch():
    f = fx()
    df = read_split({"csv": f["train"], "scheme": "eight"})
    c = load_config("configs/bd/BD1.yaml")["data"]["aug"]
    ds = AugImageTable(df, 64, "bicubic", c, seed=7, stream="train")
    smp = EpochTagged(BalancedRealFakeSampler(df.label.values, 8, steps=3, seed=7))
    dl = torch.utils.data.DataLoader(ds, batch_size=8, sampler=smp, num_workers=2, persistent_workers=True)
    for e in (0, 1, 2):
        smp.set_epoch(e)
        for b in dl:
            for j, i in enumerate(b["index"].tolist()):
                assert torch.equal(b["image"][j], ds[(e, i)]["image"]), (e, i)
    plain = list(BalancedRealFakeSampler(df.label.values, 8, steps=3, seed=7))
    smp.set_epoch(0)
    assert [i for _, i in smp] == plain, "EpochTagged changed the sampler order"


@case
def a5_pair_consistency_and_stability_view():
    f = fx()
    cfg = stub_cfg("BD2", f)
    geom = engine.build_geometry(cfg, build(cfg), "cpu")
    import pandas as pd
    pr = pd.read_csv(f["pairs"])
    same = pr.copy()
    same["real_path"] = same["fake_path"]
    p_same = os.path.join(ARGS.scratch, "pairs_identical.csv")
    same.to_csv(p_same, index=False)
    c = cfg["data"]["aug"]
    t = PairTable(p_same, 64, geom.grid, geom.core, 0.0314, align_tau=0.015, aug_cfg=c, aug_seed=7)
    seen = set()
    for e in range(4):
        for i in range(len(t)):
            it = t[(e, i)]
            assert torch.equal(it["fake"], it["real"]), f"identical pair diverged under augmentation (epoch {e}, item {i})"
            seen.add(it["aug_op"])
    assert seen >= {-1, 0, 1, 2, 3, 4, 5}, f"ops seen on pairs {seen}"
    t2 = PairTable(f["pairs"], 64, geom.grid, geom.core, 0.0314, align_tau=0.015, aug_cfg=c, aug_seed=7,
                   group_of=dict(zip(*[read_split({"csv": f["train"], "scheme": "eight"})[k] for k in ("img_path", "group")])))
    raw = PairTable(f["pairs"], 64, geom.grid, geom.core, 0.0314, align_tau=0.015)
    for i in range(len(t2)):
        it, r = t2[(3, i)], raw[i]
        prm = bd_aug.draw(bd_aug.item_rng(7, 3, "pairs", i), bd_aug.config(c))
        fu, su = load_image_u8(pr.fake_path[i], 64), load_image_u8(pr.real_path[i], 64)
        assert torch.equal(it["fake"], to_tensor(bd_aug.apply(fu, prm))) and torch.equal(it["real"], to_tensor(bd_aug.apply(su, prm)))
        assert torch.equal(it["frac"], torch.flip(r["frac"], dims=[1]) if prm["flip"] else r["frac"])
        assert int(it["fake_group"]) >= 0 and int(it["real_group"]) >= 0
    # stability pair: the nuisance view of an augmented image keeps the AUGMENTED face pixels (background kinds)
    ds = AugImageTable(read_split({"csv": f["train"], "scheme": "eight"}), 64, "bicubic", c, seed=7, stream="train")
    x = torch.stack([ds[(0, i)]["image"] for i in range(16)])
    y = torch.tensor([int(ds[(0, i)]["label"]) for i in range(16)])
    v, kind = sample_views(x, y, cfg["cf"], geom, torch.Generator().manual_seed(3))
    bg = torch.isin(kind, torch.tensor(BACKGROUND_KINDS))
    assert bg.any() and float((v[bg] - x[bg])[:, :, geom.core].abs().max()) == 0.0


@case
def a6_pipeline_range_and_identity():
    _, p = real_image()
    from PIL import Image
    a, b = load_image(p, 224), to_tensor(load_image_u8(p, 224))
    old = torch.from_numpy(np.asarray(Image.open(p).convert("RGB").resize((224, 224), Image.BICUBIC), dtype=np.uint8)
                           .copy()).permute(2, 0, 1).float() / 255.0
    assert torch.equal(a, b) and torch.equal(a, old) and a.dtype == torch.float32
    assert float(a.min()) >= 0 and float(a.max()) <= 1
    REPORT["ffpp_native_size"] = Image.open(p).size


# ================================================================================================ B. ESA
def _esa(s, y, g, levels=("intersection",), n_min=4):
    return L.esa_loss(s, y, g, {"levels": list(levels), "n_min": n_min}, "eight")


@case
def b1_esa_hand_value_gradient_and_skip():
    u = torch.tensor([0., 1, 2, 3, 10, 11, 12, 13, 20, 21, 22], dtype=torch.float64)
    s = torch.sinh(u).requires_grad_(True)
    y = torch.zeros(11, dtype=torch.long)
    g = torch.tensor([0] * 4 + [1] * 4 + [2] * 3)
    loss, st = _esa(s, y, g)
    # pool (n = 11; q 0.5/0.75/0.9/0.95 -> 1-based order stats ceil(q n) = 6, 9, 10, 11): [11, 20, 21, 22];
    # group 0 (n = 4 -> order stats 2, 3, 4, 4): [1, 2, 3, 3]; group 1: [11, 12, 13, 13]; group 2 (n = 3 < 4) skipped
    exp = ((10 + 18 + 18 + 19) + (0 + 8 + 8 + 9)) / 2
    assert abs(float(loss) - exp) < 1e-9 and st["esa_cells"] == 2, (float(loss), st)
    loss.backward()
    gu = (s.grad * torch.sqrt(1 + s.detach() ** 2)).numpy()     # d loss / d u (u = asinh s), cells averaged (/2)
    # group 0: every quantile below the pool -> -1 on u=1 (q .5), u=2 (q .75), u=3 (q .9, .95); group 1: q .5 ties the
    # pool exactly (|0| has zero subgradient), then -1 on u=12 (q .75) and u=13 (q .9, .95)
    want = np.array([0, -0.5, -0.5, -1, 0, 0, -0.5, -1, 0, 0, 0])
    assert np.allclose(gu, want, atol=1e-9), gu
    assert np.all(gu[8:] == 0), "skipped group received gradient (pool must be stop-gradient)"


@case
def b2_esa_zero_identical_positive_shift_and_levels():
    vals_r, vals_f = [-3., -1., 0.5, 2.], [1., 3., 4., 6.]
    s = torch.tensor(sum([vals_r + vals_f for _ in range(8)], []), requires_grad=True)
    y = torch.tensor(sum([[0] * 4 + [1] * 4 for _ in range(8)], []))
    g = torch.tensor(sum([[k] * 8 for k in range(8)], []))
    loss, st = L.esa_loss(s, y, g, {}, "eight")
    assert float(loss) == 0.0 and st["esa_cells"] == 2 * (2 + 4 + 8), (float(loss), st)
    s2 = s.detach().clone()
    s2[(g == 3) & (y == 0)] += 3.0                      # one group's reals shifted up (higher FPR tail)
    s2.requires_grad_(True)
    loss2, st2 = L.esa_loss(s2, y, g, {}, "eight")
    assert float(loss2) > 0.1, float(loss2)
    loss2.backward()
    gr = s2.grad[(g == 3) & (y == 0)]
    assert float(gr.sum()) > 0 and bool((gr >= 0).all()), gr      # descent lowers the shifted reals
    REPORT["esa_shift"] = {"loss": float(loss2), "cells": st2["esa_cells"]}


# ================================================================================================ D. group-AUC floor
@case
def d1_group_auc_floor_sign():
    r = torch.tensor([-2., -1, 0, 1])
    f_good, f_bad = torch.tensor([2., 3, 4, 5]), torch.tensor([-1.5, 0.5, 1.5, 2])
    s = torch.cat([r, f_good] * 7 + [r, f_bad]).requires_grad_(True)
    y = torch.tensor(([0] * 4 + [1] * 4) * 8)
    g = torch.tensor(sum([[k] * 8 for k in range(8)], []))
    loss, st = L.group_auc_floor_loss(s, y, g, {"levels": ["intersection"]}, "eight")
    assert float(loss) > 0 and st["gauc_active"] == 1, (float(loss), st)
    loss.backward()
    bad = g == 7
    assert float(s.grad[bad & (y == 1)].sum()) < 0 and float(s.grad[bad & (y == 0)].sum()) > 0, s.grad[bad]
    assert float(s.grad[~bad].abs().max()) == 0.0, "groups above the pool must get no gradient (pool stop-grad)"
    s3 = torch.cat([r, f_good] * 8)
    l3, _ = L.group_auc_floor_loss(s3, y, g, {}, "eight")
    assert float(l3) < 1e-6, float(l3)
    l4, st4 = L.group_auc_floor_loss(s3[:12], y[:12], g[:12], {"levels": ["intersection"]}, "eight")
    assert st4["gauc_groups"] == 1, st4                          # the second group has < 4 fakes -> skipped


# ================================================================================================ C. multi-layer evidence
@case
def c1_stub_hooks_shapes_start_equivalence_gradients():
    f = fx()
    cb, cm = stub_cfg("BD2", f), stub_cfg("BD3", f)
    mb, mm = build(cb), build(cm)
    sd = mb.state_dict()
    missing, unexpected = mm.load_state_dict(sd, strict=False)
    assert not unexpected and all(k.startswith(("ml_heads.", "ml_gate_raw", "ml_c0", "ml_r")) for k in missing), missing
    rec = {}
    blocks = mm.backbone.visual.transformer.resblocks
    hs = [blk.register_forward_hook(lambda m, i, o, k=k: rec.__setitem__(k, o.detach().clone())) for k, blk in enumerate(blocks)]
    x = torch.rand(6, 3, 64, 64)
    mb.eval(); mm.eval()
    with torch.no_grad():
        ob, om = mb(x), mm(x)
    for h in hs:
        h.remove()
    assert om["patch_ml"].shape == (6, mm.grid, mm.grid) and float(mm.ml_gate()) == 0.0
    man = torch.stack([head(rec[b - 1].permute(1, 0, 2)[:, 1:, :].reshape(6, mm.grid, mm.grid, -1)).squeeze(-1)
                       for b, head in zip([8, 12, 16, 20], mm.ml_heads)]).mean(0)
    assert float((man - om["patch_ml"]).abs().max()) < 1e-6, "hooked tokens are not blocks 8/12/16/20 (1-based)"
    d0 = float((ob["score"] - om["score"]).abs().max())
    mb.train(); mm.train()
    with torch.no_grad():
        mb(x), mm(x)                                     # first train-mode forward sets the local-norm references
    mb.eval(); mm.eval()
    with torch.no_grad():
        d1 = float((mb(x)["score"] - mm(x)["score"]).abs().max())
    assert d0 < 1e-4 and d1 < 1e-4, (d0, d1)
    assert float(mm.ml_c0) == float(mm.loc_c0), "ml.c0 = base must reuse the existing map's reference"
    mm.train()
    out = mm(x)
    out["score"].sum().backward()
    gh = sum(float(p.grad.abs().sum()) for p in mm.ml_heads.parameters() if p.grad is not None)
    assert float(mm.ml_gate_raw.grad.abs()) > 0 and gh == 0.0, (float(mm.ml_gate_raw.grad), gh)
    mm.zero_grad(set_to_none=True)
    with torch.no_grad():
        mm.ml_gate_raw.fill_(0.5)
    mm(x)["score"].sum().backward()
    gh = sum(float(p.grad.abs().sum()) for p in mm.ml_heads.parameters() if p.grad is not None)
    ga = sum(float(p.grad.abs().sum()) for n, p in mm.backbone.named_parameters() if p.grad is not None)
    assert gh > 0 and ga > 0, (gh, ga)
    REPORT["stub_start_maxdiff"] = [d0, d1]
    # phase-1 style load through engine.build_model: a state dict WITHOUT ml_* keys
    ck = os.path.join(ARGS.scratch, "fake_p1.pt")
    torch.save({"model": mb.state_dict()}, ck)
    m2 = build(stub_cfg("BD3", f, [f"model.init_ckpt={ck}"])).eval()
    with torch.no_grad():
        assert float((m2(x)["score"] - mb.eval()(x)["score"]).abs().max()) < 1e-4
    try:
        load_config("configs/bd/BD3.yaml", ["model.multilayer.blocks=[8, 12, 16, 25]", "model.backbone=stub",
                                            "model.stub={image: 64, patch: 8, dim: 32, layers: 24, heads: 2}"])
        build(stub_cfg("BD3", f, ["model.multilayer.blocks=[8, 12, 16, 25]"]))
        raise AssertionError("block 25 accepted on a 24-block backbone")
    except ValueError:
        pass


@case
def c2_real_clip_start_equivalence():
    if ARGS.skip_real:
        print("      (skipped: --skip_real)")
        return
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    init = f"{CFES}/runs/clip_s7_p1/last.pt"
    df, _ = real_image()
    import pandas as pd
    sel = pd.concat([g.sample(2, random_state=0) for _, g in df.groupby("label")])
    x = torch.stack([load_image(p, 224) for p in sel.img_path])
    cb = load_config("configs/bd/base_E.yaml", [f"model.init_ckpt={init}", "data.num_workers=0"])
    mb = build(cb).eval()
    with torch.no_grad():
        zb = mb(x)["score"]
    del mb
    cm = load_config("configs/bd/BD3.yaml", [f"model.init_ckpt={init}", "data.num_workers=0"])
    mm = build(cm).eval()
    with torch.no_grad():
        om = mm(x)
    d0 = float((om["score"] - zb).abs().max())
    assert om["patch_ml"].shape == (4, 16, 16) and float(mm.ml_gate()) == 0.0
    blocks = mm.backbone.visual.transformer.resblocks
    assert len(blocks) == 24 and [b for b in mm.ml_blocks] == [8, 12, 16, 20]
    assert all(len(blocks[b - 1]._forward_hooks) == 1 for b in mm.ml_blocks)
    assert sum(len(blk._forward_hooks) for blk in blocks) == 4
    mm.train()
    with torch.no_grad():
        om1 = mm(x)
    mm.eval()
    d1 = float((om1["score"] - zb).abs().max())
    assert d0 < 1e-4 and d1 < 1e-4, (d0, d1)
    mm.train()
    with torch.no_grad():
        mm.ml_gate_raw.fill_(0.3)
    mm(x[:2])["score"].sum().backward()
    gh = sum(float(p.grad.abs().sum()) for p in mm.ml_heads.parameters() if p.grad is not None)
    ga = sum(float(p.grad.abs().sum()) for n, p in mm.backbone.named_parameters() if p.grad is not None)
    assert gh > 0 and ga > 0 and mm.ml_gate_raw.grad is not None
    REPORT["real_clip_start_maxdiff"] = {"eval": d0, "first_train_forward": d1, "score_range": [float(zb.min()), float(zb.max())],
                                         "ml_c0": float(mm.ml_c0), "loc_c0": float(mm.loc_c0)}


@case
def c3_real_resnet_multistage_start_equivalence():
    if ARGS.skip_real:
        print("      (skipped: --skip_real)")
        return
    init = f"{CFES}/runs/resnet_s7_p1/last.pt"
    df, _ = real_image()
    import pandas as pd
    sel = pd.concat([g.sample(2, random_state=1) for _, g in df.groupby("label")])
    x = torch.stack([load_image(p, 224, "bilinear") for p in sel.img_path])
    mb = build(load_config("configs/bd_resnet/base_E.yaml", [f"model.init_ckpt={init}"])).eval()
    mm = build(load_config("configs/bd_resnet/BD3.yaml", [f"model.init_ckpt={init}"])).eval()
    rec = {}
    hs = [mm.net.resnet[i].register_forward_hook(lambda m, a, o, i=i: rec.__setitem__(i, tuple(o.shape))) for i in (5, 6, 7)]
    with torch.no_grad():
        zb, om = mb(x)["score"], mm(x)
    for h in hs:
        h.remove()
    assert rec == {5: (4, 512, 28, 28), 6: (4, 1024, 14, 14), 7: (4, 2048, 7, 7)}, rec
    assert om["patch_ml"].shape == (4, 7, 7) and float(mm.ml_gate()) == 0.0
    d0 = float((om["score"] - zb).abs().max())
    assert d0 < 1e-4, d0
    mm.train()
    out = mm(x)
    out["score"].sum().backward()
    assert float(mm.ml_gate_raw.grad.abs()) > 0
    assert sum(float(p.grad.abs().sum()) for p in mm.ml_heads.parameters() if p.grad is not None) == 0.0
    mm.zero_grad(set_to_none=True)
    with torch.no_grad():
        mm.ml_gate_raw.fill_(0.3)
    mm(x)["score"].sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in mm.ml_heads.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in mm.net.resnet.parameters()), "phase-2 ResNet trunk must stay frozen"
    REPORT["real_resnet_start_maxdiff"] = d0


# ================================================================================================ configs / rules
@case
def e1_every_config_loads_with_intended_flags():
    n = 0
    for d, _, fs in os.walk("configs"):
        for fn in fs:
            if fn.endswith(".yaml"):
                load_config(os.path.join(d, fn))
                n += 1
    want = {"base_E": (0, 0, 0, 0), "BD0": (0, 1, 0, 0), "BD1": (1, 0, 0, 0), "BD2": (1, 1, 0, 0), "BD3": (1, 1, 1, 0),
            "BD4": (1, 1, 1, 1)}
    for fam in ("bd", "bd_resnet"):
        for arm, (a, b, c, d) in want.items():
            cf = load_config(f"configs/{fam}/{arm}.yaml")
            got = (int(cf["data"]["aug"]["enabled"]), int(cf["loss"]["esa"]["weight"] > 0),
                   int(cf["model"]["multilayer"]["enabled"]), int(cf["loss"]["gauc"]["weight"] > 0))
            assert got == (a, b, c, d), (fam, arm, got)
            assert cf["strat"]["per_cell"] == (4 if b else 2), (fam, arm)
            assert cf["loss"]["fprgap"]["weight"] == 1.0 and cf["select"]["metric"] == "val_auc_plus_inter_es_auc"
            if b:
                assert cf["loss"]["esa"]["weight"] == 1.0 and cf["loss"]["esa"]["q_real"] == [0.5, 0.75, 0.9, 0.95]
                assert cf["loss"]["esa"]["q_fake"] == [0.05, 0.1, 0.25, 0.5] and cf["loss"]["esa"]["n_min"] == 4
            if d:
                assert cf["loss"]["gauc"]["weight"] == 0.5 and cf["loss"]["gauc"]["tau"] == 1.0
            if a:
                ag = cf["data"]["aug"]
                assert ag["p"] == 0.6 and ag["hflip"] == 0.5 and ag["jpeg_q"] == [30, 90] and ag["downup_scale"] == [0.4, 0.9]
            terms = engine.active_terms(cf, have_pairs=True)
            assert ("esa" in terms) == bool(b) and ("gauc" in terms) == bool(d) and "fprgap" in terms, (arm, terms)
    E, B0 = load_config("configs/rev4/clip_E.yaml", ["loss.fprgap.weight=1"]), load_config("configs/bd/base_E.yaml")
    keep = lambda c: {k: v for k, v in c.items() if k not in ("select", "bd")}        # noqa: E731
    e2, b2 = keep(E), keep(B0)
    for c in (e2, b2):
        c["loss"] = {k: v for k, v in c["loss"].items() if k not in ("esa", "gauc")}
        c["data"] = {k: v for k, v in c["data"].items() if k != "aug"}
        c["model"] = {k: v for k, v in c["model"].items() if k != "multilayer"}
    assert e2 == b2, "configs/bd/base_E.yaml differs from rev4 Arm E w1 beyond selection and the (off) BD blocks"
    REPORT["configs_loaded"] = n


@case
def e2_selection_metric_and_codex_fixes():
    TR = script("train")
    res = {"auc": 96.5, "ding@0.5": {"intersection": {"es_AUC": 83.25}}}
    assert TR.selection_value(res, "val_auc_plus_inter_es_auc") == 96.5 + 83.25
    # F3: validation AUC from logits under eval.rank_metrics (Codex reproduction: sigmoid ties -> 75 instead of 100)
    import pandas as pd
    z = np.array([-50, 40, 41, 42] * 8, dtype=np.float32)
    y = np.array([0, 0, 1, 1] * 8)
    gg = np.repeat(np.arange(8), 4)
    orig = engine.predict
    engine.predict = lambda *a, **k: (1 / (1 + np.exp(-z.astype(np.float64))), z)
    try:
        cfg = {"optim": {"amp": False}, "eval": {"prob_float64": True, "rank_metrics": True, "val_tpr": 0.9},
               "data": {"val": {"scheme": "eight"}}, "bd": {"log_val_ding": False}}
        r = TR.validate(None, cfg, torch.device("cpu"), pd.DataFrame({"label": y, "group": gg}), None)
    finally:
        engine.predict = orig
    assert abs(r["auc"] - 100.0) < 1e-9 and r["threshold_tpr_units"] == "logit", r["auc"]
    # F4: the val threshold is applied in logit units on every test set
    from cfes.metrics import evaluate_scores, rank_prob
    zt = np.array([-100., -1, 1, 2] * 4)
    yt = np.array([0, 0, 1, 1] * 4)
    gt = np.repeat(np.arange(4), 4)
    rr = evaluate_scores(yt, rank_prob(zt), gt, "eight", val_threshold=1.0, thr_scores=zt)
    assert rr["tpr@val"] == 100.0 and rr["fpr@val"] == 0.0 and rr["threshold_val_units"] == "logit"
    # F8: dataset-aware cluster keys
    assert cluster_key("/x/0_youtube_001_111.png") == "0_youtube_001"
    assert cluster_key("/x/1_Deepfakes_001_870_111.png") == "1_Deepfakes_001_870"
    assert cluster_key("/d/pixart_Asian_Female_105076007.jpg").startswith("img:")
    src = open(os.path.join(ROOT, "scripts", "evaluate.py")).read()
    assert "thr_scores=np.asarray(z, dtype=np.float64)[keep]" in src and "np.asarray(zv, dtype=np.float64)" in src


@case
def e3_ding_published_values_match_the_pdf_text():
    DT = script("ding_tables")
    pub = {"xception": BM.DING_PUB, "resnet": BM.DING_PUB_R50}
    key = {"DFDC": "dfdc", "Celeb-DF": "celebdf", "DFD": "dfd", "FF++": "ffpp_test"}
    n = 0
    for t, fam in ((2, "xception"), (3, "resnet")):
        for ds, meth, bb, v in DT.published(t):
            if meth == "Ours":
                assert np.allclose(v, pub[fam][key[ds]]), (t, ds, v, pub[fam][key[ds]])
                n += 1
    txt = open("refs/Ding_CVPR2026.txt").read()
    i = txt.index("Table 1. Intra-domain")
    rows = [list(map(float, re.findall(r"\d+\.\d+", l)[-4:])) for l in txt[i:i + 6000].splitlines()
            if re.match(r"^\s*(Gender|Race|Intersection)?\s*Ours\s+Xception", l.strip()) or
            (l.strip().startswith("Ours") and "Xception" in l)]
    assert len(rows) >= 3, rows
    ff = BM.DING_PUB["ffpp_test"]
    assert rows[0] == [ff[0], ff[1], ff[2], ff[9]] and rows[1][:3] == ff[3:6] and rows[2][:3] == ff[6:9], rows[:3]
    assert n == 7, n


@case
def e4_winner_and_promotion_rule():
    BR = script("bd_report")
    mk = lambda n, sc, auc: {"name": n, "val_ding_score": sc, "val_auc": auc}       # noqa: E731
    refs = [mk("CFES-D_s7", 4, 96.07), mk("armE_w1_s7", 6, 96.88), mk("armE_w3_s7", 5, 97.01), mk("armE_w10_s7", 6, 96.56)]
    d = BR.winner_rule([mk("BD0_s7", 5, 97.2), mk("BD2_s7", 7, 96.0), mk("BD3_s7", 7, 96.4)], refs)
    assert d["winner"] == "BD3_s7" and d["promoted"] and d["recipe"] == "BD3" and d["best_reference"] == "armE_w1_s7"
    d = BR.winner_rule([mk("BD1_s7", 6, 95.0)], refs)
    assert d["promoted"] and d["recipe"] == "BD1"                     # >= the best reference promotes
    d = BR.winner_rule([mk("BD1_s7", 5, 99.0), mk("BD4_s7", 3, 99.5)], refs)
    assert not d["promoted"] and d["recipe"] == "armE_w1_s7" and d["winner"] == "BD1_s7"
    assert BR.resnet_components({"decision": {"winner": "BD3_s7", "promoted": True}}) == "BD3"
    assert BR.resnet_components({"decision": {"winner": "BD1_s7", "promoted": False}}) == "base"


# ================================================================================================ dry run of the task chain
def sh(args, env_extra, timeout=3600):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", **env_extra)
    r = subprocess.run(["bash", "run/bd_run.sh", *args], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"bd_run.sh {args} rc={r.returncode}\n{(r.stdout + r.stderr)[-3000:]}")
    return r.stdout


@case
def f1_dry_run_whole_task_chain():
    if ARGS.skip_e2e:
        print("      (skipped: --skip_e2e)")
        return
    f = fx()
    out = os.path.join(ARGS.scratch, "bd_out")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "results", "refs"), exist_ok=True)
    for fn in os.listdir(os.path.join(ROOT, "results", "refs")):         # real reference rows (saved scores, read only)
        if fn.endswith(".json"):
            shutil.copy(os.path.join(ROOT, "results", "refs", fn), os.path.join(out, "results", "refs", fn))
    # CESEL_LOCK_BEAT=1: the RUNLOCK heartbeat's sleep child would otherwise hold the captured stdout pipe open for up
    # to 60 s after each bd_run.sh call (harmless under the real runner, which logs to files)
    env = {"BD_DRY": "1", "BD_OUT": out, "BD_WORKERS": "2", "CESEL_LOCK_BEAT": "1",
           "BD_DRY_SETFILE": FX.write_setfile(os.path.join(ARGS.scratch, "clip.set"), FX.clip_overrides(f)),
           "BD_DRY_RESNET_SETFILE": FX.write_setfile(os.path.join(ARGS.scratch, "resnet.set"), FX.resnet_overrides(f)),
           "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "4")}
    t0 = time.time()
    so = sh(["smoke", os.path.join(out, "work", "smoke")], env)
    sm = json.load(open(os.path.join(out, "runs", "SMOKE_OK.json")))
    assert sm["clip"]["ok"] and sm["resnet"]["ok"], so[-2000:]
    for arm in ("BD0", "BD4"):
        sh(["arm", arm, "7"], env)
        run = os.path.join(out, "runs", f"{arm}_s7")
        ev = json.load(open(os.path.join(run, "eval", "bd_eval.json")))
        assert os.path.exists(os.path.join(run, "TRAINED")) and ev["complete"], arm
        st = [json.loads(l) for l in open(os.path.join(run, "train_log.jsonl")) if '"event": "step"' in l]
        want = {"esa"} | ({"gauc"} if arm == "BD4" else set())
        assert want <= set(st[-1]) and all(math.isfinite(l["total"]) for l in st), (arm, st[-1].keys())
        assert (st[-1].get("aug_frac") is not None) == (arm == "BD4")
        assert (st[-1].get("ml_gate") is not None) == (arm == "BD4")
        assert st[-1]["bd_pool_n"] == 16 + 16 + 8 + 64, st[-1]["bd_pool_n"]    # batch + views + pair images + strat
        ep = [json.loads(l) for l in open(os.path.join(run, "train_log.jsonl")) if '"event": "epoch"' in l]
        best = max(ep, key=lambda e: e["select_value"])
        assert abs(best["select_value"] - (best["val_auc"] + best["val_ding"]["intersection"]["es_AUC"])) < 1e-9
        sh(["arm", arm, "7"], env)                                       # resumable: second call is a no-op
    r = subprocess.run([PYD, "scripts/bd_report.py", "screen", "--runs_root", os.path.join(out, "runs"),
                        "--results", os.path.join(out, "results"), "--arms", "BD0,BD4"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    sc = json.load(open(os.path.join(out, "results", "bd_screen.json")))
    for s in ("17", "27"):
        sh(["winner", s], env)
        sh(["winner_eval", s], env)
    sh(["final"], env)
    fin = json.load(open(os.path.join(out, "results", "bd_final.json")))
    assert len(fin["seeds"]) == 3 and not fin["missing"], fin.get("missing")
    for s in ("7", "17", "27"):
        sh(["resnet", s], env)
    sh(["resnet_final"], env)
    fr = json.load(open(os.path.join(out, "results", "bd_final_resnet.json")))
    assert fr["published_table"] == "Table 3 (ResNet-50)" and len(fr["seeds"]) == 3
    REPORT["dry_chain"] = {"minutes": round((time.time() - t0) / 60, 1), "screen_decision": sc["decision"],
                           "final_recipe": fin["recipe"], "final_total": fin["total_wins"],
                           "resnet_recipe": fr["recipe"], "smoke": {k: v["steps"] for k, v in sm.items()}}


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default="/tmp/claude-156900507/-home-sachin-chaudhary/3e01ae21-517f-480b-a721-cb2115803d08/scratchpad/bd_checks")
    ap.add_argument("--skip_real", action="store_true", help="skip the real CLIP / ResNet-50 checks")
    ap.add_argument("--skip_e2e", action="store_true")
    ap.add_argument("--only", default="")
    ARGS = ap.parse_args()
    ARGS.only = [x for x in ARGS.only.split(",") if x]
    os.makedirs(ARGS.scratch, exist_ok=True)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    run_cases()
    REPORT.pop("fx", None)
    print("REPORT", json.dumps(REPORT, default=float))
    n = sum(s == "PASS" for _, s, _ in RESULTS)
    print(f"{n}/{len(RESULTS)} passed")
    if n == len(RESULTS) and not ARGS.only:
        print("ALL_PASS")
    sys.exit(0 if n == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
