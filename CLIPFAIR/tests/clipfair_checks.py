"""CLIPFAIR CPU checks (no GPU, no training run). One line per check, ALL_PASS at the end.

  ~/.conda/envs/pulid/bin/python tests/clipfair_checks.py                 # incl. ONE real-CLIP pass (p1 checkpoint)
  CLIPFAIR_SKIP_CLIP=1 ~/.conda/envs/pulid/bin/python tests/clipfair_checks.py

What is checked
  1. configs: the four method configs load; data / pairs / strat / optim / select / eval / orth equal CESel B0's
     resolved config (agents/CESEL/configs/ffpp/B0.yaml, read only); budget 10 x 1976 steps.
  2. ports vs the OFFICIAL code, executed from the official source files (methods extracted with ast; CPU shims only
     for .cuda / torch.cuda.FloatTensor): DAW-FDD and DAG-FDD get_losses; PG-FDD get_train_losses (whole loss incl.
     LDAM, contrastive, reconstruction, bi-level CVaR); Ding SoftNearestNeighborLoss per channel; SAM (sam.py).
  3. CVaR dual = brute-force CVaR; released GDA fairness term carries no gradient, the gda_grad variant does.
  4. per method on the stub backbone: every loss finite, gradients reach the adapters (Ding: frozen by design, head only)
     and every head / module the official loss trains.
  5. real CLIP ViT-L/14 from CFES clip_s7_p1: backbone loads strictly, forward + backward on 2 FF++ images reach the
     rank-1 adapters; Ding's 2-way head initialised from p1's cls_head reproduces p1's CLS logit exactly.
"""
import ast
import os
import random
import sys
import textwrap

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import optimize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cfes import clipfair_engine as FE  # noqa: E402
from cfes import clipfair_losses as FL  # noqa: E402
from cfes.clipfair_models import build_model  # noqa: E402
from cfes.config import load_config  # noqa: E402

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
AIF = f"{R}/agents/CFES/third_party/ff2id/AI-Face-FairnessBench/training"
FG = f"{R}/agents/CFES/third_party/ff2id/Fairness-Generalization/training"
DM = f"{R}/agents/DINGREPRO_LIN/code/Ding_mod"
RESULTS = []


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({info})" if info else ""), flush=True)


def official(path, cls, names, ns=None, base="object", replace=()):
    """Build a class from the OFFICIAL source: the methods `names` of class `cls` in `path`, verbatim (plus optional
    textual CPU shims in `replace`)."""
    src = open(path).read()
    tree = ast.parse(src)
    node = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls][0]
    body = []
    for f in node.body:
        if isinstance(f, ast.FunctionDef) and f.name in names:
            body.append(textwrap.indent(textwrap.dedent(ast.get_source_segment(src, f)), "    "))
    code = f"class {cls}({base}):\n" + "\n".join(body)
    for a, b in replace:
        code = code.replace(a, b)
    g = dict(torch=torch, nn=nn, F=F, np=np, optimize=optimize, random=random)
    g.update(ns or {})
    exec(code, g)
    return g[cls]


def stub_cfg(method, **over):
    return load_config(os.path.join(ROOT, "configs", "ffpp", f"{method}.yaml"),
                       ["model.backend=stub", "data.input_size=64", "model.stub={dim: 32, grid: 8}", "optim.amp=false"]
                       + [f"{k}={v}" for k, v in over.items()])


def synth_pool(gen, n=8, npair=2, nst=16, size=64):
    y = torch.arange(n) % 2
    batch = dict(image=torch.rand(n, 3, size, size, generator=gen), label=y,
                 group=torch.randint(0, 8, (n,), generator=gen), index=torch.arange(n))
    pb = dict(fake=torch.rand(npair, 3, size, size, generator=gen), real=torch.rand(npair, 3, size, size, generator=gen),
              fake_group=torch.randint(0, 8, (npair,), generator=gen), real_group=torch.randint(0, 8, (npair,), generator=gen),
              index=torch.arange(npair))
    sb = dict(image=torch.rand(nst, 3, size, size, generator=gen), label=torch.arange(nst) % 2,
              group=torch.arange(nst) % 8, index=torch.arange(nst) + n)

    class Spe:
        by_index = torch.where(torch.arange(n + nst) % 2 == 1, 1 + torch.arange(n + nst) % 5, 0)
        pair_fake = torch.tensor([2, 4] * npair)[:npair]
    return FE.assemble(batch, pb, sb, torch.device("cpu"), Spe)


def main():
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    # ---------------------------------------------------------------- 1. configs
    b0 = load_config(f"{R}/agents/CESEL/configs/ffpp/B0.yaml")
    for m in ("daw", "dag", "pgfdd", "ding"):
        c = load_config(os.path.join(ROOT, "configs", "ffpp", f"{m}.yaml"))
        same = all(c[k] == b0[k] for k in ("data", "pairs", "strat", "optim", "select", "eval", "seed", "model"))
        same &= c["loss"]["orth"] == b0["loss"]["orth"]
        check(f"config {m}: loads, method set, data/samplers/optim/select/eval/model == CESel B0, budget 10 x 1976",
              c["clipfair"]["method"] == m and same and c["optim"]["epochs"] * c["data"]["steps_per_epoch"] == 19760,
              f"terms {FE.active_terms(c)}")

    # ---------------------------------------------------------------- 2. ports vs official code
    bce_none = lambda inp, t: F.binary_cross_entropy_with_logits(inp.squeeze(1), t.float(), reduction="none")  # daw_bce_loss.py:10,24
    for name, file, cls in (("DAW-FDD", "daw_fdd.py", "DawFddDetector"), ("DAG-FDD", "dag_fdd.py", "DagFddDetector")):
        Off = official(f"{AIF}/detectors/{file}", cls, {"threshplus_tensor", "search_func", "searched_lamda_loss", "get_losses"})
        off = Off()
        off.loss_func = bce_none
        z = torch.randn(48, generator=gen) * 2
        y = (torch.arange(48) % 2)
        g = torch.randint(0, 8, (48,), generator=gen)
        z1 = z.clone().requires_grad_(True)
        lo = off.get_losses({"label": y, "intersec_label": g}, {"cls": z1[:, None]})["overall"]
        lo.backward()
        z2 = z.clone().requires_grad_(True)
        lm = (FL.daw_fdd_loss(z2, y, g)[0] if name == "DAW-FDD" else FL.dag_fdd_loss(z2, y)[0])
        lm.backward()
        check(f"{name} loss == official get_losses (value and gradient)",
              abs(float(lo) - float(lm)) < 1e-5 and torch.allclose(z1.grad, z2.grad, atol=1e-6),
              f"{float(lo):.6f} vs {float(lm):.6f}")

    # PG-FDD whole training loss
    LDAM = official(f"{FG}/loss/bal_loss.py", "LDAMLoss", {"__init__", "forward"}, base="nn.Module",
                    replace=[("torch.cuda.FloatTensor(m_list)", "torch.FloatTensor(m_list)"),
                             ("index.type(torch.cuda.FloatTensor)", "index.type(torch.FloatTensor)"),
                             ])
    con_src = open(f"{FG}/loss/contrastive_regularization.py").read()
    gcon = dict(torch=torch, nn=nn, F=F, random=random)
    exec(compile("from collections import defaultdict\n" + ast.get_source_segment(
        con_src, [n for n in ast.parse(con_src).body if isinstance(n, ast.FunctionDef)][0]), "con", "exec"), gcon)
    Con = official(f"{FG}/loss/contrastive_regularization.py", "ContrastiveLoss", {"__init__", "contrastive_loss", "forward"},
                   ns={"swap_spe_features": gcon["swap_spe_features"]}, base="nn.Module")
    PG = official(f"{FG}/detectors/fair_df_detector.py", "FairDetector",
                  {"threshplus_tensor", "search_func", "search_func_smooth", "searched_lamda_loss",
                   "searched_lamda_loss_smooth", "get_train_losses"})
    cfgp = stub_cfg("pgfdd")["clipfair"]["pgfdd"]
    off = PG()
    off.loss_func = {"cls": nn.CrossEntropyLoss(), "spe": nn.CrossEntropyLoss(), "con": Con(margin=3.0), "rec": nn.L1Loss(),
                     "fair": LDAM(cls_num_list=cfgp["cls_num_list"]), "fuse": nn.CrossEntropyLoss(reduction="none")}
    B = 12
    img = torch.rand(2 * B, 3, 16, 16, generator=gen) * 2 - 1
    lab = torch.cat([torch.zeros(B), torch.ones(B)]).long()
    spe = torch.cat([torch.zeros(B), 1 + torch.arange(B) % 5]).long()
    grp = torch.randint(0, 8, (2 * B,), generator=gen)
    mk = lambda *s: (torch.randn(*s, generator=gen)).requires_grad_(True)
    pd = {"cls": mk(2 * B, 2), "cls_spe": mk(2 * B, 6), "cls_fair": mk(2 * B, 8), "cls_fused": mk(2 * B, 2),
          "feat": mk(2 * B, 256), "feat_spe": mk(2 * B, 256),
          "recontruction_imgs": tuple(torch.tanh(mk(B, 3, 16, 16)) for _ in range(4))}
    pd["prob_fused"] = torch.softmax(pd["cls_fused"], 1)[:, 1]
    random.seed(5)
    lo = off.get_train_losses({"image": img, "label": lab, "label_spe": spe, "intersec_label": grp}, pd)
    random.seed(5)
    real_img, fake_img = img.chunk(2)
    rec_1, rec_2, srec_1, srec_2 = pd["recontruction_imgs"]
    l1 = lambda a, b: F.l1_loss(b.float(), a)
    mine = {"sha": F.cross_entropy(pd["cls"], lab), "spe": F.cross_entropy(pd["cls_spe"], spe),
            "rec": l1(fake_img, srec_1) + l1(real_img, srec_2) + l1(fake_img, rec_2) + l1(real_img, rec_1),
            "con": FL.contrastive_loss(pd["feat"], pd["feat_spe"], spe, random, 3.0),
            "fair": FL.ldam_loss(pd["cls_fair"], grp, cfgp["cls_num_list"], cfgp["ldam_max_m"], cfgp["ldam_s"]),
            "fuse": FL.bilevel_fuse_loss(pd["cls_fused"], lab, grp)[0]}
    tot = FE.weighted_total(mine, {k: float(v) for k, v in cfgp["weights"].items()})
    names = {"common": "sha", "specific": "spe", "reconstruction": "rec", "contrastive": "con", "fairness": "fair",
             "fusion": "fuse"}
    diffs = {v: abs(float(lo[k]) - float(mine[v])) for k, v in names.items()}
    ga = torch.autograd.grad(lo["overall"], [pd["cls_fused"], pd["cls_fair"]], retain_graph=True)
    gb = torch.autograd.grad(tot, [pd["cls_fused"], pd["cls_fair"]], retain_graph=True)
    check("PG-FDD total loss == official get_train_losses (each term, total, gradients)",
          max(diffs.values()) < 1e-5 and abs(float(lo["overall"]) - float(tot)) < 1e-5
          and all(torch.allclose(a, b, atol=1e-6) for a, b in zip(ga, gb)),
          f"overall {float(lo['overall']):.5f} vs {float(tot):.5f}; max term diff {max(diffs.values()):.2e}")

    # Ding SNNL (official class, .cuda() shimmed to a no-op on CPU)
    SN = official(f"{DM}/SFD.py", "SoftNearestNeighborLoss", {"__init__", "pairwise_cos_distance", "forward"},
                  base="nn.Module", replace=[(".cuda()", "")])
    sn = SN()
    feat = torch.randn(32, 20, 49, generator=gen)
    lab8 = torch.randint(0, 8, (32,), generator=gen)
    ref = torch.stack([sn(feat[:, c].reshape(32, -1), lab8) for c in range(20)])
    vec = FL.snnl_per_channel(feat, lab8)
    check("Ding SFD: vectorised per-channel SNNL == official SoftNearestNeighborLoss channel loop",
          torch.allclose(ref, vec, atol=1e-5), f"max diff {float((ref - vec).abs().max()):.2e}")
    avg = np.random.RandomState(0).rand(1024)
    ch, thr, n = FL.select_sfd_channels(avg, 2.0)
    check("Ding SFD: 2% of 1024 channels -> 20 lowest-SNNL channels (SFD.py:172-188)",
          n == 20 and len(ch) == 20 and set(ch) == set(np.argsort(avg)[:20].tolist()) and ch == sorted(ch, key=lambda c: avg[c]))

    # SAM (official sam.py) vs the trainer's two-pass implementation
    sam_src = open(f"{FG}/sam.py").read()
    gs = {"torch": torch}
    exec(sam_src, gs)
    torch.manual_seed(1)
    net0 = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 1))
    xs, ys = torch.randn(16, 6), torch.randn(16, 1)
    lossf = lambda m: ((m(xs) - ys) ** 2).mean()
    import copy
    n1, n2 = copy.deepcopy(net0), copy.deepcopy(net0)
    so = gs["SAM"](n1.parameters(), torch.optim.SGD, lr=0.1, momentum=0.9, weight_decay=5e-3)
    lossf(n1).backward(); so.first_step(zero_grad=True); lossf(n1).backward(); so.second_step(zero_grad=True)
    o2 = torch.optim.SGD(n2.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-3)
    lossf(n2).backward()
    old, _ = FE.sam_first_step(list(n2.parameters()), 0.05)
    o2.zero_grad(); lossf(n2).backward(); FE.sam_restore(old); o2.step()
    check("PG-FDD SAM two-pass step == official sam.py SAM (rho 0.05)",
          all(torch.allclose(a, b, atol=1e-7) for a, b in zip(n1.parameters(), n2.parameters())))

    # ---------------------------------------------------------------- 3. CVaR and GDA gradient
    l = torch.rand(40, generator=gen) * 3
    cv, _ = FL.cvar(l, 0.5)
    check("CVaR dual (fminbound) == brute-force mean of the top 50% losses", abs(float(cv) - float(l.sort().values[20:].mean())) < 1e-4,
          f"{float(cv):.5f} vs {float(l.sort().values[20:].mean()):.5f}")
    lg = torch.randn(30, 2, generator=gen, requires_grad=True)
    gg = torch.arange(30) % 8
    f_rel, _ = FL.gda_fair_loss(lg, gg, torch.Generator().manual_seed(0), grad=False)
    f_gr, _ = FL.gda_fair_loss(lg, gg, torch.Generator().manual_seed(0), grad=True)
    gr = torch.autograd.grad(f_gr, lg)[0]
    check("Ding GDA: released fairness term has no gradient (no_grad); gda_grad variant has one; same value",
          (not f_rel.requires_grad) and float(gr.norm()) > 0 and abs(float(f_rel) - float(f_gr)) < 1e-6,
          f"sinkhorn {float(f_rel):.4f}, grad-variant |g| {float(gr.norm()):.3e}")

    # ---------------------------------------------------------------- 4. stub models: finite losses, gradient paths
    import json
    import tempfile
    tmp = tempfile.mkdtemp()
    json.dump({"channels": [3]}, open(os.path.join(tmp, "ch.json"), "w"))
    need = {"daw": ["backbone.adapter", "head."], "dag": ["backbone.adapter", "head."],
            "pgfdd": ["backbone.adapter", "encoder_f", "encoder_c", "encoder_fair", "block_spe", "block_sha", "block_fair",
                      "con_gan", "head_spe", "head_sha", "head_fair", "head_fused"],
            "ding": ["head."]}
    for m in ("daw", "dag", "pgfdd", "ding"):
        cfg = stub_cfg(m, **({"clipfair.ding.channels_json": os.path.join(tmp, "ch.json")} if m == "ding" else {}))
        torch.manual_seed(0)
        model = build_model(cfg, torch.device("cpu")).train()
        terms = FE.active_terms(cfg)
        pool = synth_pool(torch.Generator().manual_seed(1))
        vals, stats = FE.compute_losses(model, pool, cfg, torch.Generator().manual_seed(2), torch.device("cpu"), terms)
        tot = FE.weighted_total(vals, terms)
        tot.backward()
        gn = {}
        for n_, p in model.named_parameters():
            if p.grad is not None:
                gn[n_] = gn.get(n_, 0) + float(p.grad.norm())
        reach = {k: any(n_.startswith(k) and v > 0 for n_, v in gn.items()) for k in need[m]}
        extra = ""
        ok = all(torch.isfinite(v) for v in vals.values()) and all(reach.values())
        if m == "ding":
            frozen = all(p.grad is None for p in model.backbone.parameters())
            ok &= frozen and int((model.keep == 0).sum()) == 1 and float(model.keep[3]) == 0 and "orth" not in terms
            extra = f"backbone frozen {frozen}, keep[3]=0, gda_fair requires_grad {vals['gda_fair'].requires_grad}"
        if m == "pgfdd":
            ok &= pool["real_idx"].numel() == pool["fake_idx"].numel() == 6
            ok &= bool((pool["y"][pool["real_idx"]] == 0).all()) and bool((pool["y"][pool["fake_idx"]] == 1).all())
            model.eval()
            with torch.no_grad():
                s_eval = model(pool["x"])["score"]
            ok &= s_eval.shape == (pool["x"].shape[0],)
            extra = f"halves 6+6; eval score shape ok"
        check(f"stub {m}: losses finite, gradients reach {need[m]}",
              ok, f"terms {sorted(vals)} " + " ".join(f"{k}={float(v):.3f}" for k, v in vals.items()) + "; " + extra)

    # ---------------------------------------------------------------- 5. real CLIP
    if os.environ.get("CLIPFAIR_SKIP_CLIP"):
        print("[SKIP] real-CLIP checks (CLIPFAIR_SKIP_CLIP set)")
    else:
        p1 = f"{R}/agents/CFES/runs/clip_s7_p1/last.pt"
        cfg = load_config(os.path.join(ROOT, "configs", "ffpp", "dag.yaml"), [f"model.init_backbone_ckpt={p1}", "optim.amp=false"])
        model = build_model(cfg, torch.device("cpu")).train()
        from cfes.data import load_image, read_split
        df = read_split(cfg["data"]["train"])
        rows = [df[df.label == 0].iloc[0], df[df.label == 1].iloc[0]]
        x = torch.stack([load_image(r.img_path, 224) for r in rows])
        y = torch.tensor([0, 1])
        out = model(x)
        loss = FL.dag_fdd_loss(out["score"], y)[0] + 0.1 * model.orthogonality_loss()
        loss.backward()
        ad = [p for n_, p in model.backbone.named_parameters() if p.requires_grad]
        gnorm = sum(float(p.grad.norm()) for p in ad if p.grad is not None)
        check("real CLIP: p1 backbone loads strictly, DAG loss + orth backward reaches the rank-1 adapters",
              model.init_info.get("backbone_keys", 0) > 800 and gnorm > 0 and torch.isfinite(loss),
              f"{model.init_info.get('backbone_keys')} keys, {len(ad)} adapter tensors, |g| {gnorm:.3e}")
        cfgd = load_config(os.path.join(ROOT, "configs", "ffpp", "ding.yaml"), [f"model.init_backbone_ckpt={p1}", "optim.amp=false"])
        md = build_model(cfgd, torch.device("cpu")).eval()
        sd = torch.load(p1, map_location="cpu", weights_only=False)["model"]
        with torch.no_grad():
            o = md(x)
            ref = o["z"] @ sd["cls_head.weight"][0].float() + sd["cls_head.bias"][0].float()
        check("real CLIP: Ding 2-way head from p1 cls_head reproduces p1's CLS logit (no channel zeroed)",
              torch.allclose(o["score"], ref, atol=1e-4) and all(p.requires_grad is False for p in md.backbone.parameters()),
              f"scores {o['score'].tolist()} vs {ref.tolist()}")

    n_ok = sum(ok for _, ok in RESULTS)
    print(f"{n_ok}/{len(RESULTS)} checks passed")
    print("ALL_PASS" if n_ok == len(RESULTS) else "SOME_FAILED")


if __name__ == "__main__":
    main()
