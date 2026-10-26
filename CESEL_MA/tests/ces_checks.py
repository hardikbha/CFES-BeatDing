"""CES CPU smoke tests (no GPU, no training run). Prints one line per check and ALL_PASS at the end.

  ~/.conda/envs/pulid/bin/python tests/ces_checks.py            # stub-backbone checks + ONE real-CLIP pass (2 images)
  CES_SKIP_CLIP=1 ~/.conda/envs/pulid/bin/python tests/ces_checks.py   # stub checks only
Temporary files (synthetic images for the trainer smoke run) go to $TMPDIR.
"""
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cfes import ces_engine as CE  # noqa: E402
from cfes import ces_losses as CL  # noqa: E402
from cfes.ces_model import CUES, CESDetector  # noqa: E402
from cfes.ces_state import CESState  # noqa: E402
from cfes.ces_validation import decide_validated, load_validated  # noqa: E402
from cfes.config import load_config  # noqa: E402

PY = sys.executable
RESULTS = []


def check(name, ok, info=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({info})" if info else ""), flush=True)


def stub_cfg(arm="full", **over):
    cfg = load_config(os.path.join(ROOT, "configs", "ffpp", f"{arm}.yaml"),
                      ["model.backend=stub", "data.input_size=64", "model.stub={dim: 32, grid: 8}", "optim.amp=false",
                       "loss.inv.warmup_steps=0", "ces.dim=16", "ces.hidden=32", "ces.freq_hidden=16",
                       "ces.freq_widths=[8, 8, 16, 16]"] + [f"{k}={v}" for k, v in over.items()])
    return cfg


def synth_batch(n, gen, n_groups=8, n_cond=2, size=64):
    y = torch.arange(n) % 2
    g = torch.randint(0, n_groups, (n,), generator=gen)
    c = torch.where(y == 1, 1 + torch.randint(0, n_cond, (n,), generator=gen), torch.zeros(n, dtype=torch.long))
    return dict(image=torch.rand(n, 3, size, size, generator=gen), label=y, group=g, cond=c, index=torch.arange(n))


def synth_pairs(n, gen, size=64):
    return dict(fake=torch.rand(n, 3, size, size, generator=gen), real=torch.rand(n, 3, size, size, generator=gen),
                fake_group=torch.randint(0, 8, (n,), generator=gen), real_group=torch.randint(0, 8, (n,), generator=gen),
                fake_cond=torch.ones(n, dtype=torch.long), real_cond=torch.zeros(n, dtype=torch.long),
                frac=torch.zeros(n, 1), aligned=torch.zeros(n, dtype=torch.bool), index=torch.arange(n))


def warm_state(model, cfg, gen, steps=5):
    st = CE.build_state(cfg, model, torch.device("cpu"))
    with torch.no_grad():
        for _ in range(steps):
            b = synth_batch(32, gen)
            C = model(b["image"])["cues"]
            st.update(C, b["label"], b["group"])
            st.probe_step(C, b["group"])
    return st


# ------------------------------------------------------------------------------------------------------------------
def t_configs():
    arms = ["B0", "sens", "inv", "floor", "full", "C_randcue", "C_shufgroup", "C_novalid", "B0_cfesD"]
    expect = {"B0": set(), "sens": {"sens_ces"}, "inv": {"inv"}, "floor": {"floor"},
              "full": {"sens_ces", "inv", "floor"}, "C_randcue": {"sens_ces", "inv", "floor"},
              "C_shufgroup": {"sens_ces", "inv", "floor"}, "C_novalid": {"sens_ces", "inv", "floor"},
              "B0_cfesD": {"stab", "sens", "risk"}}
    ok, bad = True, []
    for ds in ("ffpp", "ff2"):
        for arm in arms:
            cfg = load_config(os.path.join(ROOT, "configs", ds, f"{arm}.yaml"))
            terms = CE.active_terms(cfg, have_pairs=True)
            extra = set(terms) - {"det", "pair_det", "strat_det", "orth"}
            good = extra == expect[arm]
            good &= (cfg["ces"]["replace"] == "random") == (arm == "C_randcue")
            good &= bool(cfg["loss"]["floor"]["shuffle_groups"]) == (arm == "C_shufgroup")
            good &= bool(cfg["loss"]["inv"]["validity_mask"]) == (arm != "C_novalid")
            good &= os.path.exists(cfg["data"]["train"]["csv"]) and os.path.exists(cfg["pairs"]["csv"])
            good &= cfg["cf"]["face_mask"]["cy"] == (0.578 if ds == "ff2" else 0.48)
            if not good:
                ok, bad = False, bad + [f"{ds}/{arm}:{sorted(extra)}"]
    check("config loading of every arm (ffpp + ff2, 9 arms each; terms, control flags, data paths)", ok, ",".join(bad))


def t_shapes(model, cfg):
    x = torch.rand(6, 3, 64, 64)
    out = model(x)
    ok = out["score"].shape == (6,) and out["cues"].shape == (6, 4, cfg["ces"]["dim"]) and out["s_k"].shape == (6, 4)
    ok &= torch.allclose(out["s_k"].sum(1) + model.head.b, out["score"], atol=1e-5)
    g = model.geom.describe()
    check("shapes: score (N,), cues (N,4,d), s_k (N,4); additive head s = sum_k s_k + b", ok,
          f"grid {model.grid}, core {g['core_cells']} ring {g['ring_cells']} cells, freq canvas {g['freq_canvas']}")


def t_additive(model, cfg, gen):
    st = warm_state(model, cfg, gen)
    b = synth_batch(10, gen)
    with torch.no_grad():
        out = model(b["image"])
        C, s, y = out["cues"], out["score"], b["label"]
        R = CL.replacement(C, y, st)
        lin = dict(cfg["ces"], margin="linear")
        U, sm = CL.utilities(model, C, s, y, R, lin, path="closed")
        U2, sm2 = CL.utilities(model, C, s, y, R, lin, path="general")
        full = CL.margin(s, y, lin) - CL.margin(model.score_from(R), y, lin)
        Ua, sma = CL.utilities(model, C, s, y, R, cfg["ces"])            # asinh margin: logit drops still additive
        drop_sum = (s[:, None] - sma).sum(1)
    e1 = float((U.sum(1) - full).abs().max())
    e2 = float((sm - sm2).abs().max())
    e3 = float((drop_sum - (s - model.score_from(R))).abs().max())
    check("additive head: sum_k u_k = full-replacement utility (linear margin); closed form = head re-evaluation",
          e1 < 1e-4 and e2 < 1e-4 and e3 < 1e-4, f"max err {e1:.1e}, {e2:.1e}, asinh logit-drop sum {e3:.1e}")


def t_self(model, cfg, gen, mlp_model):
    b = synth_batch(8, gen)
    errs = []
    with torch.no_grad():
        for m in (model, mlp_model):
            out = m(b["image"])
            for path in (("closed", "general") if m.head.kind == "additive" else ("general",)):
                U, _ = CL.utilities(m, out["cues"], out["score"], b["label"], out["cues"], cfg["ces"], path=path)
                errs.append(float(U.abs().max()))
    check("utility of a cue replaced by itself = 0 (additive closed form + general; mlp head)", max(errs) < 1e-5,
          f"max |u| {max(errs):.1e}")


def t_gradients(model, cfg, gen):
    st = warm_state(model, cfg, gen)
    model.train()
    terms = CE.active_terms(cfg, have_pairs=True)
    vals, stats = CE.compute_ces_losses(model, st, synth_batch(16, gen), synth_pairs(4, gen), cfg, gen,
                                        torch.device("cpu"), terms, strat_batch=synth_batch(32, gen))
    model.zero_grad(set_to_none=True)
    total = sum(terms[k] * v for k, v in vals.items() if k in terms)
    total.backward()
    norms = {}
    for name, mod in [("backbone_adapter", model.backbone.adapter)] + [(f"branch_{k}", model.branches[k]) for k in CUES] \
            + [("head", model.head)]:
        norms[name] = sum(float(p.grad.abs().sum()) for p in mod.parameters() if p.grad is not None)
    ok = all(v > 0 for v in norms.values()) and torch.isfinite(total)
    check("gradients reach every branch, the head and the backbone adapters (full arm loss)", ok,
          " ".join(f"{k}={v:.2e}" for k, v in norms.items()) + f" | terms {sorted(vals)} inv_items {stats.get('inv_items')}")
    # each CES term alone reaches the cue branches it should. Random stub cues have near-equal group utilities, so the
    # floor margin is set negative here (test only) to make every cell active.
    res = {}
    cfg = dict(cfg, loss=dict(cfg["loss"], floor=dict(cfg["loss"]["floor"], delta=-1.0, n_min=2)))
    for k in ("sens_ces", "inv", "floor"):
        model.zero_grad(set_to_none=True)
        st2 = warm_state(model, cfg, gen)
        v, s = CE.compute_ces_losses(model, st2, synth_batch(16, gen), synth_pairs(4, gen), cfg, gen,
                                     torch.device("cpu"), terms, strat_batch=synth_batch(32, gen))
        v[k].backward()
        res[k] = (float(v[k]), sum(float(p.grad.abs().sum()) for p in model.branches.parameters() if p.grad is not None))
    check("each CES term alone is non-zero and back-propagates into the branches",
          all(a > 0 and b > 0 for a, b in res.values()), " ".join(f"{k}: L={a:.3g} |g|={b:.2e}" for k, (a, b) in res.items()))


def t_floor(gen):
    n, K = 64, 4
    y = torch.arange(n) % 2
    g = (torch.arange(n) // 2) % 4
    cond = y.clone()
    base = torch.rand(K) + 1.0
    U = base.repeat(n, 1).clone().requires_grad_(True)
    cfg = {"n_min": 4, "delta": 0.1}
    mask = torch.ones(K, dtype=torch.bool)
    l0, _ = CL.floor_loss(U, y, g, cond, mask, cfg)
    U2 = base.repeat(n, 1).clone()
    U2[(g == 2) & (y == 0)] -= 1.0                                # group 2's reals: utility 1.0 lower on every cue
    U2.requires_grad_(True)
    l1, s1 = CL.floor_loss(U2, y, g, cond, mask, cfg)
    l1.backward()
    # pooled real mean = base - 0.25 (4 equal-size groups); deficit of group 2 = 0.75 - delta = 0.65 per cue
    expect = K * 0.65 ** 2
    gr = U2.grad
    ok = float(l0) == 0.0 and abs(float(l1) - expect) < 1e-5 and s1["floor_active_cells"] == 1
    ok &= bool((gr[(g == 2) & (y == 0)] < 0).all()) and float(gr[~((g == 2) & (y == 0))].abs().sum()) == 0.0
    check("L_floor = 0 when groups are equal; > 0 (= K (0.75 - delta)^2) for one deficient group, gradient only raises it",
          ok, f"L_equal {float(l0):.3g}, L_deficient {float(l1):.4f} (expected {expect:.4f})")
    l_sh, _ = CL.floor_loss(U2.detach(), y, g, cond, mask, dict(cfg, shuffle_groups=True), torch.Generator().manual_seed(0))
    check("shuffled-group floor control runs and changes the floor", float(l_sh) != float(l1), f"{float(l_sh):.4f}")


def t_swap(gen):
    """Planted demographics: c = noise + a_g on a 2-D group subspace of every cue (4 groups at the corners of a
    square). After EMA + probe warm-up the swap must move the probe's prediction to the partner's group."""
    K, d, G = 4, 16, 4
    cfg = {"ces": {"ema": 0.1}, "loss": {"inv": {"rank": 2, "probe_lr": 1e-2}}}
    st = CESState(K, d, G, cfg)
    basis = torch.linalg.qr(torch.randn(d, 2, generator=gen))[0]
    corners = torch.tensor([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]]) * 3.0
    A = corners @ basis.T                                          # (G,d)

    def sample(n):
        g = torch.randint(0, G, (n,), generator=gen)
        y = torch.randint(0, 2, (n,), generator=gen)
        C = 0.5 * torch.randn(n, K, d, generator=gen) + A[g][:, None, :] + y[:, None, None] * 0.7
        return C, y, g
    for _ in range(300):
        C, y, g = sample(64)
        st.update(C, y, g)
        st.probe_step(C, g)
    C, y, g = sample(256)
    partner, has = CL.demographic_partner(y, g, gen)
    Cs = CL.demographic_swap(C, st, partner, has)
    lp, lp0 = st.probe_logits(Cs), st.probe_logits(C)
    acc_clean = float((lp0.argmax(1) == g).float().mean())
    to_target = float((lp.argmax(1) == g[partner])[has].float().mean())
    valid = CL.swap_validity(st, Cs, g, partner, has)
    Pk = st.U @ st.U.transpose(1, 2)                               # estimated projector per cue (K,d,d)
    proj_err = float(torch.einsum("nkd,kde->nke", (Cs - C[partner])[has], Pk).abs().max())   # partner's projection
    resid_err = float(torch.einsum("nkd,kde->nke", Cs - C, torch.eye(d) - Pk).abs().max())  # own complement kept
    true_err = float((torch.linalg.matrix_norm(Pk - basis @ basis.T, 2)).max())             # vs planted subspace
    # with no planted directions (pure noise) the probe is at chance and validity is low
    st0 = CESState(K, d, G, cfg)
    for _ in range(100):
        g0 = torch.randint(0, G, (64,), generator=gen); C0 = torch.randn(64, K, d, generator=gen)
        st0.update(C0, torch.randint(0, 2, (64,), generator=gen), g0); st0.probe_step(C0, g0)
    g0 = torch.randint(0, G, (256,), generator=gen); y0 = torch.randint(0, 2, (256,), generator=gen)
    C0 = torch.randn(256, K, d, generator=gen)
    p0, h0 = CL.demographic_partner(y0, g0, gen)
    v0 = CL.swap_validity(st0, CL.demographic_swap(C0, st0, p0, h0), g0, p0, h0)
    ok = acc_clean > 0.95 and to_target > 0.9 and float(valid[has].float().mean()) > 0.9 and proj_err < 1e-4 \
        and resid_err < 1e-4 and true_err < 0.1 and float(v0[h0].float().mean()) < 0.7
    check("demographic swap moves the probe to the partner's group on planted group directions", ok,
          f"probe acc clean {acc_clean:.2f}, swapped->target {to_target:.2f}, validity {float(valid[has].float().mean()):.2f},"
          f" projection err {proj_err:.1e}, complement err {resid_err:.1e}, |P_est - P_true| {true_err:.3f};"
          f" no-signal validity {float(v0[h0].float().mean()):.2f}")


def t_group_baseline(gen):
    """Amendment 1 (user decision 25 Oct): an UNPAIRED fake's removal baseline is the detached EMA real-cue mean of its
    OWN group (class mean while that group has no estimate); reals keep the fake class mean; a paired fake keeps its
    source's cue; ces.fake_baseline=class restores the class mean. Also checks the wiring in compute_ces_losses."""
    K, d, G = 4, 6, 8
    st = CESState(K, d, G, {"ces": {"ema": 0.5}})
    Cr = torch.randn(40, K, d, generator=gen)
    gr = torch.arange(40) % 3                                     # reals of groups 0, 1, 2 only
    Cf = torch.randn(10, K, d, generator=gen) + 3.0
    st.update(torch.cat([Cr, Cf]), torch.cat([torch.zeros(40), torch.ones(10)]).long(),
              torch.cat([gr, torch.full((10,), 5)]))
    C = torch.randn(6, K, d, generator=gen, requires_grad=True)
    y = torch.tensor([1, 1, 1, 0, 1, 0])
    g = torch.tensor([0, 2, 7, 1, 1, 4])                          # fake of group 7: no real estimate -> class mean
    src = torch.tensor([-1, -1, -1, -1, 5, -1])                   # row 4 = a pair fake whose source is row 5
    R = CL.replacement(C, y, st, src, "counterfactual", g=g, fake_baseline="group")
    Rc = CL.replacement(C, y, st, src, "counterfactual", g=g, fake_baseline="class")
    e = [float((R[0] - st.grp_mean[0, 0]).abs().max()), float((R[1] - st.grp_mean[0, 2]).abs().max()),
         float((R[2] - st.cls_mean[0]).abs().max()), float((R[3] - st.cls_mean[1]).abs().max()),
         float((R[4] - C[5]).abs().max()), float((R[5] - st.cls_mean[1]).abs().max()),
         float((Rc[0] - st.cls_mean[0]).abs().max()), float((Rc[1] - st.cls_mean[0]).abs().max())]
    ok = max(e) < 1e-6 and not R.requires_grad and float((st.grp_mean[0, 0] - st.cls_mean[0]).abs().max()) > 1e-3
    ok &= float((R[0] - Rc[0]).abs().max()) > 1e-3
    # wiring: the trainer's L_sens changes with the flag (same draws), the detection loss does not
    out = {}
    for fb in ("group", "class"):
        torch.manual_seed(0)
        cfg = stub_cfg("full", **{"ces.fake_baseline": fb})
        model = CESDetector(cfg)
        st2 = warm_state(model, cfg, torch.Generator().manual_seed(1))
        g2 = torch.Generator().manual_seed(5)
        vals, _ = CE.compute_ces_losses(model, st2, synth_batch(16, g2), synth_pairs(4, g2), cfg, g2, torch.device("cpu"),
                                        CE.active_terms(cfg, have_pairs=True), strat_batch=synth_batch(32, g2))
        out[fb] = {k: float(v) for k, v in vals.items()}
    ok &= out["group"]["det"] == out["class"]["det"] and out["group"]["sens_ces"] != out["class"]["sens_ces"]
    ok &= load_config(os.path.join(ROOT, "configs", "ffpp", "full.yaml"))["ces"]["fake_baseline"] == "group"
    check("unpaired-fake removal baseline = detached same-group real-cue EMA mean (class-mean fallback); reals = fake "
          "class mean; pair fake = its source; trainer wiring", ok,
          f"max errs {max(e):.1e}; L_sens group {out['group']['sens_ces']:.4f} vs class {out['class']['sens_ces']:.4f}")


def t_per_family(gen):
    """Amendment 2 (FF2 only): per-family validated masks / targets with global fallback for small families; L_sens
    uses each fake's family row (reals: global); the per-image loss equals the (K,) form when rows are equal; the flag
    is on for FF2 configs and off for FF++."""
    from cfes.ces_validation import decide_validated_per_family, load_validated_family
    rng = np.random.default_rng(0)
    n = 1200
    y = np.r_[np.zeros(400, int), np.ones(800, int)]
    cond = np.r_[np.zeros(400, int), np.ones(500, int), np.full(250, 2), np.full(50, 3)]   # fam a (500), b (250), c (50)
    U = rng.normal(0.5, 0.3, (n, 4))
    U[cond == 2, 2] = rng.normal(-0.4, 0.3, 250)          # family b: boundary utility negative -> not validated
    glob = decide_validated(U, y, {"bootstrap": 200})
    pf = decide_validated_per_family(U, y, cond, ["a", "b", "c"], {"bootstrap": 200, "family_min_n": 200}, glob)
    ok = pf["source"] == {"a": "family", "b": "family", "c": "global_fallback"}
    ok &= pf["valid"]["b"]["bound"] is False and pf["valid"]["a"]["bound"] is True and glob["valid"]["bound"] is True
    ok &= pf["valid"]["c"] == glob["valid"] and pf["target"]["b"]["bound"] is None
    d = tempfile.mkdtemp()
    json.dump(dict(glob, per_family=pf), open(os.path.join(d, "v.json"), "w"))
    cfg = {"ces": {"validated_json": os.path.join(d, "v.json"), "validation": {"per_family": True}},
           "data": {"families": ["a", "b", "c"]}}
    V, T = load_validated_family(cfg, 5)
    ok &= V.shape == (5, 4) and bool(V[2, 2]) is False and bool(V[0, 2]) is True and bool(V[4].eq(V[0]).all())
    ok &= load_validated_family({"ces": {"validation": {"per_family": False}}, "data": {}}, 5) is None
    u = torch.randn(7, 4, generator=gen)
    v1, t1 = torch.tensor([True, False, True, True]), torch.tensor([0.5, 0.0, 0.3, 0.2])
    l1, _ = CL.sens_loss(u, v1, t1)
    l2, _ = CL.sens_loss(u, v1[None].repeat(7, 1), t1[None].repeat(7, 1))
    ok &= abs(float(l1) - float(l2)) < 1e-6
    ok &= load_config(os.path.join(ROOT, "configs", "ff2", "full.yaml"))["ces"]["validation"]["per_family"] is True
    ok &= load_config(os.path.join(ROOT, "configs", "ffpp", "full.yaml"))["ces"]["validation"]["per_family"] is False
    # wiring: a family row that drops every cue for condition 2 changes the trainer's L_sens
    torch.manual_seed(0)
    cfg2 = stub_cfg("full")
    model = CESDetector(cfg2)
    outs = []
    for Vf in (torch.ones(4, 4, dtype=torch.bool), torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1]]).bool()):
        st = warm_state(model, cfg2, torch.Generator().manual_seed(1))
        g2 = torch.Generator().manual_seed(5)
        vals, _ = CE.compute_ces_losses(model, st, synth_batch(16, g2), synth_pairs(4, g2), cfg2, g2, torch.device("cpu"),
                                        CE.active_terms(cfg2, have_pairs=True), strat_batch=synth_batch(32, g2),
                                        fam=(Vf, torch.full((4, 4), 0.5)))
        outs.append(float(vals["sens_ces"]))
    ok &= outs[0] != outs[1]
    check("per-family cue validation (FF2 flag): family masks, small-family global fallback, per-image L_sens = (K,) "
          "form for equal rows, trainer wiring; FF++ flag off", ok,
          f"sources {pf['source']}; fam b bound valid={pf['valid']['b']['bound']}; L_sens all-valid {outs[0]:.4f} vs "
          f"fam-2-off {outs[1]:.4f}")


def t_controls(gen):
    out = {}
    for arm in ("full", "C_randcue", "C_shufgroup", "C_novalid", "B0", "B0_cfesD"):
        torch.manual_seed(0)
        cfg = stub_cfg(arm, **{"loss.floor.delta": -0.5, "loss.floor.n_min": 2})   # test only: make floor cells active
        model = CESDetector(cfg)
        st = warm_state(model, cfg, torch.Generator().manual_seed(1))
        model.train()
        terms = CE.active_terms(cfg, have_pairs=True)
        fgeom = CE.face_geometry(cfg, model, torch.device("cpu"))
        g2 = torch.Generator().manual_seed(5)
        vals, stats = CE.compute_ces_losses(model, st, synth_batch(16, g2), synth_pairs(4, g2), cfg, g2,
                                            torch.device("cpu"), terms, strat_batch=synth_batch(32, g2), fgeom=fgeom)
        tot = sum(terms[k] * v for k, v in vals.items() if k in terms)
        tot.backward()
        out[arm] = (float(tot), {k: round(float(v), 4) for k, v in vals.items()}, stats)
    ok = all(np.isfinite(v[0]) for v in out.values())
    ok &= out["C_randcue"][1]["sens_ces"] != out["full"][1]["sens_ces"]
    ok &= out["C_shufgroup"][1]["floor"] != out["full"][1]["floor"]
    ok &= out["C_novalid"][2].get("inv_items", 0) >= out["full"][2].get("inv_items", 0)
    same = lambda a, keys: all(out[a][1][k] == out["full"][1][k] for k in keys)       # controls change only their term
    ok &= same("C_randcue", ["det", "inv"]) and same("C_shufgroup", ["det", "sens_ces", "inv"]) \
        and same("C_novalid", ["det", "sens_ces", "floor"])
    ok &= {"stab", "sens", "risk"} <= set(out["B0_cfesD"][1])
    check("controls run (random-cue, shuffled-group floor, no-validity inv) and differ from full; B0 and B0+CFES-D run", ok,
          "; ".join(f"{a}: {v[1]}" for a, v in out.items()))


def t_validated(gen):
    n = 400
    y = np.arange(n) % 2
    U = np.stack([np.random.default_rng(0).normal(1.0, 0.5, n), np.random.default_rng(1).normal(0.0, 0.5, n),
                  np.random.default_rng(2).normal(0.05, 0.02, n), np.random.default_rng(3).normal(-0.5, 0.5, n)], 1)
    res = decide_validated(U, y, {"bootstrap": 300})
    ok = [res["valid"][k] for k in CUES] == [True, False, True, False]
    ok &= abs(res["target"]["tex"] - np.clip(np.median(U[:, 0]), 0.1, 2.0)) < 1e-9 and res["target"]["bound"] == 0.1
    p = os.path.join(tempfile.mkdtemp(), "v.json")
    json.dump(res, open(p, "w"))
    v, t, _ = load_validated({"ces": {"validated_json": p}})
    ok &= v.tolist() == [True, False, True, False]
    v0, t0, src = load_validated({"ces": {}})
    ok &= bool(v0.all()) and float(t0[0]) == 0.5 and src == "default_all_valid"
    check("validated-cue rule (bootstrap CI > 0, median target clamp) and JSON round trip; default = all valid, 0.5",
          ok, json.dumps(res["valid"]))


def t_params_timing(model, cfg, gen):
    rep = model.parameter_report()
    st = warm_state(model, cfg, gen)
    model.train()
    terms = CE.active_terms(cfg, have_pairs=True)
    t0 = time.time()
    for _ in range(3):
        vals, _ = CE.compute_ces_losses(model, st, synth_batch(64, gen), synth_pairs(8, gen), cfg, gen,
                                        torch.device("cpu"), terms, strat_batch=synth_batch(96, gen))
        sum(terms[k] * v for k, v in vals.items() if k in terms).backward()
    dt = (time.time() - t0) / 3
    check("stub step timing (176 images at 64 px, full arm, CPU)", dt < 60, f"{dt:.2f} s/step; params {rep}")
    # full-size branches on a CLIP-shaped feature (no CLIP weights needed): exact parameter counts
    cfg_full = load_config(os.path.join(ROOT, "configs", "ffpp", "full.yaml"),
                           ["model.backend=stub", "model.stub={dim: 1024, grid: 16}", "data.input_size=224"])
    m = CESDetector(cfg_full)
    r = m.parameter_report()
    branch = {k: r[f"branch_{k}"] for k in CUES}
    check("full-size cue branches (D=1024, G=16, d=128): frequency CNN < 1 M parameters", branch["freq"] < 1_000_000,
          f"{branch}, head {r['head']}, geometry core {m.geom.describe()['core_cells']} ring {m.geom.describe()['ring_cells']}")
    return branch


def t_trainer_smoke():
    """End-to-end CPU run of scripts/train_ces.py and scripts/ces_validate_cues.py on 64 synthetic images (stub
    backbone, 2 epochs x 2 steps): checks the script path, checkpoint format (ces_state) and resume/init from B0."""
    from PIL import Image
    import pandas as pd
    tmp = tempfile.mkdtemp(prefix="ces_smoke_")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(64):
        lab = i % 2
        name = f"{lab}_{'Deepfakes' if i % 4 == 1 else 'Face2Face'}_{i:03d}_{i + 100:03d}_{i}.png" if lab else \
            f"0_youtube_{i:03d}_{i}.png"
        p = os.path.join(tmp, name)
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(p)
        rows.append(dict(img_path=p, label=lab, ismale=1, isasian=-1, iswhite=1, isblack=-1, intersec_label=(i // 2) % 8))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(tmp, "train.csv"), index=False)
    df.iloc[:32].to_csv(os.path.join(tmp, "val.csv"), index=False)
    pairs = pd.DataFrame({"fake_path": df.img_path[df.label == 1].values[:16], "real_path": df.img_path[df.label == 0].values[:16]})
    pairs.to_csv(os.path.join(tmp, "pairs.csv"), index=False)
    common = ["model.backend=stub", "data.input_size=64", "model.stub={dim: 16, grid: 8}", "ces.dim=8", "ces.hidden=16",
              "ces.freq_hidden=8", "ces.freq_widths=[4, 4, 8, 8]", f"data.train.csv={tmp}/train.csv",
              f"data.val.csv=[{tmp}/val.csv]", f"pairs.csv={tmp}/pairs.csv", "pairs.per_step=2", "data.batch_size=8",
              "data.num_workers=0", "data.steps_per_epoch=2", "strat.per_cell=1", "optim.epochs=2", "optim.warmup_steps=1",
              "eval.batch_size=16", "log_every=1", "loss.inv.warmup_steps=1"]
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    out_b0 = os.path.join(tmp, "run_B0")
    r1 = subprocess.run([PY, os.path.join(ROOT, "scripts", "train_ces.py"), "--config",
                         os.path.join(ROOT, "configs", "ffpp", "B0.yaml"), "--out", out_b0, "--device", "cpu",
                         "--set", *common], capture_output=True, text=True, env=env)
    vj = os.path.join(out_b0, "validated_cues.json")
    r2 = subprocess.run([PY, os.path.join(ROOT, "scripts", "ces_validate_cues.py"), "--config",
                         os.path.join(ROOT, "configs", "ffpp", "B0.yaml"), "--ckpt", os.path.join(out_b0, "last.pt"),
                         "--out", vj, "--set", *common], capture_output=True, text=True, env=env)
    out_full = os.path.join(tmp, "run_full")
    r3 = subprocess.run([PY, os.path.join(ROOT, "scripts", "train_ces.py"), "--config",
                         os.path.join(ROOT, "configs", "ffpp", "full.yaml"), "--out", out_full, "--device", "cpu",
                         "--set", *common, f"model.init_ckpt={out_b0}/last.pt", f"ces.validated_json={vj}"],
                        capture_output=True, text=True, env=env)
    ok = r1.returncode == 0 and r2.returncode == 0 and r3.returncode == 0
    info = ""
    if ok:
        ck = torch.load(os.path.join(out_full, "last.pt"), map_location="cpu", weights_only=False)
        ok &= "ces_state" in ck["state"] and os.path.exists(os.path.join(out_full, "TRAINED"))
        logs = [json.loads(l) for l in open(os.path.join(out_full, "train_log.jsonl"))]
        steps = [l for l in logs if l.get("event") == "step"]
        ok &= len(steps) == 4 and all(k in steps[-1] for k in ("sens_ces", "inv", "floor"))
        info = f"full-arm last step: " + ", ".join(f"{k}={steps[-1][k]:.4f}" for k in ("det", "sens_ces", "inv", "floor")) \
            + f"; validated {json.load(open(vj))['valid']}"
    else:
        info = (r1.stderr[-800:] + r2.stderr[-800:] + r3.stderr[-800:])
    check("trainer + validation-pass scripts run end to end on CPU (stub, synthetic images, B0 -> json -> full arm)", ok, info)


def t_real_clip():
    """ONE real CLIP ViT-L/14 forward + backward on 2 FF++ training images (OpenAI weights from ~/.cache/clip)."""
    cfg = load_config(os.path.join(ROOT, "configs", "ffpp", "full.yaml"), ["optim.amp=false"])
    import pandas as pd
    from cfes.data import load_image
    df = pd.read_csv(cfg["data"]["train"]["csv"])
    paths = [df.img_path[df.label == 0].iloc[0], df.img_path[df.label == 1].iloc[0]]
    x = torch.stack([load_image(p, 224) for p in paths])
    t0 = time.time()
    torch.manual_seed(0)
    model = CESDetector(cfg)
    t_load = time.time() - t0
    rep = model.parameter_report()
    t0 = time.time()
    out = model(x)
    t_fwd = time.time() - t0
    t0 = time.time()
    (out["score"].sum() + out["cues"].pow(2).mean()).backward()
    t_bwd = time.time() - t0
    ad = [p for p in model.backbone.adapter_parameters()]
    frac = sum(int(p.grad is not None and float(p.grad.abs().sum()) > 0) for p in ad) / len(ad)
    g = model.geom.describe()
    ok = out["cues"].shape == (2, 4, 128) and torch.isfinite(out["score"]).all() and frac > 0.99
    ok &= model.grid == 16 and model.backbone.feat_dim == 1024 and g["core_cells"] == 46 and g["ring_cells"] == 44
    check("real CLIP ViT-L/14: forward/backward on 2 FF++ images; grads reach the rank-1 adapters", ok,
          f"load {t_load:.0f}s fwd {t_fwd:.1f}s bwd {t_bwd:.1f}s (CPU, 2 imgs); adapters with grad {frac:.2f}; "
          f"scores {out['score'].tolist()}; params {rep}")
    return rep


def main():
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    t_configs()
    cfg = stub_cfg("full")
    model = CESDetector(cfg)
    mlp_model = CESDetector(stub_cfg("full", **{"ces.head": "mlp"}))
    t_shapes(model, cfg)
    t_additive(model, cfg, gen)
    t_self(model, cfg, gen, mlp_model)
    t_gradients(model, cfg, gen)
    t_floor(gen)
    t_swap(gen)
    t_controls(gen)
    t_group_baseline(gen)
    t_per_family(gen)
    t_validated(gen)
    t_params_timing(model, cfg, gen)
    t_trainer_smoke()
    if not os.environ.get("CES_SKIP_CLIP"):
        t_real_clip()
    failed = [n for n, ok in RESULTS if not ok]
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    print("ALL_PASS" if not failed else "FAILED: " + "; ".join(failed))
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
