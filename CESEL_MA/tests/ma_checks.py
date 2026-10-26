"""CESel-MA CPU checks (no GPU). One line per check, ALL_PASS at the end.

  ~/.conda/envs/pulid/bin/python tests/ma_checks.py
Covers: label attachment (age / ITA bins, missing codes, group fallback), L_floor-MA (kernel floor 0 when uniform and
> 0 for a planted deficit at one tone; age and gender floors with exact values; reals weighted 2x; the gender x race
part equals the CESel floor), frequency warm-up schedule (factor, loss present / zero, gradients), the multi-axis
selection gap and selection value, the configs (terms, guard), and an end-to-end CPU trainer run (stub backbone,
synthetic images and labels): B0 -> validated JSON -> CESel-MA.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from cfes import ces_engine as CE  # noqa: E402
from cfes import ces_losses as CL  # noqa: E402
from cfes import ma_labels as MAL  # noqa: E402
from cfes import ma_losses as ML  # noqa: E402
from cfes.ces_model import CUES, CESDetector  # noqa: E402
from cfes.config import load_config  # noqa: E402

PY = sys.executable
RES = []
FCFG = {"n_min": 4, "delta": 0.1}


def check(name, ok, info=""):
    RES.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({info})" if info else ""), flush=True)


def macfg(**k):
    d = {"n_eff_min": 4, "real_weight": 2.0, "floor_axes": ["gxr", "gender", "age", "skin"]}
    d.update(k)
    return d


def pool(n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 2
    cond = torch.where(y == 1, torch.ones(n, dtype=torch.long), torch.zeros(n, dtype=torch.long))
    gender = (torch.arange(n) // 2) % 2
    age = (torch.arange(n) // 4) % 4
    ita = torch.linspace(-30, 60, n)[torch.randperm(n, generator=g)]
    grp = (torch.arange(n) // 2) % 8
    M = torch.stack([gender.float(), age.float(), ita, torch.zeros(n), grp.float()], 1)
    return y, cond, M


# ------------------------------------------------------------------------------------------------------------------
def t_labels():
    tmp = tempfile.mkdtemp(prefix="ma_lab_")
    df = pd.DataFrame({"img_path": [f"/x/{i}.png" for i in range(8)], "label": [0, 1] * 4, "group": list(range(8))})
    lab = pd.DataFrame({"img_path": [f"/x/{i}.png" for i in range(7)], "gender": [0, 1, 0, 1, 0, 1, np.nan],
                        "age": [29, 30, 44, 45, 59, 60, np.nan], "ita": [-10, 10, 30, 50, np.nan, 5, 0],
                        "group_gr": [7, 6, np.nan, 4, 3, 2, 1]})
    lab.to_csv(f"{tmp}/l.csv", index=False)
    cfg = {"ma": {"labels": {"train": f"{tmp}/l.csv"}, "ita_edges": [0, 20, 40], "age_edges": [30, 45, 60]}}
    out, rep = MAL.attach(df, cfg, "train")
    ok = out.ma_age_bin.tolist() == [0, 1, 1, 2, 2, 3, -1, -1]
    ok &= out.ma_ita_bin.tolist() == [0, 1, 2, 3, -1, 1, 1, -1]
    ok &= out.ma_gender.tolist() == [0, 1, 0, 1, 0, 1, -1, -1]
    ok &= out.ma_group.tolist() == [7, 6, 2, 4, 3, 2, 1, 7]          # NaN / unlabelled -> the Lin group
    ok &= rep["labelled"] == 7 and list(out.img_path) == list(df.img_path)
    check("labels: age bins <30/30-44/45-59/60+, ITA bins by edges, missing -> -1, group fallback, row order kept", ok,
          f"age {out.ma_age_bin.tolist()} ita {out.ma_ita_bin.tolist()}")


def t_kernel():
    y, cond, M = pool(96)
    K = 4
    mask = torch.ones(K, dtype=torch.bool)
    anchors, h = ML.kernel_anchors(-30, 60, 8)
    u = torch.full((96, K), 0.7)
    l0, s0 = ML.floor_ma_loss(u, y, cond, M, mask, FCFG, macfg(floor_axes=["skin"]), anchors, h)
    # planted deficit at the darkest tones (ITA < -15), real class only, cue 0
    u2 = u.clone()
    dark = (M[:, 2] < -15) & (y == 0)
    u2[dark, 0] = -2.0
    u2.requires_grad_(True)
    l1, s1 = ML.floor_ma_loss(u2, y, cond, M, mask, FCFG, macfg(floor_axes=["skin"]), anchors, h)
    l1.backward()
    gd = u2.grad[dark, 0]
    ok = float(l0) == 0.0 and float(l1) > 0 and s1["floor_ma_skin_real"] > 0 and s1["floor_ma_skin_fake"] == 0.0
    ok &= bool((gd < 0).all())                    # descent raises the deficient images' utility
    # anchors with effective n < 4 are skipped: a single image far away from everything
    y3, c3 = torch.tensor([0, 0, 0, 0, 0]), torch.zeros(5, dtype=torch.long)
    M3 = torch.zeros(5, 5)
    M3[:, 2] = torch.tensor([0.0, 0.1, 0.2, 0.3, 50.0])
    u3 = torch.zeros(5, K)
    u3[4] = -3.0
    a3, h3 = ML.kernel_anchors(0, 50, 8)
    l3, _ = ML.floor_ma_loss(u3, y3, c3, M3, mask, FCFG, macfg(floor_axes=["skin"]), a3, h3)
    _, used = ML.kernel_floor(u3, y3 == 0, M3[:, 2], mask, u3.mean(0), a3, h3, 4, 0.1)
    ok &= float(l3) == 0.0 and 7 not in used
    check("skin-tone kernel floor: 0 when uniform; > 0 for a planted deficit at one tone (gradient raises it); "
          "anchors with n_eff < 4 skipped", ok, f"planted {float(l1):.4f}, cells {s1['floor_ma_cells']}")


def t_age_gender():
    K = 4
    mask = torch.tensor([True, False, True, True])
    y = torch.zeros(16, dtype=torch.long)
    cond = torch.zeros(16, dtype=torch.long)
    M = torch.zeros(16, 5)
    M[:, 1] = torch.arange(16) // 4                  # 4 age bins x 4 images
    M[:, 2] = float("nan")
    u = torch.ones(16, K)
    u[M[:, 1] == 3] = 0.0                            # 60+ deficient: pooled 0.75, bin mean 0
    l, s = ML.floor_ma_loss(u, y, cond, M, mask, FCFG, macfg(floor_axes=["age"]), *ML.kernel_anchors(0, 1))
    want = 2.0 * 3 * (0.75 - 0.1) ** 2               # real weight 2 x 3 validated cues
    ok = abs(float(l) - want) < 1e-5
    lf, _ = ML.floor_ma_loss(u, torch.ones(16, dtype=torch.long), torch.ones(16, dtype=torch.long), M, mask, FCFG,
                             macfg(floor_axes=["age"]), *ML.kernel_anchors(0, 1))
    ok &= abs(float(lf) - want / 2) < 1e-5          # fakes weight 1
    M2 = M.clone()
    M2[:, 1] = -1                                    # age missing -> no age cells
    l2, _ = ML.floor_ma_loss(u, y, cond, M2, mask, FCFG, macfg(floor_axes=["age"]), *ML.kernel_anchors(0, 1))
    ok &= float(l2) == 0.0
    # gender: groups of 3 (< n_min) do not count
    M3 = torch.zeros(6, 5)
    M3[:3, 0] = 1
    l3, s3 = ML.floor_ma_loss(torch.rand(6, K), torch.zeros(6, dtype=torch.long), torch.zeros(6, dtype=torch.long), M3,
                              mask, FCFG, macfg(floor_axes=["gender"]), *ML.kernel_anchors(0, 1))
    ok &= float(l3) == 0.0 and s3["floor_ma_gender_real_cells"] == 0
    check("age floor exact (2 x 3 x (0.75-0.1)^2 for reals, half for fakes), missing ages ignored, n < 4 groups skipped",
          ok, f"{float(l):.5f} vs {want:.5f}")


def t_gxr_equiv():
    g = torch.Generator().manual_seed(3)
    n, K = 128, 4
    y = torch.arange(n) % 2
    cond = torch.where(y == 1, 1 + torch.randint(0, 2, (n,), generator=g), torch.zeros(n, dtype=torch.long))
    grp = torch.randint(0, 8, (n,), generator=g)
    M = torch.zeros(n, 5)
    M[:, 4] = grp.float()
    u = torch.randn(n, K, generator=g)
    mask = torch.tensor([True, True, False, True])
    ref, _ = CL.floor_loss(u, y, grp, cond, mask, FCFG)
    l1, _ = ML.floor_ma_loss(u, y, cond, M, mask, FCFG, macfg(floor_axes=["gxr"], real_weight=1.0), *ML.kernel_anchors(0, 1))
    l2, _ = ML.floor_ma_loss(u, y, cond, M, mask, FCFG, macfg(floor_axes=["gxr"], real_weight=2.0), *ML.kernel_anchors(0, 1))
    lr_, _ = CL.floor_loss(u[y == 0], y[y == 0], grp[y == 0], cond[y == 0], mask, FCFG)
    ok = abs(float(l1) - float(ref)) < 1e-5 and abs(float(l2) - (float(ref) + float(lr_))) < 1e-5 and float(ref) > 0
    check("gender x race part = the CESel floor (real weight 1); real weight 2 doubles exactly the real cells", ok,
          f"CESel {float(ref):.4f}, MA {float(l1):.4f}, MA w2 {float(l2):.4f}")


def stub_cfg(name="ma_ffpp", **over):
    return load_config(os.path.join(ROOT, "configs", f"{name}.yaml"),
                       ["model.backend=stub", "data.input_size=64", "model.stub={dim: 32, grid: 8}", "optim.amp=false",
                        "loss.inv.warmup_steps=0", "ces.dim=16", "ces.hidden=32", "ces.freq_hidden=16",
                        "ces.freq_widths=[8, 8, 16, 16]"] + [f"{k}={v}" for k, v in over.items()])


def synth_batch(n, gen, size=64):
    y = torch.arange(n) % 2
    g = torch.randint(0, 8, (n,), generator=gen)
    c = torch.where(y == 1, 1 + torch.randint(0, 2, (n,), generator=gen), torch.zeros(n, dtype=torch.long))
    M = torch.stack([(g >= 4).float(), torch.randint(0, 4, (n,), generator=gen).float(),
                     torch.rand(n, generator=gen) * 90 - 30, torch.zeros(n), g.float()], 1)
    return dict(image=torch.rand(n, 3, size, size, generator=gen), label=y, group=g, cond=c, ma=M, index=torch.arange(n))


def t_freq_warmup():
    ok = ML.freq_aux_factor(0, 10) == 1.0 and ML.freq_aux_factor(9, 10) == 1.0 and ML.freq_aux_factor(10, 10) == 0.0 \
        and ML.freq_aux_factor(5000, 1976) == 0.0 and ML.freq_aux_factor(1975, 1976) == 1.0
    cfg = stub_cfg()
    ok &= CE.freq_warmup_steps(cfg, 1976) == 1976 and CE.freq_warmup_steps(cfg, 3732) == 3732
    torch.manual_seed(0)
    model = CESDetector(cfg)
    gen = torch.Generator().manual_seed(0)
    terms = {"det": 1.0, "freq_aux": 0.2, "floor_ma": 1.0}
    st = CE.build_state(cfg, model, torch.device("cpu"))
    ctx = {"anchors": ML.kernel_anchors(-30, 60)[0], "h": ML.kernel_anchors(-30, 60)[1]}
    b = synth_batch(32, gen)
    out = {}
    for step in (0, 9, 10):
        model.zero_grad(set_to_none=True)
        vals, stats = CE.compute_ces_losses(model, st, b, None, cfg, torch.Generator().manual_seed(1), torch.device("cpu"),
                                            terms, aux={"freq_warm_steps": 10}, step=step, ma_ctx=ctx)
        (0.2 * vals["freq_aux"]).backward()
        gf = sum(float(p.grad.abs().sum()) for p in model.branches["freq"].parameters() if p.grad is not None)
        ga = sum(float(p.grad.abs().sum()) for p in model.freq_aux.parameters() if p.grad is not None)
        out[step] = (float(vals["freq_aux"]), gf, ga, stats["freq_aux_on"])
    ok &= out[0][0] > 0 and out[0][1] > 0 and out[0][2] > 0 and out[9][0] > 0 and out[9][3] == 1.0
    ok &= out[10][0] == 0.0 and out[10][1] == 0.0 and out[10][2] == 0.0 and out[10][3] == 0.0
    s = model(b["image"])["score"]
    ok &= "freq_aux" not in dict(model.head.named_parameters())       # score does not use the aux head
    check("frequency warm-up: factor 1 for step < epochs x steps_per_epoch, 0 after; aux BCE and its gradient "
          "(freq branch + aux head) only during warm-up; score head unchanged", ok,
          f"step0 loss {out[0][0]:.4f}, step10 {out[10][0]}; score shape {tuple(s.shape)}")


def t_selection():
    rng = np.random.default_rng(0)
    n = 400
    y = np.r_[np.zeros(200, int), np.ones(200, int)]
    z = np.r_[rng.normal(-2, 1, 200), rng.normal(2, 1, 200)]
    clusters = np.array([f"v{i // 10}" for i in range(n)])               # 10 frames per video
    M = np.zeros((n, 5), np.float32)
    M[:, 0] = (np.arange(n) // 10) % 2                                   # gender per video
    M[:, 1] = (np.arange(n) // 20) % 4
    M[:, 3] = (np.arange(n) // 40) % 4
    M[:, 4] = (np.arange(n) // 10) % 8
    M[:10, 1] = 3                                                        # nothing special
    M[190:200, 3] = 9                                                    # an ITA bin with ONE real video -> excluded
    z2 = z.copy()
    z2[190:200] = 5.0                                                    # that video would give a 100% FPR
    r = ML.ma_gaps(y, z2, 0.0, M, clusters, 2)
    dec = z2 >= 0
    want_g = [100 * dec[(y == 0) & (M[:, 0] == v)].mean() for v in (0, 1)]
    ok = abs(r["axes"]["gender"]["gap"] - abs(want_g[0] - want_g[1])) < 1e-9
    ok &= 9 not in r["axes"]["ita_bin"]["group_FPR"]
    ok &= abs(r["max_gap"] - max(a["gap"] for a in r["axes"].values())) < 1e-12
    import train as T
    res = {"auc": 95.0, "oppoint": {"pooled_TPR": 90.0, "pooled_FPR": 10.0, "worst_FPR": 15.0, "fpr_gap": 3.0,
                                    "group_TPR": {"a": 90.0}},
           "oppoint_ma": {"max_gap": 7.0}}
    v = T.selection_value(res, "val_oppoint_ma")
    v_old = T.selection_value(res, "val_oppoint")
    ok &= abs(v - (95.0 - 3.5)) < 1e-12 and abs(v_old - (95.0 - 1.5)) < 1e-12
    ref = {"worst_FPR": 14.0, "group_TPR": {"a": 90.0}}
    ok &= T.selection_value(res, "val_oppoint_ma", ref) == -1e9 == T.selection_value(res, "val_oppoint", ref)
    res2 = dict(res, oppoint=dict(res["oppoint"], pooled_TPR=79.0))
    ok &= T.selection_value(res2, "val_oppoint_ma") == -1e9
    check("selection gap = max over gender / age / ITA-bin / gender x race FPR gaps (groups with >= 2 real val videos); "
          "value = AUC - 0.5 max gap; eligibility identical to val_oppoint", ok,
          "gaps " + ", ".join(f"{k} {a['gap']:.1f}" for k, a in r["axes"].items()))


def t_configs():
    ok, info = True, []
    for name, want in (("ma_ffpp", {"sens_ces", "inv", "floor_ma", "freq_aux"}),
                       ("ma_floor_ffpp", {"floor_ma", "freq_aux"})):
        cfg = load_config(os.path.join(ROOT, "configs", f"{name}.yaml"))
        t = CE.active_terms(cfg, have_pairs=True)
        got = {k for k in t if k in {"sens_ces", "inv", "floor", "floor_ma", "freq_aux"}}
        ok &= got == want and t.get("floor_ma") == 1.0 and t.get("freq_aux") == 0.2 and cfg["select"]["metric"] == \
            "val_oppoint_ma" and cfg["ma"]["enabled"] and cfg["seed"] == 7 and cfg["optim"]["epochs"] == 10
        ok &= cfg["data"]["steps_per_epoch"] == 1976 and cfg["data"]["train"]["csv"].endswith("ff++_ding/train.csv")
        ok &= (t.get("sens_ces") == 0.5 and t.get("inv") == 0.5) if name == "ma_ffpp" else True
        info.append(f"{name} {sorted(got)}")
    from cfes import public_guard
    for name in ("ma_ff2public", "ma_floor_ff2public"):
        cfg = load_config(os.path.join(ROOT, "configs", f"{name}.yaml"))
        ok &= cfg["guard"]["public_only"] and cfg["guard"]["role"] == "phase2" and cfg["ma"]["enabled"]
        ok &= cfg["data"]["steps_per_epoch"] == 3732 and len(cfg["data"]["families"]) == 7
        try:                                   # no public phase-1 checkpoint given -> must refuse
            public_guard.check(cfg)
            ok = False
        except SystemExit as e:
            ok &= "REFUSE" in str(e)
        info.append(f"{name} guard refuses without PUBLIC_P1")
    cfg = load_config(os.path.join(ROOT, "configs", "ma_ffpp.yaml"), ["loss.floor.weight=1.0"])
    try:
        CE.active_terms(cfg, True)
        ok = False
    except ValueError:
        pass
    check("configs: ma_ffpp (sens 0.5, inv 0.5, floor_ma 1, freq_aux 0.2, val_oppoint_ma), ma_floor_ffpp, FF2-public "
          "variants guarded (phase 2); floor + floor_ma together refused", ok, "; ".join(info))


def t_trainer():
    from PIL import Image
    tmp = tempfile.mkdtemp(prefix="ma_smoke_")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(96):
        lab = i % 2
        name = f"{lab}_{'Deepfakes' if i % 4 == 1 else 'Face2Face'}_{i:03d}_{i + 100:03d}_{i}.png" if lab else \
            f"0_youtube_{i:03d}_{i}.png"
        p = os.path.join(tmp, name)
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(p)
        rows.append(dict(img_path=p, label=lab, ismale=1, isasian=-1, iswhite=1, isblack=-1, intersec_label=(i // 2) % 8))
    df = pd.DataFrame(rows)
    df.to_csv(f"{tmp}/train.csv", index=False)
    df.iloc[:48].to_csv(f"{tmp}/val.csv", index=False)
    pd.DataFrame({"fake_path": df.img_path[df.label == 1].values[:16],
                  "real_path": df.img_path[df.label == 0].values[:16]}).to_csv(f"{tmp}/pairs.csv", index=False)
    # DEMO_AXES label format (labels_<set>.csv): cl_male 1 = male, cl_race name, age4_if, ita_vid, ita4
    grp = (np.arange(96) // 2) % 8
    ita = rng.uniform(-30, 60, 96)
    lab = pd.DataFrame({"img_path": df.img_path, "cl_male": (grp < 4).astype(int),
                        "cl_race": np.array(["asian", "white", "black", "others"])[grp % 4],
                        "age4_if": rng.integers(0, 4, 96), "ita_vid": ita, "ita4": np.searchsorted([0.0, 15.0, 30.0], ita)})
    lab.loc[5, "ita_vid"] = np.nan
    lab.loc[5, "ita4"] = np.nan
    lab.to_csv(f"{tmp}/lab.csv", index=False)
    json.dump({"edges": [0.0, 15.0, 30.0]}, open(f"{tmp}/edges.json", "w"))
    common = ["model.backend=stub", "data.input_size=64", "model.stub={dim: 16, grid: 8}", "ces.dim=8", "ces.hidden=16",
              "ces.freq_hidden=8", "ces.freq_widths=[4, 4, 8, 8]", f"data.train.csv={tmp}/train.csv",
              f"data.val.csv=[{tmp}/val.csv]", f"pairs.csv={tmp}/pairs.csv", "pairs.per_step=2", "data.batch_size=8",
              "data.num_workers=0", "data.steps_per_epoch=2", "strat.per_cell=1", "optim.epochs=2", "optim.warmup_steps=1",
              "eval.batch_size=16", "log_every=1", "loss.inv.warmup_steps=1"]
    mal = [f"ma.labels.train={tmp}/lab.csv", f"ma.labels.val={tmp}/lab.csv", f"ma.labels.ita_edges_json={tmp}/edges.json",
           "loss.floor.n_min=1", "ma.n_eff_min=1", "loss.floor.cues=all"]
    env = dict(os.environ, PYTHONWARNINGS="ignore")
    b0 = f"{tmp}/run_B0"
    r1 = subprocess.run([PY, f"{ROOT}/scripts/train_ces.py", "--config", f"{ROOT}/configs/ffpp/B0.yaml", "--out", b0,
                         "--device", "cpu", "--set", *common], capture_output=True, text=True, env=env)
    vj = f"{b0}/validated_cues.json"
    r2 = subprocess.run([PY, f"{ROOT}/scripts/ces_validate_cues.py", "--config", f"{ROOT}/configs/ffpp/B0.yaml", "--ckpt",
                         f"{b0}/last.pt", "--out", vj, "--set", *common], capture_output=True, text=True, env=env)
    out = f"{tmp}/run_ma"
    r3 = subprocess.run([PY, f"{ROOT}/scripts/train_ces.py", "--config", f"{ROOT}/configs/ma_ffpp.yaml", "--out", out,
                         "--device", "cpu", "--set", *common, *mal, f"ces.validated_json={vj}"],
                        capture_output=True, text=True, env=env)
    ok = r1.returncode == 0 and r2.returncode == 0 and r3.returncode == 0
    info = (r1.stderr[-600:] + r2.stderr[-600:] + r3.stderr[-1500:]) if not ok else ""
    if ok:
        L = [json.loads(l) for l in open(f"{out}/train_log.jsonl")]
        st = [l for l in L if l.get("event") == "step"]
        ep = [l for l in L if l.get("event") == "epoch"]
        start = [l for l in L if l.get("event") == "start"][0]
        on = [s.get("freq_aux_on") for s in st]
        ok &= len(st) == 4 and on == [1.0, 1.0, 0.0, 0.0] and all("floor_ma" in s for s in st)
        ok &= st[0]["freq_aux"] > 0 and st[-1]["freq_aux"] == 0.0 and any(x["floor_ma"] > 0 for x in st)
        ok &= all(f"floor_ma_{ax}_real_cells" in st[-1] for ax in ("gxr", "gender", "age", "skin"))
        ok &= all(e.get("val_oppoint_ma") and "max_gap" in e["val_oppoint_ma"] for e in ep)
        ok &= start["ma"]["freq_warm_steps"] == 2 and start["ma"]["val_labels"]["labelled"] == 48
        ok &= os.path.exists(f"{out}/TRAINED") and "freq_aux.weight" in torch.load(
            f"{out}/last.pt", map_location="cpu", weights_only=False)["model"]
        r4 = subprocess.run([PY, f"{ROOT}/scripts/evaluate_ces.py", "--run", out, "--ckpt", "best.pt", "--max_items", "32",
                             "--device", "cpu"], capture_output=True, text=True, env=env)
        ok &= r4.returncode == 0
        info = (f"freq_aux_on {on}, floor_ma {[round(s['floor_ma'], 4) for s in st]}, epoch max_gap "
                f"{[round(e['val_oppoint_ma']['max_gap'], 2) for e in ep]}; evaluate rc {r4.returncode}")
        if r4.returncode:
            info += r4.stderr[-800:]
    check("trainer end to end on CPU (stub, synthetic images + labels): B0 -> validated JSON -> CESel-MA (+ evaluate)",
          ok, info)


def main():
    torch.manual_seed(0)
    t_labels()
    t_kernel()
    t_age_gender()
    t_gxr_equiv()
    t_freq_warmup()
    t_selection()
    t_configs()
    t_trainer()
    failed = [n for n, ok in RES if not ok]
    print(f"{len(RES) - len(failed)}/{len(RES)} checks passed")
    print("ALL_PASS" if not failed else "FAILED: " + "; ".join(failed))
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
