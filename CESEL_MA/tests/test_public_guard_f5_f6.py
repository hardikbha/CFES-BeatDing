"""Codex review F5 / F6 checks (CPU, no training): every FF2-public config carries the provenance guard and refuses
to start without a public phase-1 lineage; cfesd_ff2_public.yaml is the rev4 CFES-D recipe (risk 0.3, val_oppoint).
Scratch: tests/_guard_scratch (removed).   CUDA_VISIBLE_DEVICES="" ~/.conda/envs/pulid/bin/python tests/test_public_guard_f5_f6.py"""
import glob
import os
import shutil
import subprocess
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cfes import public_guard as G  # noqa: E402
from cfes.config import load_config  # noqa: E402

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
CF = f"{R}/agents/CFES"
S = os.path.join(ROOT, "tests", "_guard_scratch")
shutil.rmtree(S, ignore_errors=True)
os.makedirs(S)
PY = sys.executable


def refuses(cfg, what):
    try:
        G.check(cfg)
    except SystemExit as e:
        assert "REFUSE" in str(e), e
        return str(e)
    raise AssertionError(f"guard did not refuse: {what}")


# 1. every FF2-public config is guarded; FF++ / FF2-full configs are not
pub = ["configs/ces_ff2_public.yaml", "configs/cfesd_ff2_public.yaml", "configs/ff2_public_p1.yaml"] + \
      sorted(os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "configs/ff2_public/*.yaml")))
for c in pub:
    cfg = load_config(os.path.join(ROOT, c))
    assert cfg["guard"]["public_only"] and "fairforge2_public/train.csv" in cfg["data"]["train"]["csv"], c
for c in ["configs/ces_ffpp.yaml", "configs/ces_ff2.yaml"] + glob.glob(os.path.join(ROOT, "configs/ffpp/*.yaml")):
    assert G.check(load_config(os.path.join(ROOT, c))) is None, c
print(f"PASS guard present in all {len(pub)} FF2-public configs; FF++ / FF2-full configs unguarded")

# 2. CFES-D recipe (F6): identical to the resolved rev4 clip_D_w0.3_s7 apart from the dataset keys
d = load_config(os.path.join(ROOT, "configs/cfesd_ff2_public.yaml"))
ref = yaml.safe_load(open(f"{CF}/runs_rev4/clip_D_w0.3_s7/config.yaml"))
assert d["loss"]["risk"] == {"weight": 0.3, "tau": 1.0, "include_views": True, "n_min": 2}
assert d["select"] == {"include_init": True, "metric": "val_oppoint"} and d["model"]["local_norm"] is True
for k in ("loss", "optim", "select", "strat", "model", "cf", "pairs", "ckpt", "train"):
    a, b = dict(d[k]), dict(ref[k])
    for x in (a, b):
        x.pop("init_ckpt", None); x.pop("face_mask", None); x.pop("csv", None)
    assert a == b, (k, {kk: (a.get(kk), b.get(kk)) for kk in set(a) | set(b) if a.get(kk) != b.get(kk)})
assert d["data"]["steps_per_epoch"] == 3732 and d["data"]["pairing"] == "ff2" and "ff2_test" in d["eval"]["test_sets"]
assert "runs_ff2public/clip_s7_p1/last.pt" in d["model"]["init_ckpt"]
print("PASS F6: cfesd_ff2_public = rev4 CFES-D (risk 0.3, local_norm, val_oppoint incl. start) + FF2-public data only")

# 3. refusals
print("  ", refuses(d, "no public phase-1 yet")[:150])
d_full = dict(d, model=dict(d["model"], init_ckpt=f"{CF}/runs_ff2full/clip_s7_p1/last.pt"))
print("  ", refuses(d_full, "FF2-FULL phase-1 init")[:150])
d_trfull = dict(d, data=dict(d["data"], train=dict(d["data"]["train"],
                csv=f"{R}/data_ding_schema/fairforge2_full/train.csv")))
refuses(d_trfull, "FF2-full train csv")
p1 = load_config(os.path.join(ROOT, "configs/ff2_public_p1.yaml"))
assert G.check(p1)["role"] == "phase1"
refuses(dict(p1, model=dict(p1["model"], init_ckpt=f"{CF}/runs_ff2full/clip_s7_p1/last.pt")), "phase-1 with init")
b0 = load_config(os.path.join(ROOT, "configs/ff2_public/B0.yaml"))
refuses(b0, "CESel FF2-public B0 without init_backbone_ckpt")
print("PASS F5 refusals: FF2-full phase-1 init, missing PUBLIC_P1, FF2-full TRAIN, phase-1 with an init, B0 without init")

# 4. lineage: fake phase-1 run -> PUBLIC_P1 -> CFES-D / CESel B0 pass -> PUBLIC_LINEAGE -> arm passes; tamper refuses
p1dir = os.path.join(S, "p1")
os.makedirs(p1dir)
for c in ("last.pt", "best.pt"):
    open(os.path.join(p1dir, c), "wb").write(os.urandom(1024))
G.write_p1(p1dir, G.check(p1))
assert os.path.exists(os.path.join(p1dir, "PUBLIC_P1"))
assert G.check(dict(d, model=dict(d["model"], init_ckpt=f"{p1dir}/last.pt")))["inits"]["init_ckpt"]["marker"] == "PUBLIC_P1"
b0ok = dict(b0, model=dict(b0["model"], init_backbone_ckpt=f"{p1dir}/last.pt"))
rec = G.check(b0ok)
b0dir = os.path.join(S, "b0")
os.makedirs(b0dir)
G.write_lineage(b0dir, rec)
open(os.path.join(b0dir, "best.pt"), "wb").write(b"x")
arm = load_config(os.path.join(ROOT, "configs/ff2_public/full.yaml"))
arm = dict(arm, model=dict(arm["model"], init_ckpt=f"{b0dir}/best.pt", init_backbone_ckpt=None))
assert G.check(arm)["inits"]["init_ckpt"]["marker"] == "PUBLIC_LINEAGE"
open(os.path.join(p1dir, "last.pt"), "ab").write(b"tamper")
refuses(dict(d, model=dict(d["model"], init_ckpt=f"{p1dir}/last.pt")), "tampered phase-1 checkpoint")
print("PASS F5 lineage: PUBLIC_P1 -> CFES-D and CESel B0 start; B0 PUBLIC_LINEAGE -> arm starts; tampered ckpt refused")

# 5. the real trainers refuse before building anything
for script, cfgp in (("scripts/train.py", "configs/cfesd_ff2_public.yaml"), ("scripts/train_ces.py", "configs/ff2_public/B0.yaml")):
    out = os.path.join(S, "trainer_" + os.path.basename(script))
    r = subprocess.run([PY, script, "--config", cfgp, "--out", out], cwd=ROOT, capture_output=True, text=True,
                       timeout=600, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
    assert r.returncode != 0 and "REFUSE (FF2-public provenance guard" in r.stderr + r.stdout, (r.returncode, r.stderr[-800:])
    assert not os.path.exists(out), "trainer created its output before the guard"
print("PASS trainers: scripts/train.py (CFES-D) and scripts/train_ces.py (CESel B0) refuse to start, nothing written")
shutil.rmtree(S, ignore_errors=True)
print("ALL_PASS test_public_guard_f5_f6")
