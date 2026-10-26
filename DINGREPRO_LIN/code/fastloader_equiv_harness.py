"""CPU equivalence / speed harness for GDA.py vs GDA_fastloader.py (DINGREPRO_LIN, 26 Oct, Amendment 4a D2 option b).

Runs the UNMODIFIED script file (runpy, __name__ == '__main__') with its real CLI and real data; nothing in either script
is edited. A step budget is imposed from OUTSIDE by shortening CombinedDataset.__len__ to S*batch_size (so each epoch
has S steps; both scripts see the identical change). Instrumentation (monkeypatches, read-only):
  - XceptionDetector.forward: sha256 of every input batch (combined = 1st call of a step, group = 2nd) + timings
  - XceptionDetector.get_losses: main (CE) loss value;  geomloss.SamplesLoss.forward: Sinkhorn value
  - torch.Tensor.backward: total loss value;  torch.optim.SGD.step: last_linear sha256 after every step
  - DataLoader.__iter__: iterator creations on the group (MergedDataset) loaders
  python equiv_harness.py --script GDA.py --tag orig --S 5 --E 2 --out <dir>
"""
import argparse
import hashlib
import json
import os
import runpy
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
D = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN"
CODE = f"{D}/code/Ding_mod"

ap = argparse.ArgumentParser()
ap.add_argument("--script", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--S", type=int, default=5)
ap.add_argument("--E", type=int, default=2)
ap.add_argument("--seed_ckpt", default=f"{D}/results/DING_base_s17/49.pth")
ap.add_argument("--channels", default="349 338 218 311 312 47 163 407 422 496")
ap.add_argument("--out", required=True)
ap.add_argument("--workers", type=int, default=8)
ap.add_argument("--no_compute", action="store_true",
                help="SPEED ONLY: skip the Xception trunk (features := first 512 input values, last_linear kept) to time data loading")
ap.add_argument("--group_len", type=int, default=0,
                help="TEST ONLY: cap MergedDataset.__len__ (both scripts) to exercise pass exhaustion / short-batch drop")
a = ap.parse_args()

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
sys.path.insert(0, CODE)
work = os.path.join(a.out, a.tag)
os.makedirs(work, exist_ok=True)
os.chdir(work)                                   # runs/, outputs/ of the script land here, not in the code dir
chan = os.path.join(work, "channels.txt")
open(chan, "w").write(a.channels)

import geomloss  # noqa: E402
import src.dataprocess as dp  # noqa: E402
from detectors.xception_detector import XceptionDetector  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

BS = 32
REC = {"steps": [], "group_iter_created": 0}
cur = {}
_orig_len = dp.CombinedDataset.__len__
dp.CombinedDataset.__len__ = lambda self: min(_orig_len(self), a.S * BS)
if a.group_len:
    _glen = dp.MergedDataset.__len__
    dp.MergedDataset.__len__ = lambda self: min(_glen(self), a.group_len)


def h(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


_fwd = XceptionDetector.forward


def _cheap(self, x):
    return {"cls": self.backbone.last_linear(x.flatten(1)[:, :512])}


def fwd(self, data_dict, inference=False):
    x = data_dict["image"]
    if x.shape[0] == 1:                          # fvcore FLOP trace at start-up
        return _fwd(self, data_dict, inference)
    t0 = time.perf_counter()
    if "combined" not in cur:
        cur["t_fwd1_start"] = t0
        cur["combined"] = h(x)
        cur["combined_labels"] = data_dict["label"].tolist()
        out = _fwd(self, data_dict, inference) if not a.no_compute else _cheap(self, x)
        cur["t_fwd1"] = time.perf_counter() - t0
    else:
        cur["t_fwd2_start"] = t0
        cur["group"] = h(x)
        cur["group_attr"] = sorted(set(data_dict["intersec_label"].tolist()))
        cur["group_n"] = int(x.shape[0])
        out = _fwd(self, data_dict, inference) if not a.no_compute else _cheap(self, x)
        cur["t_fwd2"] = time.perf_counter() - t0
    return out


XceptionDetector.forward = fwd
_gl = XceptionDetector.get_losses


def gl(self, d, p):
    r = _gl(self, d, p)
    cur["main"] = float(r["overall"].item())
    cur["t_main_end"] = time.perf_counter()
    return r


XceptionDetector.get_losses = gl
_sl = geomloss.SamplesLoss.forward


def sl(self, *args, **kw):
    t0 = time.perf_counter()
    r = _sl(self, *args, **kw)
    cur["fair"] = float(r.item())
    cur["t_sink"] = time.perf_counter() - t0
    return r


geomloss.SamplesLoss.forward = sl
_bw = torch.Tensor.backward


def bw(self, *args, **kw):
    t0 = time.perf_counter()
    cur["total"] = float(self.item())
    r = _bw(self, *args, **kw)
    cur["t_bwd"] = time.perf_counter() - t0
    return r


torch.Tensor.backward = bw
_step = torch.optim.SGD.step
MODEL = {}
t_last = [None]


def step(self, *args, **kw):
    t0 = time.perf_counter()
    r = _step(self, *args, **kw)
    t1 = time.perf_counter()
    ll = [p for g in self.param_groups for p in g["params"] if p.requires_grad]
    cur["last_linear_sha"] = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in ll)).hexdigest()[:16]
    cur["t_opt"] = t1 - t0
    # data time: combined fetch = previous step end -> 1st forward; group fetch = CE loss end -> 2nd forward
    cur["t_combined_fetch"] = (cur["t_fwd1_start"] - t_last[0]) if t_last[0] else None
    cur["t_group_fetch"] = cur["t_fwd2_start"] - cur["t_main_end"]
    cur["t_wall"] = (t1 - t_last[0]) if t_last[0] else None
    t_last[0] = t1
    REC["steps"].append({k: v for k, v in cur.items() if not k.startswith("t_") or k in (
        "t_fwd1", "t_fwd2", "t_sink", "t_bwd", "t_opt", "t_combined_fetch", "t_group_fetch", "t_wall")})
    cur.clear()
    return r


torch.optim.SGD.step = step
_it = DataLoader.__iter__


def it(self):
    if isinstance(self.dataset, dp.MergedDataset):
        REC["group_iter_created"] += 1
        cur["group_iter_new"] = cur.get("group_iter_new", 0) + 1
    return _it(self)


DataLoader.__iter__ = it

sys.argv = [os.path.join(CODE, a.script), "--model_name", "XceptionDetector", "--checkpoints", a.seed_ckpt,
            "--dataset_dir", f"{D}/data/ff++_ding", "--channels_file", chan, "--num_epochs", str(a.E),
            "--lr", "0.001", "--batch_size", str(BS), "--workers", str(a.workers), "--sinkhorn_blur", "5e-4",
            "--lambda_fairloss", "0.005", "--result_dir", work, "--training_title", f"equiv_{a.tag}"]
T0 = time.perf_counter()
runpy.run_path(os.path.join(CODE, a.script), run_name="__main__")
REC["total_s"] = time.perf_counter() - T0
ck = os.path.join(work, "XceptionDetector_prune2%_only_head", "lr_1.0e-03_lambda_5.0e-03", f"{a.E - 1}.pth")
m = torch.load(ck, map_location="cpu", weights_only=False)
sd = {k: v.clone() for k, v in m.state_dict().items()}
torch.save(sd, os.path.join(a.out, f"final_state_{a.tag}.pt"))
REC.update(script=a.script, S=a.S, E=a.E, workers=a.workers, seed_ckpt=a.seed_ckpt, channels=a.channels,
           torch=torch.__version__, threads=torch.get_num_threads(), host=os.uname().nodename,
           final_ckpt=ck)
json.dump(REC, open(os.path.join(a.out, f"rec_{a.tag}.json"), "w"), indent=1)
print("HARNESS_DONE", a.tag, len(REC["steps"]), "steps", f"{REC['total_s']:.1f}s", flush=True)
os._exit(0) if not os.environ.get("HARNESS_NORMAL_EXIT") else print("normal interpreter exit", flush=True)
