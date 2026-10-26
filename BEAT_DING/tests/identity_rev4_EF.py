"""Bit-identity of CFES-D's step under two code trees: same config, phase-1 checkpoint, fixed batch, RNG (CPU, fp32).
  PY identity_check.py <code_root> <out.json> [arm] [extra --set ...]
Beat-Ding copy (26 Oct): configs are read from ID_CONFIG_ROOT (default: this BEAT_DING tree; the CFES original reads
CFES's); arm "E" -> configs/rev4/clip_E.yaml, an arm containing "/" -> configs/<arm>.yaml (e.g. bd/base_E). The
phase-1 checkpoint and the FF++ manifests are CFES's / DINGREPRO_LIN's (read only)."""
import hashlib, json, os, sys
root, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
arm = sys.argv[3] if len(sys.argv) > 3 else "D"
extra = sys.argv[4:]
sys.path[:0] = [root, os.path.join(root, "scripts")]
import torch
from cfes import engine
from cfes.config import load_config
import validate as V
CF = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CFES"
P = os.environ.get("ID_CONFIG_ROOT", "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING")
os.chdir(P)
torch.set_num_threads(16)
cfg_path = f"{P}/configs/{arm}.yaml" if "/" in arm else f"{P}/configs/rev4/clip_{arm}.yaml"
cfg = load_config(cfg_path, [f"model.init_ckpt={CF}/runs/clip_s7_p1/last.pt", "data.num_workers=0", "loss.risk.weight=0.3",
                             *extra])
torch.manual_seed(0)
dev = torch.device("cpu")
model = engine.build_model(cfg, dev).train()
geom = engine.build_geometry(cfg, model, "cpu")
torch.manual_seed(1)
batch = V.sample_batch(cfg, 8)
pairs = V.sample_pairs(cfg, geom, 4)
df = engine.read_split(cfg["data"]["train"])
sb = next(iter(engine.strat_loader(cfg, df, 1, 0)))
terms = engine.active_terms(cfg, have_pairs=True)
vals, stats = engine.compute_losses(model, batch, pairs, cfg, geom, torch.Generator().manual_seed(5), dev, terms,
                                    strat_batch=sb)
tot = engine.weighted_total(vals, terms)
tot.backward()
g = torch.cat([p.grad.flatten() for p in model.trainable_parameters() if p.grad is not None])
rec = {"code_root": root, "config": cfg_path, "arm": arm, "terms": terms, "vals": {k: float(v).hex() for k, v in vals.items()},
       "vals_float": {k: float(v) for k, v in vals.items()}, "total": float(tot).hex(),
       "grad_sha256": hashlib.sha256(g.detach().numpy().tobytes()).hexdigest(), "grad_n": int(g.numel()),
       "batch_sha256": hashlib.sha256(torch.cat([batch["image"].flatten(), sb["image"].flatten()]).numpy().tobytes()).hexdigest(),
       "stats": {k: v for k, v in stats.items() if isinstance(v, (int, float))}}
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("arm", "terms", "vals_float", "grad_sha256")}))
