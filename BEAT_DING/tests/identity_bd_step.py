"""One BD training step under a given code tree, for before / after identity proofs of edits to cfes/ (CPU, fp32, fixed
seeds). Generalises tests/identity_rev4_EF.py to any config: the step's data comes from the code tree's own training
loaders (engine.train_loader / pair_loader / strat_loader, num_workers 0, first batch of epoch 0), so augmentation (A),
pair group labels (B / D) and the stratified batch are exercised exactly as train.py uses them.
  PY tests/identity_bd_step.py <code_root> <out.json> <config relative to configs/> <init_ckpt> [extra --set ...]
Configs are read from ID_CONFIG_ROOT (default: this BEAT_DING tree). Compare outputs with tests/identity_compare.py."""
import hashlib
import json
import os
import sys

root, out, cfg_rel, init = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), sys.argv[3], sys.argv[4]
extra = sys.argv[5:]
sys.path[:0] = [root, os.path.join(root, "scripts")]
import torch  # noqa: E402
from cfes import engine  # noqa: E402
from cfes.config import load_config  # noqa: E402

P = os.environ.get("ID_CONFIG_ROOT", "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING")
os.chdir(P)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "16")))
cfg = load_config(f"{P}/configs/{cfg_rel}", [f"model.init_ckpt={init}", "data.num_workers=0", *extra])
dev = torch.device("cpu")
torch.manual_seed(0)
model = engine.build_model(cfg, dev).train()
geom = engine.build_geometry(cfg, model, dev)
geom_cpu = engine.build_geometry(cfg, model, "cpu")
df, sampler, loader = engine.train_loader(cfg, 7)
batch = next(iter(loader))
pl = engine.pair_loader(cfg, geom_cpu, 7)
pairs = next(iter(pl)) if pl is not None else None
sl = engine.strat_loader(cfg, df, 1, 7)
sb = next(iter(sl)) if sl is not None else None
terms = engine.active_terms(cfg, have_pairs=pairs is not None)
aux = {"fprgap_rq": engine.fprgap_quantile(cfg)} if "fprgap" in terms else {}
vals, stats = engine.compute_losses(model, batch, pairs, cfg, geom, torch.Generator().manual_seed(5), dev, terms,
                                    strat_batch=sb, aux=aux)
tot = engine.weighted_total(vals, terms)
tot.backward()
g = torch.cat([p.grad.flatten() for p in model.trainable_parameters() if p.grad is not None])
imgs = [batch["image"].flatten()] + ([sb["image"].flatten()] if sb is not None else []) + \
    ([pairs["fake"].flatten(), pairs["real"].flatten()] if pairs is not None else [])
rec = {"code_root": root, "config": cfg_rel, "init": init, "extra": extra, "terms": terms,
       "vals": {k: float(v).hex() for k, v in vals.items()}, "vals_float": {k: float(v) for k, v in vals.items()},
       "total": float(tot).hex(), "grad_sha256": hashlib.sha256(g.detach().numpy().tobytes()).hexdigest(),
       "grad_n": int(g.numel()), "batch_sha256": hashlib.sha256(torch.cat(imgs).numpy().tobytes()).hexdigest(),
       "stats": {k: v for k, v in stats.items() if isinstance(v, (int, float))}}
json.dump(rec, open(out, "w"), indent=1)
print(json.dumps({k: rec[k] for k in ("config", "terms", "vals_float", "grad_sha256")}))
