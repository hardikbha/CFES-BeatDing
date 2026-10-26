"""Compare rec_orig.json / rec_fast.json (+ final state dicts) written by equiv_harness.py -> one JSON + a printed table."""
import json
import statistics as st
import sys

import torch

d = sys.argv[1]
out = sys.argv[2]
o = json.load(open(f"{d}/rec_orig.json"))
f = json.load(open(f"{d}/rec_fast.json"))
S = o["S"]
rows = []
for i, (a, b) in enumerate(zip(o["steps"], f["steps"])):
    rows.append({
        "step": i, "epoch": i // S, "group_iter_new_fast": b.get("group_iter_new", 0),
        "group_attr": a["group_attr"], "group_n_orig": a["group_n"], "group_n_fast": b["group_n"],
        "combined_batch_same": a["combined"] == b["combined"], "group_batch_same": a["group"] == b["group"],
        "main_orig": a["main"], "main_fast": b["main"], "main_same": a["main"] == b["main"],
        "fair_orig": a["fair"], "fair_fast": b["fair"], "total_orig": a["total"], "total_fast": b["total"],
        "total_same": a["total"] == b["total"], "last_linear_same": a["last_linear_sha"] == b["last_linear_sha"]})
so = torch.load(f"{d}/final_state_orig.pt")
sf = torch.load(f"{d}/final_state_fast.pt")
assert so.keys() == sf.keys()
diff = {k: float((so[k].double() - sf[k].double()).abs().max()) for k in so if so[k].is_floating_point()}
ll = {k: v for k, v in diff.items() if "last_linear" in k}
bn = {k: v for k, v in diff.items() if "running_" in k}
other = {k: v for k, v in diff.items() if k not in ll and k not in bn}
nbt = all(torch.equal(so[k], sf[k]) for k in so if k.endswith("num_batches_tracked"))


def steady(rec, key):
    v = [s[key] for i, s in enumerate(rec["steps"]) if i % S != 0 and s.get(key) is not None]
    return {"mean": st.mean(v), "median": st.median(v), "n": len(v)}


timing = {}
for tag, rec in (("orig", o), ("fast", f)):
    timing[tag] = {k: steady(rec, k) for k in ("t_group_fetch", "t_combined_fetch", "t_wall", "t_fwd1", "t_fwd2",
                                               "t_sink", "t_bwd")}
    timing[tag]["group_iterators_created"] = rec["group_iter_created"]
    timing[tag]["total_s"] = rec["total_s"]
res = {"setup": {k: o[k] for k in ("S", "E", "workers", "seed_ckpt", "channels", "torch", "threads", "host")},
       "steps": rows,
       "summary": {
           "combined_batch_identical_all_steps": all(r["combined_batch_same"] for r in rows),
           "main_loss_identical_all_steps": all(r["main_same"] for r in rows),
           "last_linear_identical_after_every_step": all(r["last_linear_same"] for r in rows),
           "group_batch_identical_steps": [r["step"] for r in rows if r["group_batch_same"]],
           "fresh_group_iterator_steps_fast": [r["step"] for r in rows if r["group_iter_new_fast"]],
           "total_loss_identical_steps": [r["step"] for r in rows if r["total_same"]],
           "group_batch_size_always_64": all(r["group_n_orig"] == 64 and r["group_n_fast"] == 64 for r in rows),
           "final_ckpt_max_abs_diff_last_linear": ll,
           "final_ckpt_max_abs_diff_bn_running_stats_max": max(bn.values()),
           "final_ckpt_max_abs_diff_other_float_tensors_max": max(other.values()) if other else 0.0,
           "num_batches_tracked_identical": nbt},
       "timing_s_per_step_excl_first_step_of_epoch": timing}
json.dump(res, open(out, "w"), indent=1)
print(f"{'st':>2} {'ep':>2} new comb grp {'main_orig':>12} {'main_fast':>12} {'fair_orig':>9} {'fair_fast':>9} total= ll=")
for r in rows:
    print(f"{r['step']:>2} {r['epoch']:>2} {r['group_iter_new_fast']:>3} {str(r['combined_batch_same'])[0]:>4} "
          f"{str(r['group_batch_same'])[0]:>3} {r['main_orig']:12.9f} {r['main_fast']:12.9f} {r['fair_orig']:9.6f} "
          f"{r['fair_fast']:9.6f} {str(r['total_same'])[0]:>6} {str(r['last_linear_same'])[0]:>3}  groups={r['group_attr']}")
print(json.dumps(res["summary"], indent=1))
print(json.dumps(timing, indent=1))
