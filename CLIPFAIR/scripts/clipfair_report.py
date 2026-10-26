"""CLIPFAIR report (Amendment 3): the four matched-backbone fairness baselines on FF++ (Lin), seeds 7/17/27, with the
columns of the CESel screen report (scripts/cesel_report.py of agents/CESEL): validation selection value (Amendment-9
rule, best over the run's checkpoints, start checkpoint eligible) and, per test set, AUC, TPR / FPR, gender / race /
intersectional F_FPR, F_DP, es-AUC at threshold 0.5 (raw logit >= 0) and at A1 (fair_a1.py, the CFES fair_oppoint
rule: one logit threshold per model chosen on FF++ validation), with 3-seed means (+- SD). Test metrics are reported,
never used to select anything. References: Ding et al. published Xception rows (Tables 1-2, threshold 0.5) and CESel
B0 (agents/CESEL/runs/ffpp_B0_s<seed>, read only, when finished: same backbone, data and budget).

  ~/.conda/envs/aiface/bin/python scripts/clipfair_report.py [--out_root R] [--seeds 7,17,27] [--partial]
  -> <out_root>/results/clipfair_ffpp.md (+ .json)
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import fair_a1 as A  # noqa: E402

CESEL = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL"
METHODS = ["daw", "dag", "pgfdd", "ding"]
LABEL = {"daw": "DAW-FDD (CLIP)", "dag": "DAG-FDD (CLIP)", "pgfdd": "PG-FDD (CLIP)", "ding": "Ding SFD+GDA (CLIP)",
         "B0": "CESel B0 (CLIP, ref.)"}
COLS = ["AUC", "TPR", "FPR", "g_F_FPR", "g_F_DP", "g_es_AUC", "r_F_FPR", "r_F_DP", "r_es_AUC", "i_F_FPR", "i_F_DP",
        "i_es_AUC"]


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def sel_of(run):
    t = jl(os.path.join(run, "TRAINED"))
    if t is None:
        return None
    rec = {"select": float(t["best"]), "best_epoch": int(t["best_epoch"]), "epochs": int(t["epoch"])}
    lg = os.path.join(run, "train_log.jsonl")
    if os.path.exists(lg):
        for line in open(lg):
            e = json.loads(line)
            if e.get("event") == "epoch" and e.get("epoch") == rec["best_epoch"]:
                rec["val_auc"] = e.get("val_auc")
                rec["val_fpr_gap"] = (e.get("val_oppoint") or {}).get("fpr_gap")
    return rec


def fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def ms(vals, nd=2):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "-"
    if len(v) == 1:
        return f"{v[0]:.{nd}f} (n=1)"
    return f"{np.mean(v):.{nd}f} +- {np.std(v, ddof=1):.{nd}f}" + ("" if len(v) == 3 else f" (n={len(v)})")


def _atomic_write(path, text):
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=f".{os.getpid()}.tmp", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def publish(res_dir, md, md_text, js, payload, force=False, wait_s=60, stale_s=1800):
    """Serialised (mkdir lock), atomic, monotone publication: complete beats partial, more finished runs beat fewer."""
    lock = os.path.join(res_dir, ".clipfair_report_publish.lock")
    got, t0 = False, time.time()
    while time.time() - t0 < wait_s:
        try:
            os.mkdir(lock)
            got = True
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > stale_s:
                    os.rmdir(lock)
                    continue
            except OSError:
                pass
            time.sleep(0.2)
    try:
        try:
            cur = json.load(open(js))
        except (OSError, ValueError):
            cur = None
        if cur is not None and not force:
            if cur.get("complete") and not payload["complete"]:
                print(f"kept existing COMPLETE report {js}")
                return False
            if cur.get("progress") and tuple(payload["progress"]) < tuple(cur["progress"]):
                print(f"kept existing report {js} (further along: {cur['progress']})")
                return False
        _atomic_write(md, md_text)
        _atomic_write(js, json.dumps(payload, indent=1, default=float))
        return True
    finally:
        if got:
            try:
                os.rmdir(lock)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=os.path.dirname(HERE))
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--force", action="store_true", help="publish even if the existing report is further along")
    a = ap.parse_args()
    R = a.out_root
    seeds = [int(s) for s in a.seeds.split(",")]
    runs = {(m, s): f"{R}/runs/ffpp_{m}_s{s}" for m in METHODS for s in seeds}
    runs.update({("B0", s): f"{CESEL}/runs/ffpp_B0_s{s}" for s in seeds})
    sel = {k: sel_of(d) for k, d in runs.items()}
    a1 = {k: jl(f"{d}/eval/a1.json") for k, d in runs.items()}
    sfd = {s: jl(f"{R}/runs/ffpp_ding_sfd_s{s}/sfd_channels.json") for s in seeds}
    complete = all(sel[(m, s)] is not None and a1[(m, s)] is not None for m in METHODS for s in seeds)
    rows = METHODS + ["B0"]

    L = [f"# CLIPFAIR: matched-backbone fairness baselines on FF++ (Lin protocol){' -- PARTIAL' if a.partial or not complete else ''}",
         "", "Generated by scripts/clipfair_report.py. Pre-registration: decisions/2026-10-25-cesel-plan.md, Amendment 3. "
         "Same CLIP ViT-L/14 + rank-1 SVD adapters, FF++ (Lin) data and samplers, start checkpoint (CFES clip_s<seed>_p1), "
         "budget (10 x 1976 steps) and Amendment-9 validation selection as CESel B0; adaptations in README.md. Test "
         "metrics are reported only (nothing is selected on them). Mean +- SD over seeds " + "/".join(map(str, seeds)) + ".",
         "", "## Validation selection values (Amendment-9 rule; -1e9 = no checkpoint inside the TPR>=80 / FPR<=20 box)", "",
         "| model | " + " | ".join(f"s{s}" for s in seeds) + " | mean +- SD | best epochs | val AUC at best |",
         "|---|" + "---|" * (len(seeds) + 3)]
    for m in rows:
        v = [(sel[(m, s)] or {}).get("select") for s in seeds]
        be = [str((sel[(m, s)] or {}).get("best_epoch", "")) for s in seeds]
        va = [(sel[(m, s)] or {}).get("val_auc") for s in seeds]
        L.append(f"| {LABEL[m]} | " + " | ".join(fmt(x, 3) if x is not None else "not finished" for x in v) +
                 f" | {ms([x for x in v if x is not None and x > -1e8], 3)} | {'/'.join(be)} | {ms(va)} |")
    L += ["", "Best epoch -1 = the start checkpoint (p1 backbone with the method's initial head) was never beaten.", ""]
    if any(sfd.values()):
        L += ["Ding SFD channels (2% of 1024 ln_post channels, lowest mean SNNL over FF++ train; zeroed in the CLS "
              "feature): " + "; ".join(f"s{s}: {v['channels']}" for s, v in sfd.items() if v), ""]
    L += ["Ding GDA is run AS RELEASED: the Sinkhorn fairness term is computed inside torch.no_grad() (GDA.py:232-235), so "
          "it contributes no gradient; the Ding row is therefore SFD + head fine-tuning with CE. The gradient-enabled variant "
          "(clipfair.ding.gda_grad) is implemented but not run (decision file not amended).", ""]

    def table(level):
        out = []
        for k, nm in A.NAMES.items():
            out += [f"### {nm} ({'A1 threshold' if level == 'a1' else 'threshold 0.5'})", "",
                    "| model | t | " + " | ".join(COLS) + " |", "|---|---|" + "---|" * len(COLS)]
            if level == "raw0":
                p = dict(zip(A.PUB_COLS, A.DING_PUB[k]))
                out.append("| Ding published Xception (0.5) | | " + " | ".join(fmt(p.get(c)) if c in p else "" for c in COLS) + " |")
            for m in rows:
                per = []
                for s in seeds:
                    r = a1[(m, s)]
                    if not r:
                        continue
                    if level == "a1":
                        if r.get("t") is None:
                            out.append(f"| {LABEL[m]} s{s} | no A1 point | " + " | ".join("-" for _ in COLS) + " |")
                            continue
                        t, met = r["t"], r["test"].get(k)
                    else:
                        t, met = 0.0, ((r.get("reference") or {}).get("raw0") or {}).get("test", {}).get(k)
                    if met:
                        per.append(met)
                        out.append(f"| {LABEL[m]} s{s} | {t:.3f} | " + " | ".join(fmt(met.get(c)) for c in COLS) + " |")
                if per:
                    out.append(f"| **{LABEL[m]} mean** | | " + " | ".join(ms([q.get(c) for q in per]) for c in COLS) + " |")
            out.append("")
        return out

    L += ["## Test metrics at the A1 threshold (validation-chosen; reported only)", "",
          "A1 = one logit threshold per model minimising FF++-val intersectional F_FPR s.t. val TPR >= 80%, FPR <= 20% "
          "(CFES fair_oppoint rule, scripts/fair_a1.py). g / r / i = gender / race / intersectional groups.", ""]
    L += table("a1")
    L += ["## Test metrics at threshold 0.5 (raw logit >= 0)", ""]
    L += table("raw0")
    os.makedirs(f"{R}/results", exist_ok=True)
    md = f"{R}/results/clipfair_ffpp.md"
    js = {f"{m}_s{s}": {"selection": sel[(m, s)], "a1": a1[(m, s)]} for m in rows for s in seeds}
    progress = [int(complete), sum(v is not None for v in sel.values()), sum(v is not None for v in a1.values())]
    payload = {"runs": js, "sfd": sfd, "complete": complete, "progress": progress}
    # Codex review F10 (CESel, 25 Oct): unique temp files + atomic replace, serialised, never regress to an older report
    if publish(f"{R}/results", md, "\n".join(L) + "\n", f"{R}/results/clipfair_ffpp.json", payload, force=a.force):
        print(f"wrote {md} (complete={complete}, progress={progress})")


if __name__ == "__main__":
    main()
