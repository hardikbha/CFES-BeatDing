"""CESel seed-7 FF++ screen report: applies the PRE-REGISTERED keep / kill rules (decisions/2026-10-25-cesel-plan.md,
Amendment 1) on VALIDATION selection values only, and lists test metrics (reported, never used to decide).

  ~/.conda/envs/aiface/bin/python scripts/cesel_report.py --design p1 [--partial] -> results/cesel_ffpp_screen_p1.md (+ .json)
  ~/.conda/envs/aiface/bin/python scripts/cesel_report.py [--design x5] [--partial] -> results/cesel_ffpp_screen.md (+ .json)

Designs: p1 (Amendment 5, 26 Oct; THE pre-registered comparison): every arm trained like B0 from CFES clip_s<seed>_p1
(runs/ffpp_p1_<arm>_s<seed>, 10 x 1976 steps, start checkpoint = selection reference) and compared with B0 itself
(runs/ffpp_B0_s<seed>) in place of B0-continued; same keep / drop / winner / success rules. x5 (Amendment 1 warm-start
screen, runs/ffpp_x5_*): EXPLORATORY record only (degenerate: every arm kept B0's checkpoint), not used for decisions.

Selection value = the Amendment-9 rule of train.py (val AUC - 0.5 val FPR gap at the validation-calibrated operating
point, -1e9 outside the TPR >= 80 / FPR <= 20 box or when worse than the start checkpoint), the best over the run's
checkpoints (TRAINED["best"]; the start checkpoint is eligible).
Rules: (1) keep a term only if its single-term arm (and Full) beats step-matched B0-continued on the selection value;
(2) drop a term when its negative control matches Full within the seed spread: |sel(control) - sel(Full)| <= spread,
spread = range (max - min) of the full-data CFES-D 3-seed selection values (runs_rev4/clip_D_w0.3_s{7,17,27});
C_randcue -> the counterfactual removal underlying L_sens and L_floor (sens, floor), C_shufgroup -> floor,
C_novalid -> inv; (3) winner = highest selection value among the CESel arms (sens, inv, floor, full) that beat
B0-continued and contain no dropped term; promoted to seeds 7/17/27: B0, B0+CFES-D, winner.
Codex review fixes (25 Oct, Amendment 4): F2 -- a term whose single-term arm does NOT beat B0-continued is excluded
like a dropped term (Full / any arm containing it cannot win), and the decision FAILS CLOSED (no winner, no promotion)
while any keep / drop gate is unresolved, while the reference spread is unavailable, or when the reference
selection values differ from the ones frozen in Amendment 1 (87.62 / 89.68 / 89.28, range 2.06). F10 -- md and json are
published through unique temp files + atomic replace under a publish lock, and a report never replaces one that is
further along (complete beats partial; more finished runs / evaluations beat fewer); --force overrides.
Test metrics: A1 threshold (fair_a1.py, the CFES fair_oppoint rule) and raw 0.5, vs Ding published (Tables 1-2) and
vs CFES-D s7 / Arm E w10 s7 (CFES runs, read only; their A1 recomputed here from their saved scores with the same code).
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

CF = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CFES"
ARMS = ["B0", "sens", "inv", "floor", "full", "C_randcue", "C_shufgroup", "C_novalid", "B0_cfesD"]
LABEL = {"B0": "B0-continued", "sens": "+sens", "inv": "+inv", "floor": "+floor", "full": "Full CESel",
         "C_randcue": "C_randcue", "C_shufgroup": "C_shufgroup", "C_novalid": "C_novalid", "B0_cfesD": "B0+CFES-D"}
TERMS = {"sens": {"sens"}, "inv": {"inv"}, "floor": {"floor"}, "full": {"sens", "inv", "floor"}}
CONTROL_DROPS = {"C_randcue": {"sens", "floor"}, "C_shufgroup": {"floor"}, "C_novalid": {"inv"}}
COLS_A1 = ["AUC", "TPR", "FPR", "i_F_FPR", "i_F_DP", "i_es_AUC"]
# Amendment 1 (frozen before launch): CFES-D full-data 3-seed selection values (runs_rev4/clip_D_w0.3_s{7,17,27}) and
# their range = the control-rule spread. The report recomputes them from the runs and fails closed on any mismatch.
DECLARED_REF = {"seeds": (7, 17, 27), "values": (87.62, 89.68, 89.28), "spread": 2.06, "tol": 0.005}
# Amendment 5 design: arms trained like B0; the reference is B0 itself
ARMS_P1 = ["sens", "inv", "floor", "full", "C_randcue", "C_shufgroup", "C_novalid", "B0_cfesD"]
LABEL_P1 = dict(LABEL, B0="B0")


def reference_spread(sv):
    """sv = the three recomputed selection values (None if missing) -> (spread or None, status string)."""
    if len(sv) != 3 or any(v is None for v in sv):
        return None, "unavailable (CFES-D 3-seed TRAINED records missing)"
    if any(abs(v - d) > DECLARED_REF["tol"] for v, d in zip(sv, DECLARED_REF["values"])):
        return None, f"MISMATCH with Amendment 1 ({[round(v, 3) for v in sv]} vs {list(DECLARED_REF['values'])})"
    sp = float(max(sv) - min(sv))
    if abs(sp - DECLARED_REF["spread"]) > 2 * DECLARED_REF["tol"]:
        return None, f"MISMATCH: range {sp:.4f} vs declared {DECLARED_REF['spread']}"
    return float(DECLARED_REF["spread"]), f"ok: recomputed range {sp:.4f} matches Amendment 1; the declared 2.06 is used"


def decide(S, spread):
    """Pre-registered keep / drop / winner rules on validation selection values (Codex F2 fix). S(k) -> value or None
    for k in ARMS; spread = verified reference spread or None. Returns the decision dict; dec["winner"] / ["promote"]
    are set only when EVERY gate is resolved (fail closed otherwise)."""
    dec = {"spread_rule": "range of CFES-D full-data 3-seed selection values (frozen in Amendment 1)", "spread": spread,
           "keep": {}, "drop": {}, "winner": None, "promote": None}
    b0c = S("B0")
    for k in ("sens", "inv", "floor", "full"):
        dec["keep"][k] = None if (S(k) is None or b0c is None) else bool(S(k) > b0c)
    for c, terms in CONTROL_DROPS.items():
        if S(c) is None or S("full") is None or spread is None:
            dec["drop"][c] = None
        else:
            dec["drop"][c] = {"terms": sorted(terms), "abs_diff": abs(S(c) - S("full")),
                              "matches_full": bool(abs(S(c) - S("full")) <= spread)}
    dropped = set()
    for v in dec["drop"].values():
        if v and v["matches_full"]:
            dropped |= set(v["terms"])
    failed = {t for k in ("sens", "inv", "floor") if dec["keep"][k] is False for t in TERMS[k]}
    dec["dropped_terms"] = sorted(dropped)                 # negative control matches Full
    dec["failed_terms"] = sorted(failed)                   # single-term arm does not beat B0-continued
    dec["excluded_terms"] = sorted(dropped | failed)
    unresolved = [f"keep:{k}" for k, v in dec["keep"].items() if v is None] + \
                 [f"drop:{c}" for c, v in dec["drop"].items() if v is None] + \
                 (["reference spread"] if spread is None else [])
    dec["unresolved"] = unresolved
    elig = [k for k in ("sens", "inv", "floor", "full") if dec["keep"].get(k) and not (TERMS[k] & (dropped | failed))]
    dec["eligible"] = elig
    if not unresolved:
        dec["winner"] = max(elig, key=S) if elig else None
        dec["promote"] = ["B0", "B0+CFES-D"] + ([LABEL[dec["winner"]]] if dec["winner"] else [])
    return dec


def _atomic_write(path, text):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=f".{os.getpid()}.tmp", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def publish(res_dir, md_text, payload, force=False, wait_s=60, stale_s=1800, stem="cesel_ffpp_screen"):
    """F10: serialised, atomic, monotone publication of results/<stem>.{md,json}. Returns True if written."""
    md, js = os.path.join(res_dir, f"{stem}.md"), os.path.join(res_dir, f"{stem}.json")
    lock = os.path.join(res_dir, f".cesel_report_publish{'' if stem == 'cesel_ffpp_screen' else '_' + stem}.lock")
    got, t0 = False, time.time()
    while time.time() - t0 < wait_s:
        try:
            os.mkdir(lock)
            got = True
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock) > stale_s:
                    os.rmdir(lock)                    # crashed publisher (empty dir); publication itself takes < 1 s
                    continue
            except OSError:
                pass
            time.sleep(0.2)
    if not got:
        print(f"WARNING: publish lock {lock} busy for {wait_s}s; publishing with the progress check only")
    try:
        cur = None
        try:
            cur = json.load(open(js))
        except (OSError, ValueError):
            cur = None
        if cur is not None and not force:
            old = tuple(cur.get("progress") or ())
            if cur.get("complete") and not payload["complete"]:
                print(f"kept existing COMPLETE report {js}; not replaced by a partial one")
                return False
            if old and tuple(payload["progress"]) < old:
                print(f"kept existing report {js} (progress {old} > this report's {tuple(payload['progress'])})")
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


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def sel_of(run):
    t = jl(os.path.join(run, "TRAINED"))
    if t is None:
        return None
    rec = {"select": float(t["best"]), "best_epoch": int(t["best_epoch"]), "epochs": int(t["epoch"])}
    lg = os.path.join(run, "train_log.jsonl")
    if os.path.exists(lg):
        for l in open(lg):
            e = json.loads(l)
            if e.get("event") == "epoch" and e.get("epoch") == rec["best_epoch"]:
                rec["val_auc"] = e.get("val_auc")
                rec["val_fpr_gap"] = (e.get("val_oppoint") or {}).get("fpr_gap")
    return rec


def a1_from_scores(ev):
    ld = lambda f: (lambda z: (z["logit"].astype(np.float64), z["label"].astype(np.int64), z["group"].astype(np.int64)))(
        np.load(f, allow_pickle=True))
    rec = A.a1_for(ld(f"{ev}/val_scores.npz"), {k: ld(f"{ev}/{k}_scores.npz") for k in A.SETS},
                   {k: sch for k, (_, sch) in A.SETS.items()})
    rec["reference"] = {"raw0": {"t": 0.0, "test": {k: A.full_metrics(*ld(f"{ev}/{k}_scores.npz"), sch, 0.0)
                                                     for k, (_, sch) in A.SETS.items()}}}
    return rec


def fmt(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=os.path.dirname(HERE))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--force", action="store_true", help="publish even if the existing report is further along")
    ap.add_argument("--design", choices=["x5", "p1"], default="x5",
                    help="p1 = Amendment 5 screen (arms like B0, vs B0); x5 = exploratory warm-start record")
    a = ap.parse_args()
    R, s = a.out_root, a.seed
    p1 = a.design == "p1"
    lab = LABEL_P1 if p1 else LABEL
    arms = ARMS_P1 if p1 else ARMS
    REF = "B0" if p1 else "B0-continued"
    stem = "cesel_ffpp_screen_p1" if p1 else "cesel_ffpp_screen"
    if p1:
        runs = {"B0": f"{R}/runs/ffpp_B0_s{s}"}
        runs.update({lab[k]: f"{R}/runs/ffpp_p1_{k}_s{s}" for k in ARMS_P1})
    else:
        runs = {"B0 (start)": f"{R}/runs/ffpp_B0_s{s}"}
        runs.update({LABEL[k]: f"{R}/runs/ffpp_x5_{k}_s{s}" for k in ARMS})
    sel = {n: sel_of(d) for n, d in runs.items()}
    a1 = {n: jl(f"{d}/eval/a1.json") for n, d in runs.items()}
    interp = {n: jl(f"{d}/eval/interp.json") for n, d in runs.items()}
    # references (CFES, read only)
    refs = {"CFES-D s7": f"{CF}/runs_rev4/clip_D_w0.3_s{s}", "Arm E w10 s7": f"{CF}/runs_rev4/clip_E_w10_s{s}"}
    for n, d in refs.items():
        sel[n] = sel_of(d)
        a1[n] = a1_from_scores(f"{d}/eval") if os.path.exists(f"{d}/eval/val_scores.npz") else None
    cfesd = [sel_of(f"{CF}/runs_rev4/clip_D_w0.3_s{x}") for x in DECLARED_REF["seeds"]]
    sv = [c["select"] for c in cfesd if c]
    spread, spread_status = reference_spread([c["select"] if c else None for c in cfesd])
    sd = float(np.std(sv, ddof=1)) if len(sv) == 3 else None

    # ------------------------------------------------------------------ pre-registered rules (validation only)
    S = lambda k: (sel.get(lab[k]) or {}).get("select")
    b0c = S("B0")
    dec = decide(S, spread)
    if p1 and dec["promote"]:
        dec["promote"] = ["B0", "B0+CFES-D"] + ([lab[dec["winner"]]] if dec["winner"] else [])
    dec["design"], dec["reference_run"] = a.design, REF
    dec["cfesd_sd"], dec["spread_status"] = sd, spread_status
    complete = all(S(k) is not None for k in (["B0"] + arms if p1 else arms))
    w = dec["winner"]
    if w:
        dec["success_a_seed7"] = {n: bool(S(w) > (sel.get(n) or {}).get("select", float("inf")))
                                  for n in ((("B0", "CFES-D s7", "Arm E w10 s7")) if p1 else
                                            ("B0 (start)", "B0-continued", "CFES-D s7", "Arm E w10 s7"))}

    # ------------------------------------------------------------------ markdown
    title = (f"# CESel seed-{s} FF++ screen, Amendment 5 design (arms trained like B0; Ding/Lin protocol)" if p1 else
             f"# CESel seed-{s} FF++ warm-start screen (Amendment 1) -- EXPLORATORY, not used for decisions")
    L = [f"{title}{' -- PARTIAL' if a.partial or not complete else ''}", "",
         "Generated by scripts/cesel_report.py. Pre-registration: decisions/2026-10-25-cesel-plan.md (+ Amendments 1-5). "
         "Decisions use the VALIDATION selection value only (Amendment-9 rule); test metrics are reported, never used.", ""]
    if p1:
        L += ["Design (Amendment 5): every arm trains exactly like B0 (CFES runs/clip_s7_p1/last.pt, fresh heads / "
              "branches / CES state, 10 x 1976 steps, same seed and data order, Amendment-9 selection with the start "
              "checkpoint as reference); validated-cue JSON = the one from B0 best (runs/ffpp_B0_s7). Arms are compared "
              "with B0 itself (no continued-B0 arm). The warm-start screen (results/cesel_ffpp_screen.md) is EXPLORATORY.", "",
              "B0 caveat (Amendment 4, F3): B0's checkpoint choice was made by the pre-fix validation code "
              "(probability AUC); the arms use the fixed rank-of-logit code. results/b0_selection_audit_F3.md re-scores "
              "B0 best.pt under the fixed rule: 85.624 = recorded 85.624 (0 saturated, 0 cross-class ties); other B0 "
              "epochs cannot be re-scored (logits not saved).", ""]
    else:
        L += ["EXPLORATORY (Amendment 5): in this warm-start design every arm started from B0 best.pt and no later epoch "
              "passed the start-referenced checks, so the comparison is degenerate. The pre-registered screen is "
              "results/cesel_ffpp_screen_p1.md.", ""]
    L += ["## Validation selection values", "",
         "| run | selection value | best epoch | val AUC at best | val FPR gap at best |", "|---|---|---|---|---|"]
    for n in list(runs) + list(refs):
        r = sel.get(n)
        L.append(f"| {n} | {fmt(r['select'], 3) if r else 'not finished'} | {r['best_epoch'] if r else ''} | "
                 f"{fmt(r.get('val_auc')) if r else ''} | {fmt(r.get('val_fpr_gap')) if r else ''} |")
    L += ["", (f"Best epoch -1 = the start checkpoint (CFES phase-1 backbone with fresh heads; outside the TPR/FPR box). "
               if p1 else "Best epoch -1 = the start checkpoint (B0 for the arms: no epoch beat it). ") +
              f"CFES-D 3-seed selection values "
              f"{[round(x, 3) for x in sv]}: spread (range) = {fmt(spread, 3)} [{spread_status}], SD = {fmt(sd, 3)}.", "",
          "## Pre-registered decisions", ""]
    for k in ("sens", "inv", "floor", "full"):
        v = dec["keep"][k]
        L.append(f"- {lab[k]} vs {REF}: {fmt(S(k), 3)} vs {fmt(b0c, 3)} -> "
                 f"{'pending' if v is None else ('beats (keep)' if v else 'does NOT beat (drop)')}")
    for c, v in dec["drop"].items():
        L.append(f"- {c} vs Full: " + ("pending" if v is None else
                 f"|diff| {v['abs_diff']:.3f} {'<=' if v['matches_full'] else '>'} spread {spread:.3f} -> "
                 f"{'matches Full: DROP ' + '/'.join(v['terms']) if v['matches_full'] else 'differs from Full (term kept)'}"))
    L.append(f"- Excluded terms: dropped by a control {dec['dropped_terms'] or 'none'}; single-term arm fails "
             f"{REF} {dec['failed_terms'] or 'none'}")
    if dec["unresolved"]:
        L.append(f"- **No promotion yet (fail closed)**: unresolved gates {', '.join(dec['unresolved'])}")
    else:
        L.append(f"- Winner (CESel arms that beat {REF} and contain no excluded term): **{lab[dec['winner']] if dec['winner'] else 'none'}**")
        L.append(f"- Promote to seeds 7/17/27: {', '.join(dec['promote'])}")
        if dec.get("success_a_seed7"):
            L.append(f"- Success criterion (a) at seed 7 only (indicative; the claim needs 3 seeds): winner beats "
                     + ", ".join(f"{n}: {'yes' if v else 'no'}" for n, v in dec["success_a_seed7"].items()))
    L += ["", "## Test metrics at the A1 threshold (validation-chosen; reported only)", "",
          "A1 = one logit threshold per model minimising FF++-val intersectional F_FPR s.t. val TPR >= 80%, FPR <= 20% "
          "(CFES fair_oppoint rule). Ding published values are at 0.5 (their Tables 1-2).", ""]
    order = list(runs) + list(refs)
    for k, nm in A.NAMES.items():
        p = dict(zip(A.PUB_COLS, A.DING_PUB[k]))
        L += [f"### {nm}", "", "| model | t | " + " | ".join(COLS_A1) + " |", "|---|---|" + "---|" * len(COLS_A1),
              f"| Ding published (0.5) | | {p['AUC']:.2f} | | | {p['i_F_FPR']:.2f} | {p['i_F_DP']:.2f} | {p['i_es_AUC']:.2f} |"]
        for n in order:
            r = a1.get(n)
            if not r:
                continue
            if r.get("t") is None:
                L.append(f"| {n} | no A1 point | " + " | ".join("-" for _ in COLS_A1) + " |")
                continue
            m = r["test"].get(k)
            if m:
                L.append(f"| {n} | {r['t']:.3f} | " + " | ".join(fmt(m[c]) for c in COLS_A1) + " |")
        L.append("")
    L += ["## Test metrics at 0.5 (raw logit >= 0), Ding Table-2 columns", ""]
    for k, nm in A.NAMES.items():
        L += [f"### {nm}", "", "| model | " + " | ".join(A.PUB_COLS) + " |", "|---|" + "---|" * len(A.PUB_COLS),
              "| Ding published | " + " | ".join(f"{x:.2f}" for x in A.DING_PUB[k]) + " |"]
        for n in order:
            r = a1.get(n)
            m = (((r or {}).get("reference") or {}).get("raw0") or {}).get("test", {}).get(k)
            if m:
                L.append(f"| {n} | " + " | ".join(fmt(m[c]) for c in A.PUB_COLS) + " |")
        L.append("")
    L += ["## Per-cue interpretability (from saved per-cue logits; scripts/ces_posteval.py)", "",
          "Deletion = AUC after replacing cue k by the neutral EMA baseline (alpha = 1); solo = AUC of s_k alone. "
          "Full curves (alpha grid, A1 rates, per-group utilities, cumulative deletion) in <run>/eval/interp.json.", "",
          "| run | val importance (tex/freq/bound/sem) | FF++ test AUC | delete tex/freq/bound/sem | solo tex/freq/bound/sem |",
          "|---|---|---|---|---|"]
    for n in runs:
        it = interp.get(n)
        if not it or "ffpp_test" not in it["sets"]:
            continue
        st = it["sets"]["ffpp_test"]
        dl = [st["deletion"][c][-1]["AUC"] for c in it["cues"]]
        L.append(f"| {n} | " + "/".join(f"{it['val_importance'][c]:.2f}" for c in it["cues"]) +
                 f" | {st['deletion'][it['cues'][0]][0]['AUC']:.2f} | " + "/".join(f"{x:.2f}" for x in dl) + " | " +
                 "/".join(f"{st['solo_auc'][c]:.1f}" for c in it["cues"]) + " |")
    sv_md = f"{R}/results/b0_cue_starvation/B0_cue_starvation.md"
    L += ["", "## B0 per-group cue starvation", "",
          (f"[results/b0_cue_starvation/B0_cue_starvation.md]({sv_md}) (+ .csv): per group x class x cue mean utility, "
           "bootstrap CI, deficit vs pooled, cue share; FF++ val and the four test sets." if os.path.exists(sv_md)
           else "not written yet (runs after B0).")]
    if p1:
        prom = set(dec["promote"] or [])
        L += ["", "## Per-arm cue starvation (interpretability; required for every PROMOTED arm)", "",
              "| arm | promoted | table |", "|---|---|---|"]
        for k in ARMS_P1:
            f = f"{R}/results/p1_cue_starvation/p1_{k}_cue_starvation.md"
            L.append(f"| {lab[k]} | {'yes' if lab[k] in prom else ('pending' if dec['promote'] is None else 'no')} | "
                     f"{'[' + os.path.relpath(f, R) + '](' + f + ')' if os.path.exists(f) else 'not written yet'} |")
    os.makedirs(f"{R}/results", exist_ok=True)
    md = f"{R}/results/{stem}.md"
    progress = [int(complete), sum(sel.get(n) is not None for n in runs), sum(a1.get(n) is not None for n in runs),
                sum(interp.get(n) is not None for n in runs)]          # screen runs only (references excluded)
    payload = {"selection": sel, "decisions": dec, "complete": complete, "progress": progress, "design": a.design,
               "exploratory": not p1, "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "partial_flag": bool(a.partial)}
    if publish(f"{R}/results", "\n".join(L) + "\n", payload, force=a.force, stem=stem):
        print(f"wrote {md} (complete={complete}, progress={progress}, unresolved={dec['unresolved']})")


if __name__ == "__main__":
    main()
