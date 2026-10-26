"""Step-1 gate of the final-model spec (decisions/2026-10-26-final-model-spec.md section 1), applied EXACTLY and on
VALIDATION selection values only (no test metric is read).

  ~/.conda/envs/aiface/bin/python scripts/gate_check.py [--screen <cesel_ffpp_screen_p1.json>] [--out results/gate.json]

Rule (pre-registered): the gate PASSES iff some arm A in {Full CESel, +floor} satisfies BOTH
  (1) sel(A) - sel(B0) > 2.06                                     (the declared Amendment-1 spread)
  (2) for every negative control C in {C_randcue, C_shufgroup, C_novalid}: |sel(C) - sel(A)| > 2.06
      (a control "within 2.06 of the passing arm" matches it and blocks it).
sel = the Amendment-9 validation selection value of the run's selected checkpoint (TRAINED["best"]; -1e9 = no
eligible checkpoint), as published in the Amendment-5 screen report (agents/CESEL/results/cesel_ffpp_screen_p1.json),
cross-checked against every run's TRAINED file. The report must be COMPLETE and its reference spread verified
("spread_status" ok); otherwise the gate is PENDING (exit 3) -- or FAIL once the decision date (2 Nov 2026, the
final-model freeze) has passed.
Derived (for the priority runner): ma_base_arm = the passing arm with the higher selection value (full -> CESel-MA
config "ma", floor -> "ma_floor"); single_axis_arm = the screen's pre-registered winner (report decisions.winner),
or ma_base_arm when the report has no winner.
Output: gate.json (status PASS | FAIL | PENDING, every number used). Exit 0 = decided, 3 = pending, 2 = error.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import tempfile
import time

AG = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents"
SPREAD = 2.06
ARMS = {"full": "Full CESel", "floor": "+floor"}
CONTROLS = ["C_randcue", "C_shufgroup", "C_novalid"]
LABEL2KEY = {"B0": "B0", "+sens": "sens", "+inv": "inv", "+floor": "floor", "Full CESel": "full",
             "C_randcue": "C_randcue", "C_shufgroup": "C_shufgroup", "C_novalid": "C_novalid", "B0+CFES-D": "B0_cfesD"}
DEADLINE = datetime.date(2026, 11, 2)
RULE = ("PASS iff an arm A in {Full CESel, +floor} has sel(A) - sel(B0) > 2.06 AND |sel(C) - sel(A)| > 2.06 for every "
        "negative control C in {C_randcue, C_shufgroup, C_novalid}; validation selection values only (Amendment-9 "
        "value of each run's selected checkpoint, Amendment-5 screen); report must be complete with a verified spread")


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".gate.", dir=os.path.dirname(os.path.abspath(path)))
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, default=float)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def decide(sel):
    """sel: {key: selection value or None} for B0, full, floor and the 3 controls -> (status, detail)."""
    need = ["B0", "full", "floor"] + CONTROLS
    miss = [k for k in need if sel.get(k) is None]
    if miss:
        return "PENDING", {"missing": miss}
    b0 = sel["B0"]
    cand = {}
    for k in ARMS:
        a = sel[k]
        ctr = {c: {"sel": sel[c], "abs_diff": abs(sel[c] - a), "within": bool(abs(sel[c] - a) <= SPREAD)} for c in CONTROLS}
        beats = bool(a - b0 > SPREAD)
        cand[k] = {"label": ARMS[k], "sel": a, "diff_vs_B0": a - b0, "beats_B0_by_more_than_spread": beats,
                   "controls": ctr, "blocked_by": [c for c, v in ctr.items() if v["within"]],
                   "passes": bool(beats and not any(v["within"] for v in ctr.values())),
                   "note": ("a control EXCEEDS this arm by more than the spread" if any(sel[c] - a > SPREAD for c in CONTROLS)
                            else None)}
    passing = [k for k in ARMS if cand[k]["passes"]]
    return ("PASS" if passing else "FAIL"), {"B0": b0, "candidates": cand, "passing_arms": passing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default=f"{AG}/CESEL/results/cesel_ffpp_screen_p1.json")
    ap.add_argument("--runs_root", default=None, help="root holding runs/ffpp_B0_s7 and runs/ffpp_p1_<arm>_s7 "
                    "(default: the screen json's parent of results/)")
    ap.add_argument("--out", default=f"{AG}/CESEL_MA/results/gate.json")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--today", default=None, help="test hook: ISO date used for the decision-date rule")
    a = ap.parse_args()
    today = datetime.date.fromisoformat(a.today) if a.today else datetime.date.today()
    rec = {"rule": RULE, "spread": SPREAD, "screen_json": os.path.abspath(a.screen), "generated":
           time.strftime("%Y-%m-%dT%H:%M:%S"), "test_metrics_used": False, "decision_deadline": DEADLINE.isoformat()}
    if not os.path.exists(a.screen):
        status, det = "PENDING", {"reason": "screen report not written yet"}
    else:
        raw = open(a.screen, "rb").read()
        rep = json.loads(raw)
        rec["screen_sha256"] = hashlib.sha256(raw).hexdigest()
        rec["screen_generated"] = rep.get("generated")
        dec = rep.get("decisions") or {}
        sel = {LABEL2KEY[n]: (v or {}).get("select") for n, v in (rep.get("selection") or {}).items() if n in LABEL2KEY}
        rec["selection_values"] = sel
        if rep.get("design") != "p1" or rep.get("exploratory"):
            print(f"ERROR: {a.screen} is not the Amendment-5 (p1) screen report")
            sys.exit(2)
        spread_ok = str(dec.get("spread_status", "")).startswith("ok")
        if not rep.get("complete") or not spread_ok:
            status, det = "PENDING", {"reason": f"screen report complete={rep.get('complete')}, spread_status="
                                                f"{dec.get('spread_status')}"}
        else:
            # cross-check every value used against the run's TRAINED record
            root = a.runs_root or os.path.dirname(os.path.dirname(os.path.abspath(a.screen)))
            runs = {"B0": f"{root}/runs/ffpp_B0_s{a.seed}"}
            runs.update({k: f"{root}/runs/ffpp_p1_{k}_s{a.seed}" for k in ["full", "floor"] + CONTROLS})
            bad = {}
            for k, d in runs.items():
                t = os.path.join(d, "TRAINED")
                v = float(json.load(open(t))["best"]) if os.path.exists(t) else None
                if v is None or sel.get(k) is None or abs(v - float(sel[k])) > 1e-6:
                    bad[k] = {"report": sel.get(k), "TRAINED": v}
            if bad:
                print(f"ERROR: screen report and TRAINED records disagree: {bad}")
                rec.update(status="ERROR", mismatch=bad)
                atomic_json(a.out, rec)
                sys.exit(2)
            status, det = decide(sel)
            w = dec.get("winner")
            rec["screen_winner"] = w
            if status == "PASS":
                base = max(det["passing_arms"], key=lambda k: sel[k])
                rec["ma_base_arm"] = base
                rec["ma_config"] = {"full": "ma", "floor": "ma_floor"}[base]
                rec["single_axis_arm"] = w if w else base
                rec["single_axis_run"] = runs.get(rec["single_axis_arm"], f"{root}/runs/ffpp_p1_{rec['single_axis_arm']}_s{a.seed}")
    if status == "PENDING" and today > DEADLINE:
        status, det = "FAIL", dict(det, reason_fail=f"decision date {DEADLINE} passed without a complete screen")
    rec.update(status=status, detail=det)
    atomic_json(a.out, rec)
    print(f"gate {status}: " + json.dumps({k: rec.get(k) for k in ("ma_base_arm", "single_axis_arm")}) +
          f" detail {json.dumps(det, default=float)[:600]}")
    sys.exit(3 if status == "PENDING" else 0)


if __name__ == "__main__":
    main()
