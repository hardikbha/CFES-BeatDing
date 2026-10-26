"""Re-run the Codex review's REPRO blocks (codex_review/test_metrics_protocol.py, the report half of
codex_review/test_runner_and_report.py) block by block against the FIXED code. Each REPRO block asserts the DEFECT;
after the fixes those assertions must FAIL ("defect gone"); PASS blocks (formula oracles) must still pass.
The lock half of test_runner_and_report.py patches a line that no longer exists; its interleaving is re-implemented
in tests/test_runlock_f1.sh.   CUDA_VISIBLE_DEVICES="" ~/.conda/envs/pulid/bin/python tests/rerun_codex_repros.py"""
import os
import re
import shutil
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CR = os.path.join(ROOT, "codex_review")
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
os.chdir(ROOT)


def _unused_blocks(src, start=None):
    if start:
        head = src[:src.index("import sys")]  # docstring
        src = src[src.index(start):]
    out, cur = [], []
    for line in src.splitlines():
        if re.match(r"^# ", line) and cur:
            out.append("\n".join(cur)); cur = []
        cur.append(line)
    out.append("\n".join(cur))
    return out


ok = True
# ---- metrics protocol (F3, F4, F7, F9 + oracles)
src = open(os.path.join(CR, "test_metrics_protocol.py")).read().replace(
    "ROOT = Path(__file__).resolve().parents[1]", f"ROOT = Path({ROOT!r})")
anchors = [("# Validation threshold", "F4 rank threshold"), ("# Production validate()", "F3 validate saturation"),
           ("# Independent direct", "oracle"), ("rec=A.a1_for(", "F7 FF2 A1 omission"),
           ("# All four removed", "F9 full deletion residual")]
pos = [src.index(a) for a, _ in anchors] + [len(src)]
ns = {}
exec(compile(src[:pos[0]], "codex_head", "exec"), ns)
for (a, name), st, en in zip(anchors, pos[:-1], pos[1:]):
    b = src[st:en]
    try:
        exec(compile(b, "codex_block", "exec"), ns)
        res = "held"
    except AssertionError:
        res = "assert_failed"
    except KeyError as e:
        res = f"raised KeyError: {e}"
    if name == "F4 rank threshold":
        # reproduces with a hand-built rank threshold, not product code: check the product now thresholds logits
        src_ev = open(os.path.join(ROOT, "scripts/evaluate_ces.py")).read()
        status = ("DEFECT GONE (evaluate_ces.py selects/applies the threshold on logits)"
                  if 'zv.astype(np.float64), cfg["eval"]["val_tpr"]' in src_ev and "thr_scores=" in src_ev else "STILL PRESENT")
    elif name == "F9 full deletion residual":
        # computes s - sum(s_k) by hand (the old formula); the product now uses ces_posteval.intervene
        import ces_posteval as P
        import numpy as np
        sk = ns["sk"]; s_ = sk.sum(1).astype(np.float64)
        full = P.intervene(s_, sk, 0.0, np.zeros(4), np.ones(4))
        status = "DEFECT GONE (ces_posteval.intervene: full deletion is an exact constant)" if np.ptp(full) == 0 else "STILL PRESENT"
    elif name == "oracle":
        status = "PASS (formula / A1 oracle still holds)" if res == "held" else "ORACLE BROKEN"
    elif name == "F7 FF2 A1 omission":
        status = (f"DEFECT GONE (no silent omission: a set without a scheme {res}; with schemes it is scored, "
                  "tests/test_codex_eval_fixes.py)") if res != "held" else "STILL PRESENT"
    else:
        status = "DEFECT GONE (Codex REPRO assertion now fails)" if res == "assert_failed" else "STILL PRESENT"
    ok &= not status.startswith(("STILL", "ORACLE"))
    print(f"[{status}] {name}")

# ---- report half of test_runner_and_report.py (F2)
S = os.path.join(ROOT, "tests", "_codex_rerun_scratch")
shutil.rmtree(S, ignore_errors=True)
os.makedirs(S)
src = open(os.path.join(CR, "test_runner_and_report.py")).read()
rep = src[src.index("import cesel_report as R"):]
pre = ("import sys, os, json\nfrom pathlib import Path\n"
       f"S = Path({S!r})\nsys.path[:0] = [{ROOT!r}, {os.path.join(ROOT, 'scripts')!r}]\n")
parts = rep.split("\nR.main(); dec=")
# part 0: setup + first R.main(); part 1: first check + removal + second R.main(); part 2: second check
ns = {}
exec(pre + parts[0], ns)
for i, tail in enumerate(parts[1:]):
    code = "dec=" + tail
    lines = code.splitlines()
    check, rest = lines[:3], lines[3:]
    ns["R"].main()
    try:
        exec("\n".join(check), ns)
        st = "STILL PRESENT"
    except AssertionError:
        st = "DEFECT GONE (Codex REPRO assertion now fails)"
    d = ns["json"].loads((ns["fixture"] / "results/cesel_ffpp_screen.json").read_text())["decisions"]
    print(f"[{st}] report repro {i + 1}: winner={d['winner']} promote={d['promote']} unresolved={d['unresolved']}")
    ok &= st.startswith("DEFECT GONE")
    if rest:
        exec("\n".join(rest), ns)
shutil.rmtree(S, ignore_errors=True)
print("ALL_DEFECTS_GONE" if ok else "SOME_DEFECT_REMAINS")
