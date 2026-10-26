"""Codex review F2 (promotion gates, fail closed) and F10 (atomic, monotone report publication) checks on local
synthetic TRAINED fixtures (no scores, no GPU). Scratch: tests/_report_scratch (removed at the end).
  CUDA_VISIBLE_DEVICES="" ~/.conda/envs/pulid/bin/python tests/test_report_f2_f10.py"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import cesel_report as R  # noqa: E402

S = ROOT / "tests" / "_report_scratch"
shutil.rmtree(S, ignore_errors=True)
fx, cf = S / "fx", S / "cfes"


def trained(p, v):
    p.mkdir(parents=True, exist_ok=True)
    (p / "TRAINED").write_text(json.dumps({"best": v, "best_epoch": 0, "epoch": 5}))


def setup(vals, refs=(87.62, 89.68, 89.28)):
    shutil.rmtree(fx, ignore_errors=True)
    shutil.rmtree(cf, ignore_errors=True)
    trained(fx / "runs/ffpp_B0_s7", 90)
    for arm, v in vals.items():
        trained(fx / f"runs/ffpp_x5_{arm}_s7", v)
    for seed, v in zip((7, 17, 27), refs):
        if v is not None:
            trained(cf / f"runs_rev4/clip_D_w0.3_s{seed}", v)


def run(*extra):
    R.CF = str(cf)
    sys.argv = ["cesel_report.py", "--out_root", str(fx), *extra]
    R.main()
    return json.loads((fx / "results/cesel_ffpp_screen.json").read_text())


base = {"B0": 90, "sens": 89, "inv": 91, "floor": 92, "full": 95, "C_randcue": 80, "C_shufgroup": 80, "C_novalid": 80,
        "B0_cfesD": 90}
# Codex repro 1: +sens fails B0-continued -> Full (contains sens) must NOT win
setup(base)
d = run()["decisions"]
assert d["keep"]["sens"] is False and d["failed_terms"] == ["sens"] and "full" not in d["eligible"]
assert d["winner"] == "floor" and d["promote"] == ["B0", "B0+CFES-D", "+floor"], d
print("PASS F2a: +sens fails B0-continued (89 < 90) -> Full excluded; winner +floor (92), not Full (95)")
# Codex repro 2: reference spread unavailable -> no promotion
setup(base, refs=(None, None, None))
d = run()["decisions"]
assert d["spread"] is None and d["winner"] is None and d["promote"] is None and "reference spread" in d["unresolved"]
print(f"PASS F2b: reference spread unavailable -> fail closed (unresolved {d['unresolved']})")
# reference values differ from the frozen Amendment-1 values -> fail closed
setup(base, refs=(87.62, 89.68, 88.00))
d = run()["decisions"]
assert d["spread"] is None and d["promote"] is None and d["spread_status"].startswith("MISMATCH"), d["spread_status"]
print(f"PASS F2c: reference values changed -> fail closed ({d['spread_status']})")
# a pending arm -> fail closed
setup({k: v for k, v in base.items() if k != "C_novalid"})
d = run("--partial")["decisions"]
assert d["promote"] is None and "drop:C_novalid" in d["unresolved"]
print("PASS F2d: pending control arm -> no promotion")
# all gates resolved, a control matches Full -> its terms excluded
v2 = dict(base, sens=93, C_novalid=94.5)
setup(v2)
d = run()["decisions"]
assert d["dropped_terms"] == ["inv"] and d["winner"] == "sens" and not d["unresolved"], d
print("PASS F2e: C_novalid matches Full within 2.06 -> inv dropped -> Full and +inv excluded; winner +sens")

# F10: a later partial report never replaces a complete one; concurrent publishers leave a valid, complete report
setup(base)
j = run()
assert j["complete"]
shutil.rmtree(fx / "runs/ffpp_x5_C_novalid_s7")
j2 = run("--partial")
assert j2["complete"] and j2["progress"] == j["progress"], "partial report replaced the complete one"
print("PASS F10a: partial report (one arm missing) does not replace the complete report")
drv = S / "drv.py"
drv.write_text(f"import sys; sys.path[:0]={[str(ROOT), str(ROOT / 'scripts')]!r}\nimport cesel_report as R\n"
               f"R.CF={str(cf)!r}; sys.argv=['x','--out_root',{str(fx)!r}]+sys.argv[1:]; R.main()\n")
for rnd in range(3):
    setup(base)
    (fx / "results").mkdir(parents=True, exist_ok=True)
    ps = []
    for i in range(10):
        ps.append(subprocess.Popen([sys.executable, str(drv)] + (["--partial"] if i % 2 else []),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    outs = [p.communicate(timeout=300)[0] for p in ps]
    assert all(p.returncode == 0 for p in ps), outs
    j = json.loads((fx / "results/cesel_ffpp_screen.json").read_text())
    md = (fx / "results/cesel_ffpp_screen.md").read_text()
    left = [f for f in os.listdir(fx / "results") if f.endswith(".tmp") or f.startswith(".cesel_report_publish")]
    assert j["complete"] and "## Pre-registered decisions" in md and not left, (left, j.get("complete"))
print("PASS F10b: 3 rounds x 10 concurrent publishers: every exit 0, final md+json complete and valid, no temp/lock left")
shutil.rmtree(S, ignore_errors=True)
print("ALL_PASS test_report_f2_f10")
