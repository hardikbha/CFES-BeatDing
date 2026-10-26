"""Validate a Beat-Ding task file exactly as the CESEL_MA priority runner will read it (its own normalise_external /
validate_spec, plus load_tasks' duplicate-name and unknown-dependency checks over tasks.yaml + every include).
  aiface python tests/validate_bd_tasks.py [run/bd_tasks.json]      -> prints JSON, exit 1 on any error"""
import importlib.util
import json
import os
import sys

import yaml

PR = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA/scripts/priority_runner.py"
TASKS = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA/run/tasks.yaml"


def runner():
    spec = importlib.util.spec_from_file_location("priority_runner_check", PR)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)        # module level only defines constants / functions; main() is not called
    return m


def validate(path):
    P = runner()
    spec = yaml.safe_load(open(TASKS)) or {}
    inc = [i for i in spec.get("include") or [] if "BEAT_DING/run/bd_tasks.json" in i.get("file", "")]
    if len(inc) != 1:
        return {"errors": [f"tasks.yaml includes bd_tasks.json {len(inc)} times"]}
    inc = dict(inc[0], _path=path)
    raw = yaml.safe_load(open(path))
    lst = raw.get("tasks") if isinstance(raw, dict) else raw
    norm = [P.normalise_external(t, inc) for t in lst]
    errs = [e for e in (P.validate_spec(t) for t in norm) if e]
    others = list(spec.get("tasks") or [])
    for i2 in spec.get("include") or []:          # other includes (none today): read as the runner would
        f = P.subst(i2.get("file", ""), P.ROOTS)
        if f and "BEAT_DING/run/bd_tasks.json" not in f and os.path.exists(f):
            r2 = yaml.safe_load(open(f)) or {}
            others += [P.normalise_external(t, dict(i2, _path=f)) for t in (r2.get("tasks") if isinstance(r2, dict) else r2) or []]
    allt = others + norm
    names = [t["name"] for t in allt if isinstance(t, dict) and "name" in t]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        errs.append(f"duplicate task names {dup}")
    known = set(names)
    errs += [f"{t['name']}: unknown dependency {d}" for t in allt for d in (t.get("deps") or []) if d not in known]
    return {"file": path, "n_tasks": len(norm), "errors": errs, "order": [t["name"] for t in norm],
            "kinds": {t["name"]: t["run"][0] for t in norm}, "priority": sorted({t["priority"] for t in norm})}


if __name__ == "__main__":
    r = validate(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__))), "run", "bd_tasks.json"))
    print(json.dumps(r))
    sys.exit(1 if r["errors"] else 0)
