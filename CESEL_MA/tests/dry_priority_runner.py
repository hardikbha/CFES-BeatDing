"""CPU dry run of the priority runner (scripts/priority_runner.py + run/ma_run.sh) with stubs, TWO runners sharing the
REAL task list (run/tasks.yaml, roots redirected to a scratch directory) plus a stub BEAT_DING include, for an injected
gate PASS and an injected gate FAIL (and a failing bd_smoke). ALL_PASS at the end.

  ~/.conda/envs/pulid/bin/python tests/dry_priority_runner.py [--scenario pass|fail|smokefail|all] [--keep]
Per scenario: synthetic images + demographic labels; a REAL tiny CESel B0 (stub backbone) -> validated JSON ->
evaluation, copied in as the single-axis winner; a screen report with the injected selection values; the external
tasks (clipfair / FairReL / Ding / Arm E / FF2) are ma_run.sh stubs that write their done markers, while the CESel-MA
tasks run the REAL trainer + evaluation + cue validation + ma_compare on CPU. Checks: exit codes, every task's final
state, priority order of the START lines (BD -> MA -> baselines s7 -> s17/27 -> Ding/Arm E -> FF2), no task run twice,
the injected failure retried, the HOLD file honoured, gate.json content.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/sachin.chaudhary/.conda/envs/pulid/bin/python"
PYD = "/home/sachin.chaudhary/.conda/envs/aiface/bin/python"
RES = []


def check(name, ok, info=""):
    RES.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({info})" if info else ""), flush=True)


def synth(S):
    from PIL import Image
    d = f"{S}/data"
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(96):
        lab = i % 2
        name = f"{lab}_{'Deepfakes' if i % 4 == 1 else 'Face2Face'}_{i:03d}_{i + 100:03d}_{i}.png" if lab else \
            f"0_youtube_{i:03d}_{i}.png"
        p = f"{d}/{name}"
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(p)
        rows.append(dict(img_path=p, label=lab, ismale=1, isasian=-1, iswhite=1, isblack=-1, intersec_label=(i // 2) % 8))
    df = pd.DataFrame(rows)
    df.to_csv(f"{d}/train.csv", index=False)
    df.iloc[:48].to_csv(f"{d}/val.csv", index=False)
    pd.DataFrame({"fake_path": df.img_path[df.label == 1].values[:16],
                  "real_path": df.img_path[df.label == 0].values[:16]}).to_csv(f"{d}/pairs.csv", index=False)
    L = f"{S}/labels"
    os.makedirs(L, exist_ok=True)
    # DEMO_AXES label format (labels_<set>.csv): cl_male 1 = male, cl_race name, age4_if, ita_vid, ita4
    grp = (np.arange(96) // 2) % 8
    ita = rng.uniform(-30, 60, 96)
    lab = pd.DataFrame({"img_path": df.img_path, "cl_male": (grp < 4).astype(int),
                        "cl_race": np.array(["asian", "white", "black", "others"])[grp % 4],
                        "age4_if": rng.integers(0, 4, 96), "ita_vid": ita, "ita4": np.searchsorted([0.0, 15.0, 30.0], ita)})
    for n in ("labels_ffpp_train", "labels_ffpp_val", "labels_ff2public_train", "labels_ff2public_val"):
        lab.to_csv(f"{L}/{n}.csv", index=False)
    json.dump({"ffpp_protocol": {"ita_gw": {"edges": [0.0, 15.0, 30.0]}}, "ff2": {"ita_gw": {"edges": [0.0, 15.0, 30.0]}}},
              open(f"{L}/ita_bin_edges.json", "w"))
    common = ["model.backend=stub", "data.input_size=64", "model.stub={dim:16,grid:8}", "ces.dim=8", "ces.hidden=16",
              "ces.freq_hidden=8", "ces.freq_widths=[4,4,8,8]", f"data.train.csv={d}/train.csv",
              f"data.val.csv=[{d}/val.csv]", f"pairs.csv={d}/pairs.csv", "pairs.per_step=2", "data.batch_size=8",
              "data.steps_per_epoch=2", "strat.per_cell=1", "optim.epochs=2", "optim.warmup_steps=1",
              "eval.batch_size=16", "log_every=1", "loss.inv.warmup_steps=1"]
    ma = [f"ma.labels.train={L}/labels_ffpp_train.csv", f"ma.labels.val={L}/labels_ffpp_val.csv",
          f"ma.labels.ita_edges_json={L}/ita_bin_edges.json", "loss.floor.n_min=1", "ma.n_eff_min=1"]
    return df, common, ma


def make_cesel(S, common, sel):
    """real tiny B0 (stub) -> validated JSON + evaluation; screen runs' TRAINED with the injected selection values."""
    C = f"{S}/cesel"
    b0 = f"{C}/runs/ffpp_B0_s7"
    env = dict(os.environ, PYTHONWARNINGS="ignore", CUDA_VISIBLE_DEVICES="")
    cmds = [[PY, f"{ROOT}/scripts/train_ces.py", "--config", f"{ROOT}/configs/ffpp/B0.yaml", "--out", b0, "--device", "cpu",
             "--set", "data.num_workers=0", *common],
            [PY, f"{ROOT}/scripts/ces_validate_cues.py", "--config", f"{ROOT}/configs/ffpp/B0.yaml", "--ckpt",
             f"{b0}/best.pt", "--out", f"{b0}/validated_cues.json", "--set", "data.num_workers=0", *common],
            [PY, f"{ROOT}/scripts/evaluate_ces.py", "--run", b0, "--ckpt", "best.pt", "--max_items", "64", "--device", "cpu"]]
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True, env=env)
        if r.returncode:
            raise SystemExit(f"setup failed: {' '.join(c[:3])}\n{r.stderr[-2000:]}")
    tr = json.load(open(f"{b0}/TRAINED"))
    labels = {"B0": "B0", "sens": "+sens", "inv": "+inv", "floor": "+floor", "full": "Full CESel",
              "C_randcue": "C_randcue", "C_shufgroup": "C_shufgroup", "C_novalid": "C_novalid", "B0_cfesD": "B0+CFES-D"}
    selection = {}
    for k, v in sel.items():
        d = b0 if k == "B0" else f"{C}/runs/ffpp_p1_{k}_s7"
        if k != "B0":
            os.makedirs(d, exist_ok=True)
            if k in ("full", "floor"):              # possible single-axis winners: give them B0's real evaluation
                shutil.copytree(f"{b0}/eval", f"{d}/eval", dirs_exist_ok=True)
        json.dump(dict(tr, best=v), open(f"{d}/TRAINED", "w"))
        selection[labels[k]] = {"select": v, "best_epoch": 1}
    os.makedirs(f"{C}/results", exist_ok=True)
    winner = max(("full", "floor", "sens", "inv"), key=lambda k: sel[k]) if sel["full"] - sel["B0"] > 0 else None
    json.dump({"selection": selection, "complete": True, "design": "p1", "exploratory": False, "progress": [1, 9, 9, 9],
               "decisions": {"spread_status": "ok: injected for the dry run", "winner": winner}},
              open(f"{C}/results/cesel_ffpp_screen_p1.json", "w"))
    return C


def tasks_file(S, smoke_fail):
    spec = yaml.safe_load(open(f"{ROOT}/run/tasks.yaml"))
    spec["include"] = [{"file": f"{S}/bd/bd_tasks.json", "priority": 5, "prefix": "", "optional": True, "reserve": True}]
    os.makedirs(f"{S}/bd", exist_ok=True)
    bd = {"tasks": [{"name": "bd_smoke", "cmd": ["bash", "-c", "echo smoke"], "done": [f"{S}/bd/runs/SMOKE_OK"]}] +
                   [{"name": f"BD{i}_s7", "cmd": f"echo BD{i}", "deps": ["bd_smoke"], "done": [f"{S}/bd/runs/BD{i}_s7/DONE"]}
                    for i in range(3)] +
                   [{"name": "bd_report", "cmd": "echo report", "gpu": False, "deps": [f"BD{i}_s7" for i in range(3)],
                     "done": [f"{S}/bd/results/report.md"]}]}
    json.dump(bd, open(f"{S}/bd/bd_tasks.json", "w"), indent=1)
    p = f"{S}/tasks.yaml"
    yaml.safe_dump(spec, open(p, "w"), sort_keys=False)
    return p


def scenario(name, keep):
    S = tempfile.mkdtemp(prefix=f"ma_dry_{name}_")
    print(f"--- scenario {name}: scratch {S}", flush=True)
    t0 = time.time()
    df, common, ma = synth(S)
    sel = {"B0": 85.0, "full": 88.5, "floor": 86.0, "sens": 85.5, "inv": 85.2, "C_randcue": 85.5, "C_shufgroup": 85.9,
           "C_novalid": 86.1, "B0_cfesD": 85.8}
    if name == "fail":
        sel.update(full=86.4, floor=85.5)                 # +1.4 / +0.5 over B0: not > 2.06 -> FAIL
    C = make_cesel(S, common, sel)
    tf = tasks_file(S, name == "smokefail")
    hold = f"{S}/HOLD_RELEASE"
    env = dict(os.environ, MA_DRY="1", MA_OUT=f"{S}/ma", MA_TASKS=tf, CESEL_ROOT=C, CFES_ROOT=f"{S}/cfes",
               CLIPFAIR_ROOT=f"{S}/clipfair", FAIRREL_ROOT=f"{S}/fairrel", DING_ROOT=f"{S}/ding", LABELS_ROOT=f"{S}/labels",
               GATE_SCREEN_JSON=f"{C}/results/cesel_ffpp_screen_p1.json", GATE_RUNS_ROOT=C, POLL="1", IDLE_EXIT_S="15",
               MA_DRY_SET=" ".join(common + ma), MA_DRY_SLEEP="2", MAX_CONC="2", MA_DRY_SLOTS="2", RETRY_WAIT_MIN="1",
               RETRY_WAIT_MAX="2", HOLD_UNTIL_FILES=hold, PYTHONWARNINGS="ignore", MA_WORKERS="0",
               MA_DRY_FAIL="daw_s7:1" + (",bd_smoke:9" if name == "smokefail" else ""))
    os.makedirs(f"{S}/ma", exist_ok=True)
    procs = []
    for host in ("gpu01", "gpu02"):
        e = dict(env, MA_HOST_OVERRIDE=host)
        procs.append(subprocess.Popen([PYD, f"{ROOT}/scripts/priority_runner.py"], env=e, stdout=subprocess.DEVNULL,
                                      stderr=open(f"{S}/runner_{host}.err", "w")))
        time.sleep(1)
    time.sleep(6)
    open(hold, "w").write("released\n")                 # the screen jobs 'end' 7 s after the runners start
    rcs = [p.wait(timeout=3600) for p in procs]
    logs = "".join(open(f"{S}/ma/logs/priority_{h}.log").read() for h in ("gpu01", "gpu02"))
    starts = sorted(re.findall(r"^(\S+) \[(\S+) \S+\] START (\S+) \(P([0-9.]+)\)", logs, re.M))
    started = [s[2] for s in starts]
    t_first = {}
    for ts, _, n, _ in starts:
        t_first.setdefault(n, ts)
    gate = json.load(open(f"{S}/ma/results/gate.json"))
    stat = {}
    for h in ("gpu01", "gpu02"):
        for r in json.load(open(f"{S}/ma/results/priority_status_{h}.json"))["tasks"]:
            if r["state"] in ("done", "skipped", "failed"):
                stat[r["name"]] = r["state"]
    spec = yaml.safe_load(open(tf))
    names = [t["name"] for t in spec["tasks"]] + [t["name"] for t in json.load(open(f"{S}/bd/bd_tasks.json"))["tasks"]]
    hold_ok = "HOLD: no launches" in logs and all(ts >= time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(
        os.path.getmtime(hold))) for ts, *_ in starts)
    ma_tasks = {"ma_s7", "ma_compare_s7", "ma_s17", "ma_s27"}
    ff2 = [n for n in names if n.startswith("ff2")]
    if name in ("pass", "smokefail"):
        cmpj = json.load(open(f"{S}/ma/results/ma_vs_single_s7.json"))
        nw = cmpj["not_worse"]
        want_skip = ({"ff2_final_single_s7"} if nw else {"ma_s17", "ma_s27", "ff2_final_ma_s7"})
        if name == "smokefail":
            want_skip |= {"BD0_s7", "BD1_s7", "BD2_s7", "bd_report"}
        want_fail = {"bd_smoke"} if name == "smokefail" else set()
        ok_gate = gate["status"] == "PASS" and gate["ma_config"] == "ma" and gate["single_axis_arm"] == "full" \
            and gate["detail"]["passing_arms"] == ["full"] and not gate["test_metrics_used"]
        info_g = f"gate PASS base {gate.get('ma_base_arm')}, MA {cmpj['ma']['value_ma']:.2f} vs single " \
                 f"{cmpj['single_axis']['value_ma']:.2f} -> not_worse {nw}"
    else:
        want_skip = ma_tasks | {"ff2_final_ma_s7", "ff2_final_single_s7"}
        want_fail = set()
        ok_gate = gate["status"] == "FAIL" and gate["detail"]["passing_arms"] == [] and "ma_config" not in gate
        info_g = f"gate FAIL, candidates {[(k, round(v['diff_vs_B0'], 2)) for k, v in gate['detail']['candidates'].items()]}"
    want_done = set(names) - want_skip - want_fail
    got_done = {n for n, s in stat.items() if s == "done"}
    got_skip = {n for n, s in stat.items() if s == "skipped"}
    got_fail = {n for n, s in stat.items() if s == "failed"}
    ok_states = got_done == want_done and got_skip == want_skip and got_fail == want_fail
    want_rc = [7, 7] if name == "smokefail" else [0, 0]
    if name == "smokefail" and 7 in rcs and set(rcs) <= {0, 7}:
        want_rc = rcs                  # a runner whose remaining tasks were held by the other may leave by idle exit (0)
    # priority order: for tasks that are ready from the start (no deps, when always), a higher-priority task never
    # starts later than a lower-priority one; bd_smoke starts first; FF2 phase 1 after every such FF++ task
    prio = {t["name"]: t["priority"] for t in spec["tasks"]}
    prio.update({t["name"]: 5 for t in json.load(open(f"{S}/bd/bd_tasks.json"))["tasks"]})
    free = [t["name"] for t in spec["tasks"] if not t.get("deps") and t.get("when", "always") == "always"
            and not t.get("needs")] + ["bd_smoke"]
    inv = [(a, b) for a in free for b in free if prio[a] < prio[b] and a in t_first and b in t_first
           and t_first[a] > t_first[b]]
    first = sorted(t_first.items(), key=lambda kv: kv[1])
    order_ok = not inv and all(t_first["bd_smoke"] <= ts for _, ts in first)
    acq = {}
    for n in got_done:
        f = f"{S}/ma/logs/ma_runs/{n}.log"
        acq[n] = open(f).read().count(" acquired: token ") if os.path.exists(f) else 0
    dup = {n: c for n, c in acq.items() if c != 1}
    retried = "RETRY daw_s7" in logs and t_first.get("daw_s7") is not None
    check(f"[{name}] two runners exit {rcs} (want {want_rc}); gate.json correct", rcs == want_rc and ok_gate, info_g)
    check(f"[{name}] final task states (done {len(got_done)}, skipped {sorted(got_skip)}, failed {sorted(got_fail)})",
          ok_states, f"missing done {sorted(want_done - got_done)}; unexpected skip {sorted(got_skip - want_skip)}")
    check(f"[{name}] priority order: no task starts before bd_smoke, no lower-priority ready task before a higher one; "
          f"HOLD honoured; "
          f"injected daw_s7 failure retried", order_ok and hold_ok and retried,
          f"first starts {[n for n, _ in first[:6]]}; order inversions {inv[:3]}; hold {hold_ok}; retried {retried}")
    shared = len([1 for h in ("gpu01", "gpu02") if f"[{h} " in logs and re.search(rf"\[{h} \S+\] START", logs)]) == 2
    check(f"[{name}] both runners launched tasks from the one shared list; every done task ran exactly once (one "
          f"RUNLOCK acquisition)", shared and not dup, f"{len(started)} starts in {time.time() - t0:.0f} s; dup {dup}")
    if not keep and all(ok for _, ok in RES):
        shutil.rmtree(S, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", choices=["pass", "fail", "smokefail", "all"])
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    for sc in (["pass", "fail", "smokefail"] if a.scenario == "all" else [a.scenario]):
        scenario(sc, a.keep)
    failed = [n for n, ok in RES if not ok]
    print(f"{len(RES) - len(failed)}/{len(RES)} checks passed")
    print("ALL_PASS" if not failed else "FAILED: " + "; ".join(failed))
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
