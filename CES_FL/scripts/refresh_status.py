#!/usr/bin/env python3
"""Publishable, read-only observation of explicitly tracked cluster artifacts.

This never trains, submits/kills jobs, changes a checkpoint, commits or pushes.
GitHub changes only after the operator reviews, commits and pushes the output.
"""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parent


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {"exit_code": p.returncode, "stdout": p.stdout.strip(),
                "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"error": str(e)}


def observe_stage(audit, launch, arm, run_directory=None):
    run = Path(run_directory) if run_directory is not None else Path(audit) / "full_experiment_runs" / arm
    progress_path = Path(launch) / "runtime" / (arm + ".progress.json")
    result = {"arm": arm, "run_directory": str(run), "state": "NO_ARTIFACTS"}
    try:
        verified_path = run / "RUN_VERIFIED.json"
        if verified_path.exists():
            v = read_json(verified_path)
            result["verification_receipt"] = v
            if v.get("status") != "PASS":
                result["state"] = "VERIFICATION_NOT_PASS"
                return result
            config = read_json(run / "config.json")
            history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines() if line.strip()]
            if not (run / "DONE").is_file() or not (run / "best.pt").is_file():
                raise ValueError("PASS receipt lacks DONE or best.pt")
            if len(history) != config["epochs"] or [h["epoch"] for h in history] != list(range(1, config["epochs"] + 1)):
                raise ValueError("History is not a complete sequence of configured epochs")
            values = [h["validation"]["overall"]["auc"] for h in history]
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
                raise ValueError("Undefined/nonfinite validation AUC")
            selected = max(history, key=lambda h: h["validation"]["overall"]["auc"])
            if selected["epoch"] != v["selected_epoch"] or not math.isclose(selected["validation"]["overall"]["auc"], v["validation_auc"], abs_tol=1e-12):
                raise ValueError("Verification receipt disagrees with validation selection")
            result.update(state="COMPLETED_VERIFIED", selected_epoch=v["selected_epoch"],
                          validation_auc=v["validation_auc"], test_evaluated=v.get("test_evaluated"),
                          checkpoint_sha256_from_receipt=v.get("checkpoint_sha256"),
                          checkpoint_rehashed_this_refresh=False, backbone=config["backbone"])
        elif (run / "DONE").exists():
            result["state"] = "COMPLETED_UNVERIFIED"
        elif progress_path.exists():
            p = read_json(progress_path)
            result.update(state="RECORDED_PROGRESS_UNVERIFIED", recorded_progress=p,
                          progress_mtime_utc=dt.datetime.fromtimestamp(progress_path.stat().st_mtime, dt.timezone.utc).isoformat())
        elif (run / "history.jsonl").exists():
            history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines() if line.strip()]
            result.update(state="RECORDED_PROGRESS_UNVERIFIED", completed_epochs_recorded=len(history),
                          history_mtime_utc=dt.datetime.fromtimestamp((run / "history.jsonl").stat().st_mtime, dt.timezone.utc).isoformat())
        elif run.exists():
            result["state"] = "ARTIFACTS_UNVERIFIED"
    except (OSError, ValueError, KeyError, TypeError) as e:
        result.update(state="UNVERIFIED", error=str(e))
    return result


def collect(tracking, project=PROJECT, resources=False):
    project = Path(project)
    source = project / "src" / "ces_fl_diagram_model_v021.py"
    original = read_json(project / "provenance" / "source_snapshot.json")
    current = sha(source)
    stages = [observe_stage(tracking["audit_root"], tracking["launch_root"], arm,
              tracking.get("run_directories", {}).get(arm)) for arm in tracking["arms"]]
    evaluations = []
    for record in tracking.get("evaluation_records", []):
        item = dict(record)
        try:
            p = Path(record["directory"])
            if any(part.startswith("smoke") for part in p.parts):
                raise ValueError("Synthetic smoke directories cannot supply benchmark results")
            metrics = read_json(p / "metrics.json")
            item.update(state="METRICS_PRESENT_REVIEW_REQUIRED", metrics=metrics,
                        predictions_present=(p / "predictions.csv").is_file())
        except (OSError, ValueError, KeyError, TypeError) as e:
            item.update(state="UNVERIFIED", error=str(e))
        evaluations.append(item)
    snapshot = {"observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "experiment_id": tracking["experiment_id"], "seed": tracking["seed"],
                "git_head_at_refresh": command(["git", "-C", str(project.parent), "rev-parse", "HEAD"]),
                "review_code_sha256": current, "completed_run_code_sha256": original["tested_source_sha256"],
                "review_code_matches_completed_run": current == original["tested_source_sha256"],
                "stages": stages, "evaluation_records": evaluations,
                "active_benchmarks": tracking.get("active_benchmarks", []),
                "freshness": "Point-in-time observation. No monitoring daemon or automatic GitHub publication.",
                "resources": {"state": "NOT_QUERIED"}}
    if resources:
        snapshot["resources"] = {"scheduler": command(["qstat", "-u", "sachin.chaudhary"])}
        for host in ("gpu01", "gpu02"):
            snapshot["resources"][host] = command(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
                "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader"])
    return snapshot


def render(snapshot):
    lines = ["# Current CES-FL status", "", "Last observed (UTC): **" + snapshot["observed_at_utc"] + "**.",
             "", snapshot["freshness"], "", "Experiment: `" + snapshot["experiment_id"] + "`; seed " + str(snapshot["seed"]) + ".",
             "", "| Arm | Observed state | Selected epoch | Validation AUC (%) | Real test evaluated |",
             "|---|---|---:|---:|---|"]
    for r in snapshot["stages"]:
        auc = r.get("validation_auc")
        lines.append("| " + " | ".join([r["arm"], r["state"], str(r.get("selected_epoch", "—")),
                     f"{100*auc:.4f}" if auc is not None else "—", str(r.get("test_evaluated", "UNVERIFIED"))]) + " |")
    lines += ["", "Review-source SHA256: `" + snapshot["review_code_sha256"] + "`.",
              "", "Review source matches the completed experiment source: **" + str(snapshot["review_code_matches_completed_run"]) + "**.",
              "If this becomes False, historical CPU/CUDA receipts and results do not verify the newly edited code.",
              "", "## Interpretation and pending work", "",
              "Read [the dated handoff](../docs/ces_fl/2026-10-31/HANDOFF_2026-10-31.md) for the historical fairness conclusion. This status page does not infer fairness from overall AUC.",
              "", "The current experiment is invariance-only. Full forensic selectivity is not validated. The historical CESel-MA gate remains FAIL.",
              "", "Table1 requires real FF++ test evaluation; Table2 requires DFDC/Celeb-DF/DFD adapters and evaluation; Table3 requires a separately prepared current-method ResNet-50 experiment. FairForge-2 evaluation/training is a separate phase. No training is launched by refreshing this page.",
              "", "Explicit evaluation records tracked: **" + str(len(snapshot["evaluation_records"])) + "**. Their detailed evidence/status appears in [current.json](status/current.json). Presence of files is not verification of protocol compliance.",
              "", "Checkpoint hashes above are from existing verification receipts; checkpoint bytes were not rehashed during this refresh. Recorded progress without verified completion is not proof a process is still alive.",
              "", "## Resource observation", "", "```json", json.dumps(snapshot["resources"], indent=2), "```", "",
              "Read [WORKFLOW.md](WORKFLOW.md) before proposing or launching a run. See [README.md](README.md) for code, configuration and result paths."]
    for batch in snapshot.get("active_benchmarks", []):
        lines += ["", "## Additional authorized benchmark batch", "",
                  "The stage table above describes the original pilot. Current batch: [" + batch["request_id"] + "](" + batch["live_github_url"] + "). Read that separately timestamped page for its submitted jobs and actual progress."]
    return "\n".join(lines) + "\n"


def write_snapshot(snapshot, project=PROJECT):
    project = Path(project)
    history = project / "status" / "history"
    history.mkdir(parents=True, exist_ok=True)
    stamp = snapshot["observed_at_utc"].replace(":", "").replace("+", "_")
    destination = history / (stamp + ".json")
    data = json.dumps(snapshot, indent=2, allow_nan=False) + "\n"
    with destination.open("x") as f:
        f.write(data)
    for path, content in [(project / "status" / "current.json", data), (project / "STATUS.md", render(snapshot))]:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(content)
        tmp.replace(path)
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracking", type=Path, default=PROJECT / "status" / "tracking.json")
    p.add_argument("--resources", action="store_true", help="Read qstat and GPU telemetry, never modify jobs")
    args = p.parse_args()
    snapshot = collect(read_json(args.tracking), resources=args.resources)
    print("STATUS_SNAPSHOT_WRITTEN", write_snapshot(snapshot))
    print("No training, scheduler mutations, commit or push performed.")


if __name__ == "__main__":
    main()
