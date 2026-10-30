import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("refresh", Path(__file__).resolve().parents[1] / "scripts/refresh_status.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class StatusContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "full_experiment_runs/B"
        self.run.mkdir(parents=True)
        self.launch = self.root / "launch"
        (self.launch / "runtime").mkdir(parents=True)

    def fixture(self):
        (self.run / "config.json").write_text(json.dumps({"epochs": 2, "backbone": "xception"}))
        (self.run / "history.jsonl").write_text("\n".join(json.dumps({"epoch": e, "validation": {"overall": {"auc": 0.9}}}) for e in (1, 2)))
        (self.run / "RUN_VERIFIED.json").write_text(json.dumps({"status": "PASS", "selected_epoch": 1, "validation_auc": 0.9, "test_evaluated": False}))
        (self.run / "best.pt").write_bytes(b"fixture; never loaded as a checkpoint")
        (self.run / "DONE").write_text("fixture")

    def test_verified_complete_and_earliest_auc_tie(self):
        self.fixture()
        row = M.observe_stage(self.root, self.launch, "B")
        self.assertEqual(row["state"], "COMPLETED_VERIFIED")
        self.assertFalse(row["test_evaluated"])
        self.assertFalse(row["checkpoint_rehashed_this_refresh"])

    def test_missing_artifacts_cannot_claim_completion(self):
        self.fixture()
        (self.run / "best.pt").unlink()
        self.assertEqual(M.observe_stage(self.root, self.launch, "B")["state"], "UNVERIFIED")

    def test_wrong_selected_epoch_rejected(self):
        self.fixture()
        p = self.run / "RUN_VERIFIED.json"
        v = json.loads(p.read_text())
        v["selected_epoch"] = 2
        p.write_text(json.dumps(v))
        self.assertEqual(M.observe_stage(self.root, self.launch, "B")["state"], "UNVERIFIED")

    def test_stale_running_record_is_explicitly_unverified(self):
        (self.launch / "runtime/B.progress.json").write_text('{"phase":"TRAINING"}')
        self.assertEqual(M.observe_stage(self.root, self.launch, "B")["state"], "RECORDED_PROGRESS_UNVERIFIED")

    def test_bad_json_cannot_silently_succeed(self):
        (self.run / "RUN_VERIFIED.json").write_text("broken")
        self.assertEqual(M.observe_stage(self.root, self.launch, "B")["state"], "UNVERIFIED")

    def test_done_without_receipt_is_unverified(self):
        (self.run / "DONE").write_text("fixture")
        self.assertEqual(M.observe_stage(self.root, self.launch, "B")["state"], "COMPLETED_UNVERIFIED")

    def test_explicit_template_output_path_is_supported(self):
        self.fixture()
        custom = self.root / "new_package/training"
        custom.parent.mkdir()
        self.run.rename(custom)
        self.assertEqual(M.observe_stage(self.root, self.launch, "B", custom)["state"], "COMPLETED_VERIFIED")

    def test_history_without_completion_does_not_claim_running_or_done(self):
        (self.run / "history.jsonl").write_text('{"epoch":1}\n')
        row = M.observe_stage(self.root, self.launch, "B")
        self.assertEqual(row["state"], "RECORDED_PROGRESS_UNVERIFIED")
        self.assertEqual(row["completed_epochs_recorded"], 1)

    def test_changed_review_source_does_not_inherit_old_test_claim(self):
        project = self.root / "project"
        (project / "src").mkdir(parents=True)
        (project / "provenance").mkdir()
        (project / "src/ces_fl_diagram_model_v021.py").write_text("# changed source")
        (project / "provenance/source_snapshot.json").write_text('{"tested_source_sha256":"historical"}')
        tracking = {"audit_root": str(self.root), "launch_root": str(self.launch), "arms": [], "seed": 17, "experiment_id": "fixture"}
        with patch.object(M, "command", return_value={"exit_code": 0, "stdout": "fixture"}):
            snapshot = M.collect(tracking, project)
        self.assertFalse(snapshot["review_code_matches_completed_run"])

    def test_synthetic_evaluation_cannot_become_benchmark_evidence(self):
        project = self.root / "project"
        (project / "src").mkdir(parents=True)
        (project / "provenance").mkdir()
        (project / "src/ces_fl_diagram_model_v021.py").write_text("# fixture")
        (project / "provenance/source_snapshot.json").write_text('{"tested_source_sha256":"fixture"}')
        tracking = {"audit_root": str(self.root), "launch_root": str(self.launch), "arms": [], "seed": 17, "experiment_id": "fixture",
                    "evaluation_records": [{"directory": str(self.root / "smoke_cuda/eval_test")}]}
        with patch.object(M, "command", return_value={"exit_code": 0, "stdout": "fixture"}):
            snapshot = M.collect(tracking, project)
        self.assertEqual(snapshot["evaluation_records"][0]["state"], "UNVERIFIED")

    def test_immutable_snapshot_refuses_same_timestamp_overwrite(self):
        data = {"observed_at_utc": "2026-10-31T00:00:00+00:00", "experiment_id": "fixture", "seed": 17,
                "stages": [], "evaluation_records": [], "resources": {}, "freshness": "fixture",
                "review_code_sha256": "x", "review_code_matches_completed_run": True}
        M.write_snapshot(data, self.root)
        with self.assertRaises(FileExistsError):
            M.write_snapshot(data, self.root)


if __name__ == "__main__":
    unittest.main()
