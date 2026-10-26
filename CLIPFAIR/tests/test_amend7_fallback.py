"""CPU mocked epoch loop using production selection and real checkpoint I/O."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import torch
import train_clipfair as trainer


def result(auc, tpr=85., fpr=15., group_tpr=85., gap=4.):
    return dict(auc=auc, threshold_tpr=auc / 100,
                oppoint=dict(pooled_TPR=tpr, pooled_FPR=fpr, worst_FPR=18.,
                             group_TPR={"g": group_tpr}, fpr_gap=gap))


class FallbackTests(unittest.TestCase):
    def run_epochs(self, rows, expected_fallback, expected_epoch, pooled_epoch, start_auc=90.):
        cfg = {"select": {"metric": "val_oppoint", "ref_checks": True}, "ckpt": {"full": True}}
        model = torch.nn.Linear(1, 1)
        start = result(start_auc, group_tpr=99.)
        state = dict(epoch=0, step=0, best=trainer.T.selection_value(start, "val_oppoint"),
                     best_epoch=-1, best_threshold_tpr=start["threshold_tpr"], select_ref=start["oppoint"],
                     trained_eligible_epochs=0, best_pooled=-float("inf"), best_pooled_epoch=-1)
        # All scratch writes stay within CLIPFAIR, never the system temp directory.
        with tempfile.TemporaryDirectory(dir=ROOT / "codex_ops") as out:
            with torch.no_grad():
                model.weight.fill_(-1)
            trainer.save_checkpoint(os.path.join(out, "best.pt"), model, None, None, None, state)
            original = Path(out, "best.pt").read_bytes()
            for epoch, res in enumerate(rows):
                with torch.no_grad():
                    model.weight.fill_(epoch)
                state["epoch"] = epoch + 1
                trainer.select_epoch(model, cfg, out, state, epoch, res)
                # Exercise persistence of selection state across a resume.
                trainer.save_checkpoint(os.path.join(out, "last.pt"), model, None, None, None, state)
                state = trainer.load_checkpoint(os.path.join(out, "last.pt"), model)
            trainer.finalize_selection(out, state, "val_oppoint")
            summary = json.loads(Path(out, "TRAINED").read_text())
            self.assertEqual(summary["selection_fallback"], expected_fallback)
            self.assertEqual(summary["best_epoch"], expected_epoch)
            self.assertEqual(summary["best_pooled_epoch"], pooled_epoch)
            ckpt = torch.load(Path(out, "best.pt"), map_location="cpu", weights_only=False)
            self.assertEqual(ckpt["state"]["best_epoch"], expected_epoch)
            self.assertEqual(ckpt["model"]["weight"].item(), expected_epoch)
            if expected_fallback is True:
                self.assertEqual(Path(out, "best.pt").read_bytes(), Path(out, "best_pooled.pt").read_bytes())
            if expected_epoch == -1:
                self.assertEqual(Path(out, "best.pt").read_bytes(), original)
            self.assertEqual(Path(out, "best_pooled.pt").exists(), pooled_epoch >= 0)

    def test_normal(self):
        self.run_epochs([result(93., group_tpr=98.), result(96.)], False, 0, 1)

    def test_eligible_below_start_prevents_fallback(self):
        self.run_epochs([result(88., group_tpr=98.), result(96.)], False, -1, 1)

    def test_pooled_best_uses_penalized_score_and_boundaries(self):
        self.run_epochs([result(93.), result(96., tpr=80., fpr=20.),
                         result(97., gap=12.), result(99., tpr=79.)], True, 1, 1)

    def test_none_eligible(self):
        self.run_epochs([result(93., tpr=79.), result(96., fpr=21.)], "none_eligible", -1, -1)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FallbackTests)
    outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    print("ALL_PASS" if outcome.wasSuccessful() else "SOME_FAILED", flush=True)
    sys.exit(0 if outcome.wasSuccessful() else 1)
