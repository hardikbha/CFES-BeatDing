import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPT_DIR))
import lh_resnet50_locked_table3 as locked  # noqa: E402


class LockedResNetTable3GuardTests(unittest.TestCase):
    def test_pinned_protocol_loads(self):
        core = locked.load_core()
        protocol, parent = locked.load_protocol(core)
        self.assertEqual(protocol['threshold_views'], [
            'fixed_probability_0.5', 'selected_FFPP_validation_real_empirical_10pct'])
        self.assertEqual(set(parent['datasets']), set(locked.DATASETS))

    def test_rejects_test_domain_calibration(self):
        core = locked.load_core()
        altered = json.loads(locked.PROTOCOL_PATH.read_text())
        altered['target_domain_calibration'] = True
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'protocol.json'
            path.write_text(json.dumps(altered))
            with patch.object(locked, 'PROTOCOL_PATH', path):
                with self.assertRaisesRegex(ValueError, 'TEST-dependent'):
                    locked.load_protocol(core)

    def test_rejects_changed_threshold_views(self):
        core = locked.load_core()
        altered = json.loads(locked.PROTOCOL_PATH.read_text())
        altered['threshold_views'] = ['target_TEST_FPR']
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'protocol.json'
            path.write_text(json.dumps(altered))
            with patch.object(locked, 'PROTOCOL_PATH', path):
                with self.assertRaisesRegex(ValueError, 'Threshold views'):
                    locked.load_protocol(core)

    def test_rejects_synthetic_fixture_checkpoint(self):
        core = locked.load_core()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            checkpoint = root / 'best.pt'
            config = {'backbone': 'resnet50', 'epochs': 10, 'seed': 17, 'target_fpr': 0.1}
            torch.save({'format': core.VERSION, 'source_sha256': locked.CORE_SHA,
                        'arm': 'A', 'config': config, 'synthetic_fixture': True}, checkpoint)
            history = [{'epoch': i, 'validation': {'overall': {'auc': i / 10}}} for i in range(1, 11)]
            (root / 'history.json').write_text(json.dumps(history))
            (root / 'DONE').write_text('fixture only')
            (root / 'RUN_VERIFIED.json').write_text(json.dumps({
                'status': 'COMPLETED_VERIFIED', 'test_evaluated': False,
                'selected_checkpoint_sha256': locked.sha(checkpoint), 'source_sha256': locked.CORE_SHA}))
            with self.assertRaisesRegex(ValueError, 'synthetic or non-ResNet'):
                locked.verify_selected(core, root, 'L_A')

    def test_rejects_stale_plan_runner_hash(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'plan.json'
            path.write_text(json.dumps({'format': 'CESFL_RESNET50_TABLE3_LOCKED_v1',
                                        'protocol_sha256': locked.sha(locked.PROTOCOL_PATH),
                                        'runner_sha256': '0' * 64,
                                        'core_source_sha256': locked.CORE_SHA,
                                        'threshold_views': ['fixed_probability_0.5',
                                                            'selected_FFPP_validation_real_empirical_10pct'],
                                        'test_threshold_search': False,
                                        'target_domain_calibration': False,
                                        'test_driven_selection': False}))
            with self.assertRaisesRegex(ValueError, 'protocol/runner changed'):
                locked.load_plan(path)


if __name__ == '__main__':
    unittest.main()
