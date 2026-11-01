"""Fail-closed gates for launching the original-only comparison; no real jobs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('baseline_launcher', Path(__file__).with_name('launch_baseline.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class Gates(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for folder in ['checks', 'bound_originals', 'dry_bce']:
            (self.root / folder).mkdir()
        self.patch = patch.object(m, 'ROOT', self.root); self.patch.start()
        self.write('checks/P_OPERATOR_VERIFIED.json', {'status': 'PASS', 'completed_epochs': 50, 'checkpoint_sha256': 'CHECKPOINT'})
        self.write('bound_originals/bound_config.json', {'bank': None, 'pretrain_sha256': 'CHECKPOINT'})

    def tearDown(self):
        self.patch.stop(); self.temp.cleanup()

    def write(self, name, data):
        (self.root / name).write_text(json.dumps(data))

    def add_dry_pass(self):
        self.write('dry_bce/DRY_RUN.json', {'status': 'PASS', 'arm': 'bce', 'device': 'cuda', 'optimizer_steps': 0,
            'quartets': 0, 'synthetic_fixture': False, 'losses': {'total': .5}})
        self.write('checks/dry_bce_PASS.json', {'status': 'PASS', 'binding': {'host': 'gpu02', 'gpu_uuid': 'DEVICE5',
            'bound_config_sha256': m.sha(self.root / 'bound_originals/bound_config.json'), 'pretrain_checkpoint_sha256': 'CHECKPOINT'}})

    def test_paired_arm_not_supported(self):
        with self.assertRaises(ValueError): m.gate('response', 'train', 'gpu02', 'DEVICE5')

    def test_no_training_before_configured_backward(self):
        with self.assertRaises(FileNotFoundError): m.gate('bce', 'train', 'gpu02', 'DEVICE5')

    def test_device_and_host_must_match_backward(self):
        self.add_dry_pass()
        m.gate('bce', 'train', 'gpu02', 'DEVICE5')
        with self.assertRaises(ValueError): m.gate('bce', 'train', 'gpu02', 'DEVICE4')
        with self.assertRaises(ValueError): m.gate('bce', 'train', 'gpu01', 'DEVICE5')

    def test_changed_config_or_existing_outputs_blocked(self):
        self.add_dry_pass()
        self.write('bound_originals/bound_config.json', {'bank': None, 'pretrain_sha256': 'CHECKPOINT', 'epochs': 49})
        with self.assertRaises(ValueError): m.gate('bce', 'train', 'gpu02', 'DEVICE5')
        self.write('bound_originals/bound_config.json', {'bank': None, 'pretrain_sha256': 'CHECKPOINT'})
        self.add_dry_pass()
        (self.root / 'bce').mkdir()
        with self.assertRaises(ValueError): m.gate('bce', 'train', 'gpu02', 'DEVICE5')

    def test_incomplete_p_or_nonfinite_dry_blocked(self):
        self.write('checks/P_OPERATOR_VERIFIED.json', {'status': 'PASS', 'completed_epochs': 49, 'checkpoint_sha256': 'CHECKPOINT'})
        with self.assertRaises(ValueError): m.gate('bce', 'dry', 'gpu02', None)
        self.write('checks/P_OPERATOR_VERIFIED.json', {'status': 'PASS', 'completed_epochs': 50, 'checkpoint_sha256': 'CHECKPOINT'})
        self.add_dry_pass()
        dry_path = self.root / 'dry_bce/DRY_RUN.json'
        dry = json.loads(dry_path.read_text()); dry['losses']['total'] = float('nan')
        dry_path.write_text(json.dumps(dry))
        with self.assertRaises(ValueError): m.gate('bce', 'train', 'gpu02', 'DEVICE5')


if __name__ == '__main__':
    unittest.main(verbosity=2)
