"""Direct-SSH safeguards; no CUDA or research training is executed here."""
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

spec = importlib.util.spec_from_file_location('direct', Path(__file__).with_name('direct_mr.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
UUID = 'GPU-ee335748-8273-2d67-42a5-d45463027c1b'


class Guards(unittest.TestCase):
    def test_shared_headroom_does_not_require_killing_existing_work(self):
        self.assertTrue(m.eligible(['4', UUID, '81559', '26351', '0'], {UUID}, set()))

    def test_protected_qwen_device_rejected(self):
        self.assertFalse(m.eligible(['4', UUID, '81559', '70000', '0'], set(), {UUID}))

    def test_low_headroom_and_wrong_binding_rejected(self):
        self.assertFalse(m.eligible(['4', UUID, '81559', '19000', '0'], set(), set()))
        self.assertFalse(m.eligible(['4', UUID, '81559', '26351', '0'], set(), set(), {'3'}))

    def test_uuid_normalization_is_exact(self):
        self.assertEqual(m.normalized_uuid(UUID), m.normalized_uuid(UUID[4:]))
        self.assertNotEqual(m.normalized_uuid(UUID), m.normalized_uuid('GPU-6a002609-de85-4949-c707-ee9317b284f0'))

    def test_pretraining_cannot_start_without_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'checks').mkdir()
            with patch.object(m, 'ROOT', root), patch.object(m, 'verified_sources', return_value={}), patch.object(m, 'capture') as external:
                with self.assertRaises(FileNotFoundError): m.main('P')
                external.assert_not_called()

    def test_paired_arms_have_no_ssh_launch_path(self):
        with self.assertRaises(AssertionError): m.main('response')


if __name__ == '__main__':
    unittest.main(verbosity=2)
