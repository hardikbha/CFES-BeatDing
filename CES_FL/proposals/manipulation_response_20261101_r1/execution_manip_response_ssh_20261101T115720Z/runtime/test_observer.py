"""Observer regressions with temporary fake records; never launches work."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('ssh_observer', Path(__file__).with_name('observe_direct.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class Status(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'checks').mkdir(); (self.root / 'logs').mkdir()
        self.context = patch.object(m, 'ROOT', self.root); self.context.start()

    def tearDown(self):
        self.context.stop(); self.temp.cleanup()

    def write(self, name, data):
        (self.root / 'checks' / name).write_text(json.dumps(data))

    def test_running_requires_live_worker(self):
        self.write('P_SUBMISSION.json', {'host': 'gpu02', 'pid': 17})
        self.write('P_PROGRESS.json', {'phase': 'RUNNING'})
        command = str(self.root / 'runtime/direct_mr.py')
        with patch.object(m.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout=command)):
            self.assertEqual(m.stage_state('P')['status'], 'RUNNING')
        with patch.object(m.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='')):
            self.assertEqual(m.stage_state('P')['status'], 'BLOCKED_INPUTS')

    def test_smoke_success_never_marks_scientific_arms_completed(self):
        self.write('smoke_PASS.json', {'status': 'PASS'})
        result = m.observe()
        self.assertEqual(result['stages']['smoke']['status'], 'COMPLETED')
        self.assertEqual({r['status'] for r in result['arms'].values()}, {'BLOCKED_INPUTS'})

    def test_arm_completion_requires_native_replay_and_fifty_epochs(self):
        self.write('P_PASS.json', {'status': 'PASS'})
        result = m.observe()
        self.assertEqual(result['arms']['P']['status'], 'BLOCKED_INPUTS')
        self.assertIn('completion_issue', result['arms']['P'])

    def test_failure_is_explicit(self):
        self.write('real_preflight_FAILURE.json', {'exception': 'SYNTHETIC_FAILURE'})
        result = m.observe()
        self.assertEqual(result['stages']['real_preflight']['status'], 'FAILED')
        self.assertEqual(result['arms']['response']['status'], 'BLOCKED_INPUTS')
        self.assertIn('SYNTHETIC_FAILURE', m.render(result))


if __name__ == '__main__':
    unittest.main(verbosity=2)
