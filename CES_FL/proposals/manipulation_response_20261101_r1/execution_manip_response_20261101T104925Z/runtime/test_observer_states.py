"""Status regressions: synthetic-smoke progress never becomes real arm progress."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

spec = importlib.util.spec_from_file_location('mr_observer', Path(__file__).with_name('observe_smoke.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class States(unittest.TestCase):
    def observe(self, job_state, exit_status=None, passed=False):
        record = {'job_state': job_state, 'queue': 'gpu', 'Resource_List': {'ngpus': 1}}
        if exit_status is not None:
            record['Exit_status'] = exit_status
        def read(path):
            if path.name == 'smoke_SUBMISSION.json': return {'job_id': 'SYNTHETIC_JOB'}
            if path.name == 'smoke_PASS.json' and passed: return {'status': 'PASS'}
            if path.name == 'RESULT.json' and passed: return {'status': 'PASS', 'device': 'cuda'}
            return None
        with patch.object(m, 'read', read), patch.object(m.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=json.dumps({'Jobs': {'SYNTHETIC_JOB': record}}), stderr='')):
            d = m.observe()
        self.assertEqual(set(s['status'] for s in d['arms'].values()), {'BLOCKED_INPUTS'})
        self.assertFalse(d['new_training_launched'])
        return d

    def test_queued_is_not_running(self):
        self.assertEqual(self.observe('Q')['smoke']['state'], 'QUEUED')

    def test_running_smoke_is_not_running_real_training(self):
        self.assertEqual(self.observe('R')['smoke']['state'], 'RUNNING')

    def test_failed_job_is_reported(self):
        self.assertEqual(self.observe('F', 1)['smoke']['state'], 'FAILED')

    def test_completion_requires_receipts(self):
        self.assertEqual(self.observe('F', 0)['smoke']['state'], 'FAILED_MISSING_PASS_RECEIPTS')
        self.assertEqual(self.observe('F', 0, passed=True)['smoke']['state'], 'COMPLETED')


if __name__ == '__main__':
    unittest.main(verbosity=2)
