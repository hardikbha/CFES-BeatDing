import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('ding_observer',Path(__file__).resolve().parents[1]/'scripts/watch_ding_ff2_public.py')
observer=importlib.util.module_from_spec(spec);spec.loader.exec_module(observer)


class ObserverFailures(unittest.TestCase):
    def fixture(self,root):
        (root/'STATUS.json').write_text(json.dumps(dict(state='RUNNING',host='gpu01',torchrun_pid=123)))
        (root/'logs').mkdir()
        (root/'logs/training.log').write_text(json.dumps(dict(event='TRAIN_STEP',epoch=6,batch=2000))+'\n')

    def test_ssh_timeout_is_unverified_and_keeps_observation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.fixture(root)
            with patch.object(observer.subprocess,'run',side_effect=subprocess.TimeoutExpired('ssh',30)):
                state=observer.observe(root)
            self.assertEqual(state['reported_state'],'RUNNING_UNVERIFIED_SSH')
            self.assertEqual(state['process']['verification'],'SSH_TIMED_OUT')
            self.assertEqual(state['latest_training_event']['epoch'],6)
            self.assertFalse(state['test_evaluation'])

    def test_missing_process_is_not_reported_running(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.fixture(root)
            result=subprocess.CompletedProcess([],0,json.dumps(dict(torchrun_proc_exists=False)),'')
            with patch.object(observer.subprocess,'run',return_value=result):
                state=observer.observe(root)
            self.assertEqual(state['reported_state'],'PROCESS_MISSING_INSPECT_LOGS')


if __name__=='__main__':
    unittest.main(verbosity=2)
