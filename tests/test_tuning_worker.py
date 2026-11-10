"""Exercise queue/lease exclusion in private temporary directories, no GPUs."""
import concurrent.futures
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('tuning_worker_test',SOURCE/'tuning_worker.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


class QueueTests(unittest.TestCase):
    def test_two_consumers_cannot_claim_the_same_job_or_card(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);locks=root/'locks';jobs=[dict(name='fair_only',state='PREPARED')]
            (root/'QUEUE.json').write_text(json.dumps(dict(jobs=jobs)))
            card=dict(index=0,uuid='GPU-SYNTHETIC-NOT-A-REAL-DEVICE',free_mib=81000,util=0,available_mib=81000)
            with patch.object(w,'ROOT',root),patch.object(w.pool,'LOCKS',locks),patch.object(w.pool,'candidates',return_value=[card]):
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    results=list(executor.map(w.claim,['fixture_worker0','fixture_worker1']))
            self.assertEqual(sum(r[0] is not None for r in results),1)
            self.assertEqual(json.loads((root/'QUEUE.json').read_text())['jobs'][0]['state'],'RUNNING')
            self.assertEqual(len(list(locks.rglob('task_*.json'))),1)

    def test_failed_job_is_not_automatically_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'QUEUE.json').write_text(json.dumps(dict(jobs=[dict(name='fair_only',state='FAILED')])))
            with patch.object(w,'ROOT',root),patch.object(w.pool,'candidates') as candidates:
                self.assertEqual(w.claim('fixture_worker'),(None,None,None));candidates.assert_not_called()

    def test_concurrent_status_updates_are_not_lost(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'QUEUE.json').write_text(json.dumps(dict(jobs=[dict(name=str(i),state='RUNNING') for i in range(4)])))
            with patch.object(w,'ROOT',root):
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    list(executor.map(lambda i:w.update(str(i),stage='DONE',state='COMPLETED_VERIFIED'),range(4)))
            self.assertTrue(all(j['state']=='COMPLETED_VERIFIED' for j in json.loads((root/'QUEUE.json').read_text())['jobs']))


if __name__=='__main__':unittest.main(verbosity=2)
