import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

PROPOSAL=Path(__file__).resolve().parents[1]/'proposals/ff2_baseline_suite_s17_20261103T110324Z'
spec=importlib.util.spec_from_file_location('ff2_budget',PROPOSAL/'budget.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.path=Path(self.tmp.name)/'ledger.json'
    def tearDown(self):self.tmp.cleanup()
    def test_three_gpu_concurrency(self):
        for i in range(3):b.claim(self.path,str(i),1,18,'actual-allocation',now=0)
        with self.assertRaisesRegex(ValueError,'concurrency'):b.claim(self.path,'fourth',1,1,'actual-allocation',now=0)
        self.assertEqual(len(json.loads(self.path.read_text())['jobs']),3)
    def test_cap_includes_prior_usage(self):
        b.claim(self.path,'prior',2,60,'actual-allocation',now=0);b.finish(self.path,'prior',now=30*3600)
        with self.assertRaisesRegex(ValueError,'72 additional'):b.claim(self.path,'more',1,13,'actual-allocation',now=30*3600)
        self.assertEqual(b.claim(self.path,'within',1,12,'actual-allocation',now=30*3600)['uncommitted'],0)
    def test_charge_gpu_hours_and_release_reservation(self):
        b.claim(self.path,'a',3,60,'allocation',now=0)
        r=b.finish(self.path,'a',now=2*3600)
        self.assertEqual(r,{'charged':6.,'reserved':0,'active_gpus':0,'uncommitted':66.})
    def test_duplicate_job_and_finish_rejected(self):
        b.claim(self.path,'a',1,18,'allocation',now=0)
        with self.assertRaises(ValueError):b.claim(self.path,'a',1,18,'allocation',now=0)
        b.finish(self.path,'a',now=1)
        with self.assertRaises(ValueError):b.finish(self.path,'a',now=2)
    def test_overdue_active_job_not_hidden(self):
        b.claim(self.path,'a',1,1,'allocation',now=0)
        with self.assertRaises(ValueError):b.claim(self.path,'b',1,22,'allocation',now=51*3600)
        r=b.finish(self.path,'a',now=51*3600)
        self.assertEqual(r['uncommitted'],21)
        self.assertTrue(json.loads(self.path.read_text())['jobs']['a']['reservation_overrun'])
    def test_invalid_claims(self):
        for g,h,e in [(4,1,'x'),(1,0,'x'),(1,float('nan'),'x'),(1,73,'x'),(1,1,'')]:
            with self.assertRaises(ValueError):b.claim(self.path,'a',g,h,e,now=0)


if __name__=='__main__':unittest.main()
