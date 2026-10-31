"""Operational handoff tests. qsub is mocked; no experiment jobs submitted."""
import importlib.util, json, tempfile, types, unittest
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).with_name('execute_handoff.py')
spec=importlib.util.spec_from_file_location('handoff',p);h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)

class HandoffSafety(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'runtime').mkdir();(self.root/'logs').mkdir()
        self.swap=patch.object(h,'ROOT',self.root);self.swap.start()
        h.write(self.root/'smoke_submission.json',{'job_id':'9999.mgmt01'})
    def tearDown(self):
        self.swap.stop();self.tmp.cleanup()
    def test_queued_smoke_does_not_pass(self):
        with patch.object(h,'scheduler',return_value={'job_state':'Q'}):
            self.assertFalse(h.smoke_passed())
        self.assertFalse((self.root/'CUDA_GATE_PASSED.json').exists())
    def test_failed_smoke_rejects(self):
        with patch.object(h,'scheduler',return_value={'job_state':'F','Exit_status':1}):
            with self.assertRaises(RuntimeError):h.smoke_passed()
    def test_complete_smoke_requires_eight_tests_binding_and_marker(self):
        out=self.root/'SMOKE_R1';out.mkdir()
        h.write(self.root/'runtime/SMOKE_R1.exit.json',{'exit_code':0})
        with patch.object(h,'scheduler',return_value={'job_state':'F','Exit_status':0}):
            with self.assertRaises(AssertionError):h.smoke_passed()
            (out/'CUDA_SMOKE_PASS').write_text('SYNTHETIC OPERATIONAL TEST')
            (out/'smoke_cuda.log').write_text('Ran 8 tests in 1.0s\n\nOK\n')
            h.write(out/'launch.json',{'git_commit':h.COMMIT,'smoke_only':True,'pbs_jobid':'9999.mgmt01','cuda_visible_devices':'GPU-SYNTHETIC'})
            (out/'preflight.log').write_text('{"status":"PASS"}\n')
            h.write(self.root/'runtime/SMOKE_R1.binding.json',{'state':'BOUND','gpu_uuid':'GPU-SYNTHETIC'})
            self.assertTrue(h.smoke_passed())
    def test_submit_without_cuda_gate_rejects(self):
        with patch.object(h,'pinned',return_value=None):
            with self.assertRaises(AssertionError):h.submit(h.ARMS[0])
    def test_exact_five_unique_mock_submissions_and_no_repeat(self):
        h.write(self.root/'CUDA_GATE_PASSED.json',{'pbs_job_id':'9999.mgmt01'})
        commands=[]
        def run(args,**kw):
            self.assertEqual(args[0],'qsub');commands.append(args)
            kw['stdout'].write(str(99000+len(commands))+'.mgmt01\n')
            return types.SimpleNamespace(returncode=0)
        with patch.object(h,'pinned',return_value=None),patch.object(h,'command',return_value=types.SimpleNamespace(returncode=0,stdout='{"Jobs":{}}')),patch.object(h.subprocess,'run',side_effect=run):
            for arm in h.ARMS:
                first=h.submit(arm);self.assertEqual(first['arm'],arm)
                self.assertEqual(h.submit(arm),first)
        self.assertEqual(len(commands),5)
        for arm,args in zip(h.ARMS,commands):
            variables=args[args.index('-v')+1]
            self.assertIn('CES_SMOKE_ONLY=0',variables)
            self.assertIn('configs/'+arm+'.json',variables)
            self.assertIn('CES_OUTPUT_DIR='+str(self.root/arm),variables)
    def test_prior_uncertain_attempt_never_resubmits(self):
        h.write(self.root/'CUDA_GATE_PASSED.json',{'pbs_job_id':'9999.mgmt01'})
        h.write(self.root/'runtime'/f'{h.ARMS[0]}.attempt.json',{'uncertain':True})
        with patch.object(h,'pinned',return_value=None):
            with self.assertRaises(RuntimeError):h.submit(h.ARMS[0])
    def test_existing_scheduler_job_rejects_duplicate(self):
        h.write(self.root/'CUDA_GATE_PASSED.json',{'pbs_job_id':'9999.mgmt01'})
        jobs={'Jobs':{'123.mgmt01':{'Variable_List':{'CES_SOURCE_DIR':str(h.SOURCE),'CES_UPGRADE_STAGE':h.ARMS[0]}}}}
        with patch.object(h,'pinned',return_value=None),patch.object(h,'command',return_value=types.SimpleNamespace(returncode=0,stdout=json.dumps(jobs))):
            with self.assertRaises(RuntimeError):h.submit(h.ARMS[0])
if __name__=='__main__':unittest.main(verbosity=2)
