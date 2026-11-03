"""No GPU or job launches: fixed-scope and cumulative resource guards."""
import contextlib,importlib.util,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).resolve().parents[1]/'cluster/run_hierarchical_real.py';s=importlib.util.spec_from_file_location('ho',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Guards(unittest.TestCase):
 def ledger(self,d):
  @contextlib.contextmanager
  def ctx():yield d
  return ctx
 def test_only_one_candidate(self):
  for name in ['E_support','F_support_ramp3','Ding','FF2','second_candidate']:
   with self.assertRaises(RuntimeError):m.ready(name)
 def test_duplicate_stage_rejected(self):
  d=dict(initial_total_gpu_hours=24,gpu_sessions=[dict(id='hier_smoke',state='COMPLETED',elapsed_seconds=20,ngpus=1,reserved_gpu_hours=.25)])
  with patch.object(m.budget,'ledger',self.ledger(d)):
   with self.assertRaises(RuntimeError):m.reserve('hier_smoke',.25)
 def test_original_24hour_limit(self):
  d=dict(initial_total_gpu_hours=24,gpu_sessions=[dict(id='past',state='COMPLETED',elapsed_seconds=23.9*3600,ngpus=1,reserved_gpu_hours=24)])
  with patch.object(m.budget,'ledger',self.ledger(d)):
   with self.assertRaises(RuntimeError):m.reserve('hier_smoke',.25)
 def test_new_4hour_limit(self):
  d=dict(initial_total_gpu_hours=24,gpu_sessions=[dict(id='hier_H_joint_real',state='RUNNING',elapsed_seconds=0,ngpus=1,reserved_gpu_hours=3.9)])
  with patch.object(m.budget,'ledger',self.ledger(d)):
   with self.assertRaises(RuntimeError):m.reserve('hier_checks',.25)
 def test_no_unregistered_phase(self):
  with self.assertRaises(RuntimeError):m.reserve('another_arm',.01)
 def test_reservation_binds_actual_source(self):
  d=dict(initial_total_gpu_hours=24,gpu_sessions=[])
  with patch.object(m.budget,'ledger',self.ledger(d)):m.reserve('hier_smoke',.25)
  self.assertEqual(d['gpu_sessions'][0]['scientific_commit'],'3f5f3c16839bf4b96209bea3f968255b5f351ebb')
  self.assertEqual(d['gpu_sessions'][0]['scientific_source_sha256'],m.SHA)
  self.assertFalse(d['gpu_sessions'][0]['pbs_allocation_claimed'])
 def test_no_node_git_dependency_and_detached_head_required(self):
  with tempfile.TemporaryDirectory() as tmp:
   r=Path(tmp);(r/'.git').mkdir();(r/'.git/HEAD').write_text(m.COMMIT+'\n');self.assertEqual(m.detached_head(r),m.COMMIT)
   (r/'.git/HEAD').write_text('ref: refs/heads/main\n')
   with self.assertRaises(RuntimeError):m.detached_head(r)
if __name__=='__main__':unittest.main(verbosity=2)
