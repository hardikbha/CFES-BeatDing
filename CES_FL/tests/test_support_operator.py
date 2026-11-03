import contextlib,importlib.util,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
p=Path(__file__).resolve().parents[1]/'cluster/run_support_warmup.py';s=importlib.util.spec_from_file_location('cesfl_operator',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class OperatorGuards(unittest.TestCase):
 def ledger(self,data):
  @contextlib.contextmanager
  def ctx():yield data
  return ctx
 def test_detached_head_without_git_executable(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);checkout=root/'source';metadata=root/'metadata';checkout.mkdir();metadata.mkdir();(checkout/'.git').write_text('gitdir: '+str(metadata));(metadata/'HEAD').write_text(m.COMMIT+'\n')
   self.assertEqual(m.detached_head(checkout),m.COMMIT)
   (metadata/'HEAD').write_text('ref: refs/heads/main\n')
   with self.assertRaises(RuntimeError):m.detached_head(checkout)
 def test_budget_overcommit_rejected(self):
  d=dict(initial_total_gpu_hours=24,gpu_sessions=[dict(id='old',reserved_gpu_hours=0,state='COMPLETED',elapsed_seconds=23.8*3600,ngpus=1)])
  with patch.object(m.budget,'ledger',self.ledger(d)):
   with self.assertRaises(RuntimeError):m.reserve('new',.5)
  self.assertEqual(len(d['gpu_sessions']),1)
 def test_duplicate_rejected(self):
  d=dict(initial_total_gpu_hours=24,gpu_sessions=[dict(id='same',reserved_gpu_hours=1,state='QUEUED',elapsed_seconds=0,ngpus=1)])
  with patch.object(m.budget,'ledger',self.ledger(d)):
   with self.assertRaises(RuntimeError):m.reserve('same',1)
 def test_new_scope_is_CESFL_FFPP_candidates_only(self):
  for name in ['Ding','SFD','GDA','ff2','third_candidate']:
   with self.assertRaises(RuntimeError):m.ready(name)
 def test_accepted_reservation_binds_new_source(self):
  d=dict(initial_total_gpu_hours=24,gpu_sessions=[])
  with patch.object(m.budget,'ledger',self.ledger(d)):m.reserve('support_test',.5)
  self.assertEqual(d['gpu_sessions'][0]['scientific_source_sha256'],m.SHA);self.assertFalse(d['gpu_sessions'][0]['pbs_allocation_claimed'])
if __name__=='__main__':unittest.main(verbosity=2)
