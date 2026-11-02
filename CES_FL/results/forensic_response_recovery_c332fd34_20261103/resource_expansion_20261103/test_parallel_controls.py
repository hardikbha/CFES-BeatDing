import importlib.util
from pathlib import Path
import unittest
p=Path(__file__).with_name('parallel_controls_pbs.py')
s=importlib.util.spec_from_file_location('resource_controls',p)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Guards(unittest.TestCase):
 def test_distinct_additional_devices(self):
  cards=[dict(uuid='GPU-A'),dict(uuid='GPU-C'),dict(uuid='GPU-B')]
  self.assertEqual([x['uuid'] for x in m.candidates_for(cards,{'GPU-A','GPU-B','GPU-C'},{'GPU-C'})],['GPU-A','GPU-B'])
 def test_scheduler_visibility_restricts_candidates(self):
  self.assertEqual(m.candidates_for([dict(uuid='GPU-A'),dict(uuid='GPU-B')],{'GPU-B'},set()),[dict(uuid='GPU-B')])
 def test_running_two_gpu_grant(self):
  m.verify_grant('Job_Owner = sachin.chaudhary@mgmt01\njob_state = R\nexec_vnode = (gpu01:ncpus=12:mem=25165824kb:ngpus=2)')
 def test_missing_or_wrong_grant_stops(self):
  for text in ('job_state = Q','Job_Owner = someone@host\njob_state = R\nexec_vnode = (gpu01:ngpus=2)','Job_Owner = sachin.chaudhary@host\njob_state = R\nexec_vnode = (gpu01:ncpus=12)'):
   with self.assertRaises(RuntimeError):m.verify_grant(text)
 def test_import_does_not_launch(self):
  self.assertTrue(callable(m.main));self.assertFalse(hasattr(m,'children'))
if __name__=='__main__':unittest.main(verbosity=2)
