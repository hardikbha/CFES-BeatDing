from pathlib import Path
import importlib.util,unittest
p=Path(__file__).with_name('publish_study_progress_v3.py');s=importlib.util.spec_from_file_location('status_v3',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Gates(unittest.TestCase):
 def test_rejected_review_blocks_CD(self):
  for arm in 'CD':
   state,reason=m.unstarted_state(arm,{'status':'PASS'},False,{'approved':72,'rejected':8,'uncertain':8})
   self.assertEqual(state,'BLOCKED_INPUTS');self.assertIn('rejected/uncertain',reason)
 def test_review_missing_blocks_CD(self):
  self.assertEqual(m.unstarted_state('C',{'status':'PASS'},False,None)[0],'BLOCKED_INPUTS')
 def test_control_does_not_require_operator_review(self):
  for arm in 'AB':self.assertEqual(m.unstarted_state(arm,{'status':'PASS'},False,{'rejected':8}),('PENDING',None))
 def test_no_preflight_blocks_all(self):
  for arm in 'ABCD':self.assertEqual(m.unstarted_state(arm,None,False,None)[0],'BLOCKED_INPUTS')
if __name__=='__main__':unittest.main(verbosity=2)
