"""Independent regression fixtures for support-aware CES-FL, without benchmark data."""
import importlib.util
from pathlib import Path
import unittest
import torch
spec=importlib.util.spec_from_file_location('support_fr',Path(__file__).resolve().parents[1]/'src/ces_fl_forensic_response.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class SupportWarmup(unittest.TestCase):
 def fixture(self):
  return torch.tensor([-2.,2.,2.,-2.],requires_grad=True),torch.tensor([0,1,0,1]),torch.tensor([0,0,1,1]),torch.ones(2,2)/2
 def test_exact_reference_and_no_increase_to_better_groups(self):
  z,y,g,p=self.fixture();loss=m.support_upper_error(z,y,g,p,torch.ones(2,2),0.,1.)
  self.assertAlmostEqual(loss.item(),(torch.sigmoid(torch.tensor(2.))-.5).item(),places=6)
  loss.backward();self.assertEqual(z.grad[0].item(),0.);self.assertEqual(z.grad[1].item(),0.)
  self.assertGreater(z.grad[2].item(),0.);self.assertLess(z.grad[3].item(),0.)
 def test_equal_errors_zero(self):
  z,y,g,p=self.fixture();z=torch.tensor([-2.,2.,-2.,2.],requires_grad=True)
  loss=m.support_upper_error(z,y,g,p,torch.ones(2,2),0.,1.);self.assertEqual(loss.item(),0.);loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
 def test_recorded_support_scales_loss(self):
  z,y,g,p=self.fixture();a=m.support_upper_error(z,y,g,p,torch.ones(2,2),0.,1.);b=m.support_upper_error(z,y,g,p,torch.ones(2,2)*.5,0.,1.)
  torch.testing.assert_close(a/2,b)
 def test_bad_support_is_rejected(self):
  z,y,g,p=self.fixture()
  for s in [torch.ones(2),torch.zeros(2,2),torch.ones(2,2)*1.1,torch.ones(2,2)*float('nan')]:
   with self.assertRaises(ValueError):m.support_upper_error(z,y,g,p,s,0.,1.)
 def test_equal_mean_logits_unequal_tails_not_zero(self):
  a=[-4.,1.,1.,2.];b=[-2.,-1.,-1.,4.];self.assertEqual(sum(a),sum(b))
  z=torch.tensor(a+[8.]+b+[8.],requires_grad=True);y=torch.tensor([0]*4+[1]+[0]*4+[1]);g=torch.tensor([0]*5+[1]*5)
  self.assertNotEqual(sum(v>=0 for v in a),sum(v>=0 for v in b))
  loss=m.support_upper_error(z,y,g,torch.ones(2,2)/2,torch.ones(2,2),0.,1.);self.assertGreater(loss.item(),0.);loss.backward();self.assertGreater(z.grad.abs().sum().item(),0.)
 def test_response_offset_cancellation_does_not_imply_equal_errors(self):
  s=torch.tensor([-2.,-2.]);t=s+1.;torch.testing.assert_close(m.response(s,t),m.response(s+4,t+4))
  z,y,g,p=self.fixture();z=torch.tensor([-2.,2.,2.,6.],requires_grad=True)
  self.assertGreater(m.support_upper_error(z,y,g,p,torch.ones(2,2),0.,1.).item(),0.)
 def test_lambda_changes_optimizer_update(self):
  out=[]
  for lam in [0.,1.]:
   z,y,g,p=self.fixture();o=torch.optim.SGD([z],lr=.1);loss=m.support_upper_error(z,y,g,p,torch.ones(2,2),0.,1.);o.zero_grad();(z.sum()*0+lam*loss).backward();o.step();out.append(z.detach().clone())
  self.assertFalse(torch.equal(*out));self.assertTrue(torch.isfinite(out[1]).all())
 def test_schedule_uses_absolute_epoch(self):
  self.assertEqual([m.alignment_multiplier(e,3) for e in range(5)],[1/3,2/3,1.,1.,1.]);self.assertEqual(m.alignment_multiplier(0,0),1.)
  for e,w in [(-1,3),(0,-1)]:
   with self.assertRaises(ValueError):m.alignment_multiplier(e,w)
 def test_contributor_tuple_normalization_and_missing_ids(self):
  a=dict(image='a',identity_ids='b;a',source_ids='v');b=dict(image='b',identity_ids='a;b',source_ids='v')
  self.assertEqual(m.recorded_contributor(a),m.recorded_contributor(b));self.assertNotEqual(m.recorded_contributor(dict(image='a')),m.recorded_contributor(dict(image='b')))
if __name__=='__main__':torch.set_num_threads(2);unittest.main(verbosity=2)
