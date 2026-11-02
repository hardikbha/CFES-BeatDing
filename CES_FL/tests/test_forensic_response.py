"""Independent invariants for the new standalone variant; no benchmark data."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import json
import numpy as np
import torch

spec = importlib.util.spec_from_file_location('fr', Path(__file__).resolve().parents[1] / 'src/ces_fl_forensic_response.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class ForensicResponseInvariants(unittest.TestCase):
    def test_equal_group_errors_have_zero_penalty_and_grad(self):
        z = torch.tensor([-1., 1., -1., 1.], requires_grad=True)
        result = m.soft_error_disparity(z, torch.tensor([0,1,0,1]), torch.tensor([0,0,1,1]), torch.tensor([[.1,.9],[.9,.1]]), .3, .7)
        self.assertAlmostEqual(result.item(), 0., places=6)
        result.backward(); self.assertTrue(torch.isfinite(z.grad).all())

    def test_fair_loss_rejects_missing_class_in_auxiliary(self):
        with self.assertRaises(ValueError):
            m.soft_error_disparity(torch.zeros(3),torch.tensor([0,1,0]),torch.tensor([0,0,1]),torch.ones(2,2)/2,0.,1.)

    def test_alignment_uses_distribution_not_only_mean(self):
        # Both group means are 1; the quantile distributions differ.
        r=torch.tensor([0.,2.,1.,1.],requires_grad=True)
        loss=m.response_alignment(r,torch.tensor([0,0,1,1]))
        self.assertAlmostEqual(loss.item(),.5,places=6)
        loss.backward(); self.assertGreater(r.grad.abs().sum().item(),0.)

    def test_alignment_label_permutation_invariant(self):
        r=torch.tensor([-.5,.7,1.3,2.1]); g=torch.tensor([0,0,1,1])
        self.assertAlmostEqual(m.response_alignment(r,g).item(),m.response_alignment(r,1-g).item(),places=6)
        with self.assertRaises(ValueError):m.response_alignment(r[:3],g[:3])

    def test_processing_control_is_not_zero_or_random_placeholder(self):
        x=torch.rand(3,75,75)
        s,t,mask,_=m.intervention(x,'local_warp',1.,177)
        self.assertLess((s-x).abs().mean().item(),.35)
        self.assertTrue(torch.isfinite(t).all()); self.assertGreater((t-s).abs().mean().item(),0.)
        self.assertEqual((t-s)[:,mask[0]==0].abs().sum().item(),0.)
        with self.assertRaises(ValueError):m.intervention(x,'jpeg_is_fake',1.,177)

    def test_calibration_ties_and_exact_budget(self):
        for n in (10,19,20,100):
            z=np.arange(n,dtype=float);r=m.calibrate(z,.1)
            self.assertLessEqual((z>=r['threshold_logit']).sum(),int(n*.1))
        with self.assertRaises(ValueError):m.calibrate([float('nan')])

    def test_contributor_components_are_transitive(self):
        rows=[{'image':'a','identity_ids':'person1','source_ids':'video1'},
              {'image':'b','identity_ids':'person2','source_ids':'video1'},
              {'image':'c','identity_ids':'person2','source_ids':'video2'},
              {'image':'d','identity_ids':'person3','source_ids':'video3'}]
        self.assertEqual(sorted(map(len,m.components(rows))),[1,3])

    def test_sparse_bootstrap_keeps_declared_groups(self):
        rows=[]
        for i in range(8):rows.append({'image':str(i),'label':i%2,'identity_ids':str(i),'source_ids':str(i),'family':'f',**{a:str(i//4) for a in m.AXES}})
        report=m.bootstrap(rows,np.arange(8),3.5,1,20,17,other=(np.arange(8),3.5))
        self.assertEqual(report['components'],8)
        self.assertEqual(report['eligible_groups_fixed_before_resampling']['gender'],['0','1'])
        self.assertEqual(report['intervals']['overall/auc']['estimate'],0.)
        self.assertEqual(report['intervals']['overall/auc']['ci95'],[0.,0.])

    def test_pending_review_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            m.write_json(root/'preview_PRIVATE.json',{'source_sha256':m.sha(m.__file__),'config_digest':'test','synthetic_fixture':False,'cases':[{'id':'test'}]})
            m.write_json(root/'review.json',{'preview_sha256':m.sha(root/'preview_PRIVATE.json'),'cases':[{'id':'test','approved':False}]})
            with self.assertRaises(ValueError):m.qualify(root,root/'review.json',root/'qualification.json')
            self.assertFalse((root/'qualification.json').exists())

    def test_fit_test_overlap_is_explicit(self):
        rows=[{'image':'train','identity_ids':'a','source_ids':'x'}]
        test=[{'image':'test','identity_ids':'b;a','source_ids':'y'}]
        with self.assertRaises(ValueError):m.split_audit(rows,test,evaluation=True)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'exception.json';m.write_json(p,{'declared_overlap_values':{'identity_ids:fit:test':['a'],'source_ids:fit:test':[]}})
            self.assertEqual(m.split_audit(rows,test,p,evaluation=True)['recorded_contributor_overlap']['identity_ids'],1)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
