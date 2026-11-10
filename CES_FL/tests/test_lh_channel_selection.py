"""CPU software tests; synthetic data do not establish fairness or forensic validity."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

SRC=Path(__file__).resolve().parents[1]/'src/ces_fl_lh_channel_selection.py'
spec=importlib.util.spec_from_file_location('lhmask',SRC);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(2)

class ChannelSelectionTests(unittest.TestCase):
    def test_all_one_mask_matches_plain_LH(self):
        torch.manual_seed(17)
        a=m.Detector(policy='exit'); b=m._native_detector(policy='exit')
        b.load_state_dict({k:v for k,v in a.state_dict().items() if k!='feature_mask'})
        x=torch.randn(2,3,75,75)
        aa,bb=a(x),b(x)
        torch.testing.assert_close(aa['logit'],bb['logit'],rtol=0,atol=0)
        torch.testing.assert_close(aa['local_map'],bb['local_map'],rtol=0,atol=0)

    def test_same_count_and_utility_matched_random(self):
        v=np.arange(2048,dtype=float)
        a,eligible,s=m.select_masks(v,v,v)
        self.assertEqual(len(a['informed']),41);self.assertEqual(len(a['random']),41)
        self.assertTrue(set(a['informed'])<=set(eligible));self.assertTrue(set(a['random'])<=set(eligible))
        self.assertEqual(a['informed'],list(range(983,1024)))

    def test_masks_ignore_global_response_magnitude(self):
        x=np.zeros((16,4));s=x.copy();g=np.repeat([0,1],8);q=np.zeros(16)
        n,spread,mag=m.response_diagnostics(x,s,np.ones_like(x)*100,g,q,np.ones(4))
        self.assertTrue((spread==0).all());self.assertTrue((mag>1).all())

    def test_channel_utility_zero_weight(self):
        h=np.ones((4,3));u=m.channel_utility(h,[0,1,0,1],['a','a','b','b'],np.zeros(3),0)
        np.testing.assert_array_equal(u,np.zeros(3))

    def test_missing_utility_cell_rejected(self):
        with self.assertRaises(ValueError):m.channel_utility(np.ones((2,3)),[0,1],['a','b'],np.ones(3),0)

    def test_missing_response_cell_rejected(self):
        h=np.ones((4,3))
        with self.assertRaises(ValueError):m.response_diagnostics(h,h,h,['a','a','b','b'],['x','x','y','y'],np.ones(3))

    def test_tied_ranks_and_nonfinite(self):
        np.testing.assert_array_equal(m.rank_mid([1,1,2,2]),[1/6,1/6,5/6,5/6])
        with self.assertRaises(ValueError):m.select_masks([np.nan]*8,[0]*8,[0]*8,2)

    def test_response_is_difference_then_asinh(self):
        s=torch.tensor([5.,-3.]);t=torch.tensor([6.,-1.])
        torch.testing.assert_close(m.response(s,t),torch.asinh(t-s))
        self.assertFalse(torch.allclose(m.response(s,t),torch.asinh(t)-torch.asinh(s)))

    def test_response_alignment_equal_quantiles_zero(self):
        r=torch.tensor([.3,.8,.8,.3],requires_grad=True)
        loss=m.response_alignment(r,torch.tensor([0,0,1,1]))
        self.assertEqual(float(loss.detach()),0.)

    def test_threshold_ties_budget(self):
        self.assertEqual(m.calibrate([.7]*4,.25)['achieved_fp'],0)
        self.assertEqual(m.calibrate([.1,.1,.2,.9],.25)['achieved_fp'],1)

    def test_mask_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'mask.json';c=m.defaults();c['synthetic_fixture']=True
            c['feature_selection'].update(mode='informed',mask_file=str(p))
            r=dict(format='TRAIN_CHANNEL_MASK_v1',source_sha256=m.sha(SRC),context_digest=m.digest(m.context_config(c)),synthetic_fixture=True,validation_scores_used=False,removed={'informed':list(range(41))})
            m.write_json(p,r);c['feature_selection']['mask_sha256']=m.sha(p)
            self.assertEqual(int((m.checked_mask(c)==0).sum()),41)
            r['validation_scores_used']=True;m.write_json(p,r);c['feature_selection']['mask_sha256']=m.sha(p)
            with self.assertRaises(ValueError):m.checked_mask(c)

    def test_ai_approval_rejected_without_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'review.json';out=Path(td)/'QUALIFIED.json'
            m.write_json(p,dict(cases=[dict(approved=True,reviewer='AI',reviewer_type='ai')]))
            with self.assertRaises(ValueError):m.qualify(td,p,out)
            self.assertFalse(out.exists())

    def test_no_CUDA_allocation_receipt_is_rejected(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(ValueError):m.gpu_binding(torch.device('cuda'))

    def test_fixed_mask_is_buffer_not_parameter(self):
        model=m.Detector(policy='frozen')
        self.assertNotIn('feature_mask',dict(model.named_parameters()))
        self.assertEqual(model.counts()['trainable'],2049)

if __name__=='__main__':unittest.main(verbosity=2)
