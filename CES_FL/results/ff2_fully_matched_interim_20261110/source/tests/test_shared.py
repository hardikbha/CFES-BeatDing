import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn

SOURCE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('shared_test',SOURCE/'src/shared.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Tiny(nn.Module):
    def __init__(self,*args):
        super().__init__();self.encoder=nn.Conv2d(3,3,1);self.classifier=nn.Linear(3,1);self.register_buffer('feature_mask',torch.ones(3))
    def forward(self,x):
        f=self.encoder(x)*self.feature_mask[None,:,None,None]
        return {'logit':self.classifier(f.mean((2,3))).squeeze(1)}
    def counts(self):return {'total':sum(p.numel() for p in self.parameters()),'trainable':sum(p.numel() for p in self.parameters())}

def tiny_initial(c):m.core.seed_all(c['seed']);return Tiny()

class Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(12)
    def config(self):
        c=m.core.defaults();c.update(seed=17,amp=False,suite_method='bce',group_names=['A-X','A-Y','B-X','B-Y'],
          dag_alpha=.5,daw_inner_alpha=.9,daw_outer_alpha=.5,ding_lambda=.005,sinkhorn_blur=.0005)
        return c
    def test_cvar_values_and_gradients_match_the_pinned_port(self):
        z=torch.randn(16,requires_grad=True);y=torch.tensor([0,1]*8);g=torch.arange(4).repeat_interleave(4)
        c=self.config();zero=z.sum()*0
        for method in ['dag','daw']:
            c['suite_method']=method
            got=m.method_objective(c,z,y,g,zero,zero,zero,zero,zero,False,0)['total']
            expected=m.ported.dag_fdd_loss(z,y,.5)[0] if method=='dag' else m.ported.daw_fdd_loss(z,y,g,.9,.5)[0]
            torch.testing.assert_close(got,expected,rtol=0,atol=0)
            a=torch.autograd.grad(got,z,retain_graph=True)[0];b=torch.autograd.grad(expected,z,retain_graph=True)[0]
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    def test_every_method_has_identical_forward_exposure(self):
        c=self.config();c.update(error_control='hierarchical_real')
        x=torch.randn(8,3,8,8);y=torch.tensor([0,1]*4);g=torch.arange(4).repeat_interleave(2)
        b=dict(x=x,s=x+.02,t=x+.1,y=y,index=torch.arange(8),group=g)
        aux=dict(x=x+.05,y=y,group=g,s=x[y==0]+.03,t=x[y==0]+.2,indices=list(range(8)),groups=c['group_names'],
          priors=torch.ones(4,2)/4,axis='gender_race',hierarchy={'parent_codes':torch.tensor([0,0,1,1]),
          'within_parent_real_priors':torch.tensor([[.5,.5,0,0],[0,0,.5,.5]])})
        signatures=[];gradients=[]
        initial=Tiny()
        for method in m.METHODS:
            c['suite_method']=method;model=copy.deepcopy(initial);calls=[]
            handle=model.register_forward_pre_hook(lambda model,args:calls.append((tuple(args[0].shape),float(args[0].sum()))))
            losses,count=m.train_step(model,b,aux,torch.ones(8),c,'D',.2,torch.device('cpu'))
            self.assertEqual(count,4);self.assertTrue(torch.isfinite(losses['total']))
            losses['total'].backward();self.assertGreater(float(model.classifier.weight.grad.norm()),0)
            signatures.append(calls);gradients.append(model.classifier.weight.grad.clone());handle.remove()
        self.assertTrue(all(x==signatures[0] for x in signatures))
        self.assertTrue(any(not torch.equal(x,gradients[0]) for x in gradients[1:]))
    def test_response_schedules(self):
        for method,expected in [('lh',[1]*10),('lh_ramp',[0,0,1/3,2/3,1,1,1,1,1,1]),('lh_low',[0,0]+[1/3]*8),('bce',[0]*10)]:
            self.assertEqual([m.response_factor({'suite_method':method},e) for e in range(10)],expected)
    def test_symmetric_binary_logits_preserve_bce(self):
        z=torch.randn(8);y=torch.tensor([0,1]*4)
        torch.testing.assert_close(torch.nn.functional.cross_entropy(m.symmetric_logits(z),y),
                                   torch.nn.functional.binary_cross_entropy_with_logits(z,y.float()))
    def test_padding_recipe_is_unmodified(self):
        original=Path('/home/sachin.chaudhary/ces_fl_sources/ff2-lh-ps-v2-20261106/CES_FL/src/ces_fl_head_ablation.py')
        self.assertEqual(m.core.sha(original),m.core.sha(m.core.__file__))
    def test_fast_normalisation_matches_the_historical_reader(self):
        spec=importlib.util.spec_from_file_location('cache_test',SOURCE/'data_cache.py')
        cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache)
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows=[];trusted={}
            for i in range(4):
                p=root/f'{i}.png';Image.new('RGB',(8,8),(i,5,10)).save(p);trusted[str(p)]=m.core.sha(p)
                rows.append(dict(image=str(p),label=i%2,identity_ids='b; a;b',source_ids='x; y;x',
                                 gender=' Male ',race='Asian',gender_race='Male-Asian',age_bin='',skin_bin='0',ids8='0'))
            path=root/'rows.csv'
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            a=m.core.read_manifest(path);b=cache.normalise(path,trusted)
            for x,y in zip(a,b):
                y=dict(y);y.pop('image_sha256');self.assertEqual(x,y)
    def test_test_scoring_requires_a_completed_frozen_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch.object(m,'check',return_value={}),patch.object(m,'configure'):
                with self.assertRaises(FileNotFoundError):m.evaluate({},root,root/'output',torch.device('cpu'))
    def test_sfd_selection_and_mask_do_not_add_parameters(self):
        score=np.arange(2048,dtype=float);channels,threshold,n=m.ported.select_sfd_channels(score,2.)
        self.assertEqual(n,41);self.assertEqual(channels,list(range(41)))
        model=m.SharedDetector();self.assertEqual(model.counts(),{'total':20809001,'trainable':6790433})
        before=m.initial_fingerprint(model);model.feature_mask[channels]=0
        self.assertEqual(before,m.initial_fingerprint(model))
    def test_native_lifecycle_resume_binds_method_and_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows={}
            for split in ['train','val']:
                rr=[]
                for i in range(8):
                    p=root/f'{split}_{i}.png';Image.fromarray(np.random.default_rng(i+(100 if split=='val' else 0)).integers(0,256,(75,75,3),dtype=np.uint8)).save(p)
                    gender='A' if i<4 else 'B';race='X' if i//2%2==0 else 'Y'
                    rr.append({'image':str(p),'label':i%2,'identity_ids':split+str(i),'source_ids':split+str(i),
                      'family':'fixture','gender':gender,'race':race,'gender_race':gender+'-'+race,
                      'age_bin':str(i//4),'skin_bin':str(i//4),'ids8':str(i//2),'image_sha256':m.core.sha(p)})
                rows[split]=rr
            c=self.config();c.update(epochs=3,batch_size=4,workers=0,axes=['gender_race'],size=75,amp=False,
                min_train_support=1,min_eval_support=1,group_per_class=1,aux_every=1,train_calibration_size=4,
                suite_method='lh_ramp',error_control='hierarchical_real',synthetic_fixture=True,max_gpu_hours=1.,source_sha256=m.core.sha(m.core.__file__))
            def inputs(c,path):return rows['train'],rows['val'],{'status':'PASS'}
            m.configure()
            with patch.object(m.core,'Detector',Tiny),patch.object(m.core,'load_initial',side_effect=tiny_initial),\
                 patch.object(m.core,'checked_inputs',side_effect=inputs):
                m.core.train(c,'unused',root/'normal','D',torch.device('cpu'),fixture=True)
                m.core.train(c,'unused',root/'resume','D',torch.device('cpu'),fixture=True,stop_after=2)
                m.core.train(c,'unused',root/'resume','D',torch.device('cpu'),fixture=True,resume=True)
                a=torch.load(root/'normal/last.pt',weights_only=True);b=torch.load(root/'resume/last.pt',weights_only=True)
                self.assertTrue(all(torch.equal(v,b['model'][k]) for k,v in a['model'].items()))
                self.assertEqual(a['validation'],b['validation'])
                self.assertTrue(all(h['originals_once_each'] for h in a['history']))

if __name__=='__main__':
    torch.set_num_threads(2);unittest.main(verbosity=2)
