"""Focused software checks: metrics, batch independence, pairing and dropout resume."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[2]


def load(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


r=load(ROOT/'CES_FL/src/ff2_expanded_runner.py','expanded_test_runner')
metrics=load(ROOT/'CES_FL/scripts/ff2_expanded_metrics.py','expanded_metrics')
UPSTREAM=os.environ.get('FF2_EXPANDED_SOURCE')
torch.set_num_threads(2)


class Tiny(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__();self.drop=nn.Dropout(.5);self.head=nn.Linear(3,1)
    def forward(self,x):return self.head(self.drop(x.mean((2,3)))).squeeze(1)
    def loss(self,x,y,second_view=None):return nn.functional.binary_cross_entropy_with_logits(self(x),y.float())


class Checks(unittest.TestCase):
    def test_metric_formulas_and_eer_reference(self):
        rows=[{'label':y,'gender':g} for g in ('A','B') for y in (0,0,1,1)]
        z=[1,-1,2,3,-2,-1,-.5,2];q=metrics.report(rows,z,0.,2)
        self.assertEqual(q['overall']['auc'],15/16)
        self.assertAlmostEqual(q['overall']['ap'],.95)
        self.assertEqual(q['overall']['eer'],.25)
        g=q['axes']['gender']
        for key,value in [('F_FPR',.5),('F_DP',.5),('F_MEO',.5),('F_EO',1.),('F_OAE',0.)]:self.assertAlmostEqual(g[key],value)

    def test_threshold_changes_decisions_not_auc(self):
        rows=[{'label':y} for y in (0,0,1,1)];z=[-2,1,0,3]
        a=metrics.report(rows,z,0.)['overall'];b=metrics.report(rows,z,2.)['overall']
        self.assertEqual(a['auc'],b['auc']);self.assertNotEqual(a['fpr'],b['fpr'])

    def test_unknown_and_single_class_not_fabricated(self):
        rows=[{'label':0,'gender':'A'},{'label':0,'gender':'A'},{'label':1,'gender':'B'},{'label':1,'gender':'unknown'}]
        q=metrics.report(rows,[-2,-1,1,2],0.,1)
        self.assertEqual(q['axes']['gender']['eligible_groups'],[])
        self.assertIsNone(q['axes']['gender']['F_MEO'])
        self.assertEqual(list(q['axes']['gender_skin']['groups']),['unknown'])

    def test_invalid_predictions_fail(self):
        with self.assertRaises(ValueError):metrics.report([{'label':0}],[float('nan')],0.)

    def test_spatial_fft_does_not_mix_images(self):
        x=torch.randn(3,3,32,32);a=r.zoo.spatial_phase(x)
        b=torch.cat([r.zoo.spatial_phase(t[None]) for t in x])
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        changed=x.clone();changed[2]*=5;changed[2]+=torch.randn_like(changed[2])
        torch.testing.assert_close(a[:2],r.zoo.spatial_phase(changed)[:2],rtol=0,atol=0)

    def test_paired_methods_reject_plain_bce_fallback(self):
        for method in ('ucf','pg_fdd'):
            with self.assertRaisesRegex(ValueError,'plain BCE substitution forbidden'):
                r.zoo.Adapter(method,'unused',fixture=True)

    def test_update_seed_wraps_after_many_steps(self):
        c={'seed':17,'method':'vit_b16','normalization_mean':[.5]*3,'normalization_std':[.5]*3}
        e=r.configure(c,5000,fixture=True);model=Tiny();opt=torch.optim.SGD(model.parameters(),lr=.01)
        batch=(torch.randn(2,3,8,8),torch.tensor([0,1]),torch.tensor([0,1]),torch.tensor([0,1]))
        self.assertTrue(np.isfinite(e.step(model,batch,opt,'vit_b16',torch.device('cpu'))))

    def test_dropout_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);train=[];val=[]
            rng=np.random.default_rng(17)
            for split,rows in [('train',train),('val',val)]:
                for i in range(8):
                    p=root/f'{split}{i}.png';Image.fromarray(rng.integers(0,255,(20,22,3),dtype=np.uint8)).save(p)
                    rows.append({'image':str(p),'label':i%2,'identity_ids':f'{split}{i}','source_ids':f'{split}{i}',
                                 **{axis:('A' if i<4 else 'B') for axis in r.common.AXES},
                                 'family':'fixture','native_group':i%2})
            c={'method':'vit_b16','seed':17,'epochs':2,'batch_size':2,'validation_batch':4,'size':224,'workers':0,
               'lr':.01,'momentum':.9,'weight_decay':.005,'max_gpu_hours':1.,'weights':None,'target_fpr':.1,
               'min_support':1,'val_sha256':'synthetic','normalization_mean':[.5]*3,'normalization_std':[.5]*3}
            with patch.object(r.zoo,'Adapter',Tiny):
                e=r.configure(c,fixture=True);e.train(c,train,val,'unused',root/'full',torch.device('cpu'),fixture=True)
                e=r.configure(c,fixture=True);e.train(c,train,val,'unused',root/'resume',torch.device('cpu'),fixture=True,stop_steps=3)
                ck=torch.load(root/'resume/last.pt',weights_only=True)
                self.assertEqual(ck['steps'],3)
                e=r.configure(c,ck['steps'],fixture=True);e.train(c,train,val,'unused',root/'resume',torch.device('cpu'),fixture=True,resume=True)
                full=torch.load(root/'full/last.pt',weights_only=True);resumed=torch.load(root/'resume/last.pt',weights_only=True)
                for key in full['model']:torch.testing.assert_close(full['model'][key],resumed['model'][key],rtol=0,atol=0)
                self.assertEqual(full['history'],resumed['history'])

    @unittest.skipUnless(UPSTREAM,'Source clone required')
    def test_xception_native_initialization_parity(self):
        native=r.module(ROOT/'CES_FL/src/ff2_fairness_baselines.py','parity_native')
        torch.manual_seed(17);a=native.make_model(UPSTREAM)
        torch.manual_seed(17);b=r.zoo.Adapter('xception',UPSTREAM,fixture=True)
        self.assertEqual(set(a.state_dict()),set(b.net.state_dict()))
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.net.state_dict()[k],rtol=0,atol=0)
        x=torch.randn(2,3,75,75);a.eval();b.eval()
        with torch.no_grad():torch.testing.assert_close(a(x)[0][:,0],b(x),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
