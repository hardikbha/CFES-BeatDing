import copy
import csv
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

ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('alignment_helper',ROOT/'scripts/lh_alignment_followup.py')
f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f);m=f.m
torch.set_num_threads(2)


class TinyDetector(nn.Module):
    def __init__(self,hidden=128,gate_hidden=64,policy='exit',head_type='linear'):
        super().__init__();self.encoder=nn.Sequential(nn.Conv2d(3,4,1),nn.BatchNorm2d(4),nn.ReLU())
        self.classifier=nn.Linear(4,1);self.head_type=head_type;self.policy=policy;self.train()
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    def forward(self,x):
        features=self.encoder(x);z=self.classifier(features.mean((2,3))).squeeze(1)
        spatial=(features*self.classifier.weight[0][None,:,None,None]).sum(1)
        return {'logit':z,'local_map':spatial,'global_contribution':torch.zeros_like(z),'bias':self.classifier.bias[0]}
    def counts(self):return {'total':sum(p.numel() for p in self.parameters()),'trainable':sum(p.numel() for p in self.parameters() if p.requires_grad)}
    def parameter_report(self):return {**self.counts(),'head_type':'linear','software_fixture':True}


def toy_batch(size=12):
    torch.manual_seed(17);x=torch.rand(4,3,size,size)
    b={'x':x,'s':x*.97,'t':x+torch.tensor([.15,.1,.35,.2])[:,None,None,None],
       'y':torch.tensor([0.,1.,0.,1.]),'index':torch.arange(4)}
    a={'x':x+.02,'s':x[[0,2]]*.98,'t':x[[0,2]]+torch.tensor([.25,.5])[:,None,None,None],
       'y':b['y'],'group':torch.tensor([0,0,1,1]),'priors':torch.ones(2,2)/2,
       'axis':'gender_race','hierarchy':{'parent_codes':torch.tensor([0,1]),'within_parent_real_priors':torch.eye(2)}}
    c={**m.defaults(),'amp':False,'error_control':'hierarchical_real','nuisance_mode':'matched'}
    return b,a,c


class AlignmentTests(unittest.TestCase):
    def test_protocol_and_C1_are_pinned_to_H1(self):
        p=f.protocol()
        self.assertEqual(m.sha(f.PROPOSAL/'C1.json'),p['config_sha256']['H1'])
        self.assertEqual(p['C1']['native_arm'],'C')
        self.assertEqual(len(p['diagnostic']['events']),20)

    def test_C_removes_only_alignment_value_and_gradient(self):
        b,a,c=toy_batch();model=TinyDetector();weights=torch.ones(4)
        C,_=m.train_step(model,b,a,weights,c,'C',0.,torch.device('cpu'))
        D,_=m.train_step(model,b,a,weights,c,'D',0.,torch.device('cpu'))
        for name in C:
            if name not in ('total','response_alignment'):torch.testing.assert_close(C[name],D[name],rtol=0,atol=0)
        self.assertEqual(float(C['response_alignment']),0.)
        self.assertGreater(float(D['response_alignment']),0.)
        expected=c['aux_every']*c['lambda_alignment']*D['response_alignment']
        torch.testing.assert_close(D['total']-C['total'],expected,rtol=1e-5,atol=1e-6)
        params=list(model.parameters())
        actual=torch.autograd.grad(D['total']-C['total'],params,retain_graph=True)
        reference=torch.autograd.grad(expected,params)
        for x,y in zip(actual,reference):torch.testing.assert_close(x,y,rtol=1e-4,atol=1e-6)

    def test_fixed_MAIN_plan_uniform_determinism_without_replacement(self):
        rows=[]
        for g in range(8):
            for y in (0,1):
                for j in range(5):rows.append({'image':f'train/{g}/{y}/{j}','gender_race':str(g),'label':y,
                                             'identity_ids':f'{g}_{y}_{j}','source_ids':f'{g}_{y}_{j}'})
        first=f.main_indices(rows,0);second=f.main_indices(rows,1)
        self.assertEqual(first,f.main_indices(rows,0));self.assertNotEqual(first,second)
        self.assertEqual(len(set(first)),32)
        self.assertFalse(set(first)&set(second))
        renamed=[{**r,'gender_race':'changed','label':1-r['label']} for r in rows]
        self.assertEqual(first,f.main_indices(renamed,0))
        with self.assertRaises(ValueError):f.main_indices(rows,2)

    def test_transform_is_common_and_identity_does_not_change_inputs(self):
        torch.manual_seed(9);raw=torch.rand(2,3,16,16);x=torch.stack([m.normalize(a) for a in raw])
        b={'x':x,'s':x.clone(),'t':x.clone(),'y':torch.tensor([0,1])}
        self.assertIs(f.transform_batch(b,'identity'),b)
        out=f.transform_batch(b,'resize90')
        torch.testing.assert_close(out['x'],out['s'],rtol=0,atol=0)
        torch.testing.assert_close(out['s'],out['t'],rtol=0,atol=0)
        torch.testing.assert_close(b['x'],x,rtol=0,atol=0)
        with self.assertRaises(ValueError):f.transform_batch(b,'unplanned')

    def test_engineering_trigger_needs_support_and_strength(self):
        t=f.protocol()['diagnostic']['trigger']
        def records(n=20,negative=12,ratio=.3):
            return [{'model':'H1','condition':condition,'gradients':{'alignment':{'all':{
                'cosine_to_BCE':-.5 if j<negative else .5,'norm_ratio_to_BCE':ratio}}}}
                for condition in ('identity','resize90') for j in range(n)]
        self.assertTrue(f.assess(records(),t)['run_C1'])
        self.assertFalse(f.assess(records(n=15),t)['run_C1'])
        self.assertFalse(f.assess(records(negative=11),t)['run_C1'])
        self.assertFalse(f.assess(records(ratio=.1),t)['run_C1'])
        r=records();r[0]['gradients']['alignment']['all']['cosine_to_BCE']=float('nan')
        self.assertEqual(f.assess(r,t)['conditions']['identity']['valid_batches'],19)

    def test_empty_duplicate_or_unresolved_review_inventory_rejected(self):
        with self.assertRaises(ValueError):f.verify_case_inventory({'cases':[]},{'cases':[]},{})
        expected={'a':{'image':'source','settings':{}},'b':{'image':'source2','settings':{}}}
        with self.assertRaises(ValueError):f.verify_case_inventory({'cases':[{'id':'a'},{'id':'a'}]},{'cases':[]},expected)
        manifest={'cases':[{'id':k,**v} for k,v in expected.items()]}
        review={'cases':[{'id':k,'approved':False} for k in expected]}
        with self.assertRaises(ValueError):f.verify_case_inventory(manifest,review,expected)

    def test_profile_zero_updates_and_all_component_norms(self):
        b,a,c=toy_batch();model=TinyDetector()
        before={k:v.clone() for k,v in model.state_dict().items()}
        result=f.profile_batch(model,b,a,torch.ones(4),c,0.,torch.device('cpu'))
        self.assertEqual(result['optimizer_updates'],0)
        self.assertEqual(set(result['gradients']),{'BCE','fairness','sensitivity','sham_control','alignment'})
        self.assertGreater(result['gradients']['BCE']['all']['gradient_norm'],0)
        for k,v in before.items():self.assertTrue(torch.equal(v,model.state_dict()[k]))
        b['y']=torch.ones_like(b['y'])
        result=f.profile_batch(model,b,a,torch.ones(4),c,0.,torch.device('cpu'))
        self.assertEqual(result['optimizer_updates'],0)
        for k,v in before.items():self.assertTrue(torch.equal(v,model.state_dict()[k]))

    def test_full_width_224_native_C_backward_and_frozen_BN(self):
        b,a,c=toy_batch(224);model=m.Detector();model.train()
        before={k:v.clone() for k,v in model.named_buffers()}
        loss,_=m.train_step(model,b,a,torch.ones(4),c,'C',0.,torch.device('cpu'))
        self.assertEqual(float(loss['response_alignment']),0.)
        loss['total'].backward()
        self.assertEqual(model.counts()['trainable'],14966785)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))
        self.assertTrue(all(p.grad is None for p in model.parameters() if not p.requires_grad))
        self.assertTrue(all(torch.equal(v,dict(model.named_buffers())[k]) for k,v in before.items()))

    def test_native_C_fixture_lifecycle_resume_and_coverage(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(m,'Detector',TinyDetector):
            root=Path(directory);paths={}
            for split in ('train','val'):
                rows=[]
                for i in range(16):
                    g=i//2;gender='Female' if g<4 else 'Male';race=['Asian','Black','Others','White'][g%4]
                    image=root/f'{split}_{i}.png'
                    Image.fromarray(np.random.default_rng(i+(0 if split=='train' else 100)).integers(0,256,(75,75,3),dtype=np.uint8)).save(image)
                    rows.append({'image':str(image),'label':i%2,'identity_ids':f'{split}_{i}','source_ids':f'{split}_{i}',
                                 'family':'fixture','gender':gender,'race':race,'gender_race':gender+'-'+race,
                                 'age_bin':str(g%2),'skin_bin':str(g%2),'ids8':str(g)})
                paths[split]=root/(split+'.csv')
                with paths[split].open('w',newline='') as out:
                    w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            c={**m.defaults(),'dataset_id':'SOFTWARE_FIXTURE','train':str(paths['train']),'val':str(paths['val']),
               'train_sha256':m.sha(paths['train']),'val_sha256':m.sha(paths['val']),'epochs':2,'size':75,'batch_size':4,
               'workers':0,'min_train_support':1,'min_eval_support':1,'group_per_class':1,'aux_every':1,'amp':False,
               'synthetic_fixture':True,'source_sha256':m.sha(m.__file__),'max_gpu_hours':1.,
               'nuisance_mode':'matched','error_control':'hierarchical_real','train_calibration_size':4}
            initial=TinyDetector();checkpoint=root/'initial.pt'
            torch.save({'format':'CESFL_RESNET50_P50_v1','stage':'pretrain','synthetic_fixture':True,
                        'config':{'train':c['train'],'val':c['val'],'backbone':'resnet50'},'model':initial.state_dict()},checkpoint)
            c.update(init_checkpoint=str(checkpoint),init_sha256=m.sha(checkpoint));m.check_config(c,fixture=True)
            m.preflight(c,root/'preflight')
            m.train(c,root/'preflight',root/'continuous','C',torch.device('cpu'),fixture=True)
            m.train(c,root/'preflight',root/'resumed','C',torch.device('cpu'),fixture=True,stop_after=1)
            m.train(c,root/'preflight',root/'resumed','C',torch.device('cpu'),fixture=True,resume=True)
            a=torch.load(root/'continuous/last.pt',weights_only=True);b=torch.load(root/'resumed/last.pt',weights_only=True)
            self.assertTrue(all(torch.equal(v,b['model'][k]) for k,v in a['model'].items()))
            self.assertEqual(a['calibration'],b['calibration'])
            for r in a['history']:
                self.assertTrue(r['originals_once_each']);self.assertEqual(r['unique_originals'],16)
                self.assertEqual(r['original_real_pairs'],8);self.assertEqual(r['losses_step_mean']['response_alignment'],0)
                self.assertEqual(sum(r['main_nuisance_counts'].values()),16)
            self.assertTrue((root/'continuous/DONE').exists())


if __name__=='__main__':unittest.main()
