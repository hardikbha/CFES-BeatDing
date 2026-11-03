"""No benchmark data, checkpoints, GPU allocation or automatic downloads."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn
from scipy import optimize

SRC=Path(__file__).resolve().parents[1]/'src/ff2_fairness_baselines.py'
s=importlib.util.spec_from_file_location('ff2_baselines',SRC);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
UPSTREAM=os.environ.get('FF2_BASELINE_UPSTREAM')
torch.set_num_threads(2)


class LossTests(unittest.TestCase):
    def test_bce_and_cvar_gradients(self):
        z=torch.tensor([-.8,.4,1.2,-2.,.6,.9],requires_grad=True);y=torch.tensor([0,1,0,1,0,1]);g=torch.tensor([0,0,1,1,2,2])
        per=torch.nn.functional.binary_cross_entropy_with_logits(z,y.float(),reduction='none')
        for method in m.METHODS:
            value=m.objective(z,y,g,method)
            grad=torch.autograd.grad(value,z,retain_graph=True)[0]
            self.assertTrue(torch.isfinite(grad).all());self.assertGreater(grad.abs().sum().item(),0)
            if method=='dag':self.assertAlmostEqual(value.item(),per.topk(3).values.mean().item(),places=5)
            if method=='bce':torch.testing.assert_close(value,per.mean())
        with self.assertRaises(ValueError):m.objective(z,y,g,'typo')

    def test_daw_group_code_permutation(self):
        z=torch.tensor([-.8,.4,1.2,-2.,.6,.9]);y=torch.tensor([0,1,0,1,0,1]);g=torch.tensor([0,0,1,1,2,2])
        torch.testing.assert_close(m.objective(z,y,g,'daw'),m.objective(z,y,7-g,'daw'),rtol=1e-5,atol=1e-6)

    def test_config_rejects_source_mismatch_before_io(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'c.json';p.write_text(json.dumps({'format':m.VERSION,'method':'bce','source_hashes':{}}))
            with self.assertRaisesRegex(ValueError,'Config/source'):m.config(p)

    def test_image_hash_change_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.png';Image.new('RGB',(80,75)).save(p)
            rows=[{'image':str(p),'image_sha256':'bad','label':0}]
            with self.assertRaisesRegex(ValueError,'Image changed'):m.Images(rows)[0]


@unittest.skipUnless(UPSTREAM,'Set FF2_BASELINE_UPSTREAM to verified source-only checkout')
class UpstreamTests(unittest.TestCase):
    def test_official_objective_values_and_gradients(self):
        for method,name in [('dag','DagFddDetector'),('daw','DawFddDetector')]:
            source=Path(UPSTREAM)/f'training/detectors/{method}_fdd.py'
            nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name==name]
            for n in nodes:n.decorator_list=[]
            env={'torch':torch,'nn':nn,'np':np,'optimize':optimize,'AbstractDetector':nn.Module}
            exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source),'exec'),env)
            native=env[name].__new__(env[name]);nn.Module.__init__(native)
            native.loss_func=lambda pred,target:torch.nn.functional.binary_cross_entropy_with_logits(pred.squeeze(1),target.float(),reduction='none')
            a=torch.tensor([-.8,.4,1.2,-2.,.6,.9],requires_grad=True);b=a.detach().clone().requires_grad_()
            y=torch.tensor([0,1,0,1,0,1]);g=torch.tensor([0,0,1,1,2,2])
            ref=native.get_losses({'label':y,'intersec_label':g},{'cls':a[:,None]})['overall'];ours=m.objective(b,y,g,method)
            torch.testing.assert_close(ref,ours,rtol=0,atol=0)
            torch.testing.assert_close(torch.autograd.grad(ref,a)[0],torch.autograd.grad(ours,b)[0],rtol=0,atol=0)

    def test_full_xception_and_bn_and_checkpoint_loading(self):
        m.setup(17);model=m.make_model(UPSTREAM)
        self.assertEqual(sum(p.numel() for p in model.parameters()),20809001)
        self.assertIsInstance(model.adjust_channel,nn.Identity)
        before=model.bn1.running_mean.clone()
        x=torch.randn(2,3,224,224);z,feature=model(x)
        self.assertEqual(z.shape,(2,1));self.assertFalse(torch.equal(before,model.bn1.running_mean))
        # Classifier uses official final ReLU, then GAP and linear, unlike CES's gated-head ablation.
        ref=model.last_linear(feature.relu().mean((2,3)))
        torch.testing.assert_close(z,ref)
        z.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'synthetic_imagenet.pt'
            state={k:v for k,v in model.state_dict().items() if not k.startswith('last_linear.')}
            torch.save(state,path);check=m.make_model(UPSTREAM,path)
            self.assertTrue(torch.equal(check.conv1.weight,model.conv1.weight))

    def test_changed_upstream_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'training/networks';p.mkdir(parents=True)
            (p/'xception.py').write_text('# wrong revision')
            with self.assertRaisesRegex(ValueError,'Unpinned'):m.make_model(d)


def integration(upstream,out,device):
    """Actual tiny native-model lifecycle, including bitwise mid-epoch resume."""
    out=m.common.fresh_dir(out);m.setup(17);rng=np.random.default_rng(17);sets=[]
    for split in ('train','val'):
        rows=[]
        for i in range(16):
            p=out/f'{split}_{i}.png';Image.fromarray(rng.integers(0,256,(75,75,3),dtype=np.uint8)).save(p)
            group=i//2;row=dict(image=str(p),label=i%2,native_group=group,identity_ids=f'{split}_{i}',source_ids=f'{split}_{i}',family='fixture')
            row.update({a:str(group) for a in m.common.AXES});row['image_sha256']=m.common.sha(p);rows.append(row)
        sets.append(rows)
    c=dict(format=m.VERSION,method='daw',seed=17,epochs=2,batch_size=8,validation_batch=8,workers=0,size=75,
           lr=.001,momentum=.9,weight_decay=.005,target_fpr=.1,min_support=1,max_gpu_hours=1.,val_sha256='software_fixture')
    reference=m.train(c,*sets,upstream,out/'uninterrupted',device,fixture=True)
    stopped=m.train(c,*sets,upstream,out/'resumed',device,fixture=True,stop_steps=1)
    assert stopped['status']=='INCOMPLETE_RESUMABLE'
    resumed=m.train(c,*sets,upstream,out/'resumed',device,resume=True,fixture=True)
    a=torch.load(out/'uninterrupted/last.pt',map_location='cpu',weights_only=True)
    b=torch.load(out/'resumed/last.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
    assert reference['validation']==resumed['validation'] and reference['calibration']==resumed['calibration']
    assert all(r['originals_once_each'] and r['original_count']==16 for r in a['history'])
    result={'status':'PASS','source_hashes':m.hashes(),'device':str(device),'fixture_only':True,'torch':str(torch.__version__),
            'tests':['native DAW two-epoch training','mid-epoch stop/resume bitwise identical','independent selected replay','original exposure','frozen selected calibration'],
            'scientific_results':False,'test_read':False}
    m.common.write_json(out/'SMOKE_RESULT.json',result);print(json.dumps(result))


if __name__=='__main__':
    import sys,argparse
    if len(sys.argv)>1 and sys.argv[1]=='integration':
        p=argparse.ArgumentParser();p.add_argument('integration');p.add_argument('--upstream',required=True);p.add_argument('--out',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
        a=p.parse_args();integration(a.upstream,a.out,torch.device(a.device))
    else:unittest.main()
