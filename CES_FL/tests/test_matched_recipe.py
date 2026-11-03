"""No benchmark images, pretrained downloads, TEST inference or GPU jobs."""
import ast
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

SRC=Path(__file__).resolve().parents[1]/'src'
spec=importlib.util.spec_from_file_location('matched_recipe',SRC/'ces_fl_matched_recipe.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(2)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Conv2d(3,3,1);self.classifier=nn.Linear(3,1)
    def forward(self,x):return {'logit':self.classifier(self.encoder(x).mean((2,3))).squeeze(1)}
    def parameter_report(self):return {'synthetic_tiny_only':True,'total':sum(p.numel() for p in self.parameters())}


def tiny_initial(c):
    m.core.seed_all(c['base']['seed']);model=Tiny()
    if c['stage']!='P':
        ck=torch.load(c['base']['init_checkpoint'],weights_only=True)
        m.check_parent(ck,c);model.load_state_dict(ck['model'],strict=True)
    return model


def fixture(root):
    rng=np.random.default_rng(17); manifests={}
    for split in ('train','val'):
        rows=[]
        for i in range(16):
            gender='male' if i//4%2==0 else 'female';race='r0' if i//8==0 else 'r1'
            path=root/(split+str(i)+'.png')
            Image.fromarray(rng.integers(0,256,(77,93,3),dtype=np.uint8)).save(path)
            rows.append(dict(image=str(path),label=i%2,identity_ids=split+str(i),source_ids=split+str(i),
                gender=gender,race=race,gender_race=gender+'-'+race,ids8=gender+'-'+race,
                age_bin='a'+str(i//4%2),skin_bin='s'+str(i//4%2)))
        path=root/(split+'.csv')
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        manifests[split]=str(path)
    init=root/'random_fixture.pt';torch.save({},init)
    b=m.core.defaults();b.update(manifests,train_sha256=m.core.sha(manifests['train']),val_sha256=m.core.sha(manifests['val']),
        init_checkpoint=str(init),init_sha256=m.core.sha(init),source_sha256=m.CORE_SHA,synthetic_fixture=True,
        encoder_policy='full',encoder_lr=.001,head_lr=.001,weight_decay=.005,amp=False,epochs=2,
        batch_size=4,workers=0,size=75,hidden=8,gate_hidden=4,min_train_support=1,min_eval_support=1,
        error_control='hierarchical_real',train_calibration_size=8)
    return dict(format=m.VERSION,source_sha256=m.core.sha(m.__file__),core_sha256=m.CORE_SHA,
                stage='P',protocol=copy.deepcopy(m.PROTOCOL),base=b)


class MatchedRecipeTests(unittest.TestCase):
    def test_pair_units_partial_batch_and_reproducibility(self):
        rows=[{'label':0}]*3+[{'label':1}]*5
        one=list(m.PairedBatches(rows,4,17,0));two=list(m.PairedBatches(rows,4,17,0))
        self.assertEqual(one,two);self.assertEqual([len(b) for b in one],[4,4,2])
        fake=[i for b in one for i in b if rows[i]['label']==1]
        self.assertEqual(sorted(fake),list(range(3,8)))
        self.assertEqual(sum(rows[i]['label']==0 for b in one for i in b),5)
        for b in one:self.assertEqual(sum(rows[i]['label'] for i in b),len(b)//2)
        self.assertEqual(len(m.PairedBatches([{'label':0}]*12926+[{'label':1}]*63213,128,17,0)),988)
        self.assertEqual(len(m.PairedBatches([{'label':0}]*12926+[{'label':1}]*63213,64,17,0)),1976)

    def test_val_geometry_matches_torchvision(self):
        from torchvision import transforms
        with tempfile.TemporaryDirectory() as d:
            for w,h in [(70,99),(100,71),(80,80)]:
                path=Path(d)/'image.png';Image.fromarray(np.random.default_rng(17).integers(0,256,(h,w,3),dtype=np.uint8)).save(path)
                with Image.open(path) as image:
                    ref=transforms.Compose([transforms.Resize(75),transforms.CenterCrop(75),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3)])(image)
                torch.testing.assert_close(m.val_pixels({'image':str(path)},75),ref,rtol=0,atol=0)

    def test_config_rejects_recipe_changes(self):
        with tempfile.TemporaryDirectory() as d:
            c=fixture(Path(d));m.check(c,fixture=True)
            for k,v in [('batch_size',3),('encoder_lr',1e-5),('amp',True),('lambda_fair',.2)]:
                bad=copy.deepcopy(c);bad['base'][k]=v
                with self.subTest(k=k),self.assertRaises(ValueError):m.check(bad,fixture=True)
            bad=copy.deepcopy(c);bad['protocol']['selection']='best_auc'
            with self.assertRaises(ValueError):m.check(bad,fixture=True)
            with self.assertRaises(ValueError):m.check(c,fixture=False)

    def test_actual_xception_gradients_bn_and_parameter_counts(self):
        # Production width; synthetic75px input. CUDA/224px production batch still pending.
        model=m.core.Detector(policy='exit',head_type='linear')
        self.assertEqual(model.counts(),{'total':20809001,'trainable':6790433})
        before={k:v.clone() for k,v in model.encoder.state_dict().items() if 'running_' in k}
        m.set_training(model,False);x=torch.randn(2,3,75,75)
        loss={'total':model(x)['logit'].square().mean()}
        opt=m.optimizer(model,dict(head_lr=.001,weight_decay=.005));m.step(loss,model,opt)
        self.assertIsInstance(opt,torch.optim.SGD)
        self.assertTrue(all(torch.equal(v,model.encoder.state_dict()[k]) for k,v in before.items()))
        self.assertGreater(model.classifier.weight.grad.abs().sum().item(),0)
        self.assertTrue(all(p.grad is None for k,p in model.encoder.named_parameters() if not p.requires_grad))
        full=m.core.Detector(policy='full',head_type='linear');m.set_training(full,True)
        self.assertEqual(full.counts()['trainable'],20809001)
        self.assertTrue(full.encoder.bn1.training)
        n=full.encoder.bn1.num_batches_tracked.item();full(x)
        self.assertEqual(full.encoder.bn1.num_batches_tracked.item(),n+1)

    def test_core_science_is_called_not_reimplemented(self):
        tree=ast.parse(Path(m.__file__).read_text())
        functions={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
        self.assertFalse(functions & {'train_step','response','response_alignment','hierarchical_real_error','intervention'})
        self.assertEqual(m.core.sha(m.CORE_FILE),m.CORE_SHA)

    def test_end_to_end_final_selection_transfer_resume_and_exposure(self):
        with tempfile.TemporaryDirectory() as d,patch.object(m,'initial',side_effect=tiny_initial):
            root=Path(d);c=fixture(root);m.check(c,fixture=True);m.preflight(c,root/'pre_P')
            result=m.train(c,root/'pre_P',root/'P',torch.device('cpu'),1,fixture=True)
            self.assertEqual(result['selected_epoch'],2)
            pck=torch.load(root/'P/last.pt',weights_only=True)
            self.assertEqual(pck['epoch'],2)
            h=copy.deepcopy(c);h['stage']='H';h['base'].update(encoder_policy='exit',
                init_checkpoint=str(root/'P/last.pt'),init_sha256=m.core.sha(root/'P/last.pt'))
            m.preflight(h,root/'pre_H');m.train(h,root/'pre_H',root/'H',torch.device('cpu'),1,fixture=True)
            m.train(h,root/'pre_H',root/'resume',torch.device('cpu'),1,fixture=True,stop_after=1)
            m.train(h,root/'pre_H',root/'resume',torch.device('cpu'),1,fixture=True,resume=True)
            normal=torch.load(root/'H/last.pt',weights_only=True);resumed=torch.load(root/'resume/last.pt',weights_only=True)
            self.assertTrue(all(torch.equal(v,resumed['model'][k]) for k,v in normal['model'].items()))
            self.assertEqual(normal['calibration'],resumed['calibration'])
            for e in normal['history']:
                self.assertEqual(e['steps'],4);self.assertEqual(e['main_images'],16);self.assertEqual(e['main_real_pair_draws'],8)
            a=copy.deepcopy(h);a['stage']='A';m.preflight(a,root/'pre_A')
            self.assertTrue(all(torch.equal(v,tiny_initial(h).state_dict()[k]) for k,v in tiny_initial(a).state_dict().items()))
            m.train(a,root/'pre_A',root/'A',torch.device('cpu'),1,fixture=True)
            ac=torch.load(root/'A/last.pt',weights_only=True)
            self.assertTrue(all(e['auxiliary_originals']==0 and e['main_real_pair_draws']==0 for e in ac['history']))
            self.assertFalse(result['test_evaluated'])
            with self.assertRaisesRegex(ValueError,'completed'):
                m.train(a,root/'pre_A',root/'A',torch.device('cpu'),1,fixture=True,resume=True)
            bad=copy.deepcopy(h);bad['base']['seed']=18
            with self.assertRaises(ValueError):m.check_parent(pck,bad)


if __name__=='__main__':unittest.main()
