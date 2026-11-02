"""Synthetic-only scientific counterexamples and software checks; never real data."""
import argparse
import ast
from collections import Counter
import copy
import csv
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import unittest
import tempfile

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import ces_fl_cdf_objectives as loss
import ces_fl_counterfactual_5050 as base
import ces_fl_cdf_study as study
sys.path.insert(0,str(SRC.parent/'experiments/benchmark_seed17'))
import paired_uncertainty as uncertainty

DEVICE = 'cpu'
MEASURED = {}


def tensor(values, **kw):
    return torch.tensor(values, device=DEVICE, **kw)


class ScientificFixtures(unittest.TestCase):
    def setUp(self):
        base.seed_all(17)
        torch.set_num_threads(4)

    def test_equal_means_unequal_real_tails(self):
        z = tensor([-4., 4., -1., 1.], requires_grad=True)
        y = torch.zeros_like(z); g = tensor([0, 0, 1, 1]); p = tensor([.5, .5])
        old = loss.mean_score_parity(z, y, g, 2)
        new, stats = loss.real_cdf_parity(z, y, g, p)
        self.assertAlmostEqual(float(old), 0., places=7)
        self.assertGreater(float(new), .1)
        hard = [(z[g == i].detach() >= 1.5).float().mean().item() for i in range(2)]
        self.assertEqual(hard, [.5, 0.])
        new.backward(); self.assertGreater(float(z.grad.abs().sum()), 1e-5)
        self.assertFalse(stats['knots'].requires_grad)
        MEASURED['equal_means_fixture'] = {'old_mean_loss': float(old), 'cdf_loss': float(new),
                                          'hard_FPR_at_1_5': hard, 'fairness_gradient_l1': float(z.grad.abs().sum())}

    def test_lambda_changes_updates_and_identical_groups(self):
        def step(weight):
            z = nn.Parameter(tensor([-4., 4., -1., 1.]))
            optimizer = torch.optim.SGD([z], lr=.1)
            y = torch.zeros_like(z); g = tensor([0, 0, 1, 1])
            v, _ = loss.real_cdf_parity(z, y, g, tensor([.5, .5]))
            optimizer.zero_grad(); (F.binary_cross_entropy_with_logits(z, y) + weight*v).backward(); optimizer.step()
            return z.detach()
        self.assertGreater(float((step(0)-step(1)).abs().sum()), 1e-5)
        z = tensor([-1., 2., -1., 2.], requires_grad=True)
        v, _ = loss.real_cdf_parity(z, torch.zeros_like(z), tensor([0, 0, 1, 1]), tensor([.5, .5]))
        self.assertAlmostEqual(float(v), 0., places=7)

    def test_missing_groups_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            loss.real_cdf_parity(tensor([1., 2.]), tensor([0, 0]), tensor([0, 0]), tensor([.5, .5]))
        with self.assertRaises(ValueError):
            loss.real_cdf_parity(tensor([1., float('nan')]), tensor([0, 0]), tensor([0, 1]), tensor([.5, .5]))

    def test_real_cdf_does_not_backpropagate_fake_scores(self):
        z=tensor([-4.,4.,-1.,1.,-1.,-2.,3.,4.],requires_grad=True)
        y=tensor([0,0,0,0,1,1,1,1]);g=tensor([0,0,1,1,0,0,1,1])
        value,_=loss.real_cdf_parity(z,y,g,tensor([.5,.5]));value.backward()
        self.assertTrue(torch.equal(z.grad[y==1],torch.zeros_like(z.grad[y==1])))

    def test_actual_archived_gda_block_detaches_fairness(self):
        from geomloss import SamplesLoss
        path=SRC.parent.parent/'DINGREPRO_LIN/code/Ding_mod/GDA.py'
        tree=ast.parse(path.read_text())
        candidates=[n for n in ast.walk(tree) if isinstance(n,ast.With) and 'fair_loss = fairLoss' in ast.unparse(n)]
        self.assertEqual(len(candidates),1)
        program=compile(ast.Module(body=[candidates[0]],type_ignores=[]),str(path),'exec')
        class Head(nn.Module):
            def __init__(self):super().__init__();self.fc=nn.Linear(2,2)
            def forward(self,d):return {'cls':self.fc(d['image'])}
        class Args:lambda_fairloss=0.
        model=Head().to(DEVICE);state=copy.deepcopy(model.state_dict());data={'image':tensor([[-1.,0.],[1.,1.],[-2.,1.],[2.,0.]])}
        values=[]
        for coefficient in [0.,.005,1000.]:
            torch.set_grad_enabled(True);model.load_state_dict(state);model.zero_grad(set_to_none=True)
            logits=model(data)['cls'];main=F.cross_entropy(logits,tensor([0,1,0,1]));a=Args();a.lambda_fairloss=coefficient
            scope={'torch':torch,'model':model,'group_data_dict':{'image':data['image']+1},'combined_preds':{'cls':logits},
                   'main_loss':main,'fairLoss':SamplesLoss('sinkhorn',p=2,blur=5e-4),'args':a}
            exec(program,scope)
            self.assertFalse(scope['fair_loss'].requires_grad)
            if scope['total_loss'].requires_grad:
                scope['total_loss'].backward();values.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
        if values:
            self.assertEqual(len(values),3)
            torch.testing.assert_close(values[0],values[1]);torch.testing.assert_close(values[0],values[2])
        MEASURED['archived_GDA_exact_AST_block']={'fairness_requires_grad':False,'classification_backpropagates':bool(values),
            'lambda_independent_gradients':True if values else None,'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_response_cancels_group_offset_but_cdf_detects_it(self):
        q = {k: tensor([-2., -2., -2., -2.]) for k in ['original', 'resize', 'x0', 'xa']}
        q.update(t0=tensor([2., 2., 2., 2.]), ta=tensor([2., 2., 2., 2.]))
        offset = tensor([0., 0., 6., 6.])
        shifted = {k: v + offset for k, v in q.items()}
        a, b = loss.manipulation_terms(q), loss.manipulation_terms(shifted)
        self.assertEqual(float(a['alignment'] + a['sensitivity']), 0.)
        self.assertEqual(float(b['alignment'] + b['sensitivity']), 0.)
        self.assertEqual(float(b['score']), 0.)
        cdf, _ = loss.real_cdf_parity(shifted['original'], tensor([0, 0, 0, 0]), tensor([0, 0, 1, 1]), tensor([.5, .5]))
        self.assertGreater(float(cdf), .1)
        MEASURED['response_offset_fixture'] = {'response_loss': 0., 'score_anchor_loss': 0., 'cdf_loss': float(cdf)}

    def test_demographic_offset_anchor_and_wrong_response(self):
        q = {k: tensor([-2., -2.], requires_grad=True) for k in ['original', 'resize', 'x0', 'xa']}
        q.update(t0=tensor([2., 2.], requires_grad=True), ta=tensor([2., 2.], requires_grad=True))
        shifted = {**q, 'xa': q['xa']+3, 'ta': q['ta']+3}
        terms = loss.manipulation_terms(shifted)
        self.assertEqual(float(terms['alignment']), 0.)
        self.assertGreater(float(terms['score']), 1.)
        (terms['score']+terms['alignment']+terms['sensitivity']).backward()
        self.assertIsNotNone(q['xa'].grad)
        bad = {**q, 't0': q['x0']-1, 'ta': q['xa']-1}
        self.assertGreater(float(loss.manipulation_terms(bad)['sensitivity']), .5)
        with self.assertRaises(ValueError):
            loss.manipulation_terms({**q, 'ta': tensor([1.])})

    def test_all_arms_and_complete_dro_state(self):
        z = tensor([-3., -2., -1., 2., 1., 3., -1., 4.], requires_grad=True)
        y = tensor([0, 0, 0, 0, 1, 1, 1, 1]); g = tensor([0, 0, 1, 1, 0, 0, 1, 1])
        q = {k: tensor([-2., -1.], requires_grad=True) for k in ['original', 'resize', 'x0', 'xa']}
        q.update(t0=tensor([2., 3.], requires_grad=True), ta=tensor([3., 2.], requires_grad=True))
        for arm in loss.ARMS:
            dro = loss.CompleteGroupDRO(4).to(DEVICE)
            values = loss.objective(z, y, g, tensor([.5, .5]), arm, q=q, dro=dro)
            self.assertTrue(all(torch.isfinite(v) for v in values.values()))
            grads = torch.autograd.grad(values['total'], z, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(grads).all())
        dro = loss.CompleteGroupDRO(4).to(DEVICE)
        before = copy.deepcopy(dro.state_dict()); dro(F.binary_cross_entropy_with_logits(z, y.float(), reduction='none'), y.long()*2+g)
        self.assertFalse(torch.equal(before['log_q'], dro.log_q))
        restored = loss.CompleteGroupDRO(4).to(DEVICE); restored.load_state_dict(dro.state_dict())
        torch.testing.assert_close(restored.log_q, dro.log_q)
        self.assertAlmostEqual(float(dro.log_q.exp().sum()), 1., places=6)
        with self.assertRaises(ValueError):
            dro(tensor([1., 2.]), tensor([0, 1]))

    def test_ding_adapter_and_class_conditioning(self):
        x = tensor([[-2., .4], [-1., .6], [.5, -.4], [.7, -.1],
                    [0., 1.], [.2, 1.2], [1., -.3], [1.3, -.2]], requires_grad=True)
        y = tensor([0, 0, 0, 0, 1, 1, 1, 1]); g = tensor([0, 0, 1, 1, 0, 0, 1, 1])
        torch.testing.assert_close(torch.sigmoid(loss.binary_logit(x)), x.softmax(1)[:, 1])
        torch.testing.assert_close(loss.binary_logit(x+7), loss.binary_logit(x))
        total, terms = loss.corrected_ding_cc_ot(x, y, g, 2)
        grads = torch.autograd.grad(total, x, retain_graph=True)[0]
        self.assertTrue(torch.isfinite(grads).all()); self.assertGreater(float(grads.abs().sum()), 1e-5)
        other = x.detach().clone(); other[y == 1, 1] += 3; other.requires_grad_()
        _, changed = loss.corrected_ding_cc_ot(other, y, g, 2)
        torch.testing.assert_close(terms[0], changed[0])
        self.assertGreater(abs(float(terms[1]-changed[1])), 1e-6)
        self.assertTrue(torch.is_grad_enabled())
        with torch.no_grad(), self.assertRaises(ValueError):
            loss.corrected_ding_cc_ot(x, y, g, 2)
        MEASURED['corrected_ding_fair_only_gradient_l1'] = float(grads.abs().sum())

    def test_corrected_ding_lambda_changes_head_updates(self):
        features = tensor([[-2., 0.], [-1., .2], [1., -.1], [2., .3], [-.5, 1.], [-.8, 1.3], [1., -.6], [1.2, -.4]])
        y = tensor([0, 0, 0, 0, 1, 1, 1, 1]); g = tensor([0, 0, 1, 1, 0, 0, 1, 1])
        model = nn.Linear(2, 2).to(DEVICE); initial = copy.deepcopy(model.state_dict())
        values = []
        for weight in [0., 1.]:
            model.load_state_dict(initial); opt = torch.optim.SGD(model.parameters(), lr=.1)
            out = model(features); fair, _ = loss.corrected_ding_cc_ot(out, y, g, 2)
            opt.zero_grad(); (F.cross_entropy(out, y)+weight*fair).backward(); opt.step()
            values.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
        self.assertGreater(float((values[0]-values[1]).abs().sum()), 1e-6)

    def test_corrected_ding_constant_scores_are_finite(self):
        scores=torch.zeros(8,2,device=DEVICE,requires_grad=True)
        y=tensor([0,0,0,0,1,1,1,1]);g=tensor([0,0,1,1,0,0,1,1])
        value,_=loss.corrected_ding_cc_ot(scores,y,g,2)
        value.backward()
        self.assertTrue(torch.isfinite(value));self.assertTrue(torch.isfinite(scores.grad).all())

    def test_both_backbones_gradients_frozen_bn_decomposition_reload(self):
        for backbone in ['xception', 'resnet50']:
            model = base.DiagramCES(backbone, pretrained=False, encoder_policy='trainable', bn_policy='frozen', head_seed=17).to(DEVICE)
            before_bn = {k: v.clone() for k, v in model.encoder.state_dict().items() if 'running_' in k or 'num_batches' in k}
            model.train(); x = torch.randn(8, 3, 75, 75, device=DEVICE)
            y = tensor([0., 0., 0., 0., 1., 1., 1., 1.]); g = tensor([0, 0, 1, 1, 0, 0, 1, 1])
            out = model(x)
            torch.testing.assert_close(out['logit'], out['global_contribution']+out['local_map'].mean((1, 2))+model.fusion_bias)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
            v = loss.objective(out['logit'], y, g, tensor([.5, .5]), 'cdf_fair')
            opt.zero_grad(); v['total'].backward()
            for name in ['encoder', 'global_head', 'patch_head', 'image_gate', 'patch_gate']:
                grads = [p.grad for p in getattr(model, name).parameters()]
                self.assertTrue(all(a is not None and torch.isfinite(a).all() for a in grads), name)
                self.assertGreater(sum(float(a.abs().sum()) for a in grads), 0.)
            opt.step()
            self.assertTrue(all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in before_bn.items()))
            state = copy.deepcopy(model.state_dict()); model.load_state_dict(state, strict=True)
            model.set_encoder_policy('frozen', 'frozen'); before = copy.deepcopy(model.encoder.state_dict())
            opt.zero_grad(set_to_none=True); model(x)['logit'].sum().backward(); opt.step()
            self.assertTrue(all(torch.equal(v, model.encoder.state_dict()[k]) for k, v in before.items()))
            MEASURED[backbone+'_parameters'] = model.counts()
            del model, opt, out, x, state, before
            if DEVICE == 'cuda': torch.cuda.empty_cache()

    def test_quota_rng_and_paired_cluster_uncertainty(self):
        rows=[{'label':y,'gender_race':g,'identity_ids':f'p{i}','source_ids':f'v{i}'} for i,(y,g) in enumerate(
            [(0,'a'),(0,'a'),(0,'b'),(0,'b'),(1,'a'),(1,'a'),(1,'b'),(1,'b')])]
        a=list(study.QuotaBatches(rows,['a','b'],2,17,0));b=list(study.QuotaBatches(rows,['a','b'],2,17,0))
        self.assertEqual(a,b)
        for batch in a:
            self.assertEqual(sorted(Counter((rows[i]['label'],rows[i]['gender_race']) for i in batch).values()),[2]*4)
        scores=np.array([-3,-2,-1,0,1,2,3,4.])
        paired={seed:scores for seed in [7,17,27]}; thresholds={seed:0. for seed in paired}
        result=uncertainty.paired_bootstrap(rows,paired,paired,thresholds,thresholds,['gender_race'],reps=25,min_support=1)
        for key,v in result['intervals'].items():
            if key.startswith('delta.'):
                self.assertEqual(v['low'],0.);self.assertEqual(v['high'],0.)
        self.assertEqual(uncertainty.signflip_p([1,1,1]),.25)
        connected=[{**r,'identity_ids':'same_person'} for r in rows]
        absent=uncertainty.paired_bootstrap(connected,paired,paired,thresholds,thresholds,['gender_race'],reps=5,min_support=1)
        self.assertIsNone(absent['intervals'])
        with self.assertRaises(ValueError):
            uncertainty.paired_bootstrap(rows,paired,paired,{**thresholds,7:float('nan')},thresholds,['gender_race'],reps=1,min_support=1)
        one_group=[{**r,'gender_race':'a'} for r in rows]
        single=uncertainty.paired_bootstrap(one_group,paired,paired,thresholds,thresholds,['gender_race'],reps=5,min_support=1)
        self.assertIn('gender_race.worst_group_auc',single['paired_seed_differences'])
        self.assertNotIn('gender_race.fpr_gap',single['paired_seed_differences'])

    def test_tiny_pretrain_pair_gate_training_reload_and_val_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rng=np.random.default_rng(17); paths={}; records={}
            fields=['image','label','identity_ids','source_ids','gender_race','ids8','family']
            for split in ['train','val']:
                rows=[]
                for i in range(8):
                    image=root/f'{split}_{i}.png';base.Image.fromarray(rng.integers(0,256,(48,48,3),dtype=np.uint8)).save(image)
                    rows.append({'image':str(image),'label':int(i>=4),'identity_ids':f'{split}_person{i}','source_ids':f'{split}_video{i}',
                                 'gender_race':'a' if i%4<2 else 'b','ids8':str(i%4),'family':'synthetic'})
                paths[split]=root/(split+'.csv');records[split]=rows
                with paths[split].open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
            weights=root/'dummy_weights';weights.write_text('synthetic; never loaded')
            c={'format':study.VERSION,'dataset':'ding_ffpp','seed':17,'backbone':'xception','train':str(paths['train']),'val':str(paths['val']),
               'weights':str(weights),'protocol_overlap_file':None,'groups':['a','b'],'per_cell':2,'pretrain_epochs':1,'epochs':1,
               'size':75,'hidden':8,'gate_hidden':4,'batch_size':8,'pretrain_batch_size':8,'validation_batch_size':8,'workers':0,
               'lr':.001,'encoder_lr':1e-5,'pretrain_encoder_lr':1e-4,'weight_decay':1e-4,'dro_eta':.01,'pair_batch':2,
               'target_fpr':.1,'min_support':1,'loss':{},'pretrain_run':None,'pretrain_sha256':None,'bank':None,'bank_sha256':None,
               'heldout_exclusions':None,'heldout_exclusions_sha256':None,
               'input_hashes':{k:base.sha(v) for k,v in [('train',paths['train']),('val',paths['val']),('weights',weights)]}}
            study.run(c,'pretrain',root/'P',DEVICE,synthetic=True)
            c['pretrain_run']=str(root/'P');c['pretrain_sha256']=base.sha(root/'P/best.pt')
            # Native P50 receipt uses summary/history for completed epochs.
            # Convert this explicitly synthetic fixture to that storage contract.
            original_ck=torch.load(root/'P/best.pt',map_location='cpu',weights_only=True)
            native_ck=copy.deepcopy(original_ck);native_ck['format']=base.VERSION
            native_ck['source_sha256']=study.BASE_SHA;native_ck['config']['pretrain_epochs']=50
            torch.save(native_ck,root/'P/best.pt')
            native_receipt=json.loads((root/'P/RUN_VERIFIED.json').read_text());native_receipt.pop('completed_epochs')
            native_receipt['checkpoint_sha256']=base.sha(root/'P/best.pt')
            (root/'P/RUN_VERIFIED.json').write_text(json.dumps(native_receipt))
            (root/'P/summary.json').write_text(json.dumps({'status':'COMPLETED_VERIFIED','stage':'pretrain','completed_epochs':50}))
            original_history=(root/'P/history.jsonl').read_text()
            (root/'P/history.jsonl').write_text(''.join(json.dumps({'epoch':i,'synthetic_storage_fixture_only':True})+'\n' for i in range(1,51)))
            native_c={**c,'pretrain_epochs':50,'pretrain_sha256':native_receipt['checkpoint_sha256']}
            self.assertEqual(study.load_pretrain(native_c,synthetic=True)['format'],base.VERSION)
            (root/'P/history.jsonl').write_text('{"epoch":1}\n')
            with self.assertRaises(ValueError):study.load_pretrain(native_c,synthetic=True)
            # Restore the one-epoch synthetic experiment; never release it as real P50.
            torch.save(original_ck,root/'P/best.pt');c['pretrain_sha256']=base.sha(root/'P/best.pt')
            native_receipt['completed_epochs']=1;native_receipt['checkpoint_sha256']=c['pretrain_sha256']
            (root/'P/RUN_VERIFIED.json').write_text(json.dumps(native_receipt))
            (root/'P/history.jsonl').write_text(original_history)
            study.run(c,'cdf_fair',root/'cdf',DEVICE,dry=True,synthetic=True)
            self.assertEqual(json.loads((root/'cdf/DRY_RUN.json').read_text())['optimizer_steps'],0)
            original_rows=base.read_rows(c['train']);val=base.read_rows(c['val'])
            with self.assertRaises(ValueError):study.audit_bank(c,original_rows,val,synthetic=True)
            evidence=root/'evidence.json';evidence.write_text('{"synthetic_fixture":true}')
            receipt={'path':str(evidence),'sha256':base.sha(evidence)}
            exclusions={'status':'PASS','test_scores_used':False,'synthetic_fixture':True,'heldout_contributor_ids':[r['identity_ids'] for r in val],
                        'train_component_by_image_sha256':{}}
            quartets=[]
            for ix in [0,2]:
                source=records['train'][ix];source_sha=base.sha(source['image']);component='component'+str(ix)
                exclusions['train_component_by_image_sha256'][source_sha]=component
                with base.Image.open(source['image']) as im:pixels=np.array(im)
                shared={}
                for role,shift in [('resize',0),('x0',1),('t0',2)]:
                    p=root/f'{ix}_{role}.png';base.Image.fromarray(np.roll(pixels,shift,axis=0)).save(p)
                    shared[role]={'image':str(p),'sha256':base.sha(p)}
                for direction,shift in [('older',3),('younger',4)]:
                    views={'original':{'image':source['image'],'sha256':source_sha},**copy.deepcopy(shared)}
                    for role,n in [('xa',shift),('ta',shift+2)]:
                        p=root/f'{ix}_{direction}_{role}.png';base.Image.fromarray(np.roll(pixels,n,axis=0)).save(p)
                        views[role]={'image':str(p),'sha256':base.sha(p)}
                    for role in ['x0','xa']:
                        views[role].update(parent_sha256=source_sha,editor_revision='synthetic',settings_sha256='settings',seed=17,
                                           condition='nochange' if role=='x0' else direction)
                    for role,parent in [('t0','x0'),('ta','xa')]:
                        views[role].update(parent_sha256=views[parent]['sha256'],operator_code_sha256='code',settings_sha256='operation',donor_sha256='synthetic_none',seed=17)
                    review={k:True for k in ['approved','identity_preserved','age_direction_valid','operation_valid','matched_processing','forensic_evidence_assessed','unrelated_changes_assessed','authenticity_preservation_supported','forensic_preservation_supported']}
                    review.update(evidence=receipt,image_hashes={v:views[v]['sha256'] for v in study.VIEWS})
                    quartets.append({'id':f'{ix}_{direction}','source_image':source['image'],'source_sha256':source_sha,'group':source['gender_race'],
                        'direction':direction,'contributors':[source['identity_ids'],source['source_ids']],'generators':['synthetic_generator'],
                        'reviews':[{**review,'reviewer':'synthetic_fixture_A'},{**review,'reviewer':'synthetic_fixture_B'}],
                        'processing_receipt':receipt,'operator_receipt':receipt,'operator_family':'synthetic_roll',
                        'source_component_id':component,'views':views})
            exclusion_path=root/'exclusions.json';exclusion_path.write_text(json.dumps(exclusions))
            bank_path=root/'bank.json';bank={'format':study.VERSION,'synthetic_fixture':True,'train_sha256':c['input_hashes']['train'],'quartets':quartets};bank_path.write_text(json.dumps(bank))
            c.update(bank=str(bank_path),bank_sha256=base.sha(bank_path),heldout_exclusions=str(exclusion_path),heldout_exclusions_sha256=base.sha(exclusion_path))
            checked=study.audit_bank(c,original_rows,val,synthetic=True);self.assertEqual(len(checked),4)
            with self.assertRaises(ValueError):study.audit_bank(c,original_rows,val,synthetic=False)
            study.run(c,'response',root/'response',DEVICE,synthetic=True)
            ck=torch.load(root/'response/best.pt',map_location='cpu',weights_only=True)
            self.assertEqual(ck['arm'],'response');self.assertEqual(ck['dro']['log_q'].numel(),4)
            a=json.loads((root/'cdf/provenance.json').read_text());b=json.loads((root/'response/provenance.json').read_text())
            self.assertEqual(a['heads_initial_sha256'],b['heads_initial_sha256'])
            self.assertFalse(b['test_read'])
            with self.assertRaises(ValueError):study.release_check(paths['train'],c,None)
            bank['quartets'][0]['reviews'][0]['approved']=False;bank_path.write_text(json.dumps(bank));c['bank_sha256']=base.sha(bank_path)
            with self.assertRaises(ValueError):study.audit_bank(c,original_rows,val,synthetic=True)

    def test_production_resolution_cuda_memory_backward(self):
        if DEVICE!='cuda':self.skipTest('Production224px batch64+24paired-view memory test is CUDA-only')
        y=tensor([0.]*32+[1.]*32);g=tensor([i for _ in range(2) for i in range(8) for _ in range(4)])
        for backbone in ['xception','resnet50']:
            torch.cuda.reset_peak_memory_stats();start=time.monotonic()
            model=base.DiagramCES(backbone,pretrained=False,encoder_policy='trainable',bn_policy='frozen',head_seed=17).cuda().train()
            x=torch.randn(64,3,224,224,device=DEVICE);z=model(x)['logit']
            qscore=model(torch.randn(24,3,224,224,device=DEVICE))['logit'];q={v:qscore[i*4:(i+1)*4] for i,v in enumerate(study.VIEWS)}
            v=loss.objective(z,y,g,torch.full((8,),.125,device=DEVICE),'response',q=q)
            v['total'].backward();torch.cuda.synchronize()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            MEASURED[backbone+'_production224_batch64_plus24']={'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20,
                'forward_backward_seconds':time.monotonic()-start,'optimizer_steps':0}
            del model,x,z,qscore,q,v
            torch.cuda.empty_cache()


def main():
    global DEVICE
    p = argparse.ArgumentParser(); p.add_argument('--device', choices=['cpu', 'cuda'], required=True); p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); DEVICE = args.device
    if DEVICE == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; not verified')
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScientificFixtures)
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    receipt = {'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'device': DEVICE,
               'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'skipped':len(result.skipped),
               'elapsed_seconds': time.monotonic()-start, 'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'environment': {k: importlib.metadata.version(k) for k in ['torch', 'torchvision', 'geomloss']},
               'hashes': {str(f.relative_to(SRC.parent)): hashlib.sha256(f.read_bytes()).hexdigest() for f in
                          [SRC/'ces_fl_cdf_objectives.py', SRC/'ces_fl_cdf_study.py', SRC/'ces_fl_counterfactual_5050.py',
                           SRC.parent/'experiments/benchmark_seed17/paired_uncertainty.py',Path(__file__)]},
               'measured_synthetic_fixtures': MEASURED,
               'data': 'synthetic only; no real TRAIN/VAL/TEST opened; no full training or generation'}
    receipt['hashes']['archive_GDA.py']=hashlib.sha256((SRC.parent.parent/'DINGREPRO_LIN/code/Ding_mod/GDA.py').read_bytes()).hexdigest()
    (args.out/'RESULT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('CDF_PROPOSAL_SMOKE', json.dumps(receipt), flush=True)
    if not result.wasSuccessful(): raise SystemExit(1)


if __name__ == '__main__': main()
