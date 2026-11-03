"""Architecture ablation checks without benchmark data or pretrained weights."""
import ast
import importlib.util
import json
from pathlib import Path
import copy
import tempfile
import unittest

import torch
from torch import nn

SRC = Path(__file__).resolve().parents[1] / 'src'


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


new = load('head_ablation', 'ces_fl_head_ablation.py')
old = load('hier_reference', 'ces_fl_forensic_response.py')
PROPOSAL = SRC.parent / 'proposals/plain_head_s17_20261103T101418Z'
reuse = load('reuse_review', PROPOSAL / 'reuse_review.py')
replay = load('replay_reference', PROPOSAL / 'replay_reference.py')
torch.set_num_threads(2)


class FeatureInput(nn.Module):
    def features(self, x):
        return x


class HeadAblationTests(unittest.TestCase):
    def test_exact_production_parameter_counts(self):
        for head, total, trainable, head_n in [('linear',20809001,6790433,2049),
                                             ('gated',21594159,7575591,787207)]:
            with self.subTest(head=head):
                m = new.Detector(head_type=head)
                r = m.parameter_report()
                self.assertEqual((r['total'],r['trainable'],r['head']), (total,trainable,head_n))
                self.assertEqual(r['encoder'],20806952)
                self.assertEqual(r['encoder_trainable'],6788384)
                del m

    def test_linear_removes_all_old_head_modules(self):
        m = new.Detector(head_type='linear')
        self.assertEqual([n for n,_ in m.named_parameters() if not n.startswith('encoder.')],
                         ['classifier.weight','classifier.bias'])
        for name in ('global_head','patch_head','image_gate','patch_gate','fusion_raw','fusion_bias'):
            self.assertFalse(hasattr(m,name))
        with self.assertRaises(ValueError): new.Detector(head_type='typo')

    def test_plain_formula_and_cam_are_independent_reference(self):
        m = new.Detector(head_type='linear'); m.encoder = FeatureInput()
        f = torch.randn(3,2048,3,4,requires_grad=True)
        r = m(f)
        # Explicit spatial sums rather than the implementation's pooled forward.
        ref = sum(f[:,:,i,j] @ m.classifier.weight.t() for i in range(3) for j in range(4))/12
        ref = ref.squeeze(1) + m.classifier.bias[0]
        torch.testing.assert_close(r['logit'],ref)
        torch.testing.assert_close(r['logit'],r['local_map'].mean((1,2))+r['bias'])
        self.assertNotIn('patch_gate',r)
        r['logit'].square().sum().backward()
        self.assertGreater(m.classifier.weight.grad.abs().sum().item(),0)
        self.assertGreater(f.grad.abs().sum().item(),0)

    def test_reference_gated_initialization_logits_losses_and_gradients_unchanged(self):
        torch.manual_seed(17); a = old.Detector()
        torch.manual_seed(17); b = new.Detector(head_type='gated')
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        a.encoder=FeatureInput(); b.encoder=FeatureInput()
        x=torch.randn(4,2048,2,2)
        batch={'x':x,'s':x+.02,'t':x+.1,'y':torch.tensor([0,1,0,1]),'index':torch.arange(4)}
        aux={'x':x,'s':x[[0,2]]+.02,'t':x[[0,2]]+torch.tensor([.1,.2])[:,None,None,None],
             'y':batch['y'],'group':torch.tensor([0,0,1,1]),'priors':torch.ones(2,2)/2,'axis':'gender'}
        c=new.defaults();c['amp']=False
        for arm in ('A','D'):
            a.zero_grad();b.zero_grad()
            la,_=old.train_step(a,batch,aux,torch.ones(4),c,arm,.15,torch.device('cpu'))
            lb,_=new.train_step(b,batch,aux,torch.ones(4),c,arm,.15,torch.device('cpu'))
            for k in la: torch.testing.assert_close(la[k],lb[k],rtol=0,atol=0)
            la['total'].backward();lb['total'].backward()
            for (ka,pa),(kb,pb) in zip(a.named_parameters(),b.named_parameters()):
                self.assertEqual(ka,kb)
                torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)

    def test_science_functions_are_ast_identical_to_H(self):
        def functions(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a=functions(SRC/'ces_fl_forensic_response.py');b=functions(SRC/'ces_fl_head_ablation.py')
        for name in ('Xception','Block','SeparableConv2d','intervention','Originals','auxiliary','hierarchy_layout',
                     'hierarchical_real_error','soft_error_disparity','response','sensitivity','response_alignment',
                     'original_weights','make_loader','build_optimizer','calibrate','metric_report',
                     'bootstrap','components','split_audit','read_manifest','qualify','check_qualification'):
            with self.subTest(function=name): self.assertEqual(a[name],b[name])

    def test_linear_and_gated_checkpoints_cannot_be_interchanged(self):
        a=new.Detector(head_type='linear');b=new.Detector(head_type='gated')
        with self.assertRaises(RuntimeError): a.load_state_dict(b.state_dict(),strict=True)
        with self.assertRaises(RuntimeError): b.load_state_dict(a.state_dict(),strict=True)

    def test_trainable_policy_and_bn_buffers(self):
        m=new.Detector(policy='exit',head_type='linear');m.train()
        self.assertFalse(m.encoder.training)
        for name,p in m.encoder.named_parameters():
            self.assertEqual(p.requires_grad,name.startswith(('block12.','conv3.','bn3.','conv4.','bn4.')))
        frozen=new.Detector(policy='frozen',head_type='linear')
        self.assertEqual(frozen.counts()['trainable'],2049)

    def test_config_requires_explicit_head_and_new_format(self):
        c=new.defaults();c.pop('head_type')
        with self.assertRaisesRegex(ValueError,'schema'):new.check_config(c)
        c=new.defaults();c['head_type']='unknown'
        with self.assertRaisesRegex(ValueError,'head_type'):new.check_config(c)
        c=new.defaults();c['format']=old.VERSION
        with self.assertRaisesRegex(ValueError,'format'):new.check_config(c)

    def test_exact_config_differences(self):
        for reference, filename in [('A','L_A_plain_bce.json'),('H','L_H_plain_hierarchical.json')]:
            old_c=replay.reference_config(reference)
            c=json.loads((PROPOSAL/filename).read_text())
            # Archived configs remain pinned to the original, completed source.
            expected=dict(old_c,format=new.VERSION,source_sha256='08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177',head_type='linear')
            expected.setdefault('error_control','legacy');expected.setdefault('alignment_warmup_epochs',0)
            self.assertEqual(c,expected)

    def test_review_reuse_rejects_metadata_membership_and_byte_changes(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'source';p.write_bytes(b'synthetic review test bytes')
            case={'id':'one','image':str(p),'sheet':str(p),'image_sha256':new.sha(p),
                  'sheet_sha256':new.sha(p),'settings':{'strength':.5}}
            a={'cases':[case]};b=copy.deepcopy(a)
            reuse.equivalent_cases(a,b)
            b['cases'][0]['settings']['strength']=1.
            with self.assertRaises(ValueError):reuse.equivalent_cases(a,b)
            with self.assertRaises(ValueError):reuse.equivalent_cases(a,{'cases':[]})
            b=copy.deepcopy(a);p.write_bytes(b'changed')
            with self.assertRaises(ValueError):reuse.equivalent_cases(a,b)

    def test_reference_requires_complete_pinned_run_and_earliest_auc_selection(self):
        c=replay.reference_config('H')
        history=[{'epoch':i,'validation':{'overall':{'auc':.8 if i in (2,3) else .7}},
                  'calibration':{'threshold_logit':float(i)}} for i in range(1,11)]
        best=history[1]
        ck=dict(config=c,format=c['format'],source_sha256=c['source_sha256'],arm='D',
                synthetic_fixture=False,epoch=2,validation=best['validation'],calibration=best['calibration'])
        receipt=dict(status='COMPLETED_VERIFIED',epochs_completed=10,arm='D',
                     source_sha256=c['source_sha256'],config_digest=new.digest(c),
                     selected_checkpoint_sha256='pinned',selected_epoch=2,validation=best['validation'])
        replay.verify_run(ck,receipt,history,c,'H','pinned')
        with self.assertRaises(ValueError):replay.verify_run(ck,receipt,history[:7],c,'H','pinned')
        with self.assertRaises(ValueError):replay.verify_run(ck,receipt,history,c,'H','changed')
        ck['epoch']=3
        with self.assertRaises(ValueError):replay.verify_run(ck,receipt,history,c,'H','pinned')


if __name__ == '__main__':
    unittest.main()
