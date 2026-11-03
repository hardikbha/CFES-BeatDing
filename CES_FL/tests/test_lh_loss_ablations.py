"""Independent numerical and gradient tests for the two authorized L_H ablations."""
import ast
import copy
import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from torch import nn

SOURCE = Path(__file__).resolve().parents[1] / 'src/ces_fl_head_ablation.py'
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
m = load(SOURCE, 'candidate')
old = load('/home/sachin.chaudhary/ces_fl_sources/cesfl-lh-low-lr3-execution-20261103/CES_FL/src/ces_fl_head_ablation.py', 'control')
device = torch.device(os.environ.get('CES_TEST_DEVICE', 'cpu'))
torch.set_num_threads(2)

class Toy(nn.Module):
    def __init__(self):
        super().__init__(); self.w = nn.Parameter(torch.tensor(1.3)); self.b = nn.Parameter(torch.tensor(-2.))
    def forward(self, x): return {'logit': self.w*x + self.b}

class LossAblations(unittest.TestCase):
    def fixture(self):
        c=m.defaults(); c.update(amp=False)
        x=torch.tensor([-3., 1., -1., 3.])
        b=dict(x=x,s=x+.4,t=x+.8,y=torch.tensor([0,1,0,1]),index=torch.arange(4))
        aux=dict(x=x,s=x[[0,2]]+.4,t=x[[0,2]]+torch.tensor([1.,2.]),y=b['y'],group=torch.tensor([0,0,1,1]),priors=torch.ones(2,2)/2,axis='gender')
        return c,b,aux
    def call(self, model,c,b,aux, module=m):
        return module.train_step(model,b,aux,torch.ones(4),c,'D',.15,device)[0]
    def test_legacy_losses_and_gradients_bitwise_identical(self):
        c,b,aux=self.fixture(); a=Toy().to(device); z=Toy().to(device)
        l=self.call(a,c,b,aux); ref=self.call(z,c,b,aux,old)
        for k in l: torch.testing.assert_close(l[k],ref[k],rtol=0,atol=0)
        l['total'].backward();ref['total'].backward()
        for p,q in zip(a.parameters(),z.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
    def test_sham_fix_both_terms_and_only_terms(self):
        c,b,aux=self.fixture(); a=Toy().to(device)
        ref=self.call(a,c,b,aux); c['sham_transform']='asinh_of_difference'; changed=self.call(a,c,b,aux)
        for prefix, sx, ox in [('original',b['s'][[0,2]],b['x'][[0,2]]),('aux',aux['s'],aux['x'][[0,2]])]:
            expected=torch.asinh(a(sx.to(device))['logit']-a(ox.to(device))['logit']).abs().mean()
            torch.testing.assert_close(changed[prefix+'_control_stability'],expected)
        for k in ref:
            if k not in ('total','original_control_stability','aux_control_stability'):
                torch.testing.assert_close(changed[k],ref[k],rtol=0,atol=0)
        delta=c['lambda_sensitivity']*c['sham_weight']*((changed['original_control_stability']-ref['original_control_stability'])+c['aux_every']*(changed['aux_control_stability']-ref['aux_control_stability']))
        torch.testing.assert_close(changed['total']-ref['total'],delta)
        changed['total'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters()))
    def test_difference_transform_is_offset_invariant_and_nonzero(self):
        z=torch.tensor([-2.,3.],device=device,requires_grad=True); s=z.detach()+.7
        c={'sham_transform':'asinh_of_difference'}
        a=m.sham_consistency(s,z,c); b=m.sham_consistency(s+64,z+64,c)
        torch.testing.assert_close(a,b,atol=3e-6,rtol=1e-5)
        a.backward(); self.assertTrue((z.grad.abs()>0).all())
    def test_no_align_has_exactly_zero_alignment_gradient(self):
        c,b,aux=self.fixture(); c['lambda_alignment']=0.; a=Toy().to(device)
        l=self.call(a,c,b,aux)
        self.assertGreater(l['response_alignment'].item(),0.)
        g=torch.autograd.grad(l['total'],l['response_alignment'],retain_graph=True)[0]
        self.assertEqual(g.item(),0.)
        weighted=c['aux_every']*c['lambda_alignment']*l['response_alignment']
        for g in torch.autograd.grad(weighted,tuple(a.parameters()),retain_graph=True):self.assertTrue(torch.equal(g,torch.zeros_like(g)))
        expected=l['classification']+c['lambda_sensitivity']*(l['original_sensitivity']+c['sham_weight']*l['original_control_stability'])+c['aux_every']*(c['lambda_fair']*l['error_fairness']+c['lambda_sensitivity']*(l['aux_sensitivity']+c['sham_weight']*l['aux_control_stability']))
        g1=torch.autograd.grad(l['total'],tuple(a.parameters()),retain_graph=True)
        g2=torch.autograd.grad(expected,tuple(a.parameters()))
        for x,y in zip(g1,g2):torch.testing.assert_close(x,y,rtol=0,atol=0)
    def test_zero_only_alignment_and_invalid_values_rejected(self):
        c,b,a=self.fixture(); c.update(train='x',train_sha256='h',val='x',val_sha256='h',init_checkpoint='x',init_sha256='h',overlap_file=None,overlap_sha256=None)
        with patch.object(m,'sha',return_value='h'):
            c['lambda_alignment']=0; m.check_config(c,bound=False)
            for v in (-.01,float('nan'),float('inf')):
                c['lambda_alignment']=v
                with self.assertRaises(ValueError):m.check_config(c,bound=False)
            c['lambda_alignment']=0; c['lambda_sensitivity']=0
            with self.assertRaises(ValueError):m.check_config(c,bound=False)
    def test_other_functions_ast_unchanged(self):
        def funcs(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(p).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        a,b=funcs(old.__file__),funcs(m.__file__)
        self.assertEqual(set(b)-set(a),{'sham_consistency'})
        self.assertEqual({k for k in a if a[k]!=b[k]},{'train_step','check_config'})

if __name__=='__main__':unittest.main(verbosity=2)
