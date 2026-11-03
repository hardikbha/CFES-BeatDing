"""Independent math, gradient, metadata and unchanged-component regressions."""
import ast
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import torch

P = Path(__file__).resolve().parents[1]/'src/ces_fl_forensic_response.py'
s = importlib.util.spec_from_file_location('hier', P)
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)


def inputs():
    z = torch.tensor([-2., 1., 2., -1., -1., 3., 1., 0.], requires_grad=True)
    y = torch.tensor([0, 1]*4); g = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    priors = torch.tensor([[.05, .4], [.15, .1], [.2, .3], [.6, .2]])
    layout = {'parents':['A','B'], 'parent_codes':torch.tensor([0,0,1,1]),
              'within_parent_real_priors':torch.tensor([[.25,.75,0.,0.],[0.,0.,.25,.75]])}
    return z, y, g, priors, layout


class HierarchicalReal(unittest.TestCase):
    def test_exact_independent_reference_normalization(self):
        z,y,g,p,h=inputs()
        loss=m.hierarchical_real_error(z,y,g,p,h,.3,.7)
        probabilities=1/(1+np.exp(-(z.detach().numpy()-.3)/.7))
        real=probabilities[::2];fake=1-probabilities[1::2]
        b=np.dot(real,p[:,0].numpy());parent=np.array([.25*real[0]+.75*real[1],.25*real[2]+.75*real[3]])
        expected=.5*(np.maximum(real-b,0).mean()+np.maximum(parent-b,0).mean())+.5*np.abs(fake-np.dot(fake,p[:,1].numpy())).mean()
        self.assertAlmostEqual(float(loss),float(expected),places=6)

    def test_fake_component_gradient_exactly_legacy(self):
        z,y,g,p,h=inputs()
        old=m.soft_error_disparity(z,y,g,p,.3,.7)
        new=m.hierarchical_real_error(z,y,g,p,h,.3,.7)
        a=torch.autograd.grad(old,z,retain_graph=True)[0]
        b=torch.autograd.grad(new,z)[0]
        torch.testing.assert_close(a[y==1],b[y==1],atol=1e-8,rtol=1e-7)

    def test_real_gradients_do_not_directly_raise_lower_risk(self):
        z,y,g,p,h=inputs()
        loss=m.hierarchical_real_error(z,y,g,p,h,0.,1.)
        loss.backward()
        self.assertTrue((z.grad[y==0]>=0).all())
        # Low-risk child of an above-reference parent is still pushed DOWN;
        # low-risk child of a below-reference parent has zero direct gradient.
        self.assertGreater(float(z.grad[0]),0.)
        self.assertEqual(float(z.grad[4]),0.)
        self.assertGreater(float(z.grad[y==0].sum()),0.)

    def test_parent_weights_are_real_train_not_uniform_or_fake(self):
        z,y,g,p,h=inputs();wrong={**h,'within_parent_real_priors':torch.tensor([[.5,.5,0.,0.],[0.,0.,.5,.5]])}
        with self.assertRaisesRegex(ValueError,'TRAIN class0'):
            m.hierarchical_real_error(z,y,g,p,wrong,0.,1.)

    def test_joint_order_permutation_preserves_value(self):
        z,y,g,p,h=inputs();order=torch.tensor([3,0,2,1]);inverse=torch.argsort(order)
        hp={**h,'parent_codes':h['parent_codes'][order], 'within_parent_real_priors':h['within_parent_real_priors'][:,order]}
        a=m.hierarchical_real_error(z,y,g,p,h,0.,1.)
        b=m.hierarchical_real_error(z,y,inverse[g],p[order],hp,0.,1.)
        torch.testing.assert_close(a,b)

    def test_missing_class_and_bad_group_rejected(self):
        z,y,g,p,h=inputs()
        with self.assertRaisesRegex(ValueError,'Missing auxiliary'):
            m.hierarchical_real_error(z[1:],y[1:],g[1:],p,h,0.,1.)
        with self.assertRaisesRegex(ValueError,'group code'):
            m.hierarchical_real_error(z,y,g+1,p,h,0.,1.)

    def test_lambda_changes_update(self):
        results=[]
        for lam in [0.,.1]:
            z,y,g,p,h=inputs();opt=torch.optim.SGD([z],lr=.2)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y.float())+lam*m.hierarchical_real_error(z,y,g,p,h,0.,1.)
            loss.backward();opt.step();results.append(z.detach().clone())
        self.assertFalse(torch.equal(*results))
        torch.testing.assert_close(results[0][y==1]-results[1][y==1],
                                   .02*torch.autograd.functional.jacobian(lambda zz:m.hierarchical_real_error(zz,y,g,p,h,0.,1.),inputs()[0])[y==1],atol=2e-7,rtol=1e-5)

    def test_layout_counts_and_sparse_cartesian_rejection(self):
        rows=[];groups={}
        for j,(gender,race) in enumerate([('A','X'),('A','Y'),('B','X'),('B','Y')]):
            pools={0:[],1:[]};name=gender+'-'+race
            for label,n in [(0,[1,3,2,6][j]),(1,[6,2,3,1][j])]:
                for _ in range(n):
                    pools[label].append(len(rows));rows.append(dict(gender=gender,race=race,label=label))
            groups[name]=pools
        h=m.hierarchy_layout(rows,groups)
        torch.testing.assert_close(h['within_parent_real_priors'],inputs()[-1]['within_parent_real_priors'])
        with self.assertRaisesRegex(ValueError,'Cartesian'):
            m.hierarchy_layout(rows,{k:v for k,v in groups.items() if k!='B-Y'})

    def test_train_group_inventory_fails_instead_of_dropping_sparse_cell(self):
        rows=[]
        for j,(gender,race) in enumerate([('A','X'),('A','Y'),('B','X'),('B','Y')]):
            for label in (0,1):
                for i in range(1 if j==3 else 2):
                    rows.append(dict(gender=gender,race=race,gender_race=gender+'-'+race,label=label))
        c={'axes':['gender_race'],'min_train_support':2,'error_control':'hierarchical_real'}
        with self.assertRaisesRegex(ValueError,'Missing/sparse'):
            m.group_inventory(rows,c)
        rows[0]['gender']='contradictory'
        with self.assertRaisesRegex(ValueError,'metadata disagree'):
            m.group_inventory(rows,c)

    def test_only_joint_real_term_changes_in_native_step(self):
        class Fixture(torch.nn.Module):
            def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.tensor(1.))
            def forward(self,x):return {'logit':x.mean((1,2,3))*self.w}
        z,y,g,p,h=inputs();x=z.detach()[:,None,None,None].expand(-1,3,4,4).clone()
        b={'x':x,'s':x-.1,'t':x+.2,'y':y,'index':torch.arange(8)}
        aux=dict(x=x,s=x[y==0]-.1,t=x[y==0]+.2,y=y,group=g,priors=p,hierarchy=h,axis='gender_race')
        c=m.defaults();c.update(amp=False,error_control='hierarchical_real')
        new,_=m.train_step(Fixture(),b,aux,torch.ones(8),c,'D',.3,torch.device('cpu'))
        old,_=m.train_step(Fixture(),b,aux,torch.ones(8),dict(c,error_control='legacy'),'D',.3,torch.device('cpu'))
        for key in new.keys()-{'total','error_fairness'}:
            torch.testing.assert_close(new[key],old[key],atol=0,rtol=0)
        torch.testing.assert_close(new['total']-old['total'],c['aux_every']*c['lambda_fair']*(new['error_fairness']-old['error_fairness']))
        aux['axis']='gender'
        a,_=m.train_step(Fixture(),b,aux,torch.ones(8),c,'D',.3,torch.device('cpu'))
        b,_=m.train_step(Fixture(),b,aux,torch.ones(8),dict(c,error_control='legacy'),'D',.3,torch.device('cpu'))
        for key in a:torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)

    def test_architecture_and_original_D_components_ast_unchanged(self):
        old=Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-forensic-response-c332fd34/CES_FL/src/ces_fl_forensic_response.py')
        if not old.exists():self.skipTest('Pinned old source unavailable')
        def definitions(path):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(path.read_text()).body if isinstance(x,(ast.FunctionDef,ast.ClassDef))}
        a,b=definitions(P),definitions(old)
        for name in ['Detector','Xception','Originals','intervention','normalize','load_pixels','seed_all','original_weights','build_optimizer','load_initial','calibrate','metric_report','soft_error_disparity','response','sensitivity','response_alignment','predict','split_audit']:
            self.assertEqual(a[name],b[name],name)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
