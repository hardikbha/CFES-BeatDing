import ast
import copy
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import ding_ff2_public as d

WEIGHTS='/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/data/xception-b5690688.pth'


def gather_worker(rank,port,path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2)
    # Uneven final native pair batch: rank0 has 3 pairs, rank1 has 2.
    torch.manual_seed(1)
    model=nn.Linear(2,2)
    ddp=torch.nn.parallel.DistributedDataParallel(model)
    x=torch.arange(20,dtype=torch.float32).reshape(10,2)/10
    y=torch.tensor([0,1,0,1,0,1,0,1,0,1])
    indices=[0,1,2,3,4,5] if rank==0 else [6,7,8,9]
    z=ddp(x[indices])
    full=d.gather(z,True)
    labels=d.gather(y[indices])
    loss=F.cross_entropy(z,y[indices],reduction='sum')*2/10
    loss=loss+.3*(full[:,1]-full[:,0]).square().mean()
    loss.backward()
    torch.manual_seed(1)
    reference=nn.Linear(2,2)
    rz=reference(x)
    rloss=F.cross_entropy(rz,y)+.3*(rz[:,1]-rz[:,0]).square().mean()
    rloss.backward()
    for a,b in zip(model.parameters(),reference.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=2e-6,atol=2e-7)
    torch.testing.assert_close(labels,y)
    if rank==0:
        Path(path).write_text('uneven_gather_and_global_mean_gradients_PASS\n')
    dist.destroy_process_group()


class Audit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_01_native_model_exact(self):
        from detectors.xception_detector import XceptionDetector
        d.seed(17)
        native=XceptionDetector().eval()
        d.seed(17)
        ours=d.Detector(WEIGHTS).eval()
        self.assertEqual(sum(p.numel() for p in ours.parameters()),21858090)
        self.assertEqual(set(native.state_dict()),set(ours.state_dict()))
        for k,v in native.state_dict().items():
            torch.testing.assert_close(v,ours.state_dict()[k],rtol=0,atol=0)
        x=torch.randn(2,3,224,224)
        with torch.no_grad():
            a=native({'image':x})['cls']
            b=ours(x)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        y=torch.tensor([0,1])
        torch.testing.assert_close(native.get_losses({'label':y},{'cls':b})['overall'],F.cross_entropy(b,y))

    def test_02_sfd_reference(self):
        tree=ast.parse((d.NATIVE/'SFD.py').read_text())
        nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SoftNearestNeighborLoss']
        # The only reference adaptation is device portability of .cuda().
        code=ast.unparse(ast.Module(body=nodes,type_ignores=[])).replace('.cuda()', '.to(embeddings.device)')
        ns={'torch':torch,'nn':nn}
        exec(code,ns)
        criterion=ns['SoftNearestNeighborLoss']()
        x=torch.randn(7,512,3,3)
        g=torch.tensor([0,0,1,2,2,3,4])
        expected=torch.stack([criterion(x[:,k].flatten(1),g) for k in range(512)])
        torch.testing.assert_close(d.snnl(x,g),expected,rtol=3e-6,atol=3e-7)
        m=d.Detector().eval()
        image=torch.randn(2,3,224,224)
        with torch.no_grad():
            actual=d.sfd_features(m,image)
            self.assertEqual(actual.shape,(2,512,7,7))

    def test_03_gradients_bn_and_reload(self):
        d.seed(17)
        model=d.Detector().train()
        before=model.backbone.conv1.weight.detach().clone()
        x=torch.randn(2,3,224,224)
        opt=torch.optim.SGD(model.parameters(),lr=.001)
        F.cross_entropy(model(x),torch.tensor([0,1])).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        opt.step()
        self.assertFalse(torch.equal(before,model.backbone.conv1.weight))
        model.zero_grad(set_to_none=True)
        model.policy(True).train()
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),1026)
        frozen={k:v.clone() for k,v in model.state_dict().items() if 'last_linear' not in k}
        opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001)
        F.cross_entropy(model(x),torch.tensor([0,1])).backward()
        opt.step()
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in frozen.items()))
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if 'last_linear' not in n))
        d.mask_channels(model,[1,3,9])
        state=model.state_dict()
        self.assertEqual(sum(p.numel() for p in model.parameters()),21858090)
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'checkpoint.pt'
            torch.save(state,f)
            copy_model=d.Detector()
            d.mask_channels(copy_model,[1,3,9])
            copy_model.load_state_dict(torch.load(f,weights_only=True))
            model.eval();copy_model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(x),copy_model(x),rtol=0,atol=0)

    def test_04_ot_class_conditioning_and_lambda(self):
        y=torch.tensor([0,0,1,1])
        z=torch.tensor([[0.,-2.],[0.,-1.],[0.,1.],[0.,2.]],requires_grad=True)
        a=torch.tensor([[0.,1.],[0.,2.],[0.,-2.],[0.,-1.]],requires_grad=True)
        fair=d.ot_loss(z,y,a,y,5e-4)
        fair.backward()
        self.assertGreater(float(z.grad.abs().sum()+a.grad.abs().sum()),0)
        self.assertTrue(torch.isfinite(z.grad).all())
        # Same marginal scores, swapped class distributions: positive penalty.
        self.assertGreater(float(fair.detach()),.05)
        gradients=[]
        for weight in (0.,.005):
            zz=z.detach().clone().requires_grad_()
            aa=a.detach().clone().requires_grad_()
            loss=F.cross_entropy(zz,y)+weight*d.ot_loss(zz,y,aa,y,5e-4)
            loss.backward()
            gradients.append(zz.grad)
        self.assertFalse(torch.equal(*gradients))
        constant=torch.zeros(4,2,requires_grad=True)
        self.assertTrue(torch.isfinite(d.ot_loss(constant,y,constant,y,5e-4)))

    def test_05_native_sampler_roles_and_alignment(self):
        rows=[{'label':y,'native_group':g,'image':str(i)} for g in range(8) for y in (0,1) for i in range(3)]
        for aux in (False,True):
            batches=[list(d.NativeBatches(rows,10,17,r,2,aux)) for r in range(2)]
            seen=[]
            for rank_batches in zip(*batches):
                keys=[k for b in rank_batches for k in b]
                main=[i for i,role in keys if role==0]
                self.assertEqual(sum(rows[i]['label']==0 for i in main),sum(rows[i]['label']==1 for i in main))
                seen.extend(i for i in main if rows[i]['label']==1)
                if aux:
                    aa=[i for i,role in keys if role==1]
                    self.assertEqual(len(aa),20)
                    self.assertEqual(len({rows[i]['native_group'] for i in aa}),1)
                    self.assertEqual(sum(rows[i]['label']==0 for i in aa),10)
            self.assertEqual(sorted(seen),[i for i,r in enumerate(rows) if r['label']==1])

    def test_06_global_uneven_gradients(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port=sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'pass.txt'
            mp.spawn(gather_worker,args=(port,str(f)),nprocs=2,join=True)
            self.assertIn('PASS',f.read_text())

    def test_07_calibration_finite_ties_no_test(self):
        real=np.array([0.,0.,1.,1.,2.,2.,3.,3.,4.,4.])
        cal=d.common.calibrate(real,.1)
        self.assertEqual(cal['achieved_fp'],0)
        self.assertEqual(cal['allowed_fp'],1)
        self.assertGreater(cal['threshold_logit'],4.)
        self.assertEqual(d.common.calibrate(np.arange(10),.1)['achieved_fp'],1)
        self.assertFalse(cal['test_fpr_guaranteed'])
        tree=ast.parse(Path(d.__file__).read_text())
        functions={n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        self.assertNotIn("splits['test']",ast.unparse(functions['main']))
        self.assertNotIn('test',ast.unparse(functions['validation']))

    def test_08_shared_pool_reservations_are_respected(self):
        operator=d.module(d.ROOT/'CES_FL/scripts/run_ding_ff2_ssh.py','ding_ssh_operator_test')
        with tempfile.TemporaryDirectory() as td:
            operator.LOCKS=Path(td)
            shared=Path(td)/'gpu0';shared.mkdir()
            (shared/'pool_owner.json').write_text(json.dumps({'shared':True}))
            (shared/'task_other.json').write_text(json.dumps(dict(pid=os.getpid(),pgid=os.getpgrp(),need=30000)))
            exclusive=Path(td)/'gpu1';exclusive.mkdir()
            (exclusive/'pool_owner.json').write_text(json.dumps({'shared':False}))
            def output(cmd,**kwargs):
                if any('--query-gpu' in s for s in cmd):
                    return '0, GPU-0, 50000, 30000\n1, GPU-1, 70000, 10000\n2, GPU-2, 30000, 50000\n'
                return f'GPU-0, {os.getpid()}, 10000\n'
            before=(shared/'task_other.json').read_bytes()
            with patch.object(operator.subprocess,'check_output',side_effect=output):
                choices,cards=operator.snapshot(20000)
                self.assertEqual({c['index'] for c in choices},{0,2})
                self.assertEqual(cards[0]['available_mib'],30000)
                choices,_=operator.snapshot(28000)
                self.assertEqual(choices,[])
            self.assertEqual(before,(shared/'task_other.json').read_bytes())


if __name__=='__main__':
    unittest.main(verbosity=2)
