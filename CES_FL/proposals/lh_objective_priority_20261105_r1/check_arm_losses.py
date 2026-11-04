#!/usr/bin/env python3
"""Independent synthetic assertions for the exact B/C/D native arm semantics."""
import importlib.util,json
from pathlib import Path
import torch
from torch import nn
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('arm_source',HERE.parents[1]/'src/ces_fl_nuisance_resnet50.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class Probe(nn.Module):
    def __init__(self):super().__init__();self.fc=nn.Linear(3,1);self.calls=[]
    def forward(self,x):self.calls.append(x.detach().clone());return {'logit':self.fc(x.mean((2,3))).squeeze(1)}

if __name__=='__main__':
    torch.manual_seed(17);torch.set_num_threads(2)
    c=json.loads((HERE/'configs/B0.json').read_text())
    assert c==json.loads((HERE/'configs/C0.json').read_text())==json.loads((HERE/'evidence/H0_reference_config.json').read_text())
    assert c['source_sha256']==m.sha(m.__file__)
    x=torch.randn(4,3,8,8);y=torch.tensor([0,1,0,1]);b={'x':x,'y':y,'index':torch.arange(4),'s':x+.01,'t':x+.2}
    aux={'x':x*.5,'y':y,'group':torch.tensor([0,0,1,1]),'priors':torch.ones(2,2)/2,'axis':'gender',
         's':x[[0,2]]*.5+.01,'t':x[[0,2]]*.5+torch.tensor([.1,.3])[:,None,None,None]}
    for arm in ('B','C'):
        model=Probe();loss,count=m.train_step(model,b,aux,torch.ones(4),c,arm,0.,torch.device('cpu'))
        expected=loss['classification']+.4*loss['error_fairness']
        if arm=='B':
            assert count==0 and len(model.calls)==2
            torch.testing.assert_close(model.calls[0],b['x']);torch.testing.assert_close(model.calls[1],aux['x'])
            assert all(loss[k].item()==0 for k in ('original_sensitivity','aux_sensitivity','original_control_stability','aux_control_stability','response_alignment'))
        else:
            expected=expected+.1*(loss['original_sensitivity']+.5*loss['original_control_stability'])+.4*(loss['aux_sensitivity']+.5*loss['aux_control_stability'])
            assert count==2 and len(model.calls)==6 and loss['response_alignment'].item()==0
        torch.testing.assert_close(loss['total'],expected)
        loss['total'].backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters())
        print(arm,'PASS: exact objective and forward-use semantics; finite gradients')
    print('PASS: scientific source/configs identical to pinned H0; production inputs/CUDA remain cluster checks')
