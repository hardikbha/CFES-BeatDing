"""Disposable TRAIN-only CUDA diagnostic for the one hierarchical-real config."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch


def main():
    p=argparse.ArgumentParser()
    for name in ('source','run','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();root=Path(a.run);out=Path(a.out);out.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('hier_real_cuda',Path(a.source)/'CES_FL/src/ces_fl_forensic_response.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    c=m.load_config(root/'H_joint_real/config.json');rows,_,_=m.checked_inputs(c,root/'H_joint_real/preflight')
    active,_=m.group_inventory(rows,c);aux=m.auxiliary(rows,active,c,c['axes'].index('gender_race'),0)
    device=torch.device('cuda');assert torch.cuda.is_available() and torch.cuda.device_count()==1
    model=m.load_initial(c).to(device)
    real=sorted([r for r in rows if r['label']==0],key=lambda r:m.seed_for(17,'train-calibration',r['image']))[:c['train_calibration_size']]
    threshold=m.calibrate(m.predict(model,real,c,device),c['target_fpr'])
    model.train()
    buffers={k:v.clone() for k,v in model.named_buffers()}
    with m.autocast(device,c):z=model(aux['x'].to(device))['logit']
    y,g=aux['y'].to(device),aux['group'].to(device)
    loss=m.hierarchical_real_error(z,y,g,aux['priors'],aux['hierarchy'],threshold['threshold_logit'],c['temperature'])
    old=m.soft_error_disparity(z,y,g,aux['priors'],threshold['threshold_logit'],c['temperature'])
    # Independently remove the unchanged fake summand to isolate the changed real term.
    pp=torch.sigmoid((z.float()-threshold['threshold_logit'])/c['temperature'])
    fn=torch.stack([(1-pp[(g==j)&(y==1)]).mean() for j in range(len(aux['groups']))])
    fake_ref=(fn*aux['priors'][:,1].to(device)).sum()
    real_loss=loss-.5*(fn-fake_ref).abs().mean()
    new_z_grad=torch.autograd.grad(loss,z,retain_graph=True)[0]
    old_z_grad=torch.autograd.grad(old,z,retain_graph=True)[0]
    torch.testing.assert_close(new_z_grad[y==1],old_z_grad[y==1],atol=2e-6,rtol=2e-5)
    assert (new_z_grad[y==0]>=0).all() and new_z_grad[y==0].abs().sum()>0
    params=[p for p in model.parameters() if p.requires_grad]
    gs=torch.autograd.grad(real_loss,params,retain_graph=True,allow_unused=True)
    norm=sum(float(g.float().square().sum()) for g in gs if g is not None)**.5
    assert norm>0 and all(g is None or torch.isfinite(g).all() for g in gs)
    assert all(torch.equal(v,dict(model.named_buffers())[k]) for k,v in buffers.items())
    payload=dict(status='PASS',source_sha256=m.sha(m.__file__),config_digest=m.digest(c),device='cuda',
                 actual_auxiliary_axis=aux['axis'],threshold_source='fixed natural TRAIN-real subset',
                 train_calibration=threshold,real_component_loss=float(real_loss.detach()),
                 real_component_gradient_norm=norm,fake_logits_gradients_match_D=True,
                 all_real_logit_gradients_nonnegative=True,BN_buffers_unchanged=True,
                 groups=aux['groups'],parents=aux['hierarchy']['parents'],
                 TRAIN_class_priors=aux['priors'].tolist(),
                 TRAIN_within_gender_real_priors=aux['hierarchy']['within_parent_real_priors'].tolist(),
                 optimizer_steps=0,validation_gradients=False,test_read=False,parameters=model.counts())
    m.write_json(out/'REAL_TRAIN_CHECK.json',payload)
    print(json.dumps(payload),flush=True)


if __name__=='__main__':
    torch.set_num_threads(4)
    main()
