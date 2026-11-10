"""Fixed seed17 response-schedule screen. TRAIN/SELECT only; no TEST command."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import time

import torch

SOURCE=Path(__file__).resolve().parent
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_response_tuning_20261110_r1')
spec=importlib.util.spec_from_file_location('schedule_parent',SOURCE/'src/shared.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
core=base.core
VERSION='FF2_RESPONSE_SCHEDULE_TUNING_v1'
CEILINGS={'fair_only':0.,'smooth_005':.05,'smooth_010':.10,'smooth_020':.20}
PARENT_OBJECTIVE=base.method_objective
PARENT_RESPONSE=base.response_factor


def response_factor(c,epoch):
    s=c['response_schedule']
    if not isinstance(epoch,int) or epoch<0:raise ValueError('Nonnegative zero-based epoch required')
    if epoch<s['delay_epochs']:return 0.
    t=min(1.,(epoch-s['delay_epochs']+1)/s['ramp_epochs'])
    return s['ceiling']*t*t*(3-2*t)


def objective(c,z,y,groups,fair,sens,sham,align,gda,has_aux,epoch):
    result=PARENT_OBJECTIVE(c,z,y,groups,fair,sens,sham,align,gda,has_aux,epoch)
    raw=c['lambda_sensitivity']*sens+c['lambda_sensitivity']*c['sham_weight']*sham+c['lambda_alignment']*align
    result.update(raw_response=raw,weighted_response=response_factor(c,epoch)*raw,
        weighted_fairness=c['aux_every']*c['lambda_fair']*fair,
        response_multiplier=z.float().new_tensor(response_factor(c,epoch)))
    return result


def configure():
    base.ROOT=ROOT
    base.response_factor=response_factor
    base.method_objective=objective
    base.configure()


def check(c):
    base.check(c)
    if c.get('tuning_format')!=VERSION or c.get('tuning_name') not in CEILINGS:raise ValueError('Unregistered schedule')
    if c.get('tuning_source_sha256')!=core.sha(__file__):raise ValueError('Tuning source changed')
    expected=dict(delay_epochs=2,ramp_epochs=6,ceiling=CEILINGS[c['tuning_name']],shape='smoothstep')
    if c.get('response_schedule')!=expected:raise ValueError('Schedule changed; preserve the fixed screen')
    if c['seed']!=17 or c['suite_method']!='lh' or c['epochs']!=10 or c['max_gpu_hours']!=4.:
        raise ValueError('Seed17 ten-epoch full-L_H screen required')
    if any(k in c for k in ('test','test_sha256','sfd_file')):raise ValueError('No TEST/CAL or Ding dependency belongs in this screen')
    return c


def dry(c,preflight,qualification,out,device):
    check(c);configure();out=Path(out);out.mkdir(exist_ok=False)
    started=time.monotonic()
    base.dry(c,preflight,qualification,out/'INITIAL_CUDA.json',device)
    initial=json.loads((out/'INITIAL_CUDA.json').read_text())
    rows,_,_=core.checked_inputs(c,preflight);active,_=core.group_inventory(rows,c)
    model=base.initialise(c).to(device);opt=core.build_optimizer(model,c)
    epoch=7
    indices=[next(i for i,r in enumerate(rows) if r['label']==0)]+list(range(31))
    ds=base.CommonOriginals(rows,c,epoch=epoch,pairs=True);samples=[ds[i] for i in indices]
    b={k:torch.stack([s[k] for s in samples]) if torch.is_tensor(samples[0][k]) else torch.tensor([s[k] for s in samples]) for k in samples[0]}
    aux=core.auxiliary(rows,active,c,0,epoch)
    reals=sorted([r for r in rows if r['label']==0],key=lambda r:core.seed_for(c['seed'],'train-calibration',r['image']))[:2048]
    threshold=core.calibrate(core.predict(model,reals,c,device),.1)['threshold_logit'];model.train()
    losses,_=base.train_step(model,b,aux,torch.ones(len(rows)),c,'D',threshold,device,epoch)
    params=[p for p in model.parameters() if p.requires_grad]
    def norm(loss):
        gradients=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        return math.sqrt(sum(float(g.float().square().sum()) for g in gradients if g is not None))
    cls_norm=norm(losses['classification']);response_norm=norm(losses['weighted_response'])
    core.finite_step(losses,model,opt)
    if device.type=='cuda':torch.cuda.synchronize(device)
    report=dict(status='PASS',tuning_name=c['tuning_name'],seed=17,tuning_source_sha256=core.sha(__file__),
        config_digest=core.digest(c),device=str(device),parameters=initial['parameters'],
        initial_parameter_fingerprint=initial['initial_parameter_fingerprint'],
        initial_forward_input_fingerprint=initial['data_forward_fingerprint'],bn_frozen=initial['bn_frozen'],
        active_epoch_one_based=8,active_response_multiplier=response_factor(c,epoch),
        classification_gradient_norm=cls_norm,response_gradient_norm=response_norm,
        response_to_classification_gradient_ratio=response_norm/cls_norm,
        active_losses={k:float(v.detach()) for k,v in losses.items()},
        optimizer_steps_in_discarded_models=2,elapsed_seconds=time.monotonic()-started,
        recipe_changed=False,TEST_CAL_access=False)
    if c['response_schedule']['ceiling']==0 and response_norm!=0:raise RuntimeError('Zero-response control has response gradient')
    if c['response_schedule']['ceiling']>0 and not response_norm>0:raise RuntimeError('Active response term has no gradient')
    core.write_json(out/'CUDA_BACKWARD.json',report);print(json.dumps(report),flush=True)


def train(c,preflight,qualification,out,device):
    check(c);configure()
    result=base.train(c,preflight,out,qualification,device)
    if result['status']!='COMPLETED_VERIFIED':raise RuntimeError('Ten complete epochs and selected-checkpoint replay required')
    result.update(tuning_format=VERSION,tuning_name=c['tuning_name'],tuning_source_sha256=core.sha(__file__),
        response_schedule=c['response_schedule'],selection='earliest maximum SELECT AUC across all ten epochs',
        TEST_CAL_access=False,checkpoint_selected_after_response_activation=result['selected_epoch']>=3)
    core.write_json(Path(out)/'TUNING_VERIFIED.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('dry','train'));p.add_argument('--config',required=True)
    p.add_argument('--preflight',required=True);p.add_argument('--qualification',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda');a=p.parse_args()
    c=json.loads(Path(a.config).read_text());check(c);torch.set_num_threads(2);device=torch.device(a.device)
    if device.type!='cuda':raise RuntimeError('Production dry/train requires an explicit allocated CUDA device')
    if a.stage=='dry':dry(c,a.preflight,a.qualification,a.out,device)
    else:train(c,a.preflight,a.qualification,a.out,device)


if __name__=='__main__':main()
