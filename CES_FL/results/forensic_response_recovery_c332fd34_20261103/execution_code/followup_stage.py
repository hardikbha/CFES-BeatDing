"""Explicit follow-on checks/control/resume worker for the authorized A-D study."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('direct_fr',ROOT/'direct_stage.py')
direct=importlib.util.module_from_spec(spec);spec.loader.exec_module(direct)


def root_for(dataset):
    return Path('/home/sachin.chaudhary/ces_fl_runs')/f'forensic_response_overlapfix_20261103_r1_{dataset}_s17'


def config(run):
    c=json.loads((run/'config.json').read_text())
    digest=__import__('hashlib').sha256(json.dumps(c,sort_keys=True,allow_nan=False).encode()).hexdigest()
    pre=json.loads((run/'preflight/PREFLIGHT.json').read_text())
    if pre['status']!='PASS' or pre['source_sha256']!=direct.SHA or pre['config_digest']!=digest:
        raise RuntimeError('Fresh matching CPU preflight required')
    return c,digest


def verified(run,arm):
    out=run/arm
    if not (out/'DONE').exists():return False
    _,digest=config(run);r=json.loads((out/'RUN_VERIFIED.json').read_text())
    if (r['status']!='COMPLETED_VERIFIED' or r['epochs_completed']!=10 or r['source_sha256']!=direct.SHA
        or r['config_digest']!=digest or r['synthetic_fixture'] or r['test_evaluated']
        or r['selected_checkpoint_sha256']!=direct.budget.sha(out/'best.pt')):
        raise RuntimeError('Completion/provenance/checkpoint mismatch: '+str(out))
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['ffpp','ff2'],required=True)
    p.add_argument('--mode',choices=['checks','arm'],required=True);p.add_argument('--arm',choices=list('ABCD'))
    p.add_argument('--resume',action='store_true');p.add_argument('--budget-hours',type=float)
    a=p.parse_args();run=root_for(a.dataset);c,digest=config(run)
    label=a.dataset+'_checks_all' if a.mode=='checks' else a.dataset+'_'+a.arm+('_resume'+str(a.budget_hours) if a.resume else '_initial')
    lock=(ROOT/(label+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    base=['--config',str(run/'config.json'),'--preflight',str(run/'preflight')]
    if a.mode=='checks':
        commands=[(['smoke','--device','cuda','--out',str(run/'software_cuda')],run/'software_cuda.log')]
        for arm in 'ABCD':commands.append((['dry-run',*base,'--arm',arm,'--device','cuda','--out',str(run/f'dry_{arm}')],run/f'dry_{arm}.log'))
        direct.stage(label,1.,commands)
        return
    if not a.arm:raise RuntimeError('Explicit arm required')
    if verified(run,a.arm):direct.log('SKIP_VERIFIED_COMPLETED',dataset=a.dataset,arm=a.arm);return
    smoke=json.loads((run/'software_cuda/SMOKE_RESULT.json').read_text())
    dry=json.loads((run/f'dry_{a.arm}/DRY_RUN.json').read_text())
    if smoke['status']!='PASS' or not smoke['cuda_tested'] or smoke['source_sha256']!=direct.SHA:
        raise RuntimeError('Matching CUDA smoke required')
    if dry['status']!='PASS' or dry['source_sha256']!=direct.SHA or dry['config_digest']!=digest or dry['device']!='cuda':
        raise RuntimeError('Matching real CUDA backward receipt required')
    out=run/a.arm
    cmd=['train',*base,'--arm',a.arm,'--device','cuda','--software-receipt',str(run/'software_cuda/SMOKE_RESULT.json'),
         '--dry-run-receipt',str(run/f'dry_{a.arm}/DRY_RUN.json'),'--out',str(out)]
    qhash=None
    if a.arm in 'CD':
        direct.readiness(run)
        qhash=direct.budget.sha(run/'qualification.json')
        cmd+=['--qualification',str(run/'qualification.json')]
    reserve_hours=3.5
    if a.resume:
        if not a.budget_hours:raise RuntimeError('Logged cumulative resume allowance required')
        import torch
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
        if ck['source_sha256']!=direct.SHA or ck['config']!=c or ck['arm']!=a.arm or ck['qualification_sha256']!=qhash:
            raise RuntimeError('Incompatible resume checkpoint; preserve it')
        previous=ck['elapsed_seconds']/3600
        if a.budget_hours<=previous:raise RuntimeError('Resume allowance already exhausted')
        reserve_hours=max(.75,a.budget_hours-previous+.5)
        cmd+=['--resume','--budget-hours',str(a.budget_hours)]
        direct.budget.write(ROOT/(label+'_PLAN.json'),dict(source_sha256=direct.SHA,config_digest=digest,
                  arm=a.arm,dataset=a.dataset,completed_epochs=ck['epoch'],planned_epochs=10,
                  recorded_training_hours=previous,new_cumulative_training_allowance=a.budget_hours,
                  conservative_additional_gpu_reservation_hours=reserve_hours,test_tuning=False))
        del ck
    elif out.exists():
        raise RuntimeError('Existing unverified arm must be inspected/resumed, never overwritten')
    direct.stage(label,reserve_hours,[(cmd,run/(label+'.log'))])
    direct.log('SCIENTIFIC_ARM_STATE',dataset=a.dataset,arm=a.arm,
               state='COMPLETED_VERIFIED' if verified(run,a.arm) else 'INCOMPLETE_EPOCH_BOUNDARY_STOP')


if __name__=='__main__':main()
