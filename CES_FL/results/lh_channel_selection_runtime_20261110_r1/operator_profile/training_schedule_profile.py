"""Discarded fixed-mask timing of the registered original/auxiliary stream."""
import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

SOURCE=Path('/home/sachin.chaudhary/ces_fl_sources/lh-channel-selection-1a6d63db6baf-20261110/CES_FL/src/ces_fl_lh_channel_selection.py')
spec=importlib.util.spec_from_file_location('registered_LH',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/lh_channel_selection_20261110_r1_operator_1a6d63d')
WARMUP=40
MEASURE=80


def run(out):
    torch.set_num_threads(4);device=torch.device('cuda');m.gpu_binding(device)
    assert m.sha(SOURCE)=='95574295b76f16f5e5d8f70f01502848b4616fc7437030a5dfdf2ff1219f62f0'
    mask=ROOT/'channel_audit/MASKS.json';frozen_hash=m.sha(mask)
    common=m.load_config(ROOT/'configs/none.json');m.check_qualification(common,ROOT/'QUALIFIED.json')
    started=time.perf_counter();rows,val,_=m.checked_inputs(common,ROOT/'preflight');active,support=m.group_inventory(rows,common)
    input_setup=time.perf_counter()-started
    results={}
    for arm in ('none','random','informed'):
        c=m.load_config(ROOT/'configs'/f'{arm}.json');m.check_qualification(c,ROOT/'QUALIFIED.json')
        m.seed_all(c['seed']);start=time.perf_counter();model=m.load_initial(c).to(device);opt=m.build_optimizer(model,c)
        weights=m.original_weights(rows,c['balance_axis']);torch.cuda.synchronize();model_setup=time.perf_counter()-start
        real=sorted([r for r in rows if r['label']==0],key=lambda r:m.seed_for(c['seed'],'train-calibration',r['image']))[:c['train_calibration_size']]
        start=time.perf_counter();z=m.predict(model,real,c,device);torch.cuda.synchronize();calibration_seconds=time.perf_counter()-start
        threshold=m.calibrate(z,c['target_fpr'])['threshold_logit'];del z
        model.train();m.seed_all(c['seed']);aux_iter=iter(m.auxiliary_loader(rows,active,c,0,0,math.ceil(math.ceil(len(rows)/c['batch_size'])/c['aux_every'])))
        main_iter=iter(m.make_loader(rows,c,0,pairs=True,shuffle=True))
        timings=[];groups={};main_sources=0;main_real_pairs=0;axes=[]
        torch.cuda.synchronize();epoch_start=time.perf_counter()
        for step in range(WARMUP+MEASURE):
            torch.cuda.synchronize();tick=time.perf_counter();b=next(main_iter);aux=None
            if step%c['aux_every']==0:
                aux=next(aux_iter);axes.append(aux['axis'])
            losses,pairs=m.train_step(model,b,aux,weights,c,'D',threshold,device,0)
            m.finite_step(losses,model,opt);torch.cuda.synchronize();seconds=time.perf_counter()-tick
            if step>=WARMUP:
                timings.append(seconds);main_sources+=len(b['y']);main_real_pairs+=pairs
                if aux is not None:groups[aux['axis']]=groups.get(aux['axis'],0)+1
            if step%20==19:print(json.dumps(dict(event='DISCARDED_PROFILE_PROGRESS',arm=arm,steps=step+1,last_step_seconds=seconds)),flush=True)
        blocks=np.asarray(timings).reshape(-1,20).sum(1)/20
        results[arm]=dict(status='PASS',config_digest=m.digest(c),source_sha256=m.sha(SOURCE),mask_sha256=frozen_hash,
            warmup_steps=WARMUP,measured_steps=MEASURE,optimizer_steps_in_discarded_model=WARMUP+MEASURE,
            original_samples_measured=main_sources,real_pairs_measured=main_real_pairs,auxiliary_events_by_axis=groups,
            mean_seconds_per_step=float(np.mean(timings)),block_mean_step_seconds=blocks.tolist(),
            conservative_step_seconds=float(max(blocks)*1.10),input_setup_seconds=input_setup,
            model_setup_seconds=model_setup,train_real_calibration_seconds=calibration_seconds,
            total_profile_training_seconds=time.perf_counter()-epoch_start,training_checkpoint_saved=False,
            production_updates=0,TEST_CAL_access=False,VAL_metrics_computed=False)
        assert set(groups)==set(c['axes']) and sum(groups.values())==MEASURE//c['aux_every']
        m.write_json(out,dict(status='IN_PROGRESS',masks_frozen=True,arms=results))
        del main_iter,aux_iter,model,opt,weights,losses,b,aux
        gc.collect();torch.cuda.empty_cache()
    assert m.sha(mask)==frozen_hash
    m.write_json(out,dict(status='PASS',MASKS_sha256=frozen_hash,masks_frozen=True,arms=results,
        scientific_configurations_changed=False,VAL_scores_used=False,TEST_CAL_access=False))
    print(Path(out).read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();run(a.out)
