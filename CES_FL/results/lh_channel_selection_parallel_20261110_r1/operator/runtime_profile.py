"""Timing-only profile after masks freeze; VAL logits are discarded, never scored."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import torch

parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
root=Path(args.run);source=Path('/home/sachin.chaudhary/ces_fl_sources/lh-channel-selection-1a6d63db6baf-20261110/CES_FL/src/ces_fl_lh_channel_selection.py')
spec=importlib.util.spec_from_file_location('profile_pinned_LH',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
torch.set_num_threads(4);device=torch.device('cuda');m.gpu_binding(device)
mask=root/'channel_audit/MASKS.json';assert mask.is_file();frozen_hash=m.sha(mask)
c=m.load_config(root/'configs/none.json');start=time.perf_counter();rows,val,_=m.checked_inputs(c,root/'preflight');model=m.load_initial(c).to(device);torch.cuda.synchronize();setup=time.perf_counter()-start
real=sorted([r for r in rows if r['label']==0],key=lambda r:m.seed_for(c['seed'],'train-calibration',r['image']))[:2048]
start=time.perf_counter();scores=m.predict(model,real,c,device);torch.cuda.synchronize();cal=time.perf_counter()-start;del scores
start=time.perf_counter();scores=m.predict(model,val,c,device);torch.cuda.synchronize();vt=time.perf_counter()-start;del scores
assert m.sha(mask)==frozen_hash
m.write_json(args.out,dict(status='PASS',source_sha256=m.sha(source),MASKS_sha256=frozen_hash,setup_seconds=setup,TRAIN_real_calibration_forward_seconds=cal,VAL_forward_seconds=vt,VAL_rows=len(val),TRAIN_real_calibration_rows=len(real),validation_metrics_computed=False,validation_logits_retained=False,masks_selected_before_VAL=True,TEST_CAL_access=False))
print(Path(args.out).read_text(),flush=True)
