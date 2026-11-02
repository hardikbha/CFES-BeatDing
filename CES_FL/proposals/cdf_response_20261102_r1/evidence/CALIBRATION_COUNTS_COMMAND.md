# Actual CPU heredoc command

Executed from `/home/sachin.chaudhary/ces_fl_sources/cesfl-tail-response-20261102-r1`. Exit0. This records the body actually executed through stdin, not a claim that a separate Python script was run.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/sachin.chaudhary/.conda/envs/qwen38/bin/python - <<'PY'
import json,sys,math,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('CES_FL/src').resolve()))
import ces_fl_counterfactual_5050 as base
p=Path('CES_FL/proposals/cdf_response_20261102_r1/evidence')
counts={}
for backbone in ['xception','resnet50']:
 m=base.DiagramCES(backbone,pretrained=False,encoder_policy='trainable',bn_policy='frozen',head_seed=17)
 counts[backbone]={'total':sum(v.numel() for v in m.parameters()),'trainable_downstream':sum(v.numel() for v in m.parameters() if v.requires_grad),
 'modules':{name:sum(v.numel() for v in getattr(m,name).parameters()) for name in ['encoder','global_head','patch_head','image_gate','patch_gate']},
 'fusion_parameters':m.fusion_raw.numel()+m.fusion_bias.numel(),'execution':'fresh CPU instantiation; gradient behavior covered separately by smoke'}
 m.set_encoder_policy('frozen','frozen');counts[backbone]['trainable_frozen']=sum(v.numel() for v in m.parameters() if v.requires_grad)
(p/'PARAMETER_COUNTS.json').write_text(json.dumps(counts,indent=2)+'\n')
fixtures=[]
for scores,target,expected in [([0.,1.,2.,3.,4.],.4,2),([0.,0.,0.,1.,1.],.4,2),([0.,1.,1.,1.,1.],.4,0),([5.],.1,0)]:
 result=base.calibrate(np.array(scores),target);assert result['achieved_fp']==expected
 assert int((np.array(scores)>=result['threshold_logit']).sum())==expected
 fixtures.append({'scores':scores,'target':target,'actual':result})
(p/'CALIBRATION_FIXTURES.json').write_text(json.dumps({'status':'PASS','fixtures':fixtures,'test_scores_or_labels_supplied':False},indent=2)+'\n')
print(json.dumps({'parameters':counts,'calibration_status':'PASS','achieved_fp':[v['actual']['achieved_fp'] for v in fixtures]},indent=2))
PY
```

Actual stdout's counts are in PARAMETER_COUNTS.json. The final summary was `calibration_status: PASS`, `achieved_fp: [2, 2, 0, 0]`. All four assertions were executed.
