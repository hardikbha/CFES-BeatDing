"""Aggregate existing TRAIN/SELECT history; no job launch or TEST/CAL access."""
import csv
import json
from pathlib import Path
from datetime import datetime,timezone
import time

ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_response_tuning_20261110_r1')


def main():
    jobs=json.loads((ROOT/'QUEUE.json').read_text())['jobs'];rows=[];observed=datetime.now(timezone.utc).isoformat()
    for j in jobs:
        p=ROOT/j['name']/'training/history.json';h=json.loads(p.read_text()) if p.exists() else []
        best=max(h,key=lambda r:r['validation']['overall']['auc']) if h else None
        row=dict(name=j['name'],seed=17,state=j['state'],stage=j.get('stage'),host=j.get('host'),GPU=j.get('gpu',{}).get('index'),epochs=len(h),
            selected_so_far=best['epoch'] if best else None,selected_after_response=best['epoch']>=3 if best else None,
            VAL_AUC_pct=100*best['validation']['overall']['auc'] if best else None,
            VAL_FNR_pct=100*best['validation']['overall']['fnr'] if best else None,
            VAL_joint_100F_FPR=100*best['validation']['axes']['gender_race']['F_FPR'] if best else None)
        rows.append(row)
    temp=ROOT/('STATUS.csv.tmp.'+str(__import__('os').getpid()))
    with temp.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    temp.replace(ROOT/'STATUS.csv')
    text=['# Seed17 response-strength screen','', 'Observed management UTC: '+observed,'',
        '| Candidate | State | Stage | Node/GPU | Epochs | Selected epoch | VAL AUC % | VAL FNR % | Joint100F_FPR |',
        '|---|---|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        def value(k):return f'{r[k]:.4f}' if isinstance(r[k],float) else str(r[k] if r[k] is not None else '--')
        text.append(f"| {r['name']} | {r['state']} | {r['stage']} | {r['host']}/{r['GPU']} | {r['epochs']}/10 | {value('selected_so_far')} | {value('VAL_AUC_pct')} | {value('VAL_FNR_pct')} | {value('VAL_joint_100F_FPR')} |")
    text+=['','SELECT/validation only. No new TEST or CAL inference. A pre-activation selected checkpoint does not demonstrate response-learning benefit. All rows remain exploratory after historical TEST exposure.','']
    p=ROOT/('STATUS.md.tmp.'+str(__import__('os').getpid()));p.write_text('\n'.join(text));p.replace(ROOT/'STATUS.md')
    print(json.dumps(dict(observed_management_utc=observed,states={s:sum(r['state']==s for r in rows) for s in ('PREPARED','RUNNING','COMPLETED_VERIFIED','FAILED')})))


if __name__=='__main__':main()
