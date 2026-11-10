"""Finite paired SELECT comparison after the four prespecified runs complete."""
import fcntl
import json
from pathlib import Path
import os

import schedule_tuning as m

ROOT=m.ROOT


def main():
    with (ROOT/'comparison.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        if (ROOT/'COMPARISON_STARTED.json').exists():return
        q=json.loads((ROOT/'QUEUE.json').read_text())
        if not all(j['state']=='COMPLETED_VERIFIED' for j in q['jobs']):return
        m.core.write_json(ROOT/'COMPARISON_STARTED.json',dict(pid=os.getpid(),scope='Existing selected SELECT scores only',TEST_CAL_access=False))
        summary=[]
        try:
            control=json.loads((ROOT/'fair_only/training/TUNING_VERIFIED.json').read_text())
            for name in ('smooth_005','smooth_010','smooth_020'):
                path=ROOT/(name+'_minus_fair_only.json')
                for f in (ROOT/name/'training/validation_scores_PRIVATE.json',ROOT/'fair_only/training/validation_scores_PRIVATE.json'):
                    r=json.loads(f.read_text());assert r['split']=='validation' and r['test_read'] is False
                m.core.compare(ROOT/name/'training/validation_scores_PRIVATE.json',ROOT/'fair_only/training/validation_scores_PRIVATE.json',path,2000,17)
                compare=json.loads(path.read_text())
                for interval in compare['intervals'].values():
                    valid=interval['valid_replicates']>=1900
                    interval['uncertainty_status']='VALID_CONDITIONAL_INTERVAL' if valid else 'INCONCLUSIVE'
                    if not valid:interval['ci95']=None
                compare.update(minimum_valid_fraction=.95,selection_uncertainty_included=False,
                    source_tuning_sha256=m.core.sha(m.__file__),scope='Selected-checkpoint conditional SELECT analysis after historical TEST exposure; no confirmatory superiority claim')
                m.core.write_json(path,compare)
                result=json.loads((ROOT/name/'training/TUNING_VERIFIED.json').read_text());v=result['validation'];baseline=control['validation']
                guards=v['overall']['auc']>=.995 and v['overall']['fnr']<=.005651
                better=v['axes']['gender_race']['F_FPR']<baseline['axes']['gender_race']['F_FPR']
                primary=compare['intervals']['gender_race/F_FPR']
                summary.append(dict(name=name,selected_epoch=result['selected_epoch'],selected_after_response=result['selected_epoch']>=3,
                    absolute_utility_guards_pass=guards,primary_disparity_below_control=better,
                    primary_conditional_uncertainty=primary['uncertainty_status'],
                    primary_difference=primary['estimate'],primary_ci95=primary['ci95'],
                    screen_point_criteria_pass=guards and better and result['selected_epoch']>=3,TEST_CAL_access=False))
            m.core.write_json(ROOT/'COMPARISON_SUMMARY.json',dict(status='COMPLETED',scope='Exploratory seed17 SELECT only',
                results=summary,no_automatic_extension=True))
        except Exception as error:
            m.core.write_json(ROOT/'COMPARISON_FAILURE.json',dict(error=str(error),no_automatic_retry=True));raise


if __name__=='__main__':main()
