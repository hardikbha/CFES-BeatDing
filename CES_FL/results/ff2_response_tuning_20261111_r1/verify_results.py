"""Verify stored SELECT receipts/exports only; no model or prediction access."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
NAMES=('fair_only','smooth_005','smooth_010','smooth_020')
CAPS=dict(zip(NAMES,(0.,.05,.1,.2)))


def read(name):return json.loads((ROOT/name).read_text())
def rows(name):
    with (ROOT/name).open(newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b,tol=1e-9):assert math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol),(a,b)


def main():
    primary={r['candidate']:r for r in rows('SELECTED_VALIDATION_RESULTS.csv')}
    history=rows('ALL_EPOCHS_SELECT.csv');assert len(history)==40
    axes=rows('SELECTED_ALL_AXES.csv');assert len(axes)==24
    reports={};fingerprint=None
    for name in NAMES:
        prefix='native/'+name+'/'
        v=read(prefix+'TUNING_VERIFIED.json');f=read(prefix+'FROZEN_CHECKPOINT.json');c=read(prefix+'CONFIG.json');dry=read(prefix+'CUDA_BACKWARD.json')
        assert v['status']=='COMPLETED_VERIFIED' and v['epochs_completed']==10
        assert v['test_evaluated'] is False and v['TEST_CAL_access'] is False
        assert c['seed']==17 and c['response_schedule']['ceiling']==CAPS[name]
        assert not any(k in c for k in ('test','test_sha256','sfd_file'))
        assert v['selected_checkpoint_sha256']==f['checkpoint_sha256']
        assert v['tuning_source_sha256']==c['tuning_source_sha256']==dry['tuning_source_sha256']
        close(v['validation']['threshold_logit'],f['threshold_logit'])
        h=[r for r in history if r['candidate']==name]
        assert [int(r['epoch']) for r in h]==list(range(1,11))
        assert all(int(r['steps'])==6821 and int(r['unique_originals'])==218266 and r['originals_once_each']=='True' for r in h)
        best=max(h,key=lambda r:float(r['AUC_pct']));assert int(best['epoch'])==v['selected_epoch']
        for r in h:
            e=int(r['epoch']);t=0. if e<=2 else min(1.,(e-2)/6)
            close(r['response_multiplier'],CAPS[name]*t*t*(3-2*t),tol=1e-8)
            if e<=2:close(r['weighted_response'],0)
        r=primary[name];assert int(r['selected_epoch'])==v['selected_epoch']
        close(r['response_multiplier_at_selection'],best['response_multiplier'],tol=1e-8)
        report=v['validation'];reports[name]=report
        for key,native in [('AUC_pct','auc'),('FPR_pct','fpr'),('FNR_pct','fnr')]:close(r[key],100*report['overall'][native])
        close(r['joint_100F_FPR'],100*report['axes']['gender_race']['F_FPR'])
        for x in [r for r in axes if r['candidate']==name]:
            g=report['axes'][x['axis']];groups=[g['groups'][n] for n in g['eligible_groups']]
            ffpr=sum(abs(y['fpr']-report['overall']['fpr']) for y in groups)
            dp=max(y['positive_rate'] for y in groups)-min(y['positive_rate'] for y in groups)
            es=report['overall']['auc']/(1+sum(abs(y['auc']-report['overall']['auc']) for y in groups))
            close(x['F_FPR_100'],100*ffpr);close(x['F_DP_100'],100*dp);close(x['es_AUC_pct'],100*es)
        p=[dry[k] for k in ('initial_parameter_fingerprint','initial_forward_input_fingerprint')]
        if fingerprint is None:fingerprint=p
        assert p==fingerprint and dry['status']=='PASS'
    summary=read('COMPARISON_SUMMARY.json');assert summary['status']=='COMPLETED'
    for s in summary['results']:
        name=s['name'];comparison=read(name+'_minus_fair_only.json');p=comparison['intervals']['gender_race/F_FPR']
        assert comparison['replicates']==2000 and p['valid_replicates']==964
        assert p['ci95'] is None and p['uncertainty_status']=='INCONCLUSIVE'
        delta=reports[name]['axes']['gender_race']['F_FPR']-reports['fair_only']['axes']['gender_race']['F_FPR']
        close(delta,p['estimate']);close(delta,s['primary_difference'])
        expected=reports[name]['overall']['auc']>=.995 and reports[name]['overall']['fnr']<=.005651 and delta<0 and int(primary[name]['selected_epoch'])>=3
        assert s['screen_point_criteria_pass']==expected
    assert [s['name'] for s in summary['results'] if s['screen_point_criteria_pass']]==['smooth_010']
    budget=read('FINAL_GPU_ACCOUNTING.json');assert budget['status']=='PASS_CLOSED_COHORT'
    close(budget['actual_GPU_hours'],sum(s['elapsed_seconds']*s['ngpus']/3600 for s in budget['sessions']))
    assert budget['actual_GPU_hours']<=17 and budget['remaining_cohort_reservations']==0
    for host in ('gpu01','gpu02'):
        release=read('FINAL_RELEASE_'+host+'.json');assert release['status']=='PASS' and all(r['lease_released'] for r in release['records'])
    record=read('VERIFICATION.json');assert len(record['checkpoint_binary_checks'])==4
    assert all(c['actual_binary_bytes_rehashed'] for c in record['checkpoint_binary_checks'])
    if (ROOT/'SHA256.json').exists():
        for name,h in read('SHA256.json').items():assert sha(ROOT/name)==h,name
    print(json.dumps(dict(status='PASS',completed_runs=4,history_epochs_checked=40,selected_axis_rows_checked=24,
        selected_active_response_point_candidate='smooth_010',primary_uncertainty='INCONCLUSIVE',
        primary_valid_bootstrap_replicates=964,required_valid_bootstrap_replicates=1900,
        actual_GPU_hours=budget['actual_GPU_hours'],new_inference=False,private_predictions_read=False),indent=2))


if __name__=='__main__':main()
