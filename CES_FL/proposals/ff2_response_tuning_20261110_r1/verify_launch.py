"""Verify the fixed screen and saved launch evidence without models or data."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CEILINGS={'fair_only':0.,'smooth_005':.05,'smooth_010':.10,'smooth_020':.20}


def read(name):return json.loads((ROOT/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-6,abs_tol=1e-8),(a,b)


def main():
    configs={n:read('configs/'+n+'.json') for n in CEILINGS};canonical=None;fingerprint=None
    for name,c in configs.items():
        assert c['tuning_name']==name and c['seed']==17 and c['epochs']==10
        assert c['suite_method']=='lh' and c['max_gpu_hours']==4.
        assert c['response_schedule']==dict(delay_epochs=2,ramp_epochs=6,ceiling=CEILINGS[name],shape='smoothstep')
        assert not any(k in c for k in ('test','test_sha256','sfd_file'))
        assert c['init_sha256']=='741a8eb5b60ac8efe2ca34be985b423de12facac053acac7f05803df23248a23'
        d=read('CUDA_checks/'+name+'.json');assert d['status']=='PASS' and d['bn_frozen']
        assert d['tuning_source_sha256']==c['tuning_source_sha256']
        assert d['parameters']==dict(total=20809001,trainable=6790433)
        assert d['optimizer_steps_in_discarded_models']==2 and d['TEST_CAL_access'] is False
        close(d['active_response_multiplier'],CEILINGS[name])
        p=[d[k] for k in ('initial_parameter_fingerprint','initial_forward_input_fingerprint')]
        if fingerprint is None:fingerprint=p
        assert p==fingerprint
        common=json.loads(json.dumps(c));common.pop('tuning_name');common['response_schedule'].pop('ceiling')
        if canonical is None:canonical=common
        assert common==canonical
    ratio=[read('CUDA_checks/'+n+'.json')['response_to_classification_gradient_ratio'] for n in CEILINGS]
    assert ratio[0]==0;close(ratio[2],ratio[1]*2);close(ratio[3],ratio[1]*4)
    with (ROOT/'FIXED_SCHEDULES.csv').open(newline='') as f:schedules=list(csv.DictReader(f))
    assert len(schedules)==40
    for r in schedules:
        epoch=int(r['epoch']);t=0. if epoch<=2 else min(1.,(epoch-2)/6)
        close(r['multiplier'],CEILINGS[r['candidate']]*t*t*(3-2*t))
    tests=read('CPU_CHECKS.json');assert tests['status']=='PASS' and tests['total_tests_passed']==21
    for n,h in tests['log_hashes'].items():assert sha(ROOT/n)==h,n
    audit=read('FRESH_INPUT_AUDIT.json')
    assert audit['status']=='PASS' and audit['decoded_images']==249373 and audit['cross_split_content_overlap']==0
    assert audit['inventory_sha256']=='9f5d4b587d5dd87fe04a7db9b353c3aabc043b892c931dfd7f2b059112276621'
    ready=read('READINESS_AND_LAUNCH_VERIFICATION.json');assert ready['status']=='ALL_FOUR_TRAINING' and ready['GPU_training_jobs']==4
    for r in ready['receipts']:
        assert sha(ROOT/'configs'/f"{r['candidate']}.json")==r['config_sha256']
        assert r['recipe_case_count']==88 and r['new_human_review_performed'] is False
    live=read('LIVE_gpu01.json')['active_records']+read('LIVE_gpu02.json')['active_records']
    assert len(live)==4 and all(r['worker_alive'] and r['training_process_group_alive'] and r['stage']=='TRAINING' for r in live)
    budget=read('BUDGET_RESERVATION.json')
    assert budget['cohort_max_GPU_hours']==17 and budget['recorded_uncommitted_before']>=17
    assert budget['additional_budget_granted'] is False and budget['budget_transfer'] is False
    source=read('SOURCE_CHECKOUT.json');local=Path(source['source_root']);checked=0
    if local.exists():
        assert (local/'.git/HEAD').read_text().strip()==source['commit']
        for n,h in source['source_files'].items():assert sha(local/n)==h,n;checked+=1
    if (ROOT/'SHA256.json').exists():
        for n,h in read('SHA256.json').items():assert sha(ROOT/n)==h,n
    print(json.dumps(dict(status='PASS',configs=4,only_scientific_difference='response ceiling',seed=17,
        CPU_tests_passed=21,actual_CUDA_checks_passed=4,matched_initial_parameters_and_inputs=True,
        recorded_live_training_process_groups=4,fresh_TRAIN_SELECT_images_checked=249373,
        tracked_execution_source_files_verified=checked,TEST_CAL_access=False,new_Ding_models=0,
        models_executed_by_this_verifier=False),indent=2))


if __name__=='__main__':main()
