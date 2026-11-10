"""Portable stdlib-only verification. No datasets, models, GPUs or private rows."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
ARMS=('none','random','informed')


def read(name):return json.loads((ROOT/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-6),(a,b)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def main():
    mask=read('MASKS.json');configs={a:read('configs/'+a+'.json') for a in ARMS}
    assert mask['validation_scores_used'] is False and mask['test_access'] is False
    assert mask['utility_rows']==1024 and mask['review_cases']==344
    assert mask['qualification_sha256']==sha(ROOT/'QUALIFIED.json')
    assert mask['source_sha256']=='95574295b76f16f5e5d8f70f01502848b4616fc7437030a5dfdf2ff1219f62f0'
    base=None
    for arm,c in configs.items():
        assert c['feature_selection']['mode']==arm
        assert c['feature_selection']['mask_sha256']==sha(ROOT/'MASKS.json')
        assert c['epochs']==3 and c['batch_size']==32 and c['max_gpu_hours']==.5
        assert c['lambda_fair']==c['lambda_sensitivity']==c['lambda_alignment']==.1
        assert c['init_sha256']=='cce4476491829702d60a74ed7b2044aa8436768d6e08e0129da5a1fd84b0cb6d'
        dry=read('DRY_'+arm+'.json')
        assert dry['status']=='PASS' and dry['config_digest']==digest(c)
        assert dry['optimizer_steps_in_discarded_model']==1 and dry['arm']=='D'
        assert dry['parameters']['masked_channels']==(0 if arm=='none' else 41)
        assert all(math.isfinite(v) for v in dry['losses'].values())
        context=json.loads(json.dumps(c))
        context['feature_selection'].update(mode='audit',mask_file=None,mask_sha256=None)
        if base is None:base=context
        assert context==base
    assert digest(base)==mask['context_digest']
    assert len(mask['eligible'])==1024 and len(mask['utility'])==len(mask['score'])==2048
    assert mask['eligible']==sorted(range(2048),key=lambda i:(mask['utility'][i],i))[:1024]
    assert not mask['removed']['none']
    for arm in ('random','informed'):
        r=mask['removed'][arm]
        assert len(r)==len(set(r))==41 and set(r)<=set(mask['eligible'])
    expected=sorted(sorted(mask['eligible'],key=lambda i:(-mask['score'][i],i))[:41])
    assert expected==mask['removed']['informed']
    profile=read('REPRESENTATIVE_PROFILE_attempt2.json');gate=read('REFINED_COST_GATE_attempt2.json')
    timing=read('RUNTIME_PROFILE.json');assert profile['status']=='PASS'
    assert profile['MASKS_sha256']==sha(ROOT/'MASKS.json') and profile['VAL_scores_used'] is False
    assert gate['representative_profile_sha256']==sha(ROOT/'REPRESENTATIVE_PROFILE_attempt2.json')
    assert timing['validation_metrics_computed'] is False and timing['validation_logits_retained'] is False
    steps=math.ceil(218266/32)*3;assert steps==gate['steps_per_arm']==20463
    for arm,r in profile['arms'].items():
        assert arm in ARMS and r['config_digest']==digest(configs[arm])
        assert r['warmup_steps']==40 and r['measured_steps']==80
        assert r['optimizer_steps_in_discarded_model']==120 and r['training_checkpoint_saved'] is False
        assert set(r['auxiliary_events_by_axis'].values())=={4} and len(r['auxiliary_events_by_axis'])==5
        assert r['original_samples_measured']==2560 and r['real_pairs_measured']==194
        close(sum(r['block_mean_step_seconds'])/4,r['mean_seconds_per_step'])
        close(max(r['block_mean_step_seconds'])*1.10,r['conservative_step_seconds'])
        overhead=r['input_setup_seconds']+r['model_setup_seconds']+3*r['train_real_calibration_seconds']+4*timing['VAL_forward_seconds']+60
        close(1.10*(steps*r['conservative_step_seconds']+overhead),gate['estimates_seconds'][arm])
        assert steps*r['mean_seconds_per_step']>1800
    assert gate['status']=='BLOCKED_RUNTIME_FIT' and gate['production_training_started'] is False
    ledger=read('FINAL_BUDGET_VERIFICATION.json');charges=sum(x['elapsed_seconds']*x['ngpus']/3600 for x in ledger['sessions'])
    close(charges,ledger['cohort_GPU_hours']);assert charges<2 and ledger['cohort_reserved_GPU_hours']==0
    release=read('FINAL_PROCESS_AND_RELEASE_VERIFICATION.json')
    assert release['diagnostic_GPU_leases_released'] and not release['production_training_directories_present']
    assert not any(x['alive'] for x in release['controllers']) and not any(release['lease_files_present'].values())
    result=read('GATE_STATUS.json');assert result['status']=='BLOCKED_RUNTIME_FIT'
    assert all(x=='NOT_STARTED' for x in result['production_training'].values())
    close(result['conservative_projected_total_GPU_hours_including_checks'],sum(gate['estimates_seconds'].values())/3600+charges)
    assert result['TEST_CAL_access'] is False and result['new_Ding_experiments'] is False
    manifest=ROOT/'SHA256.json'
    if manifest.exists():
        for relative,h in read('SHA256.json').items():assert sha(ROOT/relative)==h,relative
    print(json.dumps(dict(status='PASS',matched_configurations=3,real_input_backward_receipts=3,
        fixed_mask_counts=[0,41,41],representative_discarded_updates=360,verified_runtime_gate='BLOCKED_RUNTIME_FIT',
        production_runs_started=0,cohort_GPU_hours=charges,models_executed=False,private_predictions_read=False),indent=2))


if __name__=='__main__':main()
