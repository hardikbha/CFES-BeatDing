"""Verify stored aggregate results and receipts; no model/data/GPU dependency."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
METHODS=('bce','dag','daw','ding_sfd_gda','lh','lh_ramp','lh_low')
SEEDS=(7,17,27)
AXES=('gender','race','gender_race','age_bin','skin_bin','ids8')


def read(name):return json.loads((ROOT/name).read_text())
def rows(name):
    with (ROOT/name).open(newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-9),(a,b)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def computed_row(v,axis):
    g=v['axes'][axis];groups=[g['groups'][n] for n in g['eligible_groups']]
    assert all(x['n_real']>=20 and x['n_fake']>=20 for x in groups)
    fpr_gap=max(x['fpr'] for x in groups)-min(x['fpr'] for x in groups)
    fnr_gap=max(x['fnr'] for x in groups)-min(x['fnr'] for x in groups)
    fpr=sum(abs(x['fpr']-v['overall']['fpr']) for x in groups)
    dp=max(x['positive_rate'] for x in groups)-min(x['positive_rate'] for x in groups)
    es=v['overall']['auc']/(1+sum(abs(x['auc']-v['overall']['auc']) for x in groups))
    accuracy=[(x['n_real']*(1-x['fpr'])+x['n_fake']*(1-x['fnr']))/x['n'] for x in groups]
    close(fpr,g['F_FPR']);close(dp,g['F_DP']);close(es,g['es_AUC'])
    close(fpr_gap,g['fpr_gap']);close(fnr_gap,g['fnr_gap'])
    return dict(AUC_pct=100*v['overall']['auc'],achieved_FPR_pct=100*v['overall']['fpr'],FNR_pct=100*v['overall']['fnr'],
        F_FPR_100=100*fpr,F_DP_100=100*dp,es_AUC_pct=100*es,F_MEO_max_pp=100*max(fpr_gap,fnr_gap),
        F_OAE_pp=100*(max(accuracy)-min(accuracy)),worst_group_AUC_pct=100*min(x['auc'] for x in groups))


def main():
    recipe=read('COMMON_RECIPE.json');hist=rows('SELECTION_HISTORY.csv');selections=rows('SELECTED_CHECKPOINTS.csv')
    exported=rows('TEST_PER_SEED_ALL_AXES.csv');means=rows('TEST_PRIMARY_MEAN_SD.csv');families=rows('TEST_PER_FAMILY.csv')
    assert len(exported)==126 and len(means)==63 and len(hist)==210 and len(selections)==21
    proof={};context=None;reports={};axis_rows={}
    for seed in SEEDS:
        for method in METHODS:
            run=f'{method}_s{seed}';folder='native/'+run+'/'
            v=read(folder+'TRAINING_VERIFIED.json');f=read(folder+'FROZEN_CHECKPOINT.json')
            t=read(folder+'TEST_METRICS.json');c=read(folder+'CONFIG.json');d=read(folder+'CUDA_BACKWARD.json')
            assert v['status']=='COMPLETED_VERIFIED' and v['epochs_completed']==10 and v['test_evaluated'] is False
            assert not v['synthetic_fixture'] and not t['test_used_for_selection']
            assert v['selected_checkpoint_sha256']==f['checkpoint_sha256']==t['checkpoint_sha256']
            close(v['validation']['threshold_logit'],f['threshold_logit']);close(f['threshold_logit'],t['calibrated']['threshold_logit'])
            assert t['manifest_sha256']==recipe['test_sha256'] and t['calibrated']['overall']['n']==87564
            assert t['suite_hashes']==v['suite_hashes']==recipe['suite_hashes']
            assert digest(c)==v['config_digest']
            common=dict(c);common.pop('seed');common.pop('suite_method')
            if context is None:context=common
            assert common==context
            assert d['status']=='PASS' and d['bn_frozen']
            p={k:d[k] for k in ('initial_parameter_fingerprint','data_forward_fingerprint','parameters')}
            assert proof.setdefault(seed,p)==p
            h=[r for r in hist if r['method']==method and int(r['seed'])==seed]
            assert len(h)==10 and [int(x['epoch']) for x in h]==list(range(1,11))
            assert all(int(r['steps'])==6821 and int(r['unique_originals'])==218266 and r['originals_once_each']=='True' for r in h)
            best=max(h,key=lambda r:float(r['validation_AUC']))
            assert int(best['epoch'])==v['selected_epoch']
            close(best['validation_AUC'],v['validation']['overall']['auc'])
            if method in ('lh_ramp','lh_low'):
                assert v['selected_epoch']==2 and float(h[1]['response_multiplier'])==0
            reports[(method,seed)]=t['calibrated']
            for axis in AXES:
                metrics=computed_row(t['calibrated'],axis);axis_rows[(method,seed,axis)]=metrics
                match=[r for r in exported if r['method']==method and int(r['seed'])==seed and r['axis']==axis]
                assert len(match)==1
                for k,val in metrics.items():close(val,match[0][k])
            by_family=t['calibrated']['by_fake_family_vs_all_reals']
            matched=[r for r in families if r['method']==method and int(r['seed'])==seed]
            assert len(matched)==len(by_family)
            for r in matched:
                native=by_family[r['family']]
                close(r['AUC_pct'],100*native['auc']);close(r['FNR_pct'],100*native['fnr'])
                assert int(r['n_fake'])==native['n_fake'] and int(r['n_real'])==native['n_real']
    for r in means:
        values=[axis_rows[(r['method'],s,'gender_race')][r['metric']] for s in SEEDS]
        assert int(r['n_seeds'])==3
        close(statistics.mean(values),r['mean']);close(statistics.stdev(values),r['sample_sd'])
    assert all(reports[('lh_ramp',s)]==reports[('lh_low',s)] for s in SEEDS)
    source=read('SOURCE_CHECKOUT.json');copies=ROOT.parent/'ff2_fully_matched_interim_20261110/source';checked=0
    if copies.exists():
        for name,h in source['source_files'].items():assert sha(copies/name)==h;checked+=1
    if (ROOT/'SHA256.json').exists():
        for name,h in read('SHA256.json').items():assert sha(ROOT/name)==h,name
    print(json.dumps(dict(status='PASS',completed_training_TEST_receipts=21,selection_epochs_checked=210,
        matched_configurations=21,matched_initialisation_and_actual_input_seeds=3,
        TEST_method_axis_rows_checked=126,primary_means_and_sample_SDs_checked=63,
        native_family_rows_checked=len(families),all_ramp_low_selected_before_response=True,
        exact_source_copies_checked=checked,AUC_crosschecked_against_native_reports=True,
        AUC_recomputed_from_private_predictions=False,models_executed=False,checkpoint_binaries_read=False),indent=2))


if __name__=='__main__':main()
