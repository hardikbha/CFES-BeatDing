"""Verify the published aggregate snapshot without datasets, models or GPUs."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def csv_rows(name):
    with (ROOT/name).open() as f:return list(csv.DictReader(f))

def close(a,b):
    if not math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-9):
        raise AssertionError((a,b))

def main():
    snapshot=json.loads((ROOT/'SNAPSHOT.json').read_text())
    native=json.loads((ROOT/'selected_validation_aggregates.json').read_text())
    joint=csv_rows('metrics_joint.csv');axes=csv_rows('metrics_all_axes.csv')
    history=csv_rows('selection_history_prefix.csv');groups=csv_rows('group_support_and_rates.csv')
    assert snapshot['split']=='SELECT/validation' and snapshot['seed']==17
    assert len(joint)==7 and len(axes)==42 and len(groups)==217
    assert snapshot['training_incomplete_at_metric_snapshot']
    assert not snapshot['test_predictions_read_for_packaging']
    selection_checks=0
    for row in joint:
        method=row['method'];selected=native[method];v=selected['validation'];a=v['axes']['gender_race']
        prefix=[x for x in history if x['method']==method]
        assert len(prefix)==int(row['epochs_recorded'])==5
        assert [int(x['epoch']) for x in prefix]==list(range(1,6))
        best=max(prefix,key=lambda x:float(x['AUC_pct']))
        assert int(best['epoch'])==int(row['selected_so_far_epoch'])==selected['selected_so_far_epoch']
        close(best['AUC_pct'],row['AUC_pct']);close(row['AUC_pct'],v['overall']['auc']*100)
        close(row['FPR_pct'],v['overall']['fpr']*100);close(row['FNR_pct'],v['overall']['fnr']*100)
        close(row['TPR_pct'],100-float(row['FNR_pct']))
        close(row['joint_F_FPR_100'],a['F_FPR']*100)
        close(row['joint_FPR_gap_pp'],a['fpr_gap']*100)
        close(row['joint_F_MEO_max_pp'],max(a['fpr_gap'],a['fnr_gap'])*100)
        close(row['joint_F_DP_100'],a['F_DP']*100)
        close(row['joint_es_AUC_pct'],a['es_AUC']*100)
        close(row['joint_worst_group_AUC_pct'],a['worst_group_auc']*100)
        selection_checks+=1
    for row in groups:
        a=native[row['method']]['validation']['axes'][row['axis']]
        g=a['groups'][row['group']]
        assert int(row['n'])==int(row['n_real'])+int(row['n_fake'])
        assert (row['eligible']=='True')==(row['group'] in a['eligible_groups'])
        for key in ['n','n_real','n_fake','auc','fpr','fnr','positive_rate']:
            if g[key] is None:assert row[key]==''
            else:close(row[key],g[key])
    for row in axes:
        v=native[row['method']]['validation'];a=v['axes'][row['axis']]
        gs=[a['groups'][name] for name in a['eligible_groups']]
        fpr=[g['fpr'] for g in gs];fnr=[g['fnr'] for g in gs];auc=[g['auc'] for g in gs]
        acc=[(g['n_real']*(1-g['fpr'])+g['n_fake']*(1-g['fnr']))/g['n'] for g in gs]
        close(row['AUC_pct'],v['overall']['auc']*100)
        close(row['F_FPR_100'],sum(abs(x-v['overall']['fpr']) for x in fpr)*100)
        close(row['F_DP_100'],(max(g['positive_rate'] for g in gs)-min(g['positive_rate'] for g in gs))*100)
        close(row['F_MEO_max_pp'],max(max(fpr)-min(fpr),max(fnr)-min(fnr))*100)
        close(row['F_OAE_pp'],(max(acc)-min(acc))*100)
        close(row['es_AUC_pct'],v['overall']['auc']/(1+sum(abs(x-v['overall']['auc']) for x in auc))*100)
        close(row['worst_group_AUC_pct'],min(auc)*100)
    source=json.loads((ROOT/'evidence/SOURCE_CHECKOUT.json').read_text())
    for relative,expected in source['source_files'].items():
        assert hashlib.sha256((ROOT/'source'/relative).read_bytes()).hexdigest()==expected,relative
    configs=list((ROOT/'source/proposals/ff2_fully_matched_20261110_r1').glob('*.json'))
    assert len(configs)==21
    recipes=[{k:v for k,v in json.loads(p.read_text()).items() if k not in ['suite_method','seed']} for p in configs]
    assert all(c==recipes[0] for c in recipes)
    checks=list((ROOT/'evidence/cuda_checks').glob('*.json'));assert len(checks)==7
    fingerprints=[]
    for p in checks:
        c=json.loads(p.read_text());assert c['status']=='PASS' and c['bn_frozen'] and not c['test_read']
        assert c['parameters']=={'total':20809001,'trainable':6790433}
        fingerprints.append((c['initial_parameter_fingerprint'],c['data_forward_fingerprint']))
    assert len(set(fingerprints))==1
    manifest=ROOT/'ARTIFACT_SHA256.json'
    if manifest.exists():
        for relative,expected in json.loads(manifest.read_text()).items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,relative
    result={'status':'PASS','selected_checkpoint_history_rows':selection_checks,
            'method_axis_rows_recomputed':len(axes),'group_rate_rows_checked':len(groups),
            'source_files_hash_verified':len(source['source_files']),
            'common_recipe_configs_verified':len(configs),'real_cuda_fingerprints_equal':len(checks),
            'models_executed':False,'private_predictions_read':False,'test_results_claimed':False,
            'scope':'Aggregate arithmetic, recorded selection and exact evidence/source copies; no inference or checkpoint replay.'}
    (ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
