#!/usr/bin/env python3
"""Bind completed L_A/L_H, evaluate four locked TEST manifests, analyze on CPU.

Never trains, reselects a checkpoint, changes the native threshold, or tunes on
TEST. Imports the exact historical L_H source without modifying it. Native DFDC
duplicate multiplicity is retained. Missing overlap evidence is a hard failure.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import hashlib

import numpy as np
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'src/ces_fl_head_ablation.py'
SOURCE_SHA = '08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177'
if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
    raise RuntimeError('Historical L_H source changed')
spec = importlib.util.spec_from_file_location('lh_locked_core', SOURCE)
core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
MANIFEST_PROTOCOL = Path(__file__).resolve().parents[1] / 'proposals/upgrade_tests_s17_20261101T040407Z/protocol.json'
ARMS = {'L_A': 'A', 'L_H': 'D'}
DATASETS = ('ffpp', 'dfdc', 'celebdf', 'dfd')
TARGETS = (.01, .02, .05, .10)


def expected_config(arm):
    folder=SOURCE.parents[1]/'proposals/plain_head_s17_20261103T101418Z'
    name='L_A_plain_bce.json' if arm=='L_A' else 'L_H_plain_hierarchical.json'
    return json.loads((folder/name).read_text())


def verify_selected(run, arm):
    root = Path(run); path = root/'best.pt'
    proof = json.loads((root/'RUN_VERIFIED.json').read_text())
    ck = torch.load(path, map_location='cpu', weights_only=True)
    history = json.loads((root/'history.json').read_text())
    if not (root/'DONE').is_file() or proof.get('status') != 'COMPLETED_VERIFIED':
        raise ValueError('Require complete, independently replayed run')
    if len(history) != 10 or [e['epoch'] for e in history] != list(range(1,11)):
        raise ValueError('Require all ten native epochs')
    selected = max(history, key=lambda e: e['validation']['overall']['auc'])
    if ck.get('format') != core.VERSION or ck.get('source_sha256') != SOURCE_SHA or ck.get('arm') != ARMS[arm]:
        raise ValueError('Wrong model/source/objective')
    if ck['synthetic_fixture'] or ck['config']['head_type'] != 'linear':
        raise ValueError('Require production plain head')
    if proof.get('selected_checkpoint_sha256') != core.sha(path) or proof.get('source_sha256') != SOURCE_SHA:
        raise ValueError('Checkpoint/receipt mismatch')
    if ck['config'] != json.loads((root/'config.json').read_text()): raise ValueError('Config mismatch')
    if ck['config'] != expected_config(arm): raise ValueError('Not the pinned original L_A/L_H configuration')
    core.check_config(ck['config'])
    if ck['epoch'] != selected['epoch'] or ck['validation'] != selected['validation'] or ck['calibration'] != selected['calibration']:
        raise ValueError('Checkpoint is not the native earliest maximum-AUC selection')
    scores_path = root/'validation_scores_PRIVATE.json'
    scores = json.loads(scores_path.read_text())
    if scores.get('split') != 'validation' or scores['checkpoint_sha256'] != core.sha(path):
        raise ValueError('Validation cache/checkpoint mismatch')
    if scores['manifest_sha256'] != ck['config']['val_sha256'] or scores['threshold'] != ck['calibration']['threshold_logit']:
        raise ValueError('Validation membership/threshold mismatch')
    original_rows=core.read_manifest(ck['config']['val'])
    if len(original_rows)!=len(scores['rows']) or any(r!={k:s.get(k) for k in r} for r,s in zip(original_rows,scores['rows'])):
        raise ValueError('Validation cache row order/labels/metadata differ from pinned manifest')
    replay = core.metric_report(scores['rows'], scores['logits'], scores['threshold'], scores['min_support'])
    if replay != ck['validation']: raise ValueError('Validation cache does not reproduce selected report')
    return ck, scores_path


def threshold_scan(scores):
    """Only an exploratory VAL curve. Does NOT select or authorize a new threshold."""
    if scores.get('split') != 'validation' or scores.get('test_read') is not False:
        raise ValueError('Threshold scan accepts labeled validation cache only')
    rows = scores['rows']; z = np.asarray(scores['logits'], np.float64)
    real = np.array([r['label'] == 0 for r in rows]); reports = []
    for target in TARGETS:
        cal = core.calibrate(z[real], target)
        reports.append({'target_validation_fpr': target, 'calibration': cal,
                        'validation': core.metric_report(rows,z,cal['threshold_logit'],scores['min_support'])})
    return dict(scope='validation exploratory curve only', checkpoint_sha256=scores['checkpoint_sha256'],
                preserved_native_threshold=scores['threshold'], selected_new_threshold=False,
                test_used=False, results=reports,
                note='AUC/es-AUC are score-ranking metrics and do not change with a decision threshold.')


def bind(runs, overlap_files, out):
    if set(runs) != set(ARMS) or set(overlap_files) != set(DATASETS):
        raise ValueError('Bind both fixed controls and all four declared evaluation audits')
    p = Path(out)
    if p.exists(): raise ValueError('Preserve prior locked plan')
    native = json.loads(MANIFEST_PROTOCOL.read_text()); arms = {}
    for arm, run in runs.items():
        ck, scores = verify_selected(run,arm)
        arms[arm] = dict(run=str(Path(run).resolve()), checkpoint=str((Path(run)/'best.pt').resolve()),
                         checkpoint_sha256=core.sha(Path(run)/'best.pt'), selected_epoch=ck['epoch'],
                         calibration=ck['calibration'], validation_cache=str(scores.resolve()),
                         validation_cache_sha256=core.sha(scores), config_digest=core.digest(ck['config']))
    audits = {}
    for dataset in DATASETS:
        a = native['manifest_audit'][dataset]; file = Path(overlap_files[dataset])
        if core.sha(a['path']) != a['sha256']: raise ValueError('Historical test manifest changed: '+dataset)
        declared = json.loads(file.read_text())
        if set(declared.get('declared_overlap_values',{})) != {'identity_ids:fit:test','source_ids:fit:test'}:
            raise ValueError('Need exact fit=(TRAIN+VAL):TEST overlap inventory, not the TRAIN/VAL audit')
        if not isinstance(declared.get('declared_content_overlap_sha256'),list):
            raise ValueError('Byte overlap inventory must be explicit')
        # Declaration provenance is inspected by the operator before bind; no automatic exceptions.
        audits[dataset] = dict(path=a['path'], sha256=a['sha256'], rows=a['rows'],
                              native_exact_duplicate_extra_rows=a['native_exact_duplicate_extra_rows'],
                              overlap_file=str(file.resolve()), overlap_sha256=core.sha(file))
    result=dict(format='LH_LOCKED_TABLES12_v1', source_sha256=SOURCE_SHA, runner_sha256=core.sha(__file__),
                historical_manifest_protocol_sha256=core.sha(MANIFEST_PROTOCOL), arms=arms,datasets=audits,
                thresholds=['native_selected_VAL_FPR10','fixed_probability_0.5'],
                test_threshold_search=False, bootstrap=1000, seed=17,
                table3_status='Needs separately trained ResNet L_H; never substitute old CES-FL rows')
    core.write_json(p,result);return result


def load_plan(path):
    p=json.loads(Path(path).read_text())
    if p.get('format')!='LH_LOCKED_TABLES12_v1' or p['runner_sha256']!=core.sha(__file__) or p['source_sha256']!=SOURCE_SHA:
        raise ValueError('Wrong/stale locked plan')
    if core.sha(MANIFEST_PROTOCOL)!=p['historical_manifest_protocol_sha256']:raise ValueError('Historical manifests changed')
    return p


def inference(plan_file,arm,dataset,out,device):
    p=load_plan(plan_file);a=p['arms'][arm];d=p['datasets'][dataset]
    if core.sha(a['checkpoint'])!=a['checkpoint_sha256'] or core.sha(d['overlap_file'])!=d['overlap_sha256']:
        raise ValueError('Checkpoint/overlap audit changed after binding')
    ck,_=verify_selected(a['run'],arm)
    if ck['calibration']!=a['calibration']:raise ValueError('Threshold changed')
    rows=core.read_manifest(d['path'],allow_duplicates=True)
    if len(rows)!=d['rows'] or len(rows)-len({r['image'] for r in rows})!=d['native_exact_duplicate_extra_rows']:
        raise ValueError('Row multiplicity changed')
    if Path(out).exists():raise ValueError('Preserve existing evaluation, including partial failures')
    # Keep bootstrap off the GPU reservation. Each TEST is scored once per model.
    result=core.evaluate(a['checkpoint'],d['path'],d['sha256'],out,torch.device(device),d['overlap_file'],replicates=0)
    if result['calibration']!=a['calibration']:raise RuntimeError('Evaluation altered threshold')
    core.write_json(Path(out)/'INFERENCE_VERIFIED.json',dict(status='PASS',arm=arm,dataset=dataset,
        plan_sha256=core.sha(plan_file),checkpoint_sha256=a['checkpoint_sha256'],
        manifest_sha256=d['sha256'],scores_sha256=core.sha(Path(out)/'scores_PRIVATE.json'),
        metrics_sha256=core.sha(Path(out)/'metrics.json'),test_calibration=False,
        uncertainty_pending=True,table=1 if dataset=='ffpp' else 2))
    return result


def analyze(plan_file,arm,dataset,run,out):
    p=load_plan(plan_file);root=Path(run);proof=json.loads((root/'INFERENCE_VERIFIED.json').read_text())
    if proof['status']!='PASS' or proof['plan_sha256']!=core.sha(plan_file) or (proof['arm'],proof['dataset'])!=(arm,dataset):
        raise ValueError('Wrong inference receipt')
    if core.sha(root/'scores_PRIVATE.json')!=proof['scores_sha256'] or core.sha(root/'metrics.json')!=proof['metrics_sha256']:
        raise ValueError('Inference outputs changed')
    if Path(out).exists():raise ValueError('Preserve previous analysis')
    s=json.loads((root/'scores_PRIVATE.json').read_text());result={}
    for key,t in [('native_selected_VAL_FPR10',s['threshold']),('fixed_probability_0.5',0.)]:
        result[key]=dict(metrics=core.metric_report(s['rows'],s['logits'],t,s['min_support']),
                        uncertainty=core.bootstrap(s['rows'],s['logits'],t,s['min_support'],p['bootstrap'],p['seed']))
    result['all_nonempty_groups_fixed05']=core.metric_report(s['rows'],s['logits'],0.,1)
    result.update(arm=arm,dataset=dataset,checkpoint_sha256=s['checkpoint_sha256'],test_threshold_search=False,
                  usable_replicates_requirement=900,training_seed_count=1)
    core.write_json(out,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    q=sub.add_parser('bind');q.add_argument('--runs-json',required=True);q.add_argument('--audits-json',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('threshold-scan');q.add_argument('--scores',required=True);q.add_argument('--out',required=True)
    for command in ('infer','analyze'):
        q=sub.add_parser(command);q.add_argument('--plan',required=True);q.add_argument('--arm',choices=list(ARMS),required=True)
        q.add_argument('--dataset',choices=DATASETS,required=True);q.add_argument('--out',required=True)
        if command=='infer':q.add_argument('--device',choices=['cpu','cuda'],default='cuda')
        else:q.add_argument('--inference-dir',required=True)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.command=='bind':return bind(json.loads(Path(args.runs_json).read_text()),json.loads(Path(args.audits_json).read_text()),args.out)
    if args.command=='threshold-scan':
        if Path(args.out).exists():raise ValueError('Preserve earlier scan')
        return core.write_json(args.out,threshold_scan(json.loads(Path(args.scores).read_text())))
    if args.command=='infer':return inference(args.plan,args.arm,args.dataset,args.out,args.device)
    return analyze(args.plan,args.arm,args.dataset,args.inference_dir,args.out)


if __name__=='__main__':main()
