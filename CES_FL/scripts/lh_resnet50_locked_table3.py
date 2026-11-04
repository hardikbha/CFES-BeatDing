#!/usr/bin/env python3
"""Bind and evaluate the frozen ResNet-50 L_A/L_H Table-3 pair once.

This wrapper never trains, recalibrates or selects from TEST. The protocol and
four manifest/overlap bindings derive from the already frozen Xception Tables
1-2 plan; only the trained ResNet checkpoint format/source is different.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
CORE_PATH = REPO / 'src/ces_fl_head_ablation_resnet50.py'
CORE_SHA = '72d368a439658c601a5fe4d717237b1561190e85c13c70f6db3a441bc5cdbe78'
PROTOCOL_PATH = REPO / 'proposals/lh_finalization_celebdf_20261104_r1/configs/resnet50_table3_eval_protocol.json'
PARENT_PLAN_SHA = '6dffee3b9b309975d7c3844dc1052427885dc41b2375e77c15f38ae33f76c5ea'
MANIFEST_PROTOCOL_SHA = '8541a40413f278d34bd26630cec3446def3fc6f2b9fdf76a825859efe1d233fe'
DATASETS = ('ffpp', 'dfdc', 'celebdf', 'dfd')
ARM_CODES = {'L_A': 'A', 'L_H': 'D'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    p = Path(path)
    if p.exists():
        raise ValueError(f'Preserve existing output: {p}')
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)


def load_core():
    if sha(CORE_PATH) != CORE_SHA:
        raise ValueError('Pinned ResNet scientific source changed')
    spec = importlib.util.spec_from_file_location('cesfl_resnet50_locked_core', CORE_PATH)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def load_protocol(core):
    p = json.loads(PROTOCOL_PATH.read_text())
    if p.get('format') != 'CESFL_RESNET50_TABLE3_EVAL_PROTOCOL_v1':
        raise ValueError('Unknown evaluation protocol version')
    if p.get('parent_locked_plan_sha256') != PARENT_PLAN_SHA or p.get('historical_manifest_protocol_sha256') != MANIFEST_PROTOCOL_SHA:
        raise ValueError('Protocol parent hashes differ from the pre-existing locked plan')
    if p.get('resnet_downstream_source_sha256') != CORE_SHA:
        raise ValueError('Protocol source hash mismatch')
    if p.get('arms') != ARM_CODES or p.get('datasets') != list(DATASETS):
        raise ValueError('Protocol arm or dataset inventory changed')
    if p.get('threshold_views') != ['fixed_probability_0.5', 'selected_FFPP_validation_real_empirical_10pct']:
        raise ValueError('Threshold views are not the preregistered pair')
    if p.get('target_domain_calibration') is not False or p.get('test_threshold_search') is not False or p.get('test_driven_selection') is not False:
        raise ValueError('TEST-dependent calibration or selection is forbidden')
    if p.get('inference_passes_per_model_dataset') != 1:
        raise ValueError('Only one frozen inference pass is permitted')
    parent_path = Path(p['parent_locked_plan'])
    if sha(parent_path) != PARENT_PLAN_SHA:
        raise ValueError('Existing Tables 1-2 locked plan changed')
    parent = json.loads(parent_path.read_text())
    if parent.get('format') != 'LH_LOCKED_TABLES12_v1':
        raise ValueError('Parent Tables 1-2 plan version changed')
    if parent.get('historical_manifest_protocol_sha256') != MANIFEST_PROTOCOL_SHA:
        raise ValueError('Parent Tables 1-2 manifest protocol changed')
    if set(parent.get('datasets', {})) != set(DATASETS):
        raise ValueError('Parent dataset inventory differs')
    if core.sha(REPO / 'proposals/upgrade_tests_s17_20261101T040407Z/protocol.json') != MANIFEST_PROTOCOL_SHA:
        raise ValueError('Historical manifest protocol source changed')
    return p, parent


def verify_selected(core, run, arm):
    root = Path(run)
    checkpoint = root / 'best.pt'
    proof = json.loads((root / 'RUN_VERIFIED.json').read_text())
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    history = json.loads((root / 'history.json').read_text())
    if not (root / 'DONE').is_file() or proof.get('status') != 'COMPLETED_VERIFIED':
        raise ValueError(f'{arm}: run is not complete and replay-verified')
    if proof.get('test_evaluated') is not False:
        raise ValueError(f'{arm}: test evaluation already occurred')
    if len(history) != 10 or [e.get('epoch') for e in history] != list(range(1, 11)):
        raise ValueError(f'{arm}: expected exactly ten recorded epochs')
    selected = max(history, key=lambda e: e['validation']['overall']['auc'])
    if ck.get('format') != core.VERSION or ck.get('source_sha256') != CORE_SHA or ck.get('arm') != ARM_CODES[arm]:
        raise ValueError(f'{arm}: wrong checkpoint format, source or objective')
    if ck.get('synthetic_fixture') is not False or ck['config'].get('backbone') != 'resnet50':
        raise ValueError(f'{arm}: synthetic or non-ResNet checkpoint rejected')
    if ck['config'].get('epochs') != 10 or ck['config'].get('seed') != 17:
        raise ValueError(f'{arm}: configuration differs from the fixed run')
    if ck['config'].get('target_fpr') != 0.1:
        raise ValueError(f'{arm}: validation calibration target is not the frozen 10% rule')
    if proof.get('selected_checkpoint_sha256') != core.sha(checkpoint) or proof.get('source_sha256') != CORE_SHA:
        raise ValueError(f'{arm}: checkpoint receipt mismatch')
    if proof.get('config_digest') != core.digest(ck['config']):
        raise ValueError(f'{arm}: config digest receipt mismatch')
    if ck['config'] != json.loads((root / 'config.json').read_text()):
        raise ValueError(f'{arm}: checkpoint/config file mismatch')
    core.check_config(ck['config'])
    if ck.get('epoch') != selected['epoch'] or proof.get('selected_epoch') != selected['epoch']:
        raise ValueError(f'{arm}: checkpoint is not earliest maximum validation AUC')
    if ck.get('validation') != selected['validation'] or ck.get('calibration') != selected['calibration']:
        raise ValueError(f'{arm}: checkpoint validation/calibration differs from selected epoch')
    scores_path = root / 'validation_scores_PRIVATE.json'
    scores = json.loads(scores_path.read_text())
    if scores.get('split') != 'validation' or scores.get('test_read') is not False:
        raise ValueError(f'{arm}: validation cache has wrong split declaration')
    if scores.get('checkpoint_sha256') != core.sha(checkpoint) or scores.get('manifest_sha256') != ck['config']['val_sha256']:
        raise ValueError(f'{arm}: validation cache binding mismatch')
    val_rows = core.read_manifest(ck['config']['val'])
    if len(val_rows) != len(scores['rows']) or any(r != {k: s.get(k) for k in r} for r, s in zip(val_rows, scores['rows'])):
        raise ValueError(f'{arm}: validation cache rows do not match manifest')
    replay = core.metric_report(scores['rows'], scores['logits'], scores['threshold'], scores['min_support'])
    if replay != ck['validation'] or scores['threshold'] != ck['calibration']['threshold_logit']:
        raise ValueError(f'{arm}: stored validation result/threshold does not replay')
    return ck, scores_path


def bind(runs_json, out):
    core = load_core()
    protocol, parent = load_protocol(core)
    runs = json.loads(Path(runs_json).read_text())
    if set(runs) != set(ARM_CODES):
        raise ValueError('Bind exactly the selected L_A and L_H ResNet runs')
    arms = {}
    for arm, run in runs.items():
        ck, val_scores = verify_selected(core, run, arm)
        arms[arm] = {
            'run': str(Path(run).resolve()),
            'checkpoint': str((Path(run) / 'best.pt').resolve()),
            'checkpoint_sha256': core.sha(Path(run) / 'best.pt'),
            'selected_epoch': ck['epoch'],
            'calibration': ck['calibration'],
            'validation_cache': str(val_scores.resolve()),
            'validation_cache_sha256': core.sha(val_scores),
            'config_digest': core.digest(ck['config']),
            'p50_encoder_sha256': ck['config']['init_sha256'],
        }
    a, h = (json.loads((Path(runs[name]) / 'config.json').read_text()) for name in ('L_A', 'L_H'))
    differences = {k for k in a if a[k] != h[k]}
    if differences != {'error_control'} or a['init_sha256'] != h['init_sha256']:
        raise ValueError(f'L_A/L_H are not matched except for the intended objective: {sorted(differences)}')
    datasets = {}
    for name in DATASETS:
        d = parent['datasets'][name]
        overlap = Path(d['overlap_file'])
        if core.sha(d['path']) != d['sha256'] or core.sha(overlap) != d['overlap_sha256']:
            raise ValueError(f'{name}: frozen manifest/overlap audit changed')
        audit = json.loads(overlap.read_text())
        expected = {'identity_ids:fit:test', 'source_ids:fit:test'}
        if set(audit.get('declared_overlap_values', {})) != expected or not isinstance(audit.get('declared_content_overlap_sha256'), list):
            raise ValueError(f'{name}: missing exact fit:TEST contributor/byte audit')
        datasets[name] = {k: d[k] for k in ('path', 'sha256', 'rows', 'native_exact_duplicate_extra_rows', 'overlap_file', 'overlap_sha256')}
    result = {
        'format': 'CESFL_RESNET50_TABLE3_LOCKED_v1',
        'protocol_sha256': core.sha(PROTOCOL_PATH),
        'runner_sha256': core.sha(__file__),
        'core_source_sha256': CORE_SHA,
        'parent_locked_plan_sha256': PARENT_PLAN_SHA,
        'historical_manifest_protocol_sha256': MANIFEST_PROTOCOL_SHA,
        'arms': arms,
        'datasets': datasets,
        'threshold_views': protocol['threshold_views'],
        'test_threshold_search': False,
        'target_domain_calibration': False,
        'test_driven_selection': False,
        'inference_passes_per_model_dataset': 1,
        'bootstrap': protocol['bootstrap'],
        'training_seed_count': 1,
        'biological_person_independence_verified': False,
    }
    write_json(out, result)
    return result


def load_plan(path):
    core = load_core(); protocol, _ = load_protocol(core)
    plan = json.loads(Path(path).read_text())
    if plan.get('format') != 'CESFL_RESNET50_TABLE3_LOCKED_v1':
        raise ValueError('Wrong plan version')
    if plan.get('protocol_sha256') != core.sha(PROTOCOL_PATH) or plan.get('runner_sha256') != core.sha(__file__):
        raise ValueError('Evaluation protocol/runner changed after binding')
    if plan.get('core_source_sha256') != CORE_SHA or plan.get('threshold_views') != protocol['threshold_views']:
        raise ValueError('Scientific source/threshold policy mismatch')
    if plan.get('test_threshold_search') is not False or plan.get('target_domain_calibration') is not False or plan.get('test_driven_selection') is not False:
        raise ValueError('TEST tuning declaration detected')
    return core, protocol, plan


def inference(plan_path, arm, dataset, out, device):
    core, _, plan = load_plan(plan_path)
    if arm not in ARM_CODES or dataset not in DATASETS:
        raise ValueError('Unknown arm or dataset')
    a, d = plan['arms'][arm], plan['datasets'][dataset]
    if core.sha(a['checkpoint']) != a['checkpoint_sha256'] or core.sha(d['overlap_file']) != d['overlap_sha256']:
        raise ValueError('Frozen checkpoint/overlap audit changed after binding')
    ck, _ = verify_selected(core, a['run'], arm)
    if ck['calibration'] != a['calibration']:
        raise ValueError('Selected source-validation threshold changed')
    if Path(out).exists():
        raise ValueError('Preserve existing evaluation output; do not rerun TEST')
    result = core.evaluate(a['checkpoint'], d['path'], d['sha256'], out, torch.device(device),
                           d['overlap_file'], replicates=0)
    if result['calibration'] != a['calibration']:
        raise RuntimeError('Evaluation changed the frozen source-validation threshold')
    write_json(Path(out) / 'INFERENCE_VERIFIED.json', {
        'status': 'PASS', 'arm': arm, 'dataset': dataset,
        'plan_sha256': core.sha(plan_path), 'checkpoint_sha256': a['checkpoint_sha256'],
        'manifest_sha256': d['sha256'], 'metrics_sha256': core.sha(Path(out) / 'metrics.json'),
        'scores_sha256': core.sha(Path(out) / 'scores_PRIVATE.json'),
        'test_calibration': False, 'test_threshold_search': False,
        'inference_passes_per_model_dataset': 1,
    })
    return result


def analyze(plan_path, arm, dataset, inference_dir, out):
    core, protocol, plan = load_plan(plan_path)
    root = Path(inference_dir)
    proof = json.loads((root / 'INFERENCE_VERIFIED.json').read_text())
    if proof.get('status') != 'PASS' or proof.get('plan_sha256') != core.sha(plan_path) or (proof.get('arm'), proof.get('dataset')) != (arm, dataset):
        raise ValueError('Inference receipt does not match the frozen plan')
    if proof.get('test_calibration') is not False or proof.get('test_threshold_search') is not False:
        raise ValueError('Inference receipt claims forbidden TEST tuning')
    if core.sha(root / 'metrics.json') != proof['metrics_sha256'] or core.sha(root / 'scores_PRIVATE.json') != proof['scores_sha256']:
        raise ValueError('Private inference outputs changed')
    if Path(out).exists():
        raise ValueError('Preserve existing analysis output')
    scores = json.loads((root / 'scores_PRIVATE.json').read_text())
    if scores.get('checkpoint_sha256') != plan['arms'][arm]['checkpoint_sha256'] or scores.get('manifest_sha256') != plan['datasets'][dataset]['sha256']:
        raise ValueError('Private score cache binding mismatch')
    if scores.get('threshold') != plan['arms'][arm]['calibration']['threshold_logit']:
        raise ValueError('TEST score cache threshold differs from selected source validation')
    result = {}
    thresholds = [('selected_FFPP_validation_real_empirical_10pct', scores['threshold']),
                  ('fixed_probability_0.5', 0.0)]
    for name, threshold in thresholds:
        result[name] = {
            'metrics': core.metric_report(scores['rows'], scores['logits'], threshold, scores['min_support']),
            'uncertainty': core.bootstrap(scores['rows'], scores['logits'], threshold, scores['min_support'],
                                          protocol['bootstrap']['replicates'], protocol['bootstrap']['seed']),
        }
    result['all_nonempty_groups_fixed05'] = core.metric_report(scores['rows'], scores['logits'], 0.0, 1)
    result.update(arm=arm, dataset=dataset, checkpoint_sha256=scores['checkpoint_sha256'],
                  test_threshold_search=False, test_calibration=False,
                  training_seed_count=1,
                  minimum_valid_bootstrap_replicates=protocol['bootstrap']['minimum_valid_replicates_for_interval'])
    write_json(out, result)
    return result


def compare(plan_path, dataset, h_scores_path, a_scores_path, out):
    core, protocol, plan = load_plan(plan_path)
    for arm, path in (('L_H', h_scores_path), ('L_A', a_scores_path)):
        score_path = Path(path)
        proof_path = score_path.parent / 'INFERENCE_VERIFIED.json'
        proof = json.loads(proof_path.read_text())
        if (proof.get('status') != 'PASS' or proof.get('plan_sha256') != core.sha(plan_path)
                or proof.get('arm') != arm or proof.get('dataset') != dataset
                or proof.get('scores_sha256') != core.sha(score_path)):
            raise ValueError(f'{arm}: comparison score cache lacks a matching inference receipt')
    h = json.loads(Path(h_scores_path).read_text()); a = json.loads(Path(a_scores_path).read_text())
    if h['rows'] != a['rows'] or h['manifest_sha256'] != a['manifest_sha256'] or h['min_support'] != a['min_support']:
        raise ValueError('Paired comparison requires identical row order, support and annotations')
    if h['checkpoint_sha256'] != plan['arms']['L_H']['checkpoint_sha256'] or a['checkpoint_sha256'] != plan['arms']['L_A']['checkpoint_sha256']:
        raise ValueError('Score caches do not match the bound L_H/L_A checkpoints')
    if h['manifest_sha256'] != plan['datasets'][dataset]['sha256']:
        raise ValueError('Score caches do not match the requested dataset')
    result = {}
    for name, h_threshold, a_threshold in (
        ('selected_FFPP_validation_real_empirical_10pct', h['threshold'], a['threshold']),
        ('fixed_probability_0.5', 0.0, 0.0),
    ):
        result[name] = core.bootstrap(h['rows'], h['logits'], h_threshold, h['min_support'],
                                      protocol['bootstrap']['replicates'], protocol['bootstrap']['seed'],
                                      other=(a['logits'], a_threshold))
    result.update(contrast='L_H minus L_A', dataset=dataset, test_threshold_search=False,
                  test_calibration=False, training_seed_count=1)
    write_json(out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('bind'); p.add_argument('--runs-json', required=True); p.add_argument('--out', required=True)
    for command in ('infer', 'analyze'):
        p = sub.add_parser(command); p.add_argument('--plan', required=True)
        p.add_argument('--arm', choices=ARM_CODES, required=True); p.add_argument('--dataset', choices=DATASETS, required=True)
        if command == 'infer': p.add_argument('--out', required=True); p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
        else: p.add_argument('--inference-dir', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('compare'); p.add_argument('--plan', required=True); p.add_argument('--dataset', choices=DATASETS, required=True)
    p.add_argument('--h-scores', required=True); p.add_argument('--a-scores', required=True); p.add_argument('--out', required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    if args.command == 'bind': bind(args.runs_json, args.out)
    elif args.command == 'infer': inference(args.plan, args.arm, args.dataset, args.out, args.device)
    elif args.command == 'analyze': analyze(args.plan, args.arm, args.dataset, args.inference_dir, args.out)
    else: compare(args.plan, args.dataset, args.h_scores, args.a_scores, args.out)


if __name__ == '__main__':
    main()
