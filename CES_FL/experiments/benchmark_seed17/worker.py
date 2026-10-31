"""Pinned benchmark worker: preserve the CES model/loss and selected thresholds."""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import torch
from uncertainty import bootstrap

REPO = Path(__file__).resolve().parents[3]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def state_sha(model, encoder):
    return hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for k, v in model.state_dict().items() if k.startswith('encoder.') == encoder)).hexdigest()


def model_module():
    path = REPO/'CES_FL/src/ces_fl_diagram_model_v021.py'
    spec = importlib.util.spec_from_file_location('ces_benchmark', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def resnet_smoke(M, weights, device):
    result = []
    for policy, bn in [('frozen', 'frozen'), ('trainable', 'update')]:
        M.seed_all(17)
        model = M.DiagramCES('resnet50', weights, encoder_policy=policy, bn_policy=bn, head_seed=17).to(device).train()
        before = state_sha(model, True)
        x = torch.rand(2, 3, 96, 96, device=device)
        out = model(x)
        torch.testing.assert_close(out['logit'], out['global_contribution']+out['local_map'].mean((1, 2))+model.fusion_bias)
        opt = M.build_optimizer(model, {'lr': .001, 'encoder_lr': .0001, 'weight_decay': .0001})
        loss = M.ces_loss(out, torch.tensor([0., 1.], device=device), mode='bce')['total']
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        receiving = sum(p.numel() for p in model.parameters() if p.grad is not None)
        assert receiving == model.counts()['trainable']
        opt.step()
        assert (before == state_sha(model, True)) == (policy == 'frozen')
        result.append({'encoder_policy': policy, 'bn_policy': bn, 'parameters': model.counts(), 'receiving_grad': receiving,
                       'head_initialization_sha256': state_sha(M.DiagramCES('resnet50', weights, head_seed=17), False)})
        del model, opt, out, loss
    return {'status': 'PASS', 'device': device, 'policies': result, 'scope': 'Synthetic software checks, not detector/fairness results'}


def read_native_rows(M, manifest, audit):
    assert sha(audit['unique_validation_index']) == audit['unique_validation_index_sha256']
    unique = M.read_rows(audit['unique_validation_index'])
    index = {r['image']: r for r in unique}
    with Path(manifest).open() as f:
        native = list(csv.DictReader(f))
    rows = [dict(index[str(Path(r['image']).resolve())]) for r in native]
    assert len(rows) == audit['rows'] and len(rows)-len(index) == audit['native_exact_duplicate_extra_rows']
    return rows


def evaluate(M, protocol, root, arm, checkpoint, progress):
    expected = sha(checkpoint)
    for dataset in ['ffpp', 'dfdc', 'celebdf', 'dfd', 'fairforge2']:
        manifest = protocol['manifests'][dataset]
        destination = root/'evaluation'/arm/dataset
        progress('EVALUATING', arm=arm, dataset=dataset)
        audit = protocol['manifest_audit'][dataset]
        if audit['native_exact_duplicate_extra_rows']:
            # Only the audited, exactly repeated native DFDC TEST rows use this
            # explicit adapter. Strict model/training manifest parsing is unchanged.
            assert dataset == 'dfdc'
            rows = read_native_rows(M, manifest, audit)
            model, ck = M.load_model(checkpoint, 'cuda')
            threshold = ck['calibration']['threshold_logit']
            z = M.predict(model, rows, 'cuda', ck['config']['batch_size'], ck['config']['workers'], ck['config']['size'])
            destination = M.empty_out(destination)
            report = {'checkpoint_sha256': expected, 'manifest_sha256': sha(manifest),
                      'threshold_source': 'selected checkpoint validation only', 'calibration': ck['calibration'],
                      'calibrated': M.metrics(rows, z, threshold, ck['config']['min_support']),
                      'fixed_probability_0_5': M.metrics(rows, z, 0., ck['config']['min_support']),
                      'native_exact_duplicate_extra_rows_preserved': audit['native_exact_duplicate_extra_rows']}
            write(destination/'metrics.json', report)
            with (destination/'predictions.csv').open('x', newline='') as f:
                w = csv.writer(f); w.writerow(['image', 'label', 'logit', 'prediction'])
                w.writerows((r['image'], r['label'], float(s), int(s >= threshold)) for r, s in zip(rows, z))
            del model, ck
        else:
            report = M.evaluate_run(checkpoint, manifest, destination, 'cuda')
            rows = M.read_rows(manifest)
        assert report['checkpoint_sha256'] == expected and report['manifest_sha256'] == protocol['manifest_audit'][dataset]['sha256']
        with (destination/'predictions.csv').open() as f:
            predictions = list(csv.DictReader(f))
        assert [r['image'] for r in rows] == [r['image'] for r in predictions]
        z = [float(r['logit']) for r in predictions]
        progress('UNCERTAINTY', arm=arm, dataset=dataset)
        ci = bootstrap(rows, z, report['calibration']['threshold_logit'], M.AXES,
                       reps=protocol['bootstrap']['replicates'], seed=17, min_support=20)
        write(destination/'confidence_intervals.json', ci)
        write(destination/'EVALUATION_VERIFIED.json', {'status': 'PASS', 'checkpoint_sha256': expected,
              'manifest_sha256': report['manifest_sha256'], 'threshold_source': report['threshold_source'],
              'no_test_calibration_or_selection': True, 'rows': len(rows), 'inference_only': True})
        (destination/'DONE').write_text('Fixed selected checkpoint and threshold evaluated; no training\n')
        print('BENCHMARK_RESULT', arm, dataset, json.dumps(report['calibrated']['overall']), flush=True)
    assert sha(checkpoint) == expected, 'Evaluation changed checkpoint bytes'


def validation_audit(M, protocol, root, progress):
    records = {}
    for arm in ['B_detection_frozen', 'C_detection_invariance']:
        entry = protocol['xception'][arm]
        checkpoint = Path(entry['checkpoint'])
        c = json.loads((checkpoint.parent/'config.json').read_text())
        v = json.loads((checkpoint.parent/'RUN_VERIFIED.json').read_text())
        assert sha(c['val']) == protocol['validation_sha256']
        progress('VALIDATION_AUDIT', arm=arm)
        destination = root/'validation_audit'/arm
        report = M.evaluate_run(checkpoint, c['val'], destination, 'cuda')
        assert abs(report['calibrated']['overall']['auc']-v['validation_auc']) < 1e-5, 'Validation replication mismatch before test'
        assert report['calibration'] == entry['stored_calibration']
        previous = checkpoint.parents[2]/'full_experiment_diagnostics'/arm/'diagnostics.json'
        control = json.loads(previous.read_text())
        assert control['checkpoint_sha256'] == entry['sha256']
        rows = M.read_rows(c['val'])
        with (destination/'predictions.csv').open() as f:
            predictions = list(csv.DictReader(f))
        records[arm] = {'rows': rows, 'predictions': predictions}
        ci = bootstrap(rows, [float(p['logit']) for p in predictions], report['calibration']['threshold_logit'], M.AXES, reps=1000, seed=17)
        write(destination/'confidence_intervals.json', ci)
        write(destination/'control_audit.json', {'existing_diagnostics': str(previous), 'sha256': sha(previous),
              'checkpoint_hash_matches': True, 'diagnostics_reused_not_rerun': True,
              'limits': 'Editor no-change drift was previously substantial; visual approval alone does not certify forensic preservation. No loss or selection policy changed.'})
    b, c = records['B_detection_frozen'], records['C_detection_invariance']
    assert [p['image'] for p in b['predictions']] == [p['image'] for p in c['predictions']]
    groups = {}
    for row, bp, cp in zip(b['rows'], b['predictions'], c['predictions']):
        item = groups.setdefault(row['gender_race'], {'n_real': 0, 'n_fake': 0, 'B_FP': 0, 'C_FP': 0, 'B_FN': 0, 'C_FN': 0, 'prediction_disagreements': 0})
        y = row['label']; pb, pc = int(bp['prediction']), int(cp['prediction'])
        item['n_real' if y == 0 else 'n_fake'] += 1
        item['B_FP'] += int(y == 0 and pb == 1); item['C_FP'] += int(y == 0 and pc == 1)
        item['B_FN'] += int(y == 1 and pb == 0); item['C_FN'] += int(y == 1 and pc == 0)
        item['prediction_disagreements'] += int(pb != pc)
    write(root/'validation_audit/paired_error_counts.json', {'groups': groups, 'selection_or_calibration_changed': False,
          'limits': 'Paired counts are descriptive; see component-bootstrap availability/limitations for uncertainty.'})


def train(M, protocol, phase, root, progress):
    config = json.loads((REPO/protocol['configs'][phase]).read_text())
    if phase.startswith(('B_', 'C_')):
        path = root/'full_experiment_runs/P_resnet50/best.pt'
        v = json.loads((path.parent/'RUN_VERIFIED.json').read_text())
        assert v['status'] == 'PASS' and sha(path) == v['checkpoint_sha256']
        config['encoder_checkpoint_sha256'] = v['checkpoint_sha256']
        write(root/'bound_configs'/f'{phase}.json', config)
    destination = root/'full_experiment_runs'/phase
    assert not destination.exists(), 'No duplicate training or silent resume'
    model_ref, initial, step_count = [], {}, [0]
    original_build, original_predict = M.build_optimizer, M.predict
    with Path(config['val']).open() as f:
        validation_paths = {r['image'] for r in csv.DictReader(f)}
    def build(model, c):
        model_ref.append(model)
        initial.update(encoder=state_sha(model, True), heads=state_sha(model, False))
        assert initial['heads'] == '32e7535b41c78a7a0b8961d9f95c346d9ceca0ecff0965a1cc72c9f5fa13086e'
        write(root/'runtime'/f'{phase}.initialization.json', initial)
        optimizer = original_build(model, c)
        original_step = optimizer.step
        def step(*args, **kwargs):
            value = original_step(*args, **kwargs); step_count[0] += 1
            if step_count[0] == 1:
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                encoder_grad = sum(p.numel() for p in model.encoder.parameters() if p.grad is not None)
                assert (encoder_grad > 0) == (phase == 'P_resnet50')
            if step_count[0] == 1 or step_count[0] % 100 == 0:
                progress('TRAINING', optimizer_steps=step_count[0])
            return value
        optimizer.step = step
        return optimizer
    def validation(model, rows, *args, **kwargs):
        assert all(r['image'] in validation_paths for r in rows), 'Non-validation rows used for selection'
        progress('VALIDATING', optimizer_steps=step_count[0])
        return original_predict(model, rows, *args, **kwargs)
    M.build_optimizer, M.predict = build, validation
    try:
        checkpoint = M.train_run(config, destination, 'cuda')
    finally:
        M.build_optimizer, M.predict = original_build, original_predict
    history = [json.loads(s) for s in (destination/'history.jsonl').read_text().splitlines()]
    assert len(history) == config['epochs'] == 10
    selected = max(history, key=lambda h: h['validation']['overall']['auc'])
    model, ck = M.load_model(checkpoint, 'cpu')
    assert ck['epoch'] == selected['epoch'] and ck['calibration'] == selected['calibration']
    assert step_count[0] == sum(h['steps'] for h in history) == 47590
    same = state_sha(model, True) == initial['encoder']
    assert same == (phase != 'P_resnet50')
    usage = json.loads((destination/'pair_usage.json').read_text())
    assert usage['unique_pairs_used'] == (36 if phase == 'C_resnet50' else 0)
    verified = {'status': 'PASS', 'arm': phase, 'optimizer_steps': step_count[0], 'selected_epoch': ck['epoch'],
                'validation_auc': selected['validation']['overall']['auc'], 'strict_checkpoint_restore': True,
                'encoder_and_BN_match_initial_state': same, 'encoder_state_sha256': state_sha(model, True),
                'checkpoint_sha256': sha(checkpoint), 'calibration': ck['calibration'], 'pair_usage': usage,
                'test_evaluated': False, 'code_sha256': protocol['core_sha256']}
    write(destination/'TRAINING_VERIFIED.json', verified)
    write(destination/'RUN_VERIFIED.json', verified)
    del model, ck, model_ref[:]
    if phase != 'P_resnet50':
        evaluate(M, protocol, root, phase, checkpoint, progress)
        verified['test_evaluated'] = True
        verified['test_scope'] = 'FF++ test and four cross-dataset evaluations, frozen checkpoint'
        write(destination/'RUN_VERIFIED.json', verified)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--protocol', required=True); p.add_argument('--phase', required=True)
    p.add_argument('--resnet-smoke', choices=['cpu', 'cuda']); args = p.parse_args()
    protocol = json.loads(Path(args.protocol).read_text()); root = Path(protocol['run_root'])
    assert json.loads((root/'FROZEN_PROTOCOL.json').read_text()) == protocol, 'Frozen protocol changed'
    assert sha(REPO/'CES_FL/src/ces_fl_diagram_model_v021.py') == protocol['core_sha256']
    M = model_module(); torch.set_num_threads(4); M.seed_all(17)
    if args.resnet_smoke:
        result = resnet_smoke(M, protocol['resnet_weights']['path'], args.resnet_smoke)
        write(root/'preflight'/f'resnet_smoke_{args.resnet_smoke}.json', result)
        print('RESNET_SOFTWARE_PASS', json.dumps(result), flush=True); return
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    sentinel = torch.empty(1, device='cuda')
    phase = args.phase; started = time.time()
    def progress(state, **kw):
        write(root/'runtime'/f'{phase}.progress.json', {'phase': state, 'stage': phase, 'time': time.time(),
              'elapsed_s': time.time()-started, 'pbs_job_id': os.environ.get('PBS_JOBID'), **kw})
    progress('CHECKING_FROZEN_PROTOCOL')
    for dataset, path in protocol['manifests'].items():
        assert sha(path) == protocol['manifest_audit'][dataset]['sha256']
    assert sha(protocol['resnet_weights']['path']) == protocol['resnet_weights']['sha256']
    if phase == 'X_evaluation':
        validation_audit(M, protocol, root, progress)
        for arm in ['B_detection_frozen', 'C_detection_invariance', 'A_imagenet_frozen']:
            entry = protocol['xception'][arm]; assert sha(entry['checkpoint']) == entry['sha256']
            evaluate(M, protocol, root, arm, entry['checkpoint'], progress)
    else:
        result = resnet_smoke(M, protocol['resnet_weights']['path'], 'cuda')
        write(root/'preflight'/f'{phase}_cuda_smoke.json', result)
        train(M, protocol, phase, root, progress)
    progress('DONE')
    write(root/'runtime'/f'{phase}.result.json', {'status': 'DONE', 'elapsed_s': time.time()-started,
          'core_sha256': protocol['core_sha256'], 'pbs_job_id': os.environ.get('PBS_JOBID')})


if __name__ == '__main__':
    main()
