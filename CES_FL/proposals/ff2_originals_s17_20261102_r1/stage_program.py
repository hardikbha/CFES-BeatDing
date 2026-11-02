"""Execute only an explicitly selected original-image FF2 milestone."""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1')
GIT = '/home/sachin.chaudhary/ces_fl_tools/git_cluster_20261031/bin/git'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def setup():
    q = json.loads((ROOT / 'request.json').read_text())
    source = Path(q['source_dir'])
    assert subprocess.check_output([GIT, '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == q['source_commit']
    assert subprocess.run([GIT, '-C', str(source), 'symbolic-ref', '-q', 'HEAD'], capture_output=True).returncode == 1
    assert not subprocess.check_output([GIT, '-C', str(source), 'status', '--porcelain', '--', 'CES_FL'], text=True).strip()
    code = source / 'CES_FL/src/ces_fl_diagram_model_v021.py'
    assert sha(code) == q['source_sha256']
    assert sha(q['config']) == q['config_sha256']
    for name, expected in q['runtime_hashes'].items():
        assert sha(ROOT / 'runtime' / name) == expected
    config = json.loads(Path(q['config']).read_text())
    assert sha(config['weights']) == q['weights_sha256']
    assert config['seed'] == 17 and config['epochs'] == 10 and config['stage'] == 'pretrain'
    assert config['encoder_policy'] == 'trainable' and config['bn_policy'] == 'update'
    assert config['mode'] == 'bce' and config['pairs'] is None and 'test' not in config
    spec = importlib.util.spec_from_file_location('ces_core', code)
    model = importlib.util.module_from_spec(spec); spec.loader.exec_module(model)
    model.torch.set_num_threads(4)
    return q, config, model


def inputs(q, c, m):
    rows, audit = {}, {}
    for split, expected in q['splits'].items():
        assert sha(expected['path']) == expected['sha256']
        rows[split] = m.read_rows(expected['path'])
        labels = {str(k): v for k, v in Counter(r['label'] for r in rows[split]).items()}
        assert len(rows[split]) == expected['rows'] and labels == expected['labels']
        audit[split] = {**expected, 'every_image_path_is_file': True, 'all_images_decoded': False,
                        'required_columns': ['image', 'label', 'identity_ids', 'source_ids'],
                        'ids8_label_counts': dict(Counter(str(r['label']) + ':' + r['ids8'] for r in rows[split]))}
        print('FF2_SPLIT_PASS', split, len(rows[split]), labels, flush=True)
    overlaps = {}
    for left, right in itertools.combinations(rows, 2):
        overlaps[left + ':' + right] = m.check_splits(rows[left], rows[right])
    for split in ['train', 'val']:
        buckets = defaultdict(list)
        for row in rows[split]:
            buckets[(row['label'], row['ids8'], row['family'])].append(row)
        chosen = [group[0] for _, group in sorted(buckets.items())]
        for row in chosen:
            tensor = m.PairImages([row], size=c['size'], backbone=c['backbone']).load(row['image'])
            assert tuple(tensor.shape) == (3, c['size'], c['size']) and m.torch.isfinite(tensor).all()
        audit[split]['stratified_decoded_images'] = len(chosen)
    result = {'status': 'PASS', 'source_commit': q['source_commit'], 'source_sha256': q['source_sha256'],
              'config_sha256': q['config_sha256'], 'splits': audit, 'overlaps': overlaps,
              'true_person_or_hidden_donor_independence_verified': False,
              'test_images_decoded_or_scored': False, 'training_started': False,
              'counterfactuals_used': 0}
    write(ROOT / 'checks/inputs_PASS.json', result)
    print('FF2_INPUTS_PASS', flush=True)


def cuda_check(m):
    t = m.torch
    assert t.cuda.is_available() and t.cuda.device_count() == 1
    t.cuda.set_device(0); t.cuda.set_per_process_memory_fraction(.20, 0)
    properties = t.cuda.get_device_properties(0)
    expected = os.environ['CUDA_VISIBLE_DEVICES'].lower().removeprefix('gpu-')
    observed = str(properties.uuid).lower().removeprefix('gpu-')
    assert expected == observed, (expected, observed)
    return {'device': 'cuda:0', 'name': properties.name, 'uuid': str(properties.uuid), 'memory_fraction': .20}


def smoke(q, c, m):
    assert json.loads((ROOT / 'checks/inputs_PASS.json').read_text())['status'] == 'PASS'
    binding = cuda_check(m)
    m.smoke('cuda', ROOT / 'smoke_cuda')
    result = json.loads((ROOT / 'smoke_cuda/RESULT.json').read_text())
    assert result['status'] == 'PASS' and result['script_sha256'] == q['source_sha256']
    rows = m.read_rows(c['train'])
    selected = []
    for y in [0, 1]:
        selected.extend([row for row in rows if row['label'] == y][:c['batch_size'] // 2])
    assert len(selected) == c['batch_size']
    batch = m.collate([m.PairImages(selected, size=c['size'], backbone=c['backbone'])[i] for i in range(len(selected))])
    assert batch['demo'] is None and batch['forensic'] is None
    model = m.DiagramCES(c['backbone'], c['weights'], c['hidden'], c['gate_hidden'],
                        encoder_policy=c['encoder_policy'], bn_policy=c['bn_policy'], head_seed=c['head_seed']).cuda().train()
    opt = m.build_optimizer(model, c)
    before = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
    losses, grad_counts = [], {}
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        output = model(batch['x'].cuda())
        m.torch.testing.assert_close(output['logit'], output['global_contribution'] + output['local_map'].mean((1, 2)) + model.fusion_bias)
        loss = m.ces_loss(output, batch['y'].cuda(), mode='bce', selectivity=c['selectivity'], lambda_d=c['lambda_d'],
                          lambda_f=c['lambda_f'], lambda_s=c['lambda_s'], tau=c['tau'], lambda_map=c['lambda_map'])
        assert m.torch.isfinite(loss['total']) and loss['demo_count'] == loss['forensic_count'] == 0
        loss['total'].backward()
        for name in ['encoder', 'global_head', 'patch_head', 'image_gate', 'patch_gate']:
            ps = [p for p in getattr(model, name).parameters() if p.requires_grad]
            assert ps and all(p.grad is not None and m.torch.isfinite(p.grad).all() for p in ps)
            assert sum(float(p.grad.abs().sum()) for p in ps) > 0
            grad_counts[name] = sum(p.numel() for p in ps if p.grad is not None)
        assert model.fusion_raw.grad is not None and m.torch.isfinite(model.fusion_raw.grad).all()
        assert model.fusion_bias.grad is not None and m.torch.isfinite(model.fusion_bias.grad).all()
        m.torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.)
        opt.step(); losses.append(float(loss['total'].detach()))
    after = model.encoder.state_dict()
    changed = [k for k, v in before.items() if not m.torch.equal(v, after[k].detach().cpu())]
    assert any('weight' in k for k in changed) and any('running_' in k for k in changed)
    write(ROOT / 'checks/smoke_PASS.json', {'status': 'PASS', 'binding': binding,
          'source_sha256': q['source_sha256'], 'config_sha256': q['config_sha256'],
          'synthetic_suite': result, 'real_train_only_optimizer_steps': 2, 'real_losses': losses,
          'parameter_counts': model.counts(), 'parameters_receiving_gradients': grad_counts,
          'encoder_parameters_changed': True, 'BN_buffers_changed': True,
          'no_counterfactuals': True, 'no_real_validation_or_test_gradient_updates': True})
    print('FF2_CUDA_AND_REAL_BACKWARD_PASS', losses, binding, flush=True)


def train(q, c, m):
    for name in ['inputs', 'smoke']:
        receipt = json.loads((ROOT / 'checks' / (name + '_PASS.json')).read_text())
        assert receipt['status'] == 'PASS' and receipt['config_sha256'] == q['config_sha256']
    binding = cuda_check(m)
    for split in ['train', 'val']:
        assert sha(q['splits'][split]['path']) == q['splits'][split]['sha256']
    assert not (ROOT / 'training').exists()
    print('FF2_TRAIN_START', json.dumps({'seed': 17, 'epochs': 10, 'binding': binding,
          'train_rows': q['splits']['train']['rows'], 'val_rows': q['splits']['val']['rows'],
          'test_evaluation': False, 'counterfactuals': False}), flush=True)
    start = time.monotonic()
    best = m.train_run(c, ROOT / 'training', 'cuda')
    history = [json.loads(line) for line in (ROOT / 'training/history.jsonl').read_text().splitlines() if line.strip()]
    assert [h['epoch'] for h in history] == list(range(1, 11))
    expected = max(history, key=lambda h: h['validation']['overall']['auc'])
    m.torch.cuda.empty_cache()
    model, ck = m.load_model(best, 'cuda')
    assert ck['epoch'] == expected['epoch'] and ck['config'] == c
    assert all(m.torch.isfinite(v).all() for v in model.state_dict().values())
    rows = m.read_rows(c['val'])
    logits = m.predict(model, rows, 'cuda', c['batch_size'], c['workers'], c['size'])
    replay = m.metrics(rows, logits, ck['calibration']['threshold_logit'], c['min_support'])
    assert json.dumps(replay, sort_keys=True) == json.dumps(ck['validation'], sort_keys=True)
    receipt = {'status': 'PASS', 'completed_epochs': 10, 'selected_epoch': ck['epoch'],
               'validation_auc': replay['overall']['auc'], 'validation': replay, 'calibration': ck['calibration'],
               'source_sha256': q['source_sha256'], 'config_sha256': q['config_sha256'],
               'checkpoint_sha256': sha(best), 'strict_checkpoint_restore': True,
               'native_independent_validation_replay': True, 'elapsed_seconds': time.monotonic() - start,
               'test_evaluated': False, 'counterfactuals': 0, 'binding': binding}
    write(ROOT / 'training/RUN_VERIFIED.json', receipt)
    write(ROOT / 'checks/train_PASS.json', receipt)
    print('FF2_TRAIN_COMPLETED_VERIFIED', ck['epoch'], replay['overall']['auc'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['inputs', 'smoke', 'train'], required=True)
    stage = p.parse_args().stage
    request, config, core = setup()
    {'inputs': inputs, 'smoke': smoke, 'train': train}[stage](request, config, core)
