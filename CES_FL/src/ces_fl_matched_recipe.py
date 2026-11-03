#!/usr/bin/env python3
"""Prospective CES-FL-only Ding-recipe control. Never launches a Ding run.

Uses the hash-pinned L_H science implementation unchanged. New protocol: uniform
BCE, fake-once/real-replacement sampling, SGD, 50+50 epochs, final checkpoint.
This matches the archived main-stream recipe, NOT the entire Ding method or
compute cost. Downstream exit tuning/frozen BN remain explicit differences.
No TEST command, model download, automatic qualification or scheduler submission.
"""
import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Sampler

CORE_SHA = '08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177'
CORE_FILE = Path(__file__).with_name('ces_fl_head_ablation.py')
if hashlib.sha256(CORE_FILE.read_bytes()).hexdigest() != CORE_SHA:
    raise RuntimeError('Pinned L_H science source changed; re-review this adapter')
spec = importlib.util.spec_from_file_location('lh_science', CORE_FILE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
VERSION = 'CESFL_DING_RECIPE_v1'
PROTOCOL = dict(optimizer='SGD', momentum=0.9, sampling='fake_once_real_replacement',
                selection='final_epoch', classification_weighting='uniform',
                train_resize='direct_224', validation_resize='short_side_224_center_crop',
                grad_clip=None, tf32=False, scheduler='constant',
                primary_threshold='fixed_probability_0.5',
                secondary_threshold='validation_real_empirical_10pct')


def check(c, fixture=False, inputs=True):
    if set(c) != {'format', 'source_sha256', 'core_sha256', 'stage', 'protocol', 'base'}:
        raise ValueError('Adapter config schema mismatch')
    if c['format'] != VERSION or c['source_sha256'] != core.sha(__file__) or c['core_sha256'] != CORE_SHA:
        raise ValueError('Adapter format/source mismatch')
    if c['stage'] not in ('P', 'A', 'H') or c['protocol'] != PROTOCOL:
        raise ValueError('Unreviewed stage/protocol change')
    b = c['base']; pre = c['stage'] == 'P'
    expected = dict(head_type='linear', encoder_policy='full' if pre else 'exit',
                    encoder_lr=.001, head_lr=.001, weight_decay=.005, amp=False,
                    source_sha256=CORE_SHA, error_control='hierarchical_real',
                    lambda_fair=.1, lambda_sensitivity=.1, lambda_alignment=.1,
                    sham_weight=.5, aux_every=4, group_per_class=4, target_fpr=.1,
                    margin=.5, temperature=1., alignment_warmup_epochs=0)
    if not fixture:
        expected.update(epochs=50, batch_size=128 if pre else 64, seed=17, size=224,
                        dataset_id='FFPP_DING_PUBLIC', train_calibration_size=2048,
                        min_train_support=20, min_eval_support=20,
                        axes=['gender', 'race', 'age_bin', 'skin_bin', 'gender_race'])
    if any(b.get(k) != v for k, v in expected.items()):
        raise ValueError('Config differs from the frozen prospective recipe')
    if b['batch_size'] % 2 or b['batch_size'] < 2:
        raise ValueError('Batch size is an even number of IMAGES, not pairs')
    if bool(b['synthetic_fixture']) != fixture:
        raise ValueError('Synthetic/production mismatch')
    if inputs:
        core.check_config(b, fixture=fixture)
    return c


def load_config(path):
    return check(json.loads(Path(path).read_text()))


class PairedBatches(Sampler):
    """Fake rows once, a random real with replacement for each, final partial kept.

    Batch RNG is independent of workers and auxiliary RNG. A and H see identical
    main-stream indices; it matches Ding's distribution, not its exact RNG draws.
    """
    def __init__(self, rows, image_batch, seed, epoch):
        self.real = [i for i, r in enumerate(rows) if r['label'] == 0]
        self.fake = [i for i, r in enumerate(rows) if r['label'] == 1]
        if not self.real or not self.fake or image_batch < 2 or image_batch % 2:
            raise ValueError('Both classes and an even image batch required')
        self.pairs = image_batch // 2
        self.seed = core.seed_for(seed, epoch, 'matched-main-stream')

    def __len__(self):
        return math.ceil(len(self.fake) / self.pairs)

    def __iter__(self):
        rng = random.Random(self.seed); fake = list(self.fake); rng.shuffle(fake)
        for start in range(0, len(fake), self.pairs):
            part = fake[start:start + self.pairs]
            yield [rng.choice(self.real) for _ in part] + part


def loader(rows, b, epoch, pairs):
    return DataLoader(core.Originals(rows, b, epoch, pairs),
                      batch_sampler=PairedBatches(rows, b['batch_size'], b['seed'], epoch),
                      num_workers=b['workers'], generator=torch.Generator().manual_seed(b['seed'] + epoch))


def set_training(model, pretraining):
    model.train()
    # Ding classification updates BN. Frozen adaptation BN isolates A/H losses.
    if pretraining:
        model.encoder.train()


def check_parent(ck, c):
    b = c['base']; parent = ck.get('config', {})
    if ck.get('format') != VERSION or parent.get('stage') != 'P' or ck.get('source_sha256') != core.sha(__file__):
        raise ValueError('Requires this recipe\'s completed P checkpoint')
    pb = parent['base']
    if ck['epoch'] != pb['epochs'] or bool(pb['synthetic_fixture']) != bool(b['synthetic_fixture']):
        raise ValueError('Incomplete/synthetic pretraining mismatch')
    for key in ('train', 'val', 'train_sha256', 'val_sha256', 'overlap_file', 'overlap_sha256', 'seed', 'head_type'):
        if b[key] != pb[key]:
            raise ValueError('P/downstream input or initialization mismatch: ' + key)
    check(parent, fixture=b['synthetic_fixture'])


def initial(c):
    b = c['base']; core.seed_all(b['seed'])
    m = core.Detector(b['hidden'], b['gate_hidden'], b['encoder_policy'], 'linear')
    state = torch.load(b['init_checkpoint'], map_location='cpu', weights_only=True)
    if c['stage'] == 'P':
        # Public Cadene ImageNet weights, external data; unused classifier excluded.
        state = {k: (v[:, :, None, None] if 'pointwise' in k and v.ndim == 2 else v)
                 for k, v in state.items() if not k.startswith(('fc.', 'last_linear.', 'adjust_channel.'))}
        m.encoder.load_state_dict(state, strict=True)
    else:
        check_parent(state, c)
        # Continue BOTH the encoder and linear head. A/H begin identically.
        m.load_state_dict(state['model'], strict=True)
    return m


def optimizer(model, b):
    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                           lr=b['head_lr'], momentum=.9, weight_decay=b['weight_decay'])


def step(loss, model, opt):
    if not all(torch.isfinite(v) for v in loss.values()):
        raise RuntimeError('Nonfinite loss')
    opt.zero_grad(set_to_none=True); loss['total'].backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
        raise RuntimeError('Missing/nonfinite trainable gradient')
    opt.step()  # No clipping: explicitly matches archived SGD recipe.


def val_pixels(row, size):
    if row.get('image_sha256') and core.sha(row['image']) != row['image_sha256']:
        raise ValueError('Validation image changed after preflight')
    with Image.open(row['image']) as im:
        im = im.convert('RGB'); w, h = im.size
        new = (size, int(size * h / w)) if w <= h else (int(size * w / h), size)
        im = im.resize(new, Image.Resampling.BILINEAR)
        left, top = round((new[0] - size) / 2), round((new[1] - size) / 2)
        a = np.array(im.crop((left, top, left + size, top + size)), dtype=np.float32) / 255.
    return core.normalize(torch.from_numpy(a).permute(2, 0, 1))


@torch.inference_mode()
def predict_val(model, rows, b, device):
    model.eval(); logits = []
    for start in range(0, len(rows), b['batch_size']):
        x = torch.stack([val_pixels(r, b['size']) for r in rows[start:start + b['batch_size']]])
        logits.extend(model(x.to(device))['logit'].float().cpu().tolist())
    z = np.asarray(logits, np.float64)
    if not np.isfinite(z).all(): raise RuntimeError('Nonfinite validation logits')
    return z


def reports(rows, z, b):
    cal = core.calibrate(z[np.array([r['label'] == 0 for r in rows])], b['target_fpr'])
    return cal, {'fixed_probability_0_5': core.metric_report(rows, z, 0., b['min_eval_support']),
                 'validation_fpr10': core.metric_report(rows, z, cal['threshold_logit'], b['min_eval_support']),
                 'fixed05_all_nonempty_groups': core.metric_report(rows, z, 0., 1)}


def preflight(c, out):
    out = core.fresh_dir(out); b = c['base']
    core.preflight(b, out / 'inputs')
    rows, _, _ = core.checked_inputs(b, out / 'inputs')
    n = sum(r['label'] == 1 for r in rows)
    record = dict(status='PASS', source_sha256=core.sha(__file__), core_sha256=CORE_SHA,
                  config_digest=core.digest(c), input_receipt_sha256=core.sha(out / 'inputs/PREFLIGHT.json'),
                  class_counts=dict(Counter(str(r['label']) for r in rows)),
                  steps_per_epoch=math.ceil(n / (b['batch_size'] // 2)), main_images_per_epoch=2*n,
                  images_per_full_batch=b['batch_size'], real_fake_pairs_per_full_batch=b['batch_size']//2,
                  test_read=False)
    core.write_json(out / 'PREFLIGHT.json', record)
    return record


def checked_inputs(c, out):
    root = Path(out); r = json.loads((root / 'PREFLIGHT.json').read_text())
    if r['status'] != 'PASS' or r['source_sha256'] != core.sha(__file__) or r['config_digest'] != core.digest(c):
        raise ValueError('Adapter preflight stale')
    if r['input_receipt_sha256'] != core.sha(root / 'inputs/PREFLIGHT.json'):
        raise ValueError('Input receipt changed')
    rows, val, _ = core.checked_inputs(c['base'], root / 'inputs')
    return rows, val


def train(c, preflight_dir, out, device, budget_hours, qualification=None, resume=False,
          fixture=False, stop_after=None, profile_steps=0):
    check(c, fixture=fixture)
    if not math.isfinite(budget_hours) or budget_hours <= 0:
        raise ValueError('Positive cumulative per-run execution allowance required')
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    b = c['base']; stage = c['stage']; is_h = stage == 'H'
    qhash = core.check_qualification(b, qualification) if is_h and not fixture and not profile_steps else None
    rows, val = checked_inputs(c, preflight_dir); active, groups = core.group_inventory(rows, b)
    model = initial(c).to(device); opt = optimizer(model, b); out = Path(out)
    weights = torch.ones(len(rows)); start = 0; elapsed = 0.; history = []
    if resume:
        if (out / 'DONE').exists() or profile_steps: raise ValueError('Cannot resume completed/profile run')
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=True)
        if ck['config'] != c or ck['qualification_sha256'] != qhash:
            raise ValueError('Resume protocol/review mismatch')
        if json.loads((out / 'history.json').read_text()) != ck['history']:
            raise ValueError('History/checkpoint disagreement; preserve and audit interrupted write')
        model.load_state_dict(ck['model'], strict=True); opt.load_state_dict(ck['optimizer'])
        start, elapsed, history = ck['epoch'], ck['elapsed_seconds'], ck['history']
        for state in opt.state.values():
            for k, v in state.items():
                if torch.is_tensor(v): state[k] = v.to(device)
    else:
        core.fresh_dir(out); core.write_json(out / 'config.json', c)
        core.write_json(out / 'RUN.json', dict(source_sha256=core.sha(__file__), parameters=model.parameter_report(),
            config_digest=core.digest(c), groups=groups, device=str(device), torch=str(torch.__version__),
            qualification_sha256=qhash, test_read=False, profile_only=bool(profile_steps)))
    real = sorted([r for r in rows if r['label'] == 0],
                  key=lambda r: core.seed_for(b['seed'], 'train-calibration', r['image']))[:b['train_calibration_size']]
    begin = time.perf_counter(); base_elapsed = elapsed
    with (out / 'attempts.jsonl').open('a') as f:
        f.write(json.dumps(dict(resume=resume, budget_hours=budget_hours, completed_epochs_before=start))+'\n')
    for epoch in range(start, b['epochs']):
        elapsed = base_elapsed + time.perf_counter() - begin
        if elapsed + (history[-1]['epoch_seconds'] if history else 0) > budget_hours*3600 or (stop_after is not None and epoch >= stop_after):
            core.write_json(out / 'STATUS.json', dict(status='INCOMPLETE', completed_epochs=epoch))
            return {'status': 'INCOMPLETE'}
        core.seed_all(b['seed'] + epoch); epoch_start = time.perf_counter()
        threshold = core.calibrate(core.predict(model, real, b, device), b['target_fpr']) if is_h else None
        set_training(model, stage == 'P'); seen = Counter(); paired = 0; sums = Counter(); steps = 0
        main = loader(rows, b, epoch, is_h); event = epoch * math.ceil(len(main)/b['aux_every'])
        auxiliary_originals = 0; auxiliary_real_pairs = 0; aux_counts = Counter(); step_times = []
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        for idx, batch in enumerate(main):
            if device.type == 'cuda': torch.cuda.synchronize(device)
            tick = time.perf_counter(); aux = None
            if is_h and idx % b['aux_every'] == 0:
                aux = core.auxiliary(rows, active, b, event, epoch); event += 1
                auxiliary_originals += len(aux['y']); auxiliary_real_pairs += int((aux['y'] == 0).sum())
                aux_counts[aux['axis']] += 1
            loss, count = core.train_step(model, batch, aux, weights, b, 'D' if is_h else 'A',
                                          threshold['threshold_logit'] if threshold else 0., device, epoch)
            if profile_steps and aux is not None and aux['axis'] == 'gender_race':
                grads = torch.autograd.grad(loss['error_fairness'], [p for p in model.parameters() if p.requires_grad],
                                            retain_graph=True, allow_unused=True)
                norm = sum(float(g.float().square().sum()) for g in grads if g is not None)
                if not math.isfinite(norm) or norm <= 0: raise RuntimeError('Hierarchical fairness gradient absent')
            step(loss, model, opt)
            if device.type == 'cuda': torch.cuda.synchronize(device)
            step_times.append(time.perf_counter()-tick); seen.update(batch['index'].tolist()); paired += count; steps += 1
            for k, v in loss.items(): sums[k] += float(v.detach())
            if steps % 100 == 0: print(json.dumps(dict(stage=stage, epoch=epoch+1, step=steps, total_steps=len(main))), flush=True)
            if profile_steps and steps >= profile_steps:
                # A discarded warm-up/profile model, never a benchmark checkpoint.
                pv_start = time.perf_counter(); predict_val(model, val[:min(512,len(val))], b, device)
                val_sec = (time.perf_counter()-pv_start)*len(val)/min(512,len(val))
                observed = time.perf_counter()-epoch_start
                estimate = ((observed/steps)*len(main)+val_sec)*b['epochs']/3600*1.4
                rec = dict(status='PASS', profile_only=True, source_sha256=core.sha(__file__),
                    config_digest=core.digest(c), device=str(device), steps=steps,
                    auxiliary_axes_checked=dict(aux_counts), estimate_training_hours_with_40pct_margin=estimate,
                    seconds_observed=observed, mean_step_seconds=float(np.mean(step_times)),
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                    test_read=False, scientific_validity_proven=False)
                core.write_json(out / 'PROFILE.json', rec); return rec
        n_fake = sum(r['label']==1 for r in rows)
        if any(seen[i] != 1 for i,r in enumerate(rows) if r['label']==1) or sum(seen.values()) != 2*n_fake:
            raise RuntimeError('Fake-once/real-replacement exposure mismatch')
        if is_h and paired != n_fake: raise RuntimeError('Real-pair draw coverage mismatch')
        z = predict_val(model, val, b, device); cal, report = reports(val,z,b)
        record = dict(epoch=epoch+1, steps=steps, main_images=sum(seen.values()), unique_originals=len(seen),
                      unique_real_sources=sum(rows[i]['label']==0 for i in seen), main_real_pair_draws=paired,
                      auxiliary_originals=auxiliary_originals, auxiliary_real_pair_draws=auxiliary_real_pairs,
                      auxiliary_events=dict(aux_counts), losses_step_mean={k:v/steps for k,v in sums.items()},
                      train_calibration=threshold, validation=report, calibration=cal,
                      epoch_seconds=time.perf_counter()-epoch_start, selection='final_epoch')
        history.append(record); elapsed=base_elapsed+time.perf_counter()-begin
        ck = dict(format=VERSION, source_sha256=core.sha(__file__), config=c, model=model.state_dict(),
                  optimizer=opt.state_dict(), epoch=epoch+1, elapsed_seconds=elapsed, history=history,
                  validation=report, calibration=cal, qualification_sha256=qhash)
        core.atomic_save(out/'last.pt',ck); core.write_json(out/'history.json',history)
        print(json.dumps(dict(stage=stage,epoch=epoch+1,auc=report['fixed_probability_0_5']['overall']['auc'],seconds=elapsed)),flush=True)
    if profile_steps: raise RuntimeError('Profile too short to cover requested steps; no profile receipt issued')
    # Final epoch is fixed prospectively, even if an earlier epoch scored better.
    selected=torch.load(out/'last.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(selected['model'],strict=True)
    z=predict_val(model,val,b,device); cal,report=reports(val,z,b)
    if report != selected['validation'] or cal != selected['calibration']: raise RuntimeError('Final checkpoint replay failed')
    scores=dict(rows=val,logits=z.tolist(),threshold=0.,min_support=b['min_eval_support'],
                manifest_sha256=b['val_sha256'],checkpoint_sha256=core.sha(out/'last.pt'),split='validation',test_read=False)
    core.write_json(out/'validation_fixed05_PRIVATE.json',scores)
    scores['threshold']=cal['threshold_logit'];core.write_json(out/'validation_fpr10_PRIVATE.json',scores)
    result=dict(status='COMPLETED_VERIFIED',stage=stage,epochs_completed=b['epochs'],selected_epoch=selected['epoch'],
                checkpoint_sha256=core.sha(out/'last.pt'),source_sha256=core.sha(__file__),config_digest=core.digest(c),
                validation=report,elapsed_seconds=elapsed,test_evaluated=False,synthetic_fixture=fixture)
    core.write_json(out/'RUN_VERIFIED.json',result);(out/'DONE').write_text('Final epoch independently replayed; TEST not read.\n')
    return result


def bind_parent(template, parent_run, out):
    c=json.loads(Path(template).read_text());root=Path(parent_run)
    proof=json.loads((root/'RUN_VERIFIED.json').read_text())
    if not (root/'DONE').is_file() or proof['status']!='COMPLETED_VERIFIED' or proof['stage']!='P':
        raise ValueError('Completed verified P required')
    path=root/'last.pt'
    if core.sha(path)!=proof['checkpoint_sha256']:raise ValueError('P checkpoint changed')
    if c['stage'] not in ('A','H'):raise ValueError('Only downstream templates can bind P')
    c['base']['init_checkpoint']=str(path.resolve());c['base']['init_sha256']=core.sha(path)
    check(c);initial(c)
    if Path(out).exists():raise ValueError('Preserve bound config')
    core.write_json(out,c)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    for cmd in ('preflight','preview','profile','train'):
        q=sub.add_parser(cmd);q.add_argument('--config',required=True);q.add_argument('--out',required=True)
        if cmd!='preflight':q.add_argument('--preflight',required=True)
        if cmd in ('profile','train'):
            q.add_argument('--device',choices=['cpu','cuda'],default='cuda');q.add_argument('--budget-hours',type=float,required=True)
        if cmd=='profile':q.add_argument('--steps',type=int,default=24)
        if cmd=='train':
            q.add_argument('--profile-receipt',required=True);q.add_argument('--qualification');q.add_argument('--resume',action='store_true')
    q=sub.add_parser('bind-parent');q.add_argument('--template',required=True);q.add_argument('--parent-run',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('qualify');q.add_argument('--preview',required=True);q.add_argument('--review',required=True);q.add_argument('--out',required=True)
    a=p.parse_args();torch.set_num_threads(4)
    if a.cmd=='bind-parent':return bind_parent(a.template,a.parent_run,a.out)
    if a.cmd=='qualify':return core.qualify(a.preview,a.review,a.out)
    c=load_config(a.config)
    if a.cmd=='preflight':return preflight(c,a.out)
    if a.cmd=='preview':
        checked_inputs(c,a.preflight);return core.preview(c['base'],Path(a.preflight)/'inputs',a.out)
    if a.cmd=='profile' and not 24<=a.steps<=100:raise ValueError('Profile24–100steps, covering every auxiliary axis')
    if a.cmd=='train':
        rec=json.loads(Path(a.profile_receipt).read_text())
        if rec.get('status')!='PASS' or rec.get('source_sha256')!=core.sha(__file__) or rec.get('config_digest')!=core.digest(c) or rec.get('device')!=a.device:
            raise ValueError('Missing/stale target-device profile')
        if not a.resume and rec['estimate_training_hours_with_40pct_margin']>a.budget_hours:
            raise ValueError('Profile estimate exceeds explicitly supplied run allowance')
    return train(c,a.preflight,a.out,torch.device(a.device),a.budget_hours,
                 getattr(a,'qualification',None),getattr(a,'resume',False),profile_steps=a.steps if a.cmd=='profile' else 0)


if __name__=='__main__':main()
