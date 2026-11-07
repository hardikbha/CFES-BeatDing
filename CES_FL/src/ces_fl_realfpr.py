"""FF2 stretch experiment (7 Nov 2026): real-only false-alarm equalisation and train-time JPEG augmentation.

Reuses the verified head-ablation model, data checks and metrics (ces_fl_head_ablation.py, unchanged). Arms:
  A  BCE only (same as L_A)
  R  BCE + real-only false-alarm equalisation on a group-balanced real microbatch
Each arm optionally adds JPEG augmentation of the classification stream (`jpeg_aug`). No probe terms.
Checkpoint selection: minimum hard gender x race F_FPR among epochs with VAL AUC >= min_auc and FNR <= max_fnr
(ties: higher AUC, then earlier epoch). VAL only; TEST is never read.
"""
import argparse, io, json, math, random, time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
import importlib.util

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

_spec = importlib.util.spec_from_file_location('ham', Path(__file__).with_name('ces_fl_head_ablation.py'))
ham = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ham)

VERSION = 'CESFL_REALFPR_v1'
ARMS = ('A', 'R')
EXTRA_KEYS = {'rf_format', 'arm', 'jpeg_aug', 'jpeg_scale_file', 'jpeg_scale_sha256', 'reals_per_group', 'fair_every',
              'lambda_rf', 'rf_ramp_epochs', 'tau_start', 'tau_end', 'threshold_refreshes_per_epoch', 'min_auc', 'max_fnr'}


def check(c):
    base = {k: v for k, v in c.items() if k not in EXTRA_KEYS}
    ham.check_config(base, fixture=bool(c['synthetic_fixture']), bound=False)
    if set(c) - set(base) != EXTRA_KEYS or c['rf_format'] != VERSION or c['arm'] not in ARMS: raise ValueError('rf schema')
    if type(c['jpeg_aug']) is not bool or ham.sha(c['jpeg_scale_file']) != c['jpeg_scale_sha256']: raise ValueError('jpeg spec')
    for k in ('reals_per_group', 'fair_every', 'rf_ramp_epochs', 'threshold_refreshes_per_epoch'):
        if type(c[k]) is not int or c[k] < 1: raise ValueError(k)
    if not (0 < c['tau_end'] <= c['tau_start'] and c['lambda_rf'] > 0 and 0 < c['min_auc'] < 1 and 0 < c['max_fnr'] < 1):
        raise ValueError('rf numbers')
    if c['source_sha256'] != ham.sha(ham.__file__): raise ValueError('head-ablation source changed')
    return c


def rf_digest(c): return ham.digest(c)


# ---------------- JPEG augmentation (classification stream only) ----------------
def jpeg_plan(seed, epoch, index):
    """50% unchanged, 25% q75 4:2:0, 25% integer q in [70, 95] with 4:2:0 or 4:4:4 (equal). Same for every label/group."""
    rng = random.Random(ham.seed_for(seed, 'jpeg', epoch, index)); u = rng.random()
    if u < .5: return None
    if u < .75: return 75, 2, rng.random()
    return rng.randint(70, 95), rng.choice((2, 0)), rng.random()


def jpeg_emulate(img, quality, subsampling, scale):
    """Emulate a source-scale JPEG on a 256 px crop: resample by 1/scale, JPEG, resample back."""
    w, h = img.size; sw, sh = max(8, round(w / scale)), max(8, round(h / scale))
    small = img.resize((sw, sh), Image.Resampling.BILINEAR)
    buf = io.BytesIO(); small.save(buf, 'JPEG', quality=int(quality), subsampling=int(subsampling)); buf.seek(0)
    with Image.open(buf) as j: return j.convert('RGB').resize((w, h), Image.Resampling.BILINEAR)


def load_pixels_aug(row, size, plan, scales):
    data = Path(row['image']).read_bytes()
    if row.get('image_sha256') and ham.hashlib.sha256(data).hexdigest() != row['image_sha256']: raise ValueError('Image bytes changed')
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert('RGB')
        if plan is not None:
            q, sub, u = plan; s = float(scales[min(len(scales) - 1, int(u * len(scales)))])
            im = jpeg_emulate(im, q, sub, s)
        a = np.asarray(im.resize((size, size), Image.Resampling.BILINEAR), np.float32).copy() / 255.
    return torch.from_numpy(a).permute(2, 0, 1)


class AugOriginals(Dataset):
    def __init__(self, rows, c, epoch, scales): self.rows, self.c, self.epoch, self.scales = rows, c, epoch, scales
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        plan = jpeg_plan(self.c['seed'], self.epoch, i) if self.c['jpeg_aug'] else None
        x = load_pixels_aug(self.rows[i], self.c['size'], plan, self.scales)
        return {'x': ham.normalize(x), 'y': self.rows[i]['label'], 'index': i}


def train_loader(rows, c, epoch, scales):
    return DataLoader(AugOriginals(rows, c, epoch, scales), batch_size=c['batch_size'], shuffle=True, num_workers=c['workers'],
                      drop_last=False, generator=torch.Generator().manual_seed(c['seed'] + epoch))


# ---------------- group-balanced real microbatches (unchanged images) ----------------
class RealPlan(Dataset):
    def __init__(self, rows, pools, c, epoch, count):
        self.rows, self.pools, self.c, self.epoch, self.count = rows, pools, c, epoch, count
        self.names = sorted(pools)
    def __len__(self): return self.count
    def __getitem__(self, k):
        rng = random.Random(ham.seed_for(self.c['seed'], 'realplan', self.epoch, k)); xs, gs = [], []
        for gi, g in enumerate(self.names):
            pool = self.pools[g]; n = self.c['reals_per_group']
            for ix in (rng.sample(pool, n) if len(pool) >= n else rng.choices(pool, k=n)):
                xs.append(ham.normalize(ham.load_pixels(self.rows[ix], self.c['size']))); gs.append(gi)
        return {'x': torch.stack(xs), 'group': torch.tensor(gs)}


def real_loader(rows, pools, c, epoch, count):
    w = 0 if c['workers'] == 0 else max(2, c['workers'] // 2)
    return DataLoader(RealPlan(rows, pools, c, epoch, count), batch_size=None, shuffle=False, num_workers=w,
                      collate_fn=ham._identity, prefetch_factor=2 if w else None)


def real_fpr_loss(z, groups, priors, threshold, tau):
    """mean_g |r_g - stopgrad(sum_h pi_h r_h)|, r_g = mean sigmoid((z - t) / tau) over group g's reals."""
    rates = torch.stack([torch.sigmoid((z[groups == g].float() - threshold) / tau).mean() for g in range(len(priors))])
    pooled = (priors.to(rates) * rates).sum().detach()
    return (rates - pooled).abs().mean(), rates.detach()


def schedule(c, epoch_float):
    ramp = min(1., epoch_float / c['rf_ramp_epochs'])
    frac = min(1., epoch_float / max(1, c['epochs'] - 1))
    return c['lambda_rf'] * ramp, c['tau_start'] + (c['tau_end'] - c['tau_start']) * frac


# ---------------- selection rule ----------------
def feasible_key(report, c):
    o = report['overall']; f = report['axes']['gender_race']['F_FPR']
    ok = o['auc'] >= c['min_auc'] and o['fnr'] <= c['max_fnr'] and f is not None
    return ok, (f if ok else math.inf, -o['auc'])


def train(c, preflight_dir, out, device, budget_hours, resume=False):
    check(c); out = Path(out)
    rows, val, _ = ham.checked_inputs(c, preflight_dir)
    scales = np.load(c['jpeg_scale_file']).astype(np.float64)
    ham.seed_all(c['seed']); model = ham.load_initial(c).to(device); opt = ham.build_optimizer(model, c)
    weights = ham.original_weights(rows, c['balance_axis'])
    reals = [i for i, r in enumerate(rows) if r['label'] == 0]
    pools = {}
    for i in reals:
        g = rows[i]['gender_race']
        if g.lower() != 'unknown': pools.setdefault(g, []).append(i)
    pools = {g: p for g, p in pools.items() if len(p) >= c['min_train_support']}
    names = sorted(pools); priors = torch.tensor([len(pools[g]) for g in names], dtype=torch.float32); priors /= priors.sum()
    cal_rows = sorted([rows[i] for i in reals], key=lambda r: ham.seed_for(c['seed'], 'train-calibration', r['image']))[:c['train_calibration_size']]
    steps_per_epoch = math.ceil(len(rows) / c['batch_size']); events = math.ceil(steps_per_epoch / c['fair_every'])
    refresh = max(1, steps_per_epoch // c['threshold_refreshes_per_epoch'])
    best_key, history, start, elapsed = (math.inf, 0.), [], 0, 0.
    if resume:
        ck = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        if ck['config'] != c: raise ValueError('resume config mismatch')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        for st in opt.state.values():
            for k, v in st.items():
                if torch.is_tensor(v): st[k] = v.to(device)
        start, best_key, history, elapsed = ck['epoch'], tuple(ck['best_key']), ck['history'], ck['elapsed_seconds']
    else:
        ham.fresh_dir(out); ham.write_json(out / 'config.json', c)
        ham.write_json(out / 'RUN.json', {'format': VERSION, 'arm': c['arm'], 'jpeg_aug': c['jpeg_aug'], 'config_digest': rf_digest(c),
                                         'rf_source_sha256': ham.sha(__file__), 'head_ablation_sha256': ham.sha(ham.__file__),
                                         'test_read': False})
    begin = time.perf_counter(); base = elapsed
    for epoch in range(start, c['epochs']):
        est = history[-1]['epoch_seconds'] if history else 0.
        if base + time.perf_counter() - begin + est > budget_hours * 3600:
            ham.write_json(out / 'STATUS.json', {'status': 'INCOMPLETE_BUDGET_STOP', 'completed_epochs': epoch}); return
        t0 = time.perf_counter(); ham.seed_all(c['seed'] + epoch); model.train()
        thr = ham.calibrate(ham.predict(model, cal_rows, c, device), c['target_fpr'])['threshold_logit']; model.train()
        realit = iter(real_loader(rows, pools, c, epoch, events)) if c['arm'] == 'R' else None
        sums, ema, n = Counter(), None, 0; thr_log = [thr]
        for step, b in enumerate(train_loader(rows, c, epoch, scales)):
            if step and step % refresh == 0:
                thr = ham.calibrate(ham.predict(model, cal_rows, c, device), c['target_fpr'])['threshold_logit']; model.train(); thr_log.append(thr)
            lam, tau = schedule(c, epoch + step / steps_per_epoch)
            x, y, ix = b['x'].to(device), b['y'].to(device).float(), b['index']
            ctx = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' and c['amp'] else nullcontext()
            with ctx:
                z = model(x)['logit']
                cls = (F.binary_cross_entropy_with_logits(z.float(), y, reduction='none') * weights[ix].to(device)).mean()
                fair = z.float().sum() * 0
                if realit is not None and step % c['fair_every'] == 0:
                    rb = next(realit); zr = model(rb['x'].to(device))['logit']
                    fair, rates = real_fpr_loss(zr, rb['group'].to(device), priors, thr, tau)
                    ema = rates.cpu() if ema is None else .9 * ema + .1 * rates.cpu()
                total = cls + c['fair_every'] * lam * fair
            ham.finite_step({'total': total, 'classification': cls, 'fair': fair}, model, opt)
            sums['classification'] += float(cls.detach()); sums['fair'] += float(fair.detach()); n += 1
            if n % 100 == 0: print(json.dumps({'arm': c['arm'], 'jpeg': c['jpeg_aug'], 'epoch': epoch + 1, 'step': n, 'loss': float(total.detach())}), flush=True)
        zv = ham.predict(model, val, c, device); yv = np.asarray([r['label'] for r in val])
        cal = ham.calibrate(zv[yv == 0], c['target_fpr']); report = ham.metric_report(val, zv, cal['threshold_logit'], c['min_eval_support'])
        ok, key = feasible_key(report, c); improved = ok and key < best_key
        if improved: best_key = key
        rec = {'epoch': epoch + 1, 'steps': n, 'losses_step_mean': {k: v / n for k, v in sums.items()}, 'train_thresholds': thr_log,
               'ema_group_soft_fpr': dict(zip(names, ema.tolist())) if ema is not None else None, 'validation': report,
               'calibration': cal, 'feasible': ok, 'selected_this_epoch': improved, 'epoch_seconds': time.perf_counter() - t0}
        history.append(rec); elapsed = base + time.perf_counter() - begin
        ck = {'format': VERSION, 'config': c, 'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch + 1,
              'best_key': list(best_key), 'validation': report, 'calibration': cal, 'history': history, 'elapsed_seconds': elapsed}
        if improved: ham.atomic_save(out / 'best.pt', ck)
        ham.atomic_save(out / 'last.pt', ck); ham.write_json(out / 'history.json', history)
        print(json.dumps({'arm': c['arm'], 'jpeg': c['jpeg_aug'], 'epoch': epoch + 1, 'auc': report['overall']['auc'],
                          'fnr': report['overall']['fnr'], 'gxr_F_FPR': report['axes']['gender_race']['F_FPR'], 'feasible': ok}), flush=True)
    if not (out / 'best.pt').exists():
        ham.write_json(out / 'RUN_VERIFIED.json', {'status': 'NO_FEASIBLE_CHECKPOINT', 'epochs_completed': c['epochs'], 'test_evaluated': False})
        (out / 'DONE').write_text('Completed; no epoch met the AUC/FNR constraints. TEST not read.\n'); return
    sel = torch.load(out / 'best.pt', map_location='cpu', weights_only=False)
    m2 = ham.Detector(c['hidden'], c['gate_hidden'], c['encoder_policy'], c['head_type']).to(device); m2.load_state_dict(sel['model'], strict=True)
    zl = ham.predict(m2, val, c, device); rep = ham.metric_report(val, zl, sel['calibration']['threshold_logit'], c['min_eval_support'])
    if rep != sel['validation']: raise RuntimeError('Selected checkpoint replay mismatch')
    ham.write_json(out / 'validation_scores_PRIVATE.json', {'rows': val, 'logits': zl.tolist(), 'threshold': sel['calibration']['threshold_logit'],
        'min_support': c['min_eval_support'], 'manifest_sha256': c['val_sha256'], 'checkpoint_sha256': ham.sha(out / 'best.pt'),
        'calibration_source': 'selected checkpoint validation only', 'split': 'validation', 'test_read': False})
    ham.write_json(out / 'RUN_VERIFIED.json', {'status': 'COMPLETED_VERIFIED', 'arm': c['arm'], 'jpeg_aug': c['jpeg_aug'], 'epochs_completed': c['epochs'],
        'selected_epoch': sel['epoch'], 'selected_checkpoint_sha256': ham.sha(out / 'best.pt'), 'config_digest': rf_digest(c),
        'rf_source_sha256': ham.sha(__file__), 'validation': rep, 'test_evaluated': False, 'elapsed_seconds': elapsed,
        'selection_rule': 'min hard gxr F_FPR s.t. AUC>=min_auc and FNR<=max_fnr; ties AUC then earlier epoch'})
    (out / 'DONE').write_text('Completed; selected checkpoint replayed on VAL. TEST not read.\n')


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--preflight', required=True)
    p.add_argument('--out', required=True); p.add_argument('--budget-hours', type=float, required=True); p.add_argument('--resume', action='store_true')
    a = p.parse_args(); c = json.loads(Path(a.config).read_text())
    train(c, a.preflight, a.out, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), a.budget_hours, a.resume)


if __name__ == '__main__': main()
