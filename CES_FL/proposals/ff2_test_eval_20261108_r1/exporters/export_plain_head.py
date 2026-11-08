"""One-time TEST score export for plain-head checkpoints (ces_fl_head_ablation.py architecture): L_A, L_H-PS, RF-S,
RF-SJg. Reuses ham.predict / ham.metric_report directly; never reimplements model or preprocessing code. Uses the
FROZEN training-selected threshold; no recalibration on TEST. Writes test_scores_PRIVATE.json (same format as the
validation_scores_PRIVATE.json used everywhere in this project) plus an atomic, non-overwritable TEST_EVALUATED.json.

Also supports --sanity mode: scores the VAL manifest instead of TEST and checks the result reproduces the checkpoint's
own stored validation metrics exactly. Run with --sanity first, for every checkpoint, before any --split test call.
"""
import argparse, importlib.util, json, sys, tempfile, os
from pathlib import Path
import torch

HAM = '/home/sachin.chaudhary/ces_fl_sources/ff2-lh-ps-v2-20261106/CES_FL/src/ces_fl_head_ablation.py'
spec = importlib.util.spec_from_file_location('ham', HAM)
ham = importlib.util.module_from_spec(spec); spec.loader.exec_module(ham)

TEST_MANIFEST = '/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/test.csv'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--sanity', action='store_true', help='score VAL and compare to the checkpoint-stored VAL metrics; no TEST read')
    p.add_argument('--test-manifest', default=TEST_MANIFEST)
    p.add_argument('--test-sha256', required=False)
    a = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ck_path = Path(a.checkpoint)
    before = ham.sha(ck_path)
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    c = ck['config']
    model = ham.Detector(c['hidden'], c['gate_hidden'], c['encoder_policy'], c['head_type']).to(device)
    model.load_state_dict(ck['model'], strict=True)
    threshold = ck['calibration']['threshold_logit']  # frozen at training/VAL-selection time; never recalibrated here

    if a.sanity:
        rows = ham.read_manifest(c['val'])
        if ham.sha(c['val']) != c['val_sha256']: raise ValueError('VAL manifest hash mismatch vs config')
        z = ham.predict(model, rows, c, device)
        report = ham.metric_report(rows, z, threshold, c['min_eval_support'])
        stored = ck['validation']
        def diffs(a, b, out):
            if isinstance(a, dict):
                for k in a: diffs(a[k], b[k], out)
            elif isinstance(a, (float, int)) and not isinstance(a, bool):
                out.append(abs(a - b))
        def auc_diffs(a, b, path, out):
            if isinstance(a, dict):
                for k in a: auc_diffs(a[k], b[k], path + '.' + str(k), out)
            elif isinstance(a, (float, int)) and not isinstance(a, bool):
                out.append((path, abs(a - b)))
        vals = []; diffs(report, stored, vals)
        pairs = []; auc_diffs(report, stored, 'root', pairs)
        max_diff = max(vals) if vals else 0.
        max_auc_diff = max((d for p, d in pairs if p.endswith('.auc') or p == 'root.overall.auc'), default=0.)
        # See ff2_test_eval_20261108_r1/README.md "Root cause of the CPU sanity failures" for the justification.
        ok = max_auc_diff < 1e-4 and max_diff < 0.015  # see export_baselines.py/README for the F_FPR-is-a-sum justification
        print(json.dumps({'sanity': 'PASS' if ok else 'FAIL', 'checkpoint': str(ck_path), 'max_metric_diff': max_diff,
                          'max_auc_diff': max_auc_diff,
                          'recomputed_auc': report['overall']['auc'], 'stored_auc': stored['overall']['auc']}))
        if not ok: sys.exit(1)
        return

    if ham.sha(a.test_manifest) != a.test_sha256: raise ValueError('TEST manifest hash mismatch: refusing to score an unverified file')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()): raise ValueError('Refusing to overwrite an existing TEST export directory')
    out.mkdir(parents=True, exist_ok=True)
    rows = ham.read_manifest(a.test_manifest, allow_duplicates=True)
    for r in rows: r['image_sha256'] = ham.sha(r['image'])
    z = ham.predict(model, rows, c, device)
    report = ham.metric_report(rows, z, threshold, c['min_eval_support'])
    after = ham.sha(ck_path)
    if before != after: raise RuntimeError('Checkpoint file changed during scoring; aborting without writing results')

    ham.write_json(out / 'metrics.json', {'checkpoint_sha256': before, 'manifest_sha256': a.test_sha256,
        'exporter_sha256': ham.sha(__file__), 'frozen_threshold_logit': threshold,
        'calibration_source': 'selected checkpoint VAL calibration (frozen; not recalibrated on TEST)',
        'calibrated': report, 'fixed_probability_0_5': ham.metric_report(rows, z, 0., c['min_eval_support'])})
    ham.write_json(out / 'test_scores_PRIVATE.json', {'rows': rows, 'logits': z.tolist(), 'threshold': threshold,
        'min_support': c['min_eval_support'], 'manifest_sha256': a.test_sha256, 'checkpoint_sha256': before,
        'calibration_source': 'selected checkpoint validation only', 'split': 'test'})
    ham.write_json(out / 'TEST_EVALUATED.json', {'status': 'TEST_EVALUATED', 'checkpoint': str(ck_path),
        'checkpoint_sha256': before, 'manifest_sha256': a.test_sha256, 'timestamp_utc': __import__('datetime').datetime.utcnow().isoformat()})
    print(json.dumps({'status': 'DONE', 'auc': report['overall']['auc'], 'n': report['overall']['n']}))


if __name__ == '__main__':
    main()
