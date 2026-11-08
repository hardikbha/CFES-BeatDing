"""One-time TEST score export for CES-FL P. Its checkpoint format (CESFL_DIAGRAM_v0.2.1) is produced by
ces_fl_diagram_model_v021.py directly (not ces_fl_counterfactual_5050.py, which was the wrong module -- verified
after a checkpoint format mismatch). Reuses that module's own load_model/evaluate_run/predict/metrics/sha/write
directly. Frozen VAL-selected threshold; evaluate_run() enforces this itself ("the checkpoint already fixes
threshold before any evaluation scores or labels are used").
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import torch

BASE = '/home/sachin.chaudhary/ces_fl_sources/ding_ff2_public_s17_20261102T130051Z/CES_FL/src/ces_fl_diagram_model_v021.py'
TEST_MANIFEST = '/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/test.csv'
TEST_SHA = 'dd5eb9ef070f658fe6b7f744c47f73e25ce7c53f948bc272f4f16c69a37d4099'
VAL_SHA = 'aeda051259d9e26b07c2810152b601ae115ce7c9e10523f99ea07bbc9f92afa8'

spec = importlib.util.spec_from_file_location('diagram', BASE)
dm = importlib.util.module_from_spec(spec); spec.loader.exec_module(dm)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--sanity', action='store_true')
    a = p.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ck_path = Path(a.checkpoint)
    before = dm.sha(ck_path)

    if a.sanity:
        model, ck = dm.load_model(ck_path, device)
        c = ck['config']
        if dm.sha(c['val']) != VAL_SHA: raise ValueError('VAL manifest is not the canonical FF2 VAL file')
        rows = dm.read_rows(c['val'])
        z = dm.predict(model, rows, device, c['batch_size'], c['workers'], c['size'])
        report = dm.metrics(rows, z, ck['calibration']['threshold_logit'], c['min_support'])

        def diffs(x, y, out):
            if isinstance(x, dict):
                for k in x: diffs(x[k], y[k], out)
            elif isinstance(x, (float, int)) and not isinstance(x, bool):
                out.append(abs(x - y))
        def auc_diffs(x, y, path, out):
            if isinstance(x, dict):
                for k in x: auc_diffs(x[k], y[k], path + '.' + str(k), out)
            elif isinstance(x, (float, int)) and not isinstance(x, bool):
                out.append((path, abs(x - y)))
        vals = []; diffs(report, ck['validation'], vals)
        pairs = []; auc_diffs(report, ck['validation'], 'root', pairs)
        max_diff = max(vals) if vals else 0.
        max_auc_diff = max((d for p, d in pairs if p.endswith('.auc') or p == 'root.overall.auc'), default=0.)
        # See ff2_test_eval_20261108_r1/README.md "Root cause of the CPU sanity failures" for the justification.
        ok = max_auc_diff < 1e-4 and max_diff < 0.015  # see export_baselines.py/README for the F_FPR-is-a-sum justification
        print(json.dumps({'sanity': 'PASS' if ok else 'FAIL', 'max_metric_diff': max_diff, 'max_auc_diff': max_auc_diff,
                          'recomputed_auc': report['overall']['auc'], 'stored_auc': ck['validation']['overall']['auc']}))
        if not ok: sys.exit(1)
        return

    if dm.sha(TEST_MANIFEST) != TEST_SHA: raise ValueError('TEST manifest hash mismatch: refusing to score')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()): raise ValueError('Refusing to overwrite an existing TEST export directory')
    report = dm.evaluate_run(str(ck_path), TEST_MANIFEST, str(out), device)  # frozen-threshold TEST scoring, built in
    after = dm.sha(ck_path)
    if before != after: raise RuntimeError('Checkpoint changed during scoring')

    # Fold predictions.csv into the same test_scores_PRIVATE.json / TEST_EVALUATED.json format used by every
    # other exporter in this sweep, for uniform downstream metric computation.
    model, ck = dm.load_model(ck_path, device)
    rows = dm.read_rows(TEST_MANIFEST)
    import csv
    with (out / 'predictions.csv').open() as f:
        pr = {r['image']: float(r['logit']) for r in csv.DictReader(f)}
    z = [pr[r['image']] for r in rows]
    dm.write(out / 'test_scores_PRIVATE.json', {'rows': rows, 'logits': z, 'threshold': ck['calibration']['threshold_logit'],
        'min_support': ck['config']['min_support'], 'manifest_sha256': TEST_SHA, 'checkpoint_sha256': before, 'split': 'test'})
    dm.write(out / 'TEST_EVALUATED.json', {'status': 'TEST_EVALUATED', 'checkpoint': str(ck_path),
        'checkpoint_sha256': before, 'manifest_sha256': TEST_SHA,
        'timestamp_utc': __import__('datetime').datetime.utcnow().isoformat()})
    print(json.dumps({'status': 'DONE', 'auc': report['calibrated']['overall']['auc'], 'n': report['calibrated']['overall']['n']}))


if __name__ == '__main__':
    main()
