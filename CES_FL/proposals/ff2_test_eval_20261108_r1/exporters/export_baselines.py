"""One-time TEST score export for BCE / DAG-FDD / DAW-FDD (ff2_fairness_baselines.py architecture). Reuses that
script's own make_model/load_xception/predict/common.metric_report directly. Frozen VAL-selected threshold.
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import torch

SRC = '/home/sachin.chaudhary/ces_fl_sources/cesfl-ff2-suite-815b4d4/CES_FL/src/ff2_fairness_baselines.py'
UPSTREAM = '/home/sachin.chaudhary/ces_fl_dependencies/ff2_baselines_pinned_20261103/dag_daw'
TEST_MANIFEST = '/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/test.csv'
TEST_SHA = 'dd5eb9ef070f658fe6b7f744c47f73e25ce7c53f948bc272f4f16c69a37d4099'

spec = importlib.util.spec_from_file_location('baselines', SRC)
bl = importlib.util.module_from_spec(spec); spec.loader.exec_module(bl)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--sanity', action='store_true')
    a = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ck_path = Path(a.checkpoint)
    before = bl.common.sha(ck_path)
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    c = ck['config']
    model = bl.make_model(UPSTREAM, None).to(device)
    model.load_state_dict(ck['model'], strict=True)
    threshold = ck['calibration']['threshold_logit']

    if a.sanity:
        val = bl.common.read_manifest(c['val'])
        if bl.common.sha(c['val']) != c['val_sha256']: raise ValueError('VAL manifest hash mismatch')
        z = bl.predict(model, val, c, device)
        report = bl.common.metric_report(val, z, threshold, c['min_support'])
        def diffs(a, b, out):
            if isinstance(a, dict):
                for k in a: diffs(a[k], b[k], out)
            elif isinstance(a, (float, int)) and not isinstance(a, bool):
                out.append(abs(a - b))
        vals = []; diffs(report, ck['validation'], vals)
        max_diff = max(vals) if vals else 0.
        ok = max_diff < 1e-5  # same replay tolerance used elsewhere in this project (device/precision float noise)
        print(json.dumps({'sanity': 'PASS' if ok else 'FAIL', 'max_metric_diff': max_diff,
                          'recomputed_auc': report['overall']['auc'], 'stored_auc': ck['validation']['overall']['auc']}))
        if not ok: sys.exit(1)
        return

    if bl.common.sha(TEST_MANIFEST) != TEST_SHA: raise ValueError('TEST manifest hash mismatch: refusing to score')
    out = Path(a.out)
    if out.exists() and any(out.iterdir()): raise ValueError('Refusing to overwrite an existing TEST export directory')
    out.mkdir(parents=True, exist_ok=True)
    rows = bl.common.read_manifest(TEST_MANIFEST, allow_duplicates=True)
    for r in rows: r['image_sha256'] = bl.common.sha(r['image'])
    z = bl.predict(model, rows, c, device)
    report = bl.common.metric_report(rows, z, threshold, c['min_support'])
    after = bl.common.sha(ck_path)
    if before != after: raise RuntimeError('Checkpoint changed during scoring; aborting without writing results')

    bl.common.write_json(out / 'metrics.json', {'checkpoint_sha256': before, 'manifest_sha256': TEST_SHA,
        'exporter_sha256': bl.common.sha(__file__), 'frozen_threshold_logit': threshold,
        'calibration_source': 'selected checkpoint VAL calibration (frozen; not recalibrated on TEST)',
        'calibrated': report, 'fixed_probability_0_5': bl.common.metric_report(rows, z, 0., c['min_support'])})
    bl.common.write_json(out / 'test_scores_PRIVATE.json', {'rows': rows, 'logits': list(z), 'threshold': threshold,
        'min_support': c['min_support'], 'manifest_sha256': TEST_SHA, 'checkpoint_sha256': before, 'split': 'test'})
    bl.common.write_json(out / 'TEST_EVALUATED.json', {'status': 'TEST_EVALUATED', 'checkpoint': str(ck_path),
        'checkpoint_sha256': before, 'manifest_sha256': TEST_SHA,
        'timestamp_utc': __import__('datetime').datetime.utcnow().isoformat()})
    print(json.dumps({'status': 'DONE', 'auc': report['overall']['auc'], 'n': report['overall']['n']}))


if __name__ == '__main__':
    main()
