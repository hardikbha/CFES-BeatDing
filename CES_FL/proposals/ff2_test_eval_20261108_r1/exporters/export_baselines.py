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
        def auc_diffs(a, b, path, out):
            if isinstance(a, dict):
                for k in a: auc_diffs(a[k], b[k], path + '.' + str(k), out)
            elif isinstance(a, (float, int)) and not isinstance(a, bool):
                out.append((path, abs(a - b)))
        vals = []; diffs(report, ck['validation'], vals)
        pairs = []; auc_diffs(report, ck['validation'], 'root', pairs)
        max_diff = max(vals) if vals else 0.
        auc_pairs = [(p, d) for p, d in pairs if p.endswith('.auc') or p == 'root.overall.auc']
        max_auc_diff = max((d for _, d in auc_pairs), default=0.)
        # AUC (rank-based, insensitive to a couple of near-tied flips) must match tightly; other fields (built from
        # per-group FPR/FNR, sensitive to single-image threshold flips in small groups) get a wider, measured bound.
        # For DAW-FDD specifically, a direct per-image logit comparison against the stored VAL scores found 5/31107
        # threshold flips (same bf16-train/fp32-replay mechanism as every other checkpoint here); the largest report
        # diff (0.0114) is F_FPR, which SUMS |group fpr - overall fpr| over up to 8 groups, so it amplifies several
        # independent single-group flips (each <=0.0052) rather than reflecting one outsized error. See the
        # pre-registration's "Root cause of the CPU sanity failures" section.
        ok = max_auc_diff < 1e-4 and max_diff < 0.015
        print(json.dumps({'sanity': 'PASS' if ok else 'FAIL', 'max_metric_diff': max_diff, 'max_auc_diff': max_auc_diff,
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
