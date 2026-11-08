"""One-time TEST score export for the Ding-style corrected-GDA checkpoint. Reuses the original script's own Detector,
mask_channels, validation(), common.calibrate/common.metrics directly (no reimplementation). Single-process (no DDP
init -> gather() is a no-op, confirmed by reading gather()'s source). Frozen VAL-selected threshold; no recalibration
on TEST.
"""
import argparse, importlib.util, json, sys
from pathlib import Path
import torch

DING = '/home/sachin.chaudhary/ces_fl_sources/ding_ff2_public_s17_20261102T130051Z/CES_FL/src/ding_ff2_public.py'
spec = importlib.util.spec_from_file_location('ding', DING)
ding = importlib.util.module_from_spec(spec); spec.loader.exec_module(ding)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)  # gda_corrected/best.pt
    p.add_argument('--out', required=True)
    p.add_argument('--sanity', action='store_true')
    a = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ck_path = Path(a.checkpoint)
    before = ding.sha(ck_path)
    ck = torch.load(ck_path, map_location='cpu', weights_only=False)
    c = ck['config']

    model = ding.Detector().to(device)
    ding.mask_channels(model, ck['channels'])
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    threshold = ck['calibration']['threshold_logit']

    if a.sanity:
        splits, receipt = ding.read_inputs(c, stat=False)
        z, cal, report = ding.validation(model, splits['val'], c, device, 0, 1)
        ok = report == ck['validation']
        print(json.dumps({'sanity': 'PASS' if ok else 'FAIL', 'recomputed_auc': report['overall']['auc'],
                          'stored_auc': ck['validation']['overall']['auc']}))
        if not ok: sys.exit(1)
        return

    splits, receipt = ding.read_inputs(c, stat=False)  # also re-verifies train/val/test manifest hashes and overlap=0
    test_rows = splits['test']
    out = Path(a.out)
    if out.exists() and any(out.iterdir()): raise ValueError('Refusing to overwrite an existing TEST export directory')
    out.mkdir(parents=True, exist_ok=True)
    # ding.validation() recalibrates internally on whatever rows it's given; we need the FROZEN VAL threshold applied
    # to TEST, not a TEST-derived one, so the scoring loop (loader + forward pass) is reproduced here exactly, and
    # only the final metric call uses the frozen threshold. No model/preprocessing code is reimplemented.
    import numpy as np
    batches = ding.SequentialBatches(len(test_rows), c['validation_batch'], 0, 1)
    loader = torch.utils.data.DataLoader(ding.Images(test_rows, True), batch_sampler=batches, num_workers=c['workers_per_rank'])
    scores, indices = [], []
    with torch.no_grad():
        for x, _, _, _, idx in loader:
            zz = model(x.to(device))
            scores.extend((zz[:, 1] - zz[:, 0]).double().cpu().tolist())
            indices.extend(idx.tolist())
    order = np.argsort(indices)
    if sorted(indices) != list(range(len(test_rows))): raise ValueError('TEST scoring missing/duplicate rows')
    z = np.asarray(scores)[order]
    report = ding.common.metrics(test_rows, z, threshold)  # min_support=20 default, matches every other FF2 table
    after = ding.sha(ck_path)
    if before != after: raise RuntimeError('Checkpoint changed during scoring; aborting without writing results')

    ding.write(out / 'metrics.json', {'checkpoint_sha256': before, 'manifest_sha256': c['input_hashes']['test'],
        'exporter_sha256': ding.sha(__file__), 'frozen_threshold_logit': threshold,
        'calibration_source': 'selected checkpoint VAL calibration (frozen; not recalibrated on TEST)', 'calibrated': report})
    ding.write(out / 'test_scores_PRIVATE.json', {'rows': test_rows, 'logits': z.tolist(), 'threshold': threshold,
        'manifest_sha256': c['input_hashes']['test'], 'checkpoint_sha256': before, 'split': 'test'})
    ding.write(out / 'TEST_EVALUATED.json', {'status': 'TEST_EVALUATED', 'checkpoint': str(ck_path),
        'checkpoint_sha256': before, 'manifest_sha256': c['input_hashes']['test'],
        'timestamp_utc': __import__('datetime').datetime.utcnow().isoformat()})
    print(json.dumps({'status': 'DONE', 'auc': report['overall']['auc'], 'n': report['overall']['n']}))


if __name__ == '__main__':
    main()
