"""Independent arithmetic/provenance checks on public aggregates; no private data access."""
import csv
import hashlib
import json
import math
from pathlib import Path

root = Path(__file__).resolve().parents[1]
e = root / 'evidence'
with (e / 'METRICS_LONG_TEST.csv').open() as f:
    summary = list(csv.DictReader(f))
with (e / 'GROUP_SUPPORT_TEST.csv').open() as f:
    groups = list(csv.DictReader(f))


def close(a, b):
    assert math.isclose(float(a), float(b), abs_tol=1e-8), (a, b)


checks = 0
for row in summary:
    match = [g for g in groups if (g['model'], g['seed'], g['axis']) == (row['model'], row['seed'], row['axis'])]
    eligible = []
    minimum = int(row['min_support_per_class'])
    for g in match:
        assert int(g['n']) == int(g['n_real']) + int(g['n_fake'])
        valid = int(g['n_real']) >= minimum and int(g['n_fake']) >= minimum
        assert valid == (g['eligible'] == 'True')
        for rate in ('fpr_pct', 'tpr_pct', 'fnr_pct', 'accuracy_pct', 'positive_rate_pct', 'auc_pct'):
            assert math.isfinite(float(g[rate])) and 0 <= float(g[rate]) <= 100
        close(float(g['tpr_pct']) + float(g['fnr_pct']), 100)
        for n, rate in ((int(g['n_real']), 'fpr_pct'), (int(g['n_fake']), 'fnr_pct')):
            count = n * float(g[rate]) / 100
            close(count, round(count))
        if valid:
            eligible.append(g)
    assert len(eligible) == int(row['eligible_groups'])
    fprs = [float(g['fpr_pct']) for g in eligible]
    tprs = [float(g['tpr_pct']) for g in eligible]
    prs = [float(g['positive_rate_pct']) for g in eligible]
    acc = [float(g['accuracy_pct']) for g in eligible]
    aucs = [float(g['auc_pct']) for g in eligible]
    close(sum(abs(x-float(row['overall_fpr_pct'])) for x in fprs), row['F_FPR_pct'])
    close(max(fprs)-min(fprs), row['FPR_gap_pp'])
    close(max(tprs)-min(tprs), row['TPR_gap_pp'])
    close(max(max(fprs)-min(fprs), max(tprs)-min(tprs)), row['F_MEO_max_pp'])
    close(max(prs)-min(prs), row['F_DP_pct'])
    close(max(acc)-min(acc), row['F_OAE_pp'])
    close(min(aucs), row['worst_group_auc_pct'])
    a = float(row['overall_auc_pct'])
    close(a/(1+sum(abs(x-a)/100 for x in aucs)), row['es_AUC_pct'])
    checks += 1

provenance = json.loads((e/'PROVENANCE.json').read_text())
expected = {'CES_FL/src/ces_fl_realfpr.py': '2d99295cc0b8190a571695b581ef5ba73e7b5fc18994461bdf1a21c1d83089d1',
            'CES_FL/src/ces_fl_head_ablation.py': '8c25ba8cb844b059adbbce75b57785a156333182afa2d5d576985997f2d36733'}
assert provenance['source_files'] == expected
for relative, digest in expected.items():
    assert hashlib.sha256((root.parents[1]/relative).read_bytes()).hexdigest() == digest
assert len(json.loads((e/'TEST_RECEIPTS.json').read_text())) == 10
assert len(json.loads((e/'VALIDATION_SELECTED.json').read_text())) == 6
assert len(json.loads((e/'EPOCH_HISTORY_AGGREGATE.json').read_text())) == 60
print(json.dumps({'status':'PASS','summary_rows_recomputed':checks,'subgroup_rows':len(groups),
                  'selected_runs':6,'epoch_records':60,'model_source_hashes_unchanged':True,
                  'inference_or_training_executed':False,'private_scores_or_images_read':False},indent=2))
