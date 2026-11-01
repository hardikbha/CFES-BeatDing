"""Render Ding-layout tables from the committed numeric snapshot, without ML execution.

python build_tables.py --snapshot source_snapshot.json --out /new/output/path
Optional --pdf uses ReportLab and DejaVu fonts. Existing outputs are never overwritten.
"""
import argparse
import html
import json
import math
from pathlib import Path

AXES = ('gender', 'race', 'gender_race')
LABELS = ('Gender', 'Race', 'Intersection')
DATASETS = {'FF++': 'ffpp', 'DFDC': 'dfdc', 'Celeb-DF': 'celebdf', 'DFD': 'dfd'}
POLICY = ('Our rows: seed 17 TEST, stored validation-real FPR10% thresholds; '
          'no test recalibration. Published rows retain Ding\'s reported numbers. '
          'Shared format does not establish identical training/evaluation protocols.')


def verify_metrics(m):
    assert m['units'] == 'fractions'
    overall = m['overall']
    for axis in AXES:
        a = m['axes'][axis]
        groups = [a['groups'][g] for g in a['eligible_groups']]
        assert all(g['n_real'] >= a['min_support_per_class'] and
                   g['n_fake'] >= a['min_support_per_class'] for g in groups)
        recorded = a['ding_style_fractions']
        if len(groups) <= 1:
            assert recorded is None
            continue
        expected = {
            'F_FPR': sum(abs(g['fpr'] - overall['fpr']) for g in groups),
            'F_DP': max(g['positive_rate'] for g in groups) - min(g['positive_rate'] for g in groups),
            'es_AUC': overall['auc'] / (1 + sum(abs(g['auc'] - overall['auc']) for g in groups)),
        }
        for key, value in expected.items():
            assert math.isfinite(recorded[key]) and math.isclose(value, recorded[key], abs_tol=1e-12), (axis, key)


def vector(m, axis=None):
    values = []
    for name in ([axis] if axis else AXES):
        d = m['axes'][name]['ding_style_fractions']
        values += [None if d is None else d[k] for k in ['F_FPR', 'F_DP', 'es_AUC']]
    values.append(m['overall']['auc'])
    return [None if v is None else v * 100 for v in values]


def rows_for(s, table, score='calibrated'):
    published = s['paper']['tables'][str(table)]
    rows = []
    for block in dict.fromkeys(r['block'] for r in published):
        rows += [r for r in published if r['block'] == block]
        backbone = 'ResNet-50' if table == 3 else 'Xception'
        for exp in s['experiments']:
            if exp['backbone'] != backbone:
                continue
            ds = 'ffpp' if table == 1 else DATASETS[block]
            metric = exp['datasets'][ds][score]
            axis = AXES[LABELS.index(block)] if table == 1 else None
            rows.append({'block': block, 'method': exp['label'] + ' [s17]', 'backbone': backbone,
                         'values_pct': vector(metric, axis), 'kind': 'measured_test', 'arm': exp['id']})
    return rows


def fmt(v):
    return '—' if v is None else f'{v:.2f}'


def md_table(rows, table):
    headers = (['Attribute', 'Method', 'Backbone', 'F_FPR ↓', 'F_DP ↓', 'es-AUC ↑', 'AUC ↑'] if table == 1 else
               ['Dataset', 'Method', 'Backbone'] +
               [f'{g} {k}' for g in LABELS for k in ['F_FPR ↓', 'F_DP ↓', 'es-AUC ↑']] + ['Overall AUC ↑'])
    result = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    for r in rows:
        result.append('| ' + ' | '.join([r['block'], r['method'], r['backbone']] +
                                       [fmt(v) for v in r['values_pct']]) + ' |')
    return '\n'.join(result)


def html_table(rows, table):
    n = 7 if table == 1 else 13
    start = '<table><thead>'
    if table == 1:
        start += ('<tr><th rowspan="2">Attributes</th><th rowspan="2">Methods</th>'
                  '<th rowspan="2">Backbone</th><th colspan="3">Fairness metrics (%)</th>'
                  '<th>Detection (%)</th></tr><tr><th>F_FPR ↓</th><th>F_DP ↓</th>'
                  '<th>es-AUC ↑</th><th>AUC ↑</th></tr>')
    else:
        start += ('<tr><th rowspan="3">Datasets</th><th rowspan="3">Methods</th>'
                  '<th rowspan="3">Backbone</th><th colspan="9">Fairness metrics (%)</th>'
                  '<th>Detection (%)</th></tr><tr>' +
                  ''.join(f'<th colspan="3">{g}</th>' for g in LABELS) + '<th>Overall</th></tr><tr>' +
                  '<th>F_FPR ↓</th><th>F_DP ↓</th><th>es-AUC ↑</th>' * 3 + '<th>AUC ↑</th></tr>')
    start += '</thead><tbody>'
    for i, row in enumerate(rows):
        style = ' style="background:#eef6ff"' if row['kind'] == 'measured_test' else ''
        cells = [row['block'], row['method'], row['backbone']] + [fmt(v) for v in row['values_pct']]
        start += '<tr' + style + '>' + ''.join('<td>' + html.escape(c) + '</td>' for c in cells) + '</tr>'
    return start + '</tbody></table>'


def make_pdf(out, tables):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, LongTable, PageBreak, Spacer
    font = next(p for p in [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
                            Path('/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf')]
                if p.is_file())
    pdfmetrics.registerFont(TTFont('DV', str(font)))
    small = ParagraphStyle('small', fontName='DV', fontSize=6.0, leading=7.2)
    title = ParagraphStyle('title', fontName='DV', fontSize=12, leading=15, spaceAfter=5)
    note = ParagraphStyle('note', fontName='DV', fontSize=7.0, leading=9.0, spaceAfter=5)
    doc = SimpleDocTemplate(str(out / 'TABLES_1_2_3.pdf'), pagesize=landscape(A4),
                            leftMargin=22, rightMargin=22, topMargin=22, bottomMargin=22,
                            title='CES-FL seed17: Tables 1, 2 and 3 in Ding format',
                            author='CES-FL research project')
    flow = []
    captions = {1: 'Table 1. FF++ intra-domain evaluation', 2: 'Table 2. Cross-domain evaluation',
                3: 'Table 3. ResNet-50 backbone evaluation'}
    for number in [1, 2, 3]:
        if flow:
            flow.append(PageBreak())
        flow += [Paragraph(captions[number] + ' — Ding layout', title), Paragraph(POLICY, note),
                 Paragraph('A/B/C: earlier controlled pilot. R0/R1/R2/RN/RD: completed upgrade. '
                           'C is invariance-only; RN/RD use six sources. Current 50+50 results are VAL only and excluded.', note)]
        if number == 1:
            headers = ['Attributes', 'Methods', 'Backbone', 'F_FPR ↓', 'F_DP ↓', 'es-AUC ↑', 'AUC ↑']
            data = [[Paragraph(h, small) for h in headers]]
            styles = []
            widths = [65, 310, 65, 70, 70, 70, 70]
        else:
            data = [['Datasets', 'Methods', 'Backbone', 'Gender', '', '', 'Race', '', '', 'Intersection', '', '', 'Overall'],
                    ['', '', '', 'F_FPR ↓', 'F_DP ↓', 'es-AUC ↑', 'F_FPR ↓', 'F_DP ↓', 'es-AUC ↑',
                     'F_FPR ↓', 'F_DP ↓', 'es-AUC ↑', 'AUC ↑']]
            data = [[Paragraph(c, small) for c in row] for row in data]
            styles = [('SPAN', (i, 0), (i + 2, 0)) for i in [3, 6, 9]]
            styles += [('SPAN', (i, 0), (i, 1)) for i in [0, 1, 2]]
            widths = [38, 203, 49] + [50.0] * 10
        header_count = len(data)
        for row in tables[number]:
            values = [row['block'], row['method'], row['backbone']] + [fmt(v) for v in row['values_pct']]
            index = len(data)
            data.append([Paragraph(html.escape(v), small) for v in values])
            if row['kind'] == 'measured_test':
                styles.append(('BACKGROUND', (0, index), (-1, index), colors.HexColor('#edf5ff')))
            if index == header_count or row['block'] != tables[number][index - header_count - 1]['block']:
                styles.append(('LINEABOVE', (0, index), (-1, index), .6, colors.black))
        table = LongTable(data, colWidths=widths, repeatRows=header_count)
        table.setStyle(styles + [('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                       ('TOPPADDING', (0, 0), (-1, -1), .6), ('BOTTOMPADDING', (0, 0), (-1, -1), .6),
                       ('LEFTPADDING', (0, 0), (-1, -1), 2), ('RIGHTPADDING', (0, 0), (-1, -1), 2),
                       ('BACKGROUND', (0, 0), (-1, header_count - 1), colors.HexColor('#e8e8e8')),
                       ('LINEBELOW', (0, -1), (-1, -1), .5, colors.black)])
        flow += [table, Spacer(1, 5), Paragraph('Blue rows: our measured single-seed results. '
                 'No best-method/significance claim. F_FPR is summed disparity, not max–min FPR gap. '
                 'See README.md for group coverage, thresholds, hashes, caveats and latest VAL-only link.', note)]
    doc.build(flow)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--pdf', action='store_true')
    args = ap.parse_args()
    s = json.loads(args.snapshot.read_text())
    assert [len(s['paper']['tables'][str(i)]) for i in [1, 2, 3]] == [21, 21, 20]
    verified = 0
    for exp in s['experiments']:
        assert exp['seed'] == 17
        for ds, d in exp['datasets'].items():
            assert d['verification']['status'] == 'PASS'
            assert d['checkpoint_sha256'] == exp['checkpoint_sha256']
            assert d['manifest_sha256'] == s['datasets'][ds]['sha256']
            for field in ['calibrated', 'fixed_probability_0_5']:
                verify_metrics(d[field])
            verified += 1
    assert s['latest_validation_only']['test_evaluated'] is False
    tables = {i: rows_for(s, i) for i in [1, 2, 3]}
    assert [len(tables[i]) for i in [1, 2, 3]] == [45, 45, 32]
    for i, rows in tables.items():
        assert all(len(r['values_pct']) == (4 if i == 1 else 10) for r in rows)
        assert all(r['kind'] in ['published', 'measured_test'] for r in rows)
    # Formatting regression checks: missing values, no double percent conversion, exact axis ordering.
    assert fmt(None) == '—' and fmt(98.3466811660372) == '98.35'
    b = next(e for e in s['experiments'] if e['id'] == 'B_detection_frozen')
    v = vector(b['datasets']['ffpp']['calibrated'])
    assert math.isclose(v[-1], 98.3466811660372, abs_tol=1e-10)
    assert math.isclose(v[6], 37.89402321846326, abs_tol=1e-10)
    assert all(not (args.out / name).exists() for name in
               ['TABLE_1.md', 'TABLE_2.md', 'TABLE_3.md', 'TABLES_1_2_3.html', 'TABLES_1_2_3.pdf', 'VERIFY.json']), 'Refuse overwrite'
    args.out.mkdir(parents=True, exist_ok=True)
    captions = {1: 'FF++ intra-domain fairness and detection', 2: 'Cross-domain fairness and detection',
                3: 'ResNet-50 fairness and detection'}
    html_parts = ['<!doctype html><html><head><meta charset="utf-8"><title>CES-FL Tables 1–3</title>'
                  '<style>body{font:13px Arial;margin:24px}table{border-collapse:collapse;font-size:11px}'
                  'td,th{padding:5px;border:1px solid #bbb;text-align:right}td:nth-child(2){text-align:left}'
                  'thead{background:#eee}section{page-break-before:always}section:first-of-type{page-break-before:auto}'
                  '@media print{body{margin:0}table{font-size:8px}td,th{padding:2px}@page{size:A4 landscape;margin:12mm}}</style></head><body>']
    for number, rows in tables.items():
        body = f'# Table {number}. {captions[number]}\n\n{POLICY}\n\n'
        body += ('All values are percentages. ↓ lower; ↑ higher. Intersection = gender × race. '
                 'Published baselines appear first in each block; our single-seed rows follow. '
                 '[Arm definitions and limitations](README.md) · [Combined PDF](TABLES_1_2_3.pdf)\n\n')
        if number != 1:
            body += html_table(rows, number) + '\n\n<details><summary>Plain Markdown version</summary>\n\n'
            body += md_table(rows, number) + '\n\n</details>\n'
        else:
            body += md_table(rows, number) + '\n'
        body += '\nLatest source `58773bb` 50+50 runs have VAL results only and are excluded. '
        body += '[Latest validation-only results](LATEST_VALIDATION_ONLY.md).\n'
        (args.out / f'TABLE_{number}.md').write_text(body)
        html_parts.append(f'<section><h1>Table {number}. {captions[number]}</h1><p>{html.escape(POLICY)}</p>' +
                          html_table(rows, number) + '</section>')
    (args.out / 'TABLES_1_2_3.html').write_text('\n'.join(html_parts) + '</body></html>\n')
    if args.pdf:
        make_pdf(args.out, tables)
    receipt = {'status': 'PASS', 'operation': 'Static formatting/numeric verification only',
               'paper_rows': [21, 21, 20], 'output_rows': [len(tables[i]) for i in [1, 2, 3]],
               'existing_test_records_verified': verified, 'score_reports_checked': verified * 2,
               'ding_style_formula_checks': verified * 2 * 3 * 3,
               'source_receipt_manifest_checkpoint_agreement': True,
               'all_latest_validation_rows_excluded_from_test_tables': True,
               'training_or_inference_launched': False, 'calibration_or_selection_performed': False}
    (args.out / 'VERIFY.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
