"""Private audit of stored CES-FL validation errors; no inference or calibration.

The user-supplied ZIP was unavailable in the inspected workspace. This exporter
is independently written and must not be attributed to that bundle.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scores', 'checkpoint', 'source', 'out'):
        p.add_argument('--' + name, required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=False)
    scores = read(a.scores)
    assert scores['split'] == 'validation' and scores['test_read'] is False
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    c = ck['config']
    assert c['dataset_id'] == 'FFPP_DING_PUBLIC' and c['seed'] == 17
    assert ck['arm'] == 'D' and not ck['synthetic_fixture']
    assert sha(a.checkpoint) == scores['checkpoint_sha256']
    assert ck['source_sha256'] == sha(a.source) == c['source_sha256']
    assert sha(c['val']) == c['val_sha256'] == scores['manifest_sha256']
    assert scores['threshold'] == ck['calibration']['threshold_logit']
    assert scores['min_support'] == c['min_eval_support']
    spec = importlib.util.spec_from_file_location('audited_cesfl_D', a.source)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    rows, logits = scores['rows'], np.asarray(scores['logits'])
    current = m.read_manifest(c['val'])
    assert [{k: v for k, v in r.items() if k != 'image_sha256'} for r in rows] == current
    assert m.metric_report(rows, logits, scores['threshold'], scores['min_support']) == ck['validation']
    receipt = read(Path(a.checkpoint).parent / 'RUN_VERIFIED.json')
    assert receipt['status'] == 'COMPLETED_VERIFIED'
    assert receipt['selected_checkpoint_sha256'] == scores['checkpoint_sha256']
    assert receipt['selected_epoch'] == ck['epoch'] == 1
    assert receipt['validation'] == ck['validation'] and not receipt['test_evaluated']
    with ThreadPoolExecutor(max_workers=8) as pool:
        measured = list(pool.map(lambda r: sha(r['image']), rows))
    bad = [i for i, (r, h) in enumerate(zip(rows, measured)) if r['image_sha256'] != h]
    if bad:
        write(out / 'IMAGE_HASH_FAILURE_PRIVATE.json', {'bad_row_indices': bad})
        raise RuntimeError('Stored validation image bytes changed; training blocked')
    pred = logits >= scores['threshold']
    component = {}
    for ci, members in enumerate(m.components(rows)):
        component.update({int(i): ci for i in members})
    specifications = [
        ('male_white_real', 'gender_race', 'Male-White', 0, 193, 1264),
        ('female_black_real', 'gender_race', 'Female-Black', 0, 9, 30),
        ('age60plus_real', 'age_bin', '60+', 0, 24, 63),
        ('male_others_fake', 'gender_race', 'Male-Others', 1, None, None),
        ('age45to59_fake', 'age_bin', '45-59', 1, None, None),
    ]
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 12) if Path(font_path).exists() else ImageFont.load_default()
    index, aggregates = [], {}
    for name, axis, group, label, expected_errors, expected_n in specifications:
        members = [i for i, r in enumerate(rows) if r[axis] == group and r['label'] == label]
        errors = [i for i in members if pred[i] != label]
        correct = [i for i in members if pred[i] == label]
        if expected_n is not None:
            assert (len(errors), len(members)) == (expected_errors, expected_n)

        def choose(candidates):
            ordered = sorted(candidates, key=lambda i: m.seed_for(17, name, rows[i]['image']))
            selected, used = [], set()
            for i in ordered:
                if component[i] not in used:
                    selected.append(i)
                    used.add(component[i])
                if len(selected) == 4:
                    break
            selected += [i for i in ordered if i not in selected][:4-len(selected)]
            return selected

        picks = {'error': choose(errors), 'correct': choose(correct)}
        sheet = Image.new('RGB', (920, 4*360+48), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 6), name + ' | manifest labels only | stored D threshold; no recalibration', fill='black', font=font)
        for col, kind in enumerate(('error', 'correct')):
            for j, i in enumerate(picks[kind]):
                row = rows[i]
                with Image.open(row['image']) as im:
                    original_size, image_format = im.size, im.format
                    rgb = im.convert('RGB')
                    # Exact network geometry: square224 bilinear, no new crop.
                    panel = rgb.resize((224, 224), Image.Resampling.BILINEAR)
                    jpeg_tables = bool(getattr(im, 'quantization', None))
                x, y = col*460+8, j*360+48
                sheet.paste(panel, (x, y))
                case = f'{name}_{kind}_{j+1}'
                text = [case, f'CSV y={label}; pred={int(pred[i])}; z={logits[i]:.5f}',
                        f'threshold={scores["threshold"]:.5f}; component={component[i]}',
                        f'gender={row["gender"]}; race={row["race"]}; age={row["age_bin"]}',
                        f'raw={original_size}; format={image_format}; sha={measured[i][:12]}',
                        f'family={row["family"]}; method={row.get("method", "unknown")}']
                draw.multiline_text((x, y+230), '\n'.join(text), fill='black', font=font, spacing=2)
                index.append(dict(case_id=case, row_index=i, image=row['image'],
                                  image_sha256=measured[i], label=label, predicted=int(pred[i]),
                                  logit=float(logits[i]), component=component[i], original_size=original_size,
                                  file_format=image_format, jpeg_tables_present=jpeg_tables,
                                  metadata={k:row[k] for k in row if k not in ('image','image_sha256')}))
        sheet.save(out / (name + '_PRIVATE.png'))
        aggregates[name] = dict(axis=axis, group=group, label=label, n=len(members),
                               errors=len(errors), correct=len(correct), error_rate=len(errors)/len(members),
                               recorded_components=len({component[i] for i in members}),
                               selected_errors=len(picks['error']), selected_correct=len(picks['correct']),
                               error_components_shown=len({component[i] for i in picks['error']}),
                               correct_components_shown=len({component[i] for i in picks['correct']}))
    write(out / 'CONTACT_INDEX_PRIVATE.json', index)
    write(out / 'D_SELECTED_PREDICTIONS_VERIFIED.json', dict(
        status='PASS', observed_management_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        model='Existing CES-FL D; not Ding', exporter_sha256=sha(__file__),
        source_sha256=sha(a.source), checkpoint_sha256=sha(a.checkpoint),
        scores_sha256=sha(a.scores), manifest_sha256=sha(c['val']),
        selected_epoch=1, threshold=scores['threshold'], threshold_recalibrated=False,
        rows_verified=len(rows), image_hashes_verified=len(measured), row_metadata_matches_manifest=True,
        aggregate_metrics_exactly_replayed=True, validation=ck['validation'], strata=aggregates,
        contact_cases=len(index), unique_shown_images=len({r['image'] for r in index}),
        test_read=False, model_inference_performed=False, human_review_completed=False,
        bundle='Supplied diagnostic ZIP unavailable; independent exporter explicitly identified',
        limitations=['Recorded IDs are not verified biological identities',
                     'Contact sampling maximizes recorded-component variety, then deterministic image variety',
                     'Visual appearance cannot establish demographic label or real/fake ground truth',
                     'JPEG container metadata does not establish original codec/compression lineage']))
    print(json.dumps(dict(status='PASS', image_hashes_verified=len(measured), strata=aggregates, test_read=False)))


if __name__ == '__main__':
    main()
