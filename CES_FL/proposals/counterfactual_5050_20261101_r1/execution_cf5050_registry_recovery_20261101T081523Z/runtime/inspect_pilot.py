"""Private contact sheets and pixel diagnostics; never writes review decisions."""
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/cf5050_registry_recovery_20261101T081523Z')
CONDITIONS = ['nochange', 'younger', 'older', 'gender_feminine', 'gender_masculine', 'skin_lighter', 'skin_darker']


def sharpness(image):
    gray = np.asarray(image.convert('L'), dtype=np.float64)
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    return float(lap.var())


def main():
    assert (ROOT / 'checks/pilot_PASS.json').is_file(), 'Wait for real generation and review-template completion'
    out = ROOT / 'pilot_inspection'
    out.mkdir(exist_ok=False)
    font_path = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    rows, sheets = [], []
    records = sorted((ROOT / 'generation/items').glob('*/record.json'))
    assert len(records) == 2
    for i, path in enumerate(records):
        record = json.loads(path.read_text())
        assert set(record['variants']) == set(CONDITIONS)
        source = record['source']
        original = Image.open(source['original_image']).convert('RGB')
        base = np.asarray(original, dtype=np.float64) / 255
        original_sharpness = sharpness(original)
        paths = [('original', source['original_image'])] + [(k, record['variants'][k]['image']) for k in CONDITIONS] + [('resize_geometry', record['resize_control']['image'])]
        sheet = Image.new('RGB', (3 * 290, 3 * 325 + 40), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 10), 'Source ' + str(i) + ' | source label: ' + ('FAKE' if source['label'] else 'REAL'), fill='black', font=font)
        for j, (condition, file) in enumerate(paths):
            image = Image.open(file).convert('RGB')
            assert image.size == original.size
            delta = np.asarray(image, dtype=np.float64) / 255 - base
            mse = float((delta**2).mean())
            rows.append({'source_index': i, 'source_label': source['label'], 'condition': condition,
                         'image': file, 'original_image': source['original_image'],
                         'mean_absolute_pixel_change': float(np.abs(delta).mean()),
                         'psnr_db': float(-10 * np.log10(mse)) if mse > 0 else None,
                         'laplacian_variance': sharpness(image),
                         'sharpness_ratio_to_original': sharpness(image) / original_sharpness if original_sharpness else None,
                         'label_preservation_verified': False})
            x, y = (j % 3) * 290, 40 + (j // 3) * 325
            thumb = ImageOps.contain(image, (280, 280))
            sheet.paste(thumb, (x + (290 - thumb.width) // 2, y))
            draw.text((x + 6, y + 284), condition, fill='black', font=font)
        target = out / ('source' + str(i) + '_' + ('fake' if source['label'] else 'real') + '_contact.png')
        sheet.save(target)
        sheets.append(str(target))
    with (out / 'pixel_diagnostics_private.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {'status': 'CONTACT_SHEETS_AND_DIAGNOSTICS_PREPARED', 'contact_sheets': sheets,
              'source_count': 2, 'generated_outputs': 14, 'geometric_controls': 2,
              'human_review': str(ROOT / 'pilot_review_pending.csv'), 'approvals_written': 0,
              'limits': 'Pixel/sharpness differences do not prove identity, target attainment, authenticity or forensic preservation. Source labels are not ground-truth labels for generated candidates.'}
    (out / 'INSPECTION_ARTIFACTS.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
