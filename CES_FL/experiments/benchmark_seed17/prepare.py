"""Prepare one explicitly authorized benchmark; no model inference or training."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
BASE = Path('/home/sachin.chaudhary')
OLD = BASE / 'ces_fl_pipeline/audits/encoder_controls_20261030T202634'
CORE_HASH = '0e001dd1251570ce43fb82bde307c519dea4045d9734506e1ce90d2ed014f8de'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def cross_rows(dataset, source):
    with source.open() as f:
        raw = list(csv.DictReader(f))
    rows = []
    races = ['Asian', 'White', 'Black', 'Others'] if dataset == 'dfdc' else ['White', 'Black', 'Others']
    for r in raw:
        assert r['label'] in ('0', '1') and r['ismale'] in ('-1', '1')
        keys = ['isasian', 'iswhite', 'isblack'] if dataset == 'dfdc' else ['iswhite', 'isblack']
        assert all(r[k] in ('-1', '1') for k in keys)
        flags = [k for k in keys if r[k] == '1']
        assert len(flags) <= 1
        race = {'isasian': 'Asian', 'iswhite': 'White', 'isblack': 'Black'}.get(flags[0] if flags else '', 'Others')
        gender = 'Male' if r['ismale'] == '1' else 'Female'
        assert int(r['intersec_label']) == races.index(race) + (0 if gender == 'Male' else len(races))
        path = Path(r['img_path'])
        assert path.is_absolute()
        video = path.stem.rsplit('_', 1)[0]
        proxy = dataset + ':video_proxy:' + video
        rows.append(dict(image=str(path), label=r['label'], gender=gender, race=race,
                         gender_race=gender+'-'+race, ids8='unknown', age_bin='unknown', skin_bin='unknown',
                         identity_ids='unverified_identity_'+proxy, source_ids=proxy, cluster=proxy,
                         family='real' if r['label'] == '0' else 'unknown', native_intersection_id=r['intersec_label'],
                         label_provenance='Archived Ding/Lin row/label/group unchanged; video filename proxy, person/donor identity unverified'))
    assert Counter((r['image'], r['label']) for r in rows) == Counter((r['img_path'], r['label']) for r in raw)
    return rows


def audit_manifest(path, allow_native_duplicates=False):
    with Path(path).open() as f:
        rows = list(csv.DictReader(f))
    assert {r['label'] for r in rows} == {'0', '1'}
    paths = [r['image'] for r in rows]
    unique = {}
    for row in rows:
        if row['image'] in unique:
            assert row == unique[row['image']], 'Conflicting duplicate image metadata'
        unique[row['image']] = row
    duplicates = len(paths)-len(unique)
    assert not duplicates or allow_native_duplicates, 'Duplicate image path outside declared native DFDC multiplicity'
    unique_path = None
    if duplicates:
        unique_path = Path(path).with_name(Path(path).stem+'_unique_validation_index.csv')
        with unique_path.open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            w.writeheader(); w.writerows(unique.values())
    with ThreadPoolExecutor(max_workers=24) as pool:
        present = list(pool.map(lambda p: Path(p).is_file(), paths))
    assert all(present), 'Missing images: ' + str([p for p, ok in zip(paths, present) if not ok][:5])
    cells = defaultdict(list)
    for r in rows:
        cells[(r['label'], r.get('gender_race', 'unknown'))].append(r)
    rng = random.Random(17)
    sample = [r for key in sorted(cells) for r in rng.sample(cells[key], min(2, len(cells[key])))]
    for r in sample:
        with Image.open(r['image']) as im:
            im.convert('RGB').load()
    return {'path': str(path), 'sha256': sha(path), 'rows': len(rows),
            'labels': dict(Counter(r['label'] for r in rows)), 'unique_images': len(unique),
            'native_exact_duplicate_extra_rows': duplicates,
            'duplicate_policy': 'Preserve native row multiplicity in inference/metrics; never allow conflicting labels/groups or duplicate TRAIN rows',
            'unique_validation_index': str(unique_path) if unique_path else None,
            'unique_validation_index_sha256': sha(unique_path) if unique_path else None,
            'group_class_counts': {str(k): len(v) for k, v in cells.items()},
            'all_referenced_files_exist': True, 'decoded_stratified_samples': len(sample),
            'all_images_decoded_before_run': False, 'person_or_donor_ground_truth_verified': False}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request-id', required=True)
    p.add_argument('--run-root', type=Path, required=True)
    args = p.parse_args()
    root = args.run_root
    assert not root.exists(), 'Preserve previous runs'
    proposal = REPO / 'CES_FL/proposals' / args.request_id
    assert not proposal.exists()
    assert sha(REPO / 'CES_FL/src/ces_fl_diagram_model_v021.py') == CORE_HASH
    root.mkdir(parents=True)
    for rel in ['manifests', 'logs', 'runtime', 'preflight', 'bound_configs', 'evaluation', 'full_experiment_runs']:
        (root / rel).mkdir()
    manifests = {'ffpp': str(BASE/'ces_fl_pipeline/manifests/ding_ffpp/test.csv'),
                 'fairforge2': str(BASE/'ces_fl_pipeline/manifests/ff2/test.csv')}
    for dataset in ['dfdc', 'celebdf', 'dfd']:
        source = REPO / 'DINGREPRO_LIN/data' / dataset / 'test.csv'
        rows = cross_rows(dataset, source)
        path = root/'manifests'/f'{dataset}_test.csv'
        with path.open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            w.writeheader(); w.writerows(rows)
        manifests[dataset] = str(path)
        print('ADAPTED', dataset, len(rows), flush=True)
    reports = {}
    for dataset, path in manifests.items():
        reports[dataset] = audit_manifest(path, allow_native_duplicates=dataset == 'dfdc')
        print('MANIFEST_CHECKED', dataset, reports[dataset]['rows'], flush=True)
    train = BASE/'ces_fl_pipeline/manifests/ding_ffpp/train.csv'
    val = BASE/'ces_fl_pipeline/manifests/ding_ffpp/val.csv'
    with train.open() as f:
        train_paths = {r['image'] for r in csv.DictReader(f)}
    with val.open() as f:
        val_paths = {r['image'] for r in csv.DictReader(f)}
    for path in manifests.values():
        with Path(path).open() as f:
            assert not ({r['image'] for r in csv.DictReader(f)} & (train_paths | val_paths)), 'TRAIN/VAL versus evaluation image-path overlap'
    weights = BASE/'.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'
    weight_hash = sha(weights)
    assert weight_hash.startswith('19c8e357')
    configs = {}
    mapping = {'P_resnet50': 'P_pretrain', 'A_resnet50': 'A_imagenet_frozen',
               'B_resnet50': 'B_detection_frozen', 'C_resnet50': 'C_detection_invariance'}
    for new, old in mapping.items():
        c = json.loads((OLD/'full_experiment_runs'/old/'config.json').read_text())
        c.update(backbone='resnet50', weights=str(weights))
        if new.startswith(('B_', 'C_')):
            c.update(encoder_checkpoint=str(root/'full_experiment_runs/P_resnet50/best.pt'), encoder_checkpoint_sha256=None)
        write(proposal/'configs'/f'{new}.json', c)
        configs[new] = 'CES_FL/proposals/'+args.request_id+'/configs/'+new+'.json'
    x = {}
    for arm in ['A_imagenet_frozen', 'B_detection_frozen', 'C_detection_invariance']:
        path = OLD/'full_experiment_runs'/arm/'best.pt'
        v = json.loads((path.parent/'RUN_VERIFIED.json').read_text())
        assert sha(path) == v['checkpoint_sha256']
        x[arm] = {'checkpoint': str(path), 'sha256': v['checkpoint_sha256'], 'selected_epoch': v['selected_epoch'], 'stored_calibration': v['calibration']}
    protocol = {'request_id': args.request_id, 'prepared_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'authorization': 'User: why did not you start the experiments; start them now (2026-10-31)',
                'run_root': str(root), 'core_sha256': CORE_HASH, 'seed': 17,
                'manifests': manifests, 'manifest_audit': reports,
                'train_sha256': sha(train), 'validation_sha256': sha(val),
                'resnet_weights': {'path': str(weights), 'sha256': weight_hash, 'initialization': 'Cached ImageNet1K V1, consistent with archived torchvision pretrained=True default'},
                'configs': configs, 'xception': x,
                'selection': 'Earliest highest validation AUC; no test selection',
                'thresholds': 'Each selected checkpoint stored validation-real empirical FPR10% threshold, plus separate fixed probability0.5 reporting; no test calibration or threshold choice',
                'support_rule': 'Existing executable metrics: minimum20 real and20 fake per eligible group; report all counts and excluded groups',
                'counterfactual_scope': 'Existing user-approved36 TRAIN edits/six sources; invariance only; lambda_f=0 and lambda_map=0; no new generation',
                'comparison_scope': 'Exploratory seed17; report protocol differences versus published Ding; no promise of superiority',
                'fairforge_scope': 'Cross-dataset evaluation of FF++-trained selected checkpoints; no FairForge2 in-domain training in this request',
                'bootstrap': {'replicates': 1000, 'seed': 17, 'unit': 'Connected components of recorded source contributors; unknown donors/person identity disclosed; fewer than2 components yields no CI'},
                'limitations': ['Exact archived membership is not independent verification of unpublished paper runs', 'Native test groups with one class have undefined AUC and are excluded by the existing support rule', 'Cross-domain filename video provenance is a proxy; hidden donors/people may connect videos', 'Later model development must disclose that these exploratory test sets have been inspected']}
    write(proposal/'protocol.json', protocol)
    write(root/'FROZEN_PROTOCOL.json', protocol)
    write(proposal/'request.json', {'request_id': args.request_id, 'state': 'USER_AUTHORIZED_PREFLIGHT', 'seed': 17,
          'purpose': 'Complete selected Xception A/B/C benchmark and matched ResNet P/A/B/C benchmark; FairForge2 cross-dataset evaluation',
          'authorization': protocol['authorization'], 'run_root': str(root), 'no_ding_retraining': True, 'no_new_image_generation': True})
    write(root/'preflight/manifest_audit.json', reports)
    print('PREPARED', proposal, root, flush=True)


if __name__ == '__main__':
    main()
