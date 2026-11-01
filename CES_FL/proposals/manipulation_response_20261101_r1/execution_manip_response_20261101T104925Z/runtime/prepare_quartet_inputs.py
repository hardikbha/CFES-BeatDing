"""Prepare prospective TRAIN-only quartet provenance/review inputs; no generation."""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/manip_response_20261101T104925Z')
SOURCE = Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb')
QWEN = Path('/home/sachin.chaudhary/ces_fl_runs/qwen_quality32_20261101T095157Z')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokens(row, key):
    return set(row[key].split(';')) - {''}


def main():
    c = json.loads((SOURCE / 'CES_FL/proposals/manipulation_response_20261101_r1/config.json').read_text())
    train_path, val_path = (Path(c['pretrain'][k]) for k in ('train', 'val'))
    assert sha(train_path) == c['pretrain']['input_hashes']['train']
    assert sha(val_path) == c['pretrain']['input_hashes']['val']
    train, val = list(csv.DictReader(train_path.open())), list(csv.DictReader(val_path.open()))
    vi = {v for r in val for v in tokens(r, 'identity_ids')}
    vs = {v for r in val for v in tokens(r, 'source_ids')}
    real = [r for r in train if r['label'] == '0']
    eligible = [r for r in real if not (tokens(r, 'identity_ids') & vi or tokens(r, 'source_ids') & vs)]
    by_path = {r['image']: r for r in real}
    eligible_paths = {r['image'] for r in eligible}
    plan = json.loads((QWEN / 'generation/plan.json').read_text())
    selected = json.loads((QWEN / 'generation/sources.json').read_text())
    assert sha(QWEN / 'generation/sources.json') == plan['sources_sha256']
    existing = [by_path[s['original_image']] for s in selected
                if s['label'] == 0 and s['original_image'] in eligible_paths]
    groups = sorted({r['gender_race'] for r in real})
    assert len(groups) == 8 and 'unknown' not in groups
    targets = []
    used_i, used_s = set(), set()
    ordering = lambda r: hashlib.sha256(('17:quartet:' + r['image']).encode()).hexdigest()
    # Reserve reusable candidate sources first; this is not an approval or bank.
    for group in groups:
        for row in sorted((r for r in existing if r['gender_race'] == group), key=ordering)[:2]:
            assert not (tokens(row, 'identity_ids') & used_i or tokens(row, 'source_ids') & used_s)
            targets.append((group, row, True))
            used_i.update(tokens(row, 'identity_ids')); used_s.update(tokens(row, 'source_ids'))
    supplementation = collections.Counter()
    for group in groups:
        missing = 2 - sum(g == group for g, _, _ in targets)
        for row in sorted((r for r in eligible if r['gender_race'] == group), key=ordering):
            if not missing:
                break
            if tokens(row, 'identity_ids') & used_i or tokens(row, 'source_ids') & used_s:
                continue
            targets.append((group, row, False)); supplementation[group] += 1; missing -= 1
            used_i.update(tokens(row, 'identity_ids')); used_s.update(tokens(row, 'source_ids'))
        assert missing == 0, ('Insufficient eligible distinct target proxies', group)
    assert len(targets) == 16
    donors = []
    used_di, used_ds = set(), set()
    for group, source, reuse in targets:
        pool = [r for r in eligible if r['gender_race'] == group and not (
                tokens(r, 'identity_ids') & used_i or tokens(r, 'source_ids') & used_s)]
        distinct = [r for r in pool if not (tokens(r, 'identity_ids') & used_di or tokens(r, 'source_ids') & used_ds)]
        chosen = sorted(distinct or pool, key=ordering)
        assert chosen, ('No donor candidate without target/VAL contributor overlap', group)
        donor = chosen[0]
        used_di.update(tokens(donor, 'identity_ids')); used_ds.update(tokens(donor, 'source_ids'))
        donors.append(donor)
    output = ROOT / 'quartet_preparation'
    output.mkdir(exist_ok=False)
    manifest = []
    group_visit = collections.Counter()
    available_base_pairs = collections.Counter()
    for (group, row, reuse), donor in zip(targets, donors):
        direction = ('older', 'younger')[group_visit[group]]
        group_visit[group] += 1
        sid = hashlib.sha256(row['image'].encode()).hexdigest()[:24]
        qid = sid + '_' + direction
        source_sha, donor_sha = sha(row['image']), sha(donor['image'])
        record_path = QWEN / 'generation/items' / sid / 'record.json'
        record = json.loads(record_path.read_text()) if record_path.exists() else None
        x0 = xa = ''
        x0_sha = xa_sha = ''
        seed = int(hashlib.sha256(('17:' + sid).encode()).hexdigest()[:15], 16)
        if record:
            assert record['source']['original_sha256'] == source_sha
            for role, condition in [('x0', 'nochange'), ('xa', direction)]:
                v = record['variants'].get(condition)
                if v:
                    assert v['input_sha256'] == source_sha and v['independent_from_original'] is True
                    assert v['seed'] == seed and sha(v['image']) == v['sha256']
                    if role == 'x0': x0, x0_sha = v['image'], v['sha256']
                    else: xa, xa_sha = v['image'], v['sha256']
        if x0 and xa:
            available_base_pairs[group] += 1
        identities = tokens(row, 'identity_ids') | tokens(donor, 'identity_ids')
        contributors = tokens(row, 'source_ids') | tokens(donor, 'source_ids')
        assert not (identities & vi or contributors & vs)
        manifest.append({'quartet_id': qid, 'group': group, 'direction': direction,
            'source_image': row['image'], 'source_sha256': source_sha,
            'donor_image': donor['image'], 'donor_sha256': donor_sha,
            'identity_ids': ';'.join(sorted(identities)), 'source_ids': ';'.join(sorted(contributors)),
            'potential_reuse_of_qwen_source': int(reuse), 'source_requires_supplementation': int(not reuse),
            'expected_editor_seed': seed, 'actual_x0_image': x0, 'actual_x0_sha256': x0_sha,
            'actual_xa_image': xa, 'actual_xa_sha256': xa_sha,
            'actual_t0_image': '', 'actual_ta_image': '',
            'operator_validated_on_this_quartet': 0, 'approved': 0, 'reviewer': '',
            'identity_preserved': 0, 'attribute_valid': 0, 'operation_valid': 0,
            'matched_processing': 0, 'unrelated_changes_assessed': 0, 'evidence': '',
            'status': 'BLOCKED_INPUTS_OPERATOR_ADAPTER_MANIPULATION_OUTPUTS_AND_INDEPENDENT_REVIEW'})
    with (output / 'prospective_quartets_PRIVATE.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0])); writer.writeheader(); writer.writerows(manifest)
    summary = {'status': 'BLOCKED_INPUTS', 'proposed_unique_sources': 16, 'expected_groups': groups,
        'directions_per_group': ['older', 'younger'],
        'eligible_qwen32_real_sources': len(existing),
        'eligible_qwen32_real_source_counts_by_gender_race': dict(collections.Counter(r['gender_race'] for r in existing)),
        'potentially_reusable_selected_source_count': sum(reuse for _, _, reuse in targets),
        'currently_generated_selected_x0_xa_pairs_by_group': dict(available_base_pairs),
        'supplementation_sources_needed': sum(supplementation.values()),
        'supplementation_counts_by_group': dict(supplementation),
        'real_train_counts_by_group': dict(collections.Counter(r['gender_race'] for r in real)),
        'eligible_real_train_counts_by_group': dict(collections.Counter(r['gender_race'] for r in eligible)),
        'selected_source_and_donor_recorded_val_contributor_overlap': 0,
        'unique_target_proxies': len(used_i), 'unique_donor_proxies': len(used_di),
        'unique_donor_images': len({d['image'] for d in donors}),
        'donor_reuse_max': max(collections.Counter(d['image'] for d in donors).values()),
        'prospective_manifest_sha256': sha(output / 'prospective_quartets_PRIVATE.csv'),
        'real_manipulation_outputs_generated': 0, 'approved_quartets': 0,
        'new_qwen_images_generated': 0, 'qwen_running_job_modified': False,
        'source_selection_depends_on_model_scores': False,
        'no_population_or_biological_identity_validity_claimed': True,
        'blocking_dependencies': ['TRAIN-only adaptation of the discovered SimSwap pipeline',
            'recorded matched manipulation outputs and geometry-only controls',
            'required supplemental age/no-change candidates, if individually approved for generation',
            'independent substantive quartet review', 'bank audit and allocated per-arm backward PASS',
            'verified P50 dependency']}
    (output / 'AGGREGATE.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
