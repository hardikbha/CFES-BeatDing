#!/usr/bin/env python3
"""Metadata-only source/taxonomy/dependency inventory. No pixels or scores read.

This reports evidence for a provenance review; it cannot certify human labels,
biological identities or permission to split dependency components.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUEST = ROOT / 'proposals/lh_gradient_conflict_20261110_r1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def tokens(row, field):
    return set(filter(None, (s.strip() for s in row.get(field, '').split(';'))))


def acquisition(row):
    values = tokens(row, 'source_ids')
    # Real-source keys only; never infer demographic/source labels from pixels.
    if len(values) != 1:
        return 'UNRESOLVED'
    value = next(iter(values))
    if value.startswith('ffhq:') and value[5:].isdigit():
        return 'FFHQ'
    if value.startswith('img:images/real/C_fairface/'):
        return 'FairFace'
    return 'UNRESOLVED'


def components(rows):
    parent = list(range(len(rows)))
    seen = {}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, row in enumerate(rows):
        keys = [('image', row['image'])]
        keys += [(field, value) for field in ('identity_ids', 'source_ids') for value in tokens(row, field)]
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    return [root(i) for i in range(len(rows))]


def inventory(rows):
    graph = components(rows)
    reals = [i for i, r in enumerate(rows) if r['label'] == '0']
    real_graph = components([rows[i] for i in reals])
    cells = defaultdict(list)
    namespaces = {field: Counter() for field in ('identity_ids', 'source_ids')}
    real_identity_images, real_identity_groups = defaultdict(set), defaultdict(set)
    for i, row in enumerate(rows):
        for field in namespaces:
            namespaces[field].update(v.split(':', 1)[0] if ':' in v else 'unstructured' for v in tokens(row, field))
        if row['label'] != '0':
            continue
        source, group = acquisition(row), row.get('gender_race', 'unknown') or 'unknown'
        cells[source, group].append(i)
        for identifier in tokens(row, 'identity_ids'):
            real_identity_images[identifier].add(row['image'])
            real_identity_groups[identifier].add(group)
    groups = sorted({r.get('gender_race', 'unknown') or 'unknown' for r in rows})
    counts = []
    for source in ('FFHQ', 'FairFace', 'UNRESOLVED'):
        for group in groups:
            indices = cells[source, group]
            counts.append({'source': source, 'group': group, 'real_rows': len(indices),
                           'full_graph_real_components': len({graph[i] for i in indices})})
    sizes = Counter(graph)
    return {
        'rows': len(rows), 'class_counts': dict(Counter(r['label'] for r in rows)),
        'real_source_counts': dict(Counter(acquisition(rows[i]) for i in reals)),
        'source_joint_support': counts,
        'global_joint_class_support': {f'{g}:{y}': sum((r.get('gender_race') or 'unknown') == g and r['label'] == y for r in rows)
                                       for g in groups for y in ('0', '1')},
        'label_provenance_distinct_values': len({r.get('label_provenance', '') for r in rows}),
        'label_provenance_frequency_histogram': dict(Counter(str(v) for v in Counter(r.get('label_provenance', '') for r in rows).values())),
        'identifier_namespaces': {k: dict(v) for k, v in namespaces.items()},
        'full_graph_components': len(sizes), 'largest_component_rows': max(sizes.values(), default=0),
        'real_bearing_full_graph_components': len({graph[i] for i in reals}),
        'real_only_graph_components_DIAGNOSTIC_ONLY': len(set(real_graph)),
        'real_only_graph_changes_no_uncertainty_protocol': True,
        'real_identity_token_image_count_histogram': dict(Counter(str(len(v)) for v in real_identity_images.values())),
        'real_identity_tokens_crossing_joint_groups': sum(len(v) > 1 for v in real_identity_groups.values()),
        'duplicate_image_paths': len(rows) - len({r['image'] for r in rows}),
        'interpretation': 'Identifier reuse and graph bridging are diagnostics, not verified errors or permission to split components.',
    }


def run(out):
    frozen = json.loads((REQUEST / 'FROZEN_FILES.json').read_text())
    for name in ('scripts/ff2_provenance_inventory.py', 'proposals/lh_gradient_conflict_20261110_r1/ff2_provenance_inputs.json'):
        if sha(ROOT / name) != frozen[name]:
            raise ValueError('Pinned metadata helper/input specification changed')
    spec = json.loads((REQUEST / 'ff2_provenance_inputs.json').read_text())
    destination = Path(out)
    if destination.exists():
        raise ValueError('Preserve prior output; choose a new directory')
    reports, rows_by_split = {}, {}
    for split in ('train', 'val'):
        bound = spec[split]
        if sha(bound['path']) != bound['sha256']:
            raise ValueError('Manifest hash mismatch: ' + split)
        with Path(bound['path']).open(newline='') as f:
            reader = csv.DictReader(f)
            if not {'image', 'label', 'identity_ids', 'source_ids', 'gender_race'}.issubset(reader.fieldnames or []):
                raise ValueError('Required provenance columns missing')
            rows = list(reader)
        if len(rows) != bound['rows'] or any(r['label'] not in ('0', '1') for r in rows):
            raise ValueError('Changed row count or class labels')
        if any(not tokens(r, 'identity_ids') or not tokens(r, 'source_ids') for r in rows):
            raise ValueError('Missing recorded contributor IDs')
        allowed = {sex + '-' + race for sex in ('Female', 'Male') for race in ('Asian', 'Black', 'White', 'Others')} | {'unknown', ''}
        if any(r.get('gender_race', '') not in allowed for r in rows):
            raise ValueError('Unexpected group taxonomy; inspect privately before publication')
        reports[split] = inventory(rows)
        rows_by_split[split] = rows
        if sha(bound['path']) != bound['sha256']:
            raise ValueError('Input changed during review')
    overlap = {}
    for field in ('identity_ids', 'source_ids'):
        sets = [{v for row in rows_by_split[s] for v in tokens(row, field)} for s in ('train', 'val')]
        overlap[field] = len(sets[0] & sets[1])
    # Keep the all-contributor graph intact; project it to SELECT as the prior
    # audit did. The real-only diagnostic above never replaces this graph.
    together = components(rows_by_split['train'] + rows_by_split['val'])
    projected = together[len(rows_by_split['train']):]
    result = {'status': 'METADATA_INVENTORY_COMPLETE_SEMANTIC_REVIEW_PENDING',
              'input_spec_sha256': sha(REQUEST / 'ff2_provenance_inputs.json'), 'script_sha256': sha(__file__),
              'inputs': spec, 'splits': reports, 'recorded_overlap_counts': overlap,
              'train_plus_val_graph_val_components': len(set(projected)),
              'pixels_or_predictions_read': False, 'test_cal_read': False,
              'labels_or_results_modified': False, 'identity_validity_established': False,
              'required_manual_evidence': [
                  'Trace source-specific race taxonomy/filtering from original builder/annotation code; do not guess from faces.',
                  'Trace identity/source token construction and donor edges; distinguish repeated IDs from legitimate shared contributors.',
                  'Record source paths/hashes/line references and unresolved semantics. No automatic relabeling or component splitting.',
              ]}
    destination.mkdir(parents=True)
    (destination / 'METADATA_INVENTORY.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'val_components': result['train_plus_val_graph_val_components'],
                      'test_cal_read': False}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    run(parser.parse_args().out)
