#!/usr/bin/env python3
"""Rebind an existing H recipe review only when every reviewed byte is unchanged.

Does not create approvals, change decisions, or certify real-forgery validity.
All input/output review artifacts are PRIVATE and must stay off GitHub.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('head_model', HERE.parents[1] / 'src/ces_fl_head_ablation.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def equivalent_cases(old, new):
    def index(manifest):
        result = {c['id']: c for c in manifest['cases']}
        if not result or len(result) != len(manifest['cases']):
            raise ValueError('Empty/duplicate preview cases')
        return result
    a, b = index(old), index(new)
    if set(a) != set(b):
        raise ValueError('Preview case membership changed; fresh review required')
    for key in a:
        # Only the contact-sheet LOCATION may change; all content/metadata must agree.
        x = {k: v for k, v in a[key].items() if k != 'sheet'}
        y = {k: v for k, v in b[key].items() if k != 'sheet'}
        if x != y:
            raise ValueError('Reviewed content/settings changed: ' + key)
        for case in (a[key], b[key]):
            for field in ('image', 'sheet'):
                if m.sha(case[field]) != case[field + '_sha256']:
                    raise ValueError('Reviewed bytes changed: ' + key)


def reuse(old_qualification, preview_dir, out):
    root = Path(preview_dir)
    new_path = root / 'preview_PRIVATE.json'
    new = json.loads(new_path.read_text())
    q = json.loads(Path(old_qualification).read_text())
    old_config = json.loads((HERE.parent / 'hierarchical_real_risk_s17_20261103_r1/H_joint_real_bound.json').read_text())
    new_config = json.loads((HERE / 'L_H_plain_hierarchical.json').read_text())
    if q['status'] != 'OPERATOR_REVIEW_RECORDED' or q['source_sha256'] != old_config['source_sha256'] or q['config_digest'] != m.digest(old_config):
        raise ValueError('Not the pinned H qualification')
    for field in ('preview', 'review'):
        if m.sha(q[field + '_file']) != q[field + '_sha256']:
            raise ValueError('Prior qualification evidence changed')
    old = json.loads(Path(q['preview_file']).read_text())
    if old['config_digest'] != q['config_digest'] or old['source_sha256'] != q['source_sha256']:
        raise ValueError('Prior preview binding mismatch')
    if new['source_sha256'] != m.sha(m.__file__) or new['config_digest'] != m.digest(new_config):
        raise ValueError('New preview binding mismatch')
    if old['synthetic_fixture'] or new['synthetic_fixture']:
        raise ValueError('Software fixtures cannot qualify production')
    equivalent_cases(old, new)
    review = json.loads(Path(q['review_file']).read_text())
    if review['preview_sha256'] != q['preview_sha256']:
        raise ValueError('Prior review binding mismatch')
    copied = copy.deepcopy(review)
    copied['preview_sha256'] = m.sha(new_path)
    copied['reuse_provenance'] = {
        'qualification_sha256': m.sha(old_qualification),
        'qualification_path': str(Path(old_qualification).resolve()),
        'basis': 'Every case, source byte and contact-sheet byte unchanged; reviewer decisions copied verbatim',
        'new_visual_review_performed': False,
    }
    out = m.fresh_dir(out)
    review_path = out / 'review_PRIVATE.json'
    m.write_json(review_path, copied)
    m.qualify(root, review_path, out / 'qualification.json')
    print('REVIEW_REUSED_IDENTICAL_BYTES', out / 'qualification.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-qualification', required=True)
    p.add_argument('--preview', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    reuse(a.old_qualification, a.preview, a.out)
