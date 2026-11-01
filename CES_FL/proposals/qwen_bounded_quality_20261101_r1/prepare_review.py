"""Read-only pilot/source audit and private review preparation; never launches jobs."""
import base64
import collections
import csv
import hashlib
import html
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

HOME = Path('/home/sachin.chaudhary')
PINNED = HOME / 'ces_fl_sources/cesfl-cf5050-fc4e63b'
PILOT = HOME / 'ces_fl_runs/cf5050_registry_recovery_20261101T081523Z'
PUBLISH = HOME / 'ces_fl_sources/publish_cf5050_20261101T065603Z'
STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
RUN = HOME / ('ces_fl_runs/qwen_review_revision_' + STAMP)
RUN.mkdir(exist_ok=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    (RUN / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def command(argv, timeout=40):
    completed = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    result = {'argv': argv, 'returncode': completed.returncode,
              'stdout': completed.stdout, 'stderr': completed.stderr,
              'observed_management_utc': datetime.now(timezone.utc).isoformat()}
    commands.append(result)
    return result


commands = []
source = PINNED / 'CES_FL/src/ces_fl_counterfactual_5050.py'
config = PINNED / 'CES_FL/proposals/counterfactual_5050_20261101_r1/protocol_config.json'
c = json.loads(config.read_text())
assert digest(source) == '3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab'
assert digest(config) == '56e77211a966ae12a449c47cdaf8e3fc271f149c6031513747813a1d1abfaef0'
input_checks = {}
for key, expected in c['input_hashes'].items():
    actual = digest(c[key])
    assert actual == expected, (key, expected, actual)
    input_checks[key] = actual

plan_dir = PILOT / 'generation'
plan = json.loads((plan_dir / 'plan.json').read_text())
assert digest(plan_dir / 'sources.json') == plan['sources_sha256']
assert plan['config'] == c and plan['source_sha256'] == digest(source)
sources = json.loads((plan_dir / 'sources.json').read_text())
inventory = {entry['original_image']: entry for entry in sources}
train = list(csv.DictReader(Path(c['train']).open()))
validation = list(csv.DictReader(Path(c['val']).open()))
assert [r['image'] for r in train] == [s['original_image'] for s in sources]
assert not ({r['image'] for r in train} & {r['image'] for r in validation})

pending_path = PILOT / 'pilot_review_pending.csv'
pending_hash = digest(pending_path)
reviews = list(csv.DictReader(pending_path.open()))
review_by_id = {r['pair_id']: r for r in reviews}
assert len(review_by_id) == len(reviews) == 14
audited = []
originals = {}
for record_path in sorted((plan_dir / 'items').glob('*/record.json')):
    record = json.loads(record_path.read_text())
    s = record['source']
    assert s == inventory[s['original_image']]
    assert record['plan_sha256'] == digest(plan_dir / 'plan.json')
    assert digest(s['original_image']) == s['original_sha256']
    with Image.open(s['original_image']) as im:
        im.load()
        size = im.size
    originals[s['source_id']] = s
    assert len(record['variants']) == 7
    rc = record['resize_control']
    assert digest(rc['image']) == rc['sha256']
    with Image.open(rc['image']) as im:
        im.load()
        assert im.size == size
    for condition, candidate in record['variants'].items():
        assert candidate['input_image'] == s['original_image']
        assert candidate['input_sha256'] == s['original_sha256']
        assert candidate['independent_from_original'] is True
        assert digest(candidate['image']) == candidate['sha256']
        assert candidate['image'] == str(record_path.parent / (condition + '.png'))
        with Image.open(candidate['image']) as im:
            im.load()
            assert im.size == size
        q = review_by_id[candidate['pair_id']]
        assert q['candidate_image'] == candidate['image']
        assert q['original_image'] == s['original_image'] and q['condition'] == condition
        audited.append({'pair_id': candidate['pair_id'], 'condition': condition,
                        'source_label': s['label'], 'hash_and_alignment_pass': True,
                        'approved_recorded': q['approved'], 'reviewer_recorded': q['reviewer'],
                        'human_validity_established_by_this_audit': False})
assert len(audited) == 14

# Exact, deterministic candidate selection for a NEW QUALITY-ONLY proposal.
# This does not change the original full-coverage plan, its TRAIN manifest, or approvals.
counts = collections.Counter((r['label'], r['ids8']) for r in train)
assert len(counts) == 16
used_identity, used_source, selected = set(), set(), []
for cell in sorted(counts, key=lambda k: (counts[k], k)):
    candidates = sorted((r for r in train if (r['label'], r['ids8']) == cell),
                        key=lambda r: hashlib.sha256(('17|' + r['image']).encode()).hexdigest())
    chosen = []
    for row in candidates:
        identities = set(row['identity_ids'].split(';'))
        contributors = set(row['source_ids'].split(';'))
        if identities & used_identity or contributors & used_source:
            continue
        chosen.append(dict(row))
        used_identity.update(identities)
        used_source.update(contributors)
        if len(chosen) == 2:
            break
    assert len(chosen) == 2, ('Insufficient distinct contributor proxies', cell)
    selected.extend(chosen)
assert len(selected) == 32
assert set(collections.Counter((r['label'], r['ids8']) for r in selected).values()) == {2}
dimensions = collections.Counter()
for row in selected:
    row['original_sha256'] = digest(row['image'])
    assert row['original_sha256'] == inventory[row['image']]['original_sha256']
    with Image.open(row['image']) as im:
        im.load()
        dimensions[str(im.size)] += 1
    assert row['image'] not in {r['image'] for r in validation}
selected_path = RUN / 'proposed_32_sources_PRIVATE.csv'
with selected_path.open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(selected[0]))
    writer.writeheader()
    writer.writerows(selected)
val_i = {v for r in validation for v in r['identity_ids'].split(';')}
val_s = {v for r in validation for v in r['source_ids'].split(';')}
exposure = {'selected_originals': 32, 'label_counts': dict(collections.Counter(r['label'] for r in selected)),
            'label_ids8_counts': {str(k): v for k, v in sorted(collections.Counter((r['label'], r['ids8']) for r in selected).items())},
            'family_counts': dict(collections.Counter(r['family'] for r in selected)),
            'within_selected_disjoint_contributor_sets': True,
            'unique_identity_proxies': len(used_identity), 'unique_source_video_ids': len(used_source),
            'selected_image_path_overlap_with_validation': 0,
            'selected_identity_proxy_overlap_with_validation': len(used_identity & val_i),
            'selected_source_id_overlap_with_validation': len(used_source & val_s),
            'person_ground_truth_verified': False, 'dimensions': dict(dimensions),
            'private_manifest_sha256': digest(selected_path),
            'selection_rule': 'Seed-17 SHA256 image-path order, scarce strata first, two per label x ids8; reject contributors used by any previously selected original',
            'selection_depends_on_detector_or_test_scores': False}

# A private, portable contact-review page; no review decisions or approvals are generated.
def thumbnail(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        im.thumbnail((320, 320))
        b = io.BytesIO()
        im.save(b, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode()

blocks = []
for q in reviews:
    folder = Path(q['candidate_image']).parent
    paths = [('Original', q['original_image']), ('Resize only', str(folder / 'resize_geometry.png')),
             ('Editor no-change', str(folder / 'nochange.png')), ('Candidate', q['candidate_image'])]
    imgs = ''.join('<figure><img src="' + thumbnail(p) + '"><figcaption>' + html.escape(label) + '</figcaption></figure>'
                   for label, p in paths)
    blocks.append('<section><h2>' + html.escape(q['pair_id']) + '</h2><div class="strip">' + imgs +
                  '</div><p>Condition: ' + html.escape(q['condition']) + '</p><p>Original: ' +
                  html.escape(q['original_image']) + '</p><p>Candidate: ' + html.escape(q['candidate_image']) +
                  '</p><p>Record your decision and evidence in the review CSV. This page assigns no validity label.</p></section>')
page = '<!doctype html><html><meta charset="utf-8"><title>Private Qwen pilot review</title><style>body{font:16px system-ui;margin:24px}section{border-top:1px solid #aaa;margin-top:24px}.strip{display:flex;flex-wrap:wrap}figure{margin:8px}img{width:260px;max-width:100%}p{overflow-wrap:anywhere}</style><h1>14 pending Qwen candidates: human review</h1><p>Default state: unreviewed. Appearance alone does not establish identity or preservation of forgery evidence. Compare with both controls. Do not approve based on detector scores.</p><p>Review CSV: ' + html.escape(str(pending_path)) + '</p>' + ''.join(blocks) + '</html>'
(RUN / 'PRIVATE_REVIEW_14.html').write_text(page)
assert page.count('<section>') == 14 and page.count('<figure>') == 56
assert digest(pending_path) == pending_hash

command(['qstat', '-u', 'sachin.chaudhary'])
probe = '''import json,os,subprocess
from pathlib import Path
r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu','--format=csv'],capture_output=True,text=True)
active=[]
names={'ces_fl_counterfactual_5050.py','pilot_profile.py','ces_fl_diagram_model_v021.py','ces_fl_diagram_model.py'}
for p in Path('/proc').iterdir():
 if not p.name.isdigit(): continue
 try:
  if p.stat().st_uid != os.getuid(): continue
  args=(p/'cmdline').read_bytes().decode().split('\\0')
  if args and Path(args[0]).name.startswith('python') and any(Path(a).name in names for a in args[1:] if a): active.append({'pid':int(p.name),'argv':args})
 except (OSError,UnicodeError): pass
print(json.dumps({'gpu_query_returncode':r.returncode,'gpu_query':r.stdout,'stderr':r.stderr,'registry_directories':sorted(p.name for p in Path('/tmp/scf_gpu_locks').glob('gpu*')),'matching_python_workers':active},indent=2))
'''
gpu_states = {}
for host in ('gpu01', 'gpu02'):
    result = command(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=12', host,
                      'python - <<\'CESPROBE\'\n' + probe + '\nCESPROBE'])
    assert result['returncode'] == 0, (host, result)
    gpu_states[host] = json.loads(result['stdout'])
remote = command(['git', '-C', str(PUBLISH), 'ls-remote', 'origin', 'refs/heads/main'])
assert remote['returncode'] == 0
profile = json.loads((PILOT / 'checks/PILOT_PROFILE.json').read_text())
times = [x['seconds'] for x in profile['output_records_aggregate']]
outputs = 32 * 7
hours = outputs * profile['mean_seconds_per_output_excluding_load'] / 3600
estimate = {'candidate_sources': 32, 'diffusion_outputs': outputs, 'resize_controls': 32,
            'projected_gpu_hours_from_existing_two_source_mean': hours,
            'per_output_observed_range_seconds': [min(times), max(times)],
            'wall_hours_if_one_equivalently_fast_gpu_continuously_available': hours,
            'planning_allowance_hours_one_gpu_before_review': [4, 6],
            'training_eta_measured': False,
            'limits': 'Two correlated sources, shared GPU. Not a confidence interval or booking; excludes new setup, queue, retries, rejects, review and training. No full-study or training ETA is established.'}
report = {'status': 'AUTOMATIC_RECHECK_PASS_HUMAN_VALIDITY_PENDING',
          'observed_management_utc': datetime.now(timezone.utc).isoformat(),
          'run_root': str(RUN), 'source_commit': 'fc4e63b8811c5813afa0d298a8172ba9555b7e40',
          'source_sha256': digest(source), 'config_sha256': digest(config), 'input_hash_checks': input_checks,
          'generation_inventory_sha256': plan['sources_sha256'],
          'attempted_completed_missing': [14, 14, 0],
          'approved_rows_in_current_artifact': sum(r['approved'] == '1' for r in reviews),
          'formal_rejections_recorded': sum(r['approved'] == '0' and bool(r['reviewer'].strip()) for r in reviews),
          'pilot_review_file_unchanged': digest(pending_path) == pending_hash,
          'all_generated_bytes_and_original_associations_reverified': True,
          'semantic_validity_established': False,
          'gpu_states': gpu_states, 'new_training_or_generation_launched': False,
          'automatic_test_evaluation': False, 'existing_jobs_modified': False,
          'ff2': 'DO NOT RUN', 'proposal_coverage_is_not_full_train_coverage': True,
          'bound_training_configuration_created': False,
          'reviewed_candidate_details': audited, 'bounded_quality_sample': exposure,
          'tentative_compute': estimate, 'github_main_before_this_work': remote['stdout'].strip()}
save('AUTOMATIC_RECHECK.json', report)
save('COMMAND_RECEIPTS.json', commands)
save('BOUNDED_SOURCE_SUMMARY.json', exposure)
save('TENTATIVE_COMPUTE.json', estimate)
save('PROPOSED_QUALITY_SCOPE.json', {
    'proposal_id': 'qwen_bounded_quality_20261101_r1',
    'status': 'DRAFT_NOT_APPROVED_FOR_GENERATION_OR_TRAINING',
    'purpose': 'Resolve editor/no-change and validity uncertainty at bounded cost; no fairness efficacy claim.',
    'selected_private_manifest': str(selected_path), 'selected_private_manifest_sha256': digest(selected_path),
    'seed': 17, 'sources': 32, 'stratum': 'label x ids8', 'sources_per_stratum': 2,
    'source_commit': report['source_commit'], 'source_sha256': report['source_sha256'],
    'parent_protocol_config_sha256': report['config_sha256'],
    'generator': {k: c[k] for k in ('model_snapshot', 'generation_steps', 'generation_cfg', 'generation_area')},
    'conditions': list(plan['prompts']), 'independent_original_inputs': True,
    'full_train_bank_replaced': False, 'training_configuration': None,
    'launch_prerequisites': ['Complete current 14-pair human decisions with evidence/uncertainty',
                             'Determine whether editor confounds justify any additional same-recipe generation',
                             'Explicit review of bounded-quality scope and resources',
                             'Source/config/input/selected-byte hashes and CUDA binding verified',
                             'Implement and test an explicit selected-source generation interface; do not weaken the production full-coverage guard'],
    'training_launch_permitted': False,
    'training_requires': 'A separate explicit reviewed design with appropriate full source coverage, matched B/N/C initialization/budgets, validation-only selection/calibration, and the original approval checks. This sample cannot bind the existing 76,139-original production study.'})
print(json.dumps({'run_root': str(RUN), 'audit_status': report['status'],
                  'pilot_outputs_verified': len(audited), 'approved_rows': report['approved_rows_in_current_artifact'],
                  'selected_quality_sources': len(selected), 'unique_contributor_proxies': len(used_identity),
                  'private_review_page': str(RUN / 'PRIVATE_REVIEW_14.html'),
                  'projected_32_source_gpu_hours': hours,
                  'new_training_or_generation_launched': False}, indent=2))
