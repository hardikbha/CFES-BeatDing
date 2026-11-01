"""CPU tests for generation selection; no Qwen calls, optimizer steps or approvals."""
import collections
import csv
import importlib.util
import json
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path('/home/sachin.chaudhary/ces_fl_runs/qwen_quality32_20261101T095157Z')
SCRIPT = Path('/home/sachin.chaudhary/ces_fl_sources/cesfl-cf5050-fc4e63b/CES_FL/src/ces_fl_counterfactual_5050.py')
spec = importlib.util.spec_from_file_location('quality_pinned_core', SCRIPT)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)
passed = []


def check(name, fn):
    fn()
    passed.append(name)
    print('PASS', name, flush=True)


def rejects(fn, text):
    try:
        fn()
    except ValueError as exc:
        assert text in str(exc), (text, str(exc))
    else:
        raise AssertionError('Expected rejection: ' + text)


def real_selection():
    root, plan, sources = core.load_plan(ROOT / 'generation')
    parent = json.loads(Path(plan['parent_plan']).read_text())
    inventory = json.loads((Path(plan['parent_plan']).parent / 'sources.json').read_text())
    lookup = {s['source_id']: s for s in inventory}
    rows = list(csv.DictReader(Path(plan['selected_private_manifest']).open()))
    assert plan['config'] == parent['config'] and plan['source_count'] == 32
    assert plan['quality_only'] and plan['full_train_binding_prohibited']
    assert not plan['synthetic_fixture']
    assert sources == [lookup[core.source_id(r['image'])] for r in rows]
    assert collections.Counter((r['label'], r['ids8']) for r in rows) == {
        (str(y), str(g)): 2 for y in range(2) for g in range(8)}
    for s in sources:
        assert core.sha(s['original_image']) == s['original_sha256']


def production_binding_guard():
    plan = json.loads((ROOT / 'generation/plan.json').read_text())
    rejects(lambda: core.audit_bank(ROOT / 'generation', ROOT / 'nonexistent_review.csv', plan['config']),
            'Plan must cover exact full TRAIN inventory')


def synthetic_generation():
    with tempfile.TemporaryDirectory(prefix='ces_quality_selection_fixture_') as temporary:
        root = Path(temporary)
        plan_root = root / 'generation'
        plan_root.mkdir()
        plan = json.loads((ROOT / 'generation/plan.json').read_text())
        sources = []
        for i in range(2):
            original = root / ('original_' + str(i) + '.png')
            Image.new('RGB', (96, 80), (20 + 40 * i, 10, 5)).save(original)
            sources.append({'source_id': core.source_id(original), 'original_image': str(original),
                            'original_sha256': core.sha(original), 'label': i,
                            'identity_ids': 'SYNTHETIC_TEST_ONLY_' + str(i),
                            'source_ids': 'SYNTHETIC_TEST_ONLY_' + str(i)})
        core.write(plan_root / 'sources.json', sources)
        plan.update(synthetic_fixture=True, source_count=2, generated_outputs=14, resize_controls=2,
                    sources_sha256=core.sha(plan_root / 'sources.json'))
        core.write(plan_root / 'plan.json', plan)
        rejects(lambda: core.load_plan(plan_root), 'Synthetic generation plan cannot be used in production')
        passed.append('fixture_rejected_by_production_loader')
        print('PASS fixture_rejected_by_production_loader', flush=True)
        pixels = []

        def render(source, condition, width, height, seed):
            original_pixel = source.getpixel((0, 0))[0]
            assert original_pixel in (20, 60)
            pixels.append(original_pixel)
            source.putpixel((0, 0), (250, 250, 250))
            return source

        first = core.generate(plan_root, shard_index=0, num_shards=2, synthetic_generator=render)
        assert first['new_generated_outputs'] == 7 and first['selected_sources'] == 1
        second = core.generate(plan_root, shard_index=1, num_shards=2, synthetic_generator=render)
        assert second['new_generated_outputs'] == 7 and second['selected_sources'] == 1
        assert collections.Counter(pixels) == {20: 7, 60: 7}
        assert len(list((plan_root / 'items').glob('*/record.json'))) == 2
        hashes = {str(p): core.sha(p) for p in (plan_root / 'items').rglob('*.png')}
        before = len(pixels)
        resume = core.generate(plan_root, synthetic_generator=render)
        assert resume['new_generated_outputs'] == 0 and len(pixels) == before
        assert hashes == {str(p): core.sha(p) for p in (plan_root / 'items').rglob('*.png')}
        for s in sources:
            record = json.loads((plan_root / 'items' / s['source_id'] / 'record.json').read_text())
            assert core.sha(s['original_image']) == s['original_sha256']
            assert len(record['variants']) == 7 and 'resize_control' in record
            assert all(v['approved'] is False and v['independent_from_original'] is True
                       and v['input_sha256'] == s['original_sha256'] for v in record['variants'].values())
        passed.append('disjoint_shards_original_only_inputs_seven_conditions_controls_zero_approvals_resume')
        print('PASS disjoint_shards_original_only_inputs_seven_conditions_controls_zero_approvals_resume', flush=True)
        original = Path(sources[0]['original_image'])
        original.write_bytes(b'CORRUPT_SYNTHETIC_TEST_ONLY')
        rejects(lambda: core.generate(plan_root, synthetic_generator=render), 'Original changed')
        passed.append('changed_original_rejected')
        print('PASS changed_original_rejected', flush=True)
        with (plan_root / 'sources.json').open('a') as f:
            f.write(' ')
        rejects(lambda: core.load_plan(plan_root, synthetic=True), 'Generation plan changed')
        passed.append('changed_source_inventory_rejected')
        print('PASS changed_source_inventory_rejected', flush=True)


check('real_32_source_exact_stratification_input_bytes_unchanged_config', real_selection)
check('real_quality_subset_cannot_bind_full_train_production_study', production_binding_guard)
synthetic_generation()
assert core.sha(SCRIPT) == '3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab'
result = {'status': 'PASS', 'checks': passed, 'count': len(passed), 'device': 'cpu',
          'real_generation_tested_by_this_command': False, 'qwen_called': False,
          'production_core_changed': False, 'training_started': False,
          'real_quality_selection_validated': True,
          'production_full_coverage_guard_verified': True}
core.write(ROOT / 'checks/SELECTION_TESTS.json', result)
print(json.dumps(result), flush=True)
