import copy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


r = load('gradient_runner_test', ROOT / 'scripts/lh_gradient_conflict.py')
inv = load('provenance_inventory_test', ROOT / 'scripts/ff2_provenance_inventory.py')
gp, m = r.gp, r.m
fixture = load('existing_alignment_tests', ROOT / 'tests/test_lh_alignment_followup.py')
torch.set_num_threads(2)


class TwoParameters(nn.Module):
    def __init__(self):
        super().__init__(); self.w = nn.Parameter(torch.zeros(2))


class ProjectionTests(unittest.TestCase):
    def test_independent_two_dimensional_reference_and_global_projection(self):
        # b=[1,2], a=[-2,-1], total=b+a. The orthogonal alignment
        # component is [-1.2,.6], and the new total is [-.2,2.6].
        total = [torch.tensor([-1.], dtype=torch.float64), torch.tensor([1.], dtype=torch.float64)]
        bce = [torch.tensor([1.], dtype=torch.float64), torch.tensor([2.], dtype=torch.float64)]
        align = [torch.tensor([-2.], dtype=torch.float64), torch.tensor([-1.], dtype=torch.float64)]
        out, stats = gp.project_alignment(total, bce, align)
        np.testing.assert_allclose(torch.cat(out).numpy(), [-.2, 2.6], rtol=0, atol=1e-14)
        self.assertAlmostEqual(stats['projection_coefficient'], .8)
        self.assertAlmostEqual(stats['projected_alignment_bce_dot'], 0.)
        self.assertTrue(torch.equal(total[0], torch.tensor([-1.], dtype=torch.float64)))

    def test_disabled_nonconflicting_and_zero_cases_preserve_native_gradient(self):
        total = [torch.tensor([.3, -.4])]; bce = [torch.tensor([1., 2.])]
        for alignment, enabled in [([torch.tensor([2., 1.])], True),
                                   ([torch.tensor([-2., -1.])], False),
                                   ([torch.zeros(2)], True)]:
            out, stats = gp.project_alignment(total, bce, alignment, enabled)
            self.assertTrue(torch.equal(out[0], total[0])); self.assertFalse(stats['projection_applied'])
        out, stats = gp.project_alignment(total, [torch.zeros(2)], [torch.ones(2)])
        self.assertIsNone(stats['raw_alignment_bce_cosine']); self.assertTrue(torch.equal(out[0], total[0]))

    def test_invalid_gradient_input_fails(self):
        with self.assertRaises(ValueError): gp.project_alignment([], [], [])
        with self.assertRaises(ValueError): gp.project_alignment([torch.ones(2)], [torch.ones(3)], [torch.ones(2)])
        with self.assertRaises(ValueError): gp.project_alignment([torch.tensor([float('nan')])], [torch.ones(1)], [torch.ones(1)])

    def test_actual_adamw_moments_follow_projected_gradient(self):
        model = TwoParameters(); opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=0.)
        cls = F.binary_cross_entropy_with_logits(model.w.sum(), torch.ones(()))
        align = (model.w[0] + 1.).square()
        step = gp.ProjectedStep({'aux_every': 1, 'lambda_alignment': 1., 'alignment_warmup_epochs': 0})
        stats = step({'classification': cls, 'response_alignment': align, 'total': cls + align}, model, opt)
        # Independent derivatives at w=0: BCE=[-.5,-.5], alignment=[2,0].
        # Corrected total=[.5,-1.5]; AdamW beta1=.9 gives first moment [.05,-.15].
        torch.testing.assert_close(opt.state[model.w]['exp_avg'], torch.tensor([.05, -.15]))
        self.assertTrue(stats['projection_applied']); self.assertAlmostEqual(stats['projection_coefficient'], 2.)

    def test_native_mode_matches_original_adamw_step_and_bn(self):
        b, a, c = fixture.toy_batch()
        baseline = fixture.TinyDetector(); alternative = copy.deepcopy(baseline)
        before = {k: v.clone() for k, v in baseline.named_buffers()}
        outputs = []
        for model, mode in [(baseline, 'native_core'), (alternative, 'native')]:
            losses, _ = m.train_step(model, b, a, torch.ones(4), c, 'D', 0., torch.device('cpu'))
            opt = m.build_optimizer(model, c)
            if mode == 'native_core': m.finite_step(losses, model, opt)
            else: gp.ProjectedStep(c, 'native')(losses, model, opt)
            outputs.append(opt.state_dict())
        for key, value in baseline.state_dict().items(): self.assertTrue(torch.equal(value, alternative.state_dict()[key]), key)
        for key, value in before.items(): self.assertTrue(torch.equal(value, dict(alternative.named_buffers())[key]))
        for index, state in outputs[0]['state'].items():
            for key, value in state.items():
                if torch.is_tensor(value): self.assertTrue(torch.equal(value, outputs[1]['state'][index][key]))

    def test_hook_restores_native_after_failure_and_deadline(self):
        c = {**m.defaults(), 'alignment_warmup_epochs': 0}
        original = m.finite_step
        with self.assertRaises(RuntimeError):
            with gp.installed(m, c): raise RuntimeError('fixture')
        self.assertIs(m.finite_step, original)
        with self.assertRaises(TimeoutError): gp.ProjectedStep(c, deadline_epoch=0)({}, None, None)
        with self.assertRaises(ValueError): gp.ProjectedStep({**c, 'alignment_warmup_epochs': 2})

    def test_trigger_requires_repeated_supported_conflict(self):
        settings = {'minimum_valid_batches': 16, 'minimum_negative_fraction': .6,
                    'minimum_median_alignment_to_bce_ratio': .25}
        def records(n, cosine, ratio):
            return [{'gradients': {'alignment': {'all': {'cosine_to_BCE': cosine, 'norm_ratio_to_BCE': ratio}}}} for _ in range(n)]
        self.assertTrue(r.assess(records(20, -.2, .3), settings)['run_candidate'])
        self.assertFalse(r.assess(records(15, -.2, .3), settings)['run_candidate'])
        self.assertFalse(r.assess(records(20, .2, .3), settings)['run_candidate'])
        self.assertFalse(r.assess(records(20, -.2, .2), settings)['run_candidate'])
        self.assertFalse(r.assess(records(20, None, None), settings)['run_candidate'])

    def test_gate_rejects_duplicates_and_changed_main_plan(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); binding = root / 'binding.json'; binding.write_text('{}')
            rows = [{'image': f'train/{i}.png', 'label': i % 2} for i in range(640)]
            settings = {'events': list(range(20)), 'minimum_valid_batches': 16,
                        'minimum_negative_fraction': .6, 'minimum_median_alignment_to_bce_ratio': .25}
            records = []
            for event in range(20):
                indices = r.f.main_indices(rows, event)
                op, strength = [(o, s) for o in m.OPERATORS for s in m.STRENGTHS][event // 5]
                records.append({'event': event, 'axis': ['gender', 'race', 'age_bin', 'skin_bin', 'gender_race'][event % 5],
                                'operator': op, 'strength': strength, 'optimizer_updates': 0,
                                'main_plan_digest': m.digest([rows[i]['image'] for i in indices]),
                                'main_class_counts': dict(r.Counter(str(rows[i]['label']) for i in indices)),
                                'gradients': {'alignment': {'all': {'cosine_to_BCE': -.5, 'norm_ratio_to_BCE': .3}}}})
            d = {'status': 'COMPLETE', 'binding_sha256': m.sha(binding), 'fingerprints': {'fixture': 'hash'},
                 'optimizer_updates': 0, 'test_cal_read': False, 'state_unchanged': True,
                 'records': records, 'decision': r.assess(records, settings)}
            diagnostic = root / 'diagnostic.json'; diagnostic.write_text(json.dumps(d))
            with patch.object(r, 'fingerprints', return_value={'fixture': 'hash'}):
                r.gate_diagnostic(binding, diagnostic, {'diagnostic': settings}, rows)
                bad = copy.deepcopy(d); bad['records'][1] = bad['records'][0]; diagnostic.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): r.gate_diagnostic(binding, diagnostic, {'diagnostic': settings}, rows)
                bad = copy.deepcopy(d); bad['records'][1]['main_plan_digest'] = 'changed'; diagnostic.write_text(json.dumps(bad))
                with self.assertRaises(ValueError): r.gate_diagnostic(binding, diagnostic, {'diagnostic': settings}, rows)

    def test_metadata_missing_cells_and_donor_graph_are_preserved(self):
        def row(i, y, group, source, identities):
            return {'image': str(i), 'label': str(y), 'gender_race': group, 'source_ids': source,
                    'identity_ids': identities, 'label_provenance': 'private-provenance/' + str(i)}
        rows = [row(1, 0, 'Female-White', 'ffhq:00001', 'person:a'),
                row(2, 0, 'Female-Others', 'img:images/real/C_fairface/2.jpg', 'person:b'),
                row(3, 0, 'Female-Others', 'img:images/real/C_fairface/3.jpg', 'person:c'),
                row(4, 1, 'Female-Others', 'img:images/real/C_fairface/2.jpg', 'person:b;person:c')]
        result = inv.inventory(rows)
        missing = next(c for c in result['source_joint_support'] if c['source'] == 'FFHQ' and c['group'] == 'Female-Others')
        self.assertEqual(missing['real_rows'], 0)
        self.assertEqual(result['real_bearing_full_graph_components'], 2)
        self.assertEqual(result['real_only_graph_components_DIAGNOSTIC_ONLY'], 3)
        self.assertNotIn('person:a', json.dumps(result)); self.assertNotIn('private-provenance/1', json.dumps(result))
        self.assertEqual(inv.acquisition({'source_ids': 'unknown'}), 'UNRESOLVED')
        unknown = inv.inventory(rows + [row(5, 0, '', 'img:images/real/C_fairface/5.jpg', 'person:d')])
        self.assertEqual(unknown['global_joint_class_support']['unknown:0'], 1)

    def test_reviewed_pixel_replay_detects_runtime_operator_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); size = 75; image = root / 'image.png'; sheet = root / 'sheet.png'
            Image.fromarray(np.random.default_rng(17).integers(0, 256, (size, size, 3), dtype=np.uint8)).save(image)
            row = {'image': str(image), 'image_sha256': m.sha(image)}
            x = m.load_pixels(row, size); s, t, _, _ = m.intervention(x, 'local_warp', .5, 17)
            canvas = Image.new('RGB', (4 * size, size + 24), 'white')
            for i, panel in enumerate((x, s, t, (t - s).abs() * 4)):
                canvas.paste(Image.fromarray(m.image_uint8(panel)), (i * size, 24))
            canvas.save(sheet)
            preview = {'cases': [{'image': str(image), 'sheet': str(sheet), 'settings':
                       {'operator': 'local_warp', 'strength': .5, 'seed': 17, 'common_post_nuisance': 'identity'}}]}
            self.assertEqual(r.verify_reviewed_pixels({'size': size}, [row], preview), 4)
            native = m.intervention
            def changed(*args, **kwargs):
                a, b, mask, settings = native(*args, **kwargs)
                return a, (b + .01).clamp(0, 1), mask, settings
            with patch.object(m, 'intervention', side_effect=changed):
                with self.assertRaises(ValueError): r.verify_reviewed_pixels({'size': size}, [row], preview)

    def test_allocation_rejects_unverified_wrong_ledger_and_reset_deadline(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td); ledger = p / 'ledger.json'; ledger.write_text('{}')
            now = time.time(); receipt = p / 'allocation.json'
            a = {'kind': 'PBS', 'allocation_id': 'software-test-only',
                 'gpu_uuid': 'GPU-00000000-0000-0000-0000-000000000000',
                 'deadline_epoch': now + 300, 'cohort_started_epoch': now - 30,
                 'cohort_id': 'lh_gradient_conflict_20261110_r1', 'ledger_scope': 'ORIGINAL_LH_STUDY',
                 'ledger_path': str(ledger), 'ledger_sha256': m.sha(ledger),
                 'verified_remaining_gpu_hours': 1., 'operator_verified': True}
            receipt.write_text(json.dumps(a))
            with patch.dict(r.os.environ, {'CES_GPU_BINDING_RECEIPT': str(receipt), 'CUDA_VISIBLE_DEVICES': a['gpu_uuid']}), patch.object(torch.cuda, 'device_count', return_value=1):
                self.assertEqual(r.allocation_guard(torch.device('cuda'))['allocation_id'], 'software-test-only')
                for change in [{'operator_verified': False}, {'ledger_scope': 'FF2'}, {'synthetic_fixture': True},
                               {'cohort_started_epoch': now - 8000}, {'deadline_epoch': now - 1},
                               {'gpu_uuid': 'GPU-other'}, {'ledger_sha256': 'changed'}]:
                    receipt.write_text(json.dumps({**a, **change}))
                    with self.assertRaises(ValueError): r.allocation_guard(torch.device('cuda'))

    def test_native_D_fixture_lifecycle_with_projection_and_checkpoint_provenance(self):
        with tempfile.TemporaryDirectory() as td, patch.object(m, 'Detector', fixture.TinyDetector):
            root = Path(td); paths = {}
            for split in ('train', 'val'):
                rows = []
                for i in range(16):
                    g = i // 2; sex = 'Female' if g < 4 else 'Male'; race = ['Asian', 'Black', 'Others', 'White'][g % 4]
                    image = root / f'{split}_{i}.png'
                    Image.fromarray(np.random.default_rng(i + (100 if split == 'val' else 0)).integers(0, 256, (75, 75, 3), dtype=np.uint8)).save(image)
                    rows.append({'image': str(image), 'label': i % 2, 'identity_ids': f'{split}_{i}', 'source_ids': f'{split}_{i}',
                                 'family': 'fixture', 'gender': sex, 'race': race, 'gender_race': sex + '-' + race,
                                 'age_bin': str(g % 2), 'skin_bin': str(g % 2), 'ids8': str(g)})
                paths[split] = root / (split + '.csv')
                with paths[split].open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            c = {**m.defaults(), 'dataset_id': 'SOFTWARE_FIXTURE', 'train': str(paths['train']), 'val': str(paths['val']),
                 'train_sha256': m.sha(paths['train']), 'val_sha256': m.sha(paths['val']), 'epochs': 2, 'size': 75,
                 'batch_size': 4, 'workers': 0, 'min_train_support': 1, 'min_eval_support': 1, 'group_per_class': 1,
                 'aux_every': 1, 'amp': False, 'synthetic_fixture': True, 'source_sha256': m.sha(m.__file__),
                 'max_gpu_hours': 1., 'nuisance_mode': 'none', 'error_control': 'hierarchical_real', 'train_calibration_size': 4}
            initial = fixture.TinyDetector(); checkpoint = root / 'initial.pt'
            torch.save({'format': 'CESFL_RESNET50_P50_v1', 'stage': 'pretrain', 'synthetic_fixture': True,
                        'config': {'train': c['train'], 'val': c['val'], 'backbone': 'resnet50'}, 'model': initial.state_dict()}, checkpoint)
            c.update(init_checkpoint=str(checkpoint), init_sha256=m.sha(checkpoint)); m.check_config(c, fixture=True)
            m.preflight(c, root / 'preflight')
            metadata = {'format': gp.VERSION, 'mode': 'projected', 'fixture': True}
            with gp.installed(m, c) as hook, r.checkpoint_metadata(metadata):
                result = m.train(c, root / 'preflight', root / 'run', 'D', torch.device('cpu'), fixture=True)
            ck = torch.load(root / 'run/best.pt', weights_only=True)
            self.assertEqual(result['status'], 'COMPLETED_VERIFIED')
            self.assertEqual(ck['gradient_update_rule'], metadata)
            self.assertEqual(hook.steps, 8)
            history = r.read(root / 'run/history.json')
            self.assertTrue(all(h['originals_once_each'] and h['original_real_pairs'] == 8 for h in history))
            restored, saved = m.load_trained(root / 'run/best.pt', torch.device('cpu'), fixture=True)
            validation = m.read_manifest(c['val']); z = m.predict(restored, validation, c, torch.device('cpu'))
            self.assertEqual(m.metric_report(validation, z, ck['calibration']['threshold_logit'], 1), ck['validation'])
            bootstrap = m.bootstrap(validation, z, ck['calibration']['threshold_logit'], 1, 20, 17,
                                    other=(z, ck['calibration']['threshold_logit']))
            self.assertEqual(bootstrap['intervals']['overall/auc']['estimate'], 0.)


if __name__ == '__main__':
    unittest.main()
