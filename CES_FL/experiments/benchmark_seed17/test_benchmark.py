import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
def module(name):
    s = importlib.util.spec_from_file_location(name, ROOT/(name+'.py'))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
U, P = module('uncertainty'), module('prepare')


class BenchmarkTests(unittest.TestCase):
    def test_monitor_does_not_claim_training_from_pbs_running_alone(self):
        import json
        monitor = module('monitor')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'JOBS.json').write_text(json.dumps({'jobs': {'P': {'job_id': '1'}}, 'source_commit': 'fixture', 'request_id': 'fixture'}))
            with patch.object(monitor.subprocess, 'run', return_value=SimpleNamespace(stdout='job_state = R\n', returncode=0)):
                result = monitor.observe(root)
            self.assertEqual(result['jobs'][0]['state'], 'STARTING_OR_UNVERIFIED')

    def test_monitor_reports_failed_dependency_without_cancelling_jobs(self):
        import json
        monitor = module('monitor')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'runtime').mkdir()
            (root/'runtime/P.failure.json').write_text('{"state":"FAILED"}')
            (root/'JOBS.json').write_text(json.dumps({'jobs': {'P': {'job_id': '1'}, 'B': {'job_id': '2', 'dependency_phase': 'P'}}, 'source_commit': 'fixture', 'request_id': 'fixture'}))
            with patch.object(monitor.subprocess, 'run', return_value=SimpleNamespace(stdout='job_state = H\n', returncode=0)):
                result = monitor.observe(root)
            self.assertEqual(result['jobs'][1]['state'], 'BLOCKED_UPSTREAM_FAILURE')
    def test_weighted_tied_auc_matches_independent_sklearn(self):
        y = np.array([0, 1, 0, 1, 0, 1]); z = np.array([.5, .5, 1, -2, 3, 3])
        w = np.array([2, 1, 4, 0, 3, 5])
        m = U.WeightedMetric(y, z, .5, np.ones(6, dtype=bool)).calculate(w)
        self.assertAlmostEqual(m['auc'], roc_auc_score(y, z, sample_weight=w))
        self.assertAlmostEqual(m['fpr'], np.average(z[y == 0] >= .5, weights=w[y == 0]))

    def test_shared_donor_connects_components_and_blocks_false_ci(self):
        rows = [{'identity_ids': 'a', 'source_ids': 'x', 'label': 0},
                {'identity_ids': 'b', 'source_ids': 'x;y', 'label': 1},
                {'identity_ids': 'c', 'source_ids': 'y', 'label': 0}]
        self.assertEqual(len(set(U.components(rows))), 1)
        self.assertIsNone(U.bootstrap(rows, [0., 1., .2], .5, [], reps=10)['intervals'])

    def test_zero_class_weight_has_no_auc(self):
        m = U.WeightedMetric(np.array([0, 1]), np.array([0., 1.]), .5, np.array([True, True]))
        self.assertIsNone(m.calculate(np.array([1., 0.])))

    def test_bootstrap_is_reproducible(self):
        rows = [{'identity_ids': str(i), 'source_ids': str(i), 'label': i % 2} for i in range(12)]
        a = U.bootstrap(rows, np.arange(12), 5., [], reps=20, seed=17)
        self.assertEqual(a, U.bootstrap(rows, np.arange(12), 5., [], reps=20, seed=17))

    def test_native_duplicate_test_rows_are_preserved_and_conflicts_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); image = root/'image.png'; Image.new('RGB', (8, 8)).save(image)
            other = root/'other.png'; Image.new('RGB', (8, 8)).save(other)
            path = root/'test.csv'
            def save(rows):
                with path.open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['image', 'label', 'gender_race']); w.writeheader(); w.writerows(rows)
            rows = [{'image': str(image), 'label': '0', 'gender_race': 'g'},
                    {'image': str(image), 'label': '0', 'gender_race': 'g'},
                    {'image': str(other), 'label': '1', 'gender_race': 'g'}]
            save(rows)
            with self.assertRaises(AssertionError): P.audit_manifest(path)
            result = P.audit_manifest(path, allow_native_duplicates=True)
            self.assertEqual((result['rows'], result['unique_images'], result['native_exact_duplicate_extra_rows']), (3, 2, 1))
            rows[1]['label'] = '1'; save(rows)
            with self.assertRaises(AssertionError): P.audit_manifest(path, allow_native_duplicates=True)

    def test_native_adapter_keeps_order_and_training_reader_remains_strict(self):
        W = module('worker'); M = W.model_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ['a.png', 'b.png']: Image.new('RGB', (8, 8)).save(root/name)
            rows = [{'image': str(root/'a.png'), 'label': '0', 'identity_ids': 'a', 'source_ids': 'a'},
                    {'image': str(root/'a.png'), 'label': '0', 'identity_ids': 'a', 'source_ids': 'a'},
                    {'image': str(root/'b.png'), 'label': '1', 'identity_ids': 'b', 'source_ids': 'b'}]
            path = root/'native.csv'
            with path.open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            audit = P.audit_manifest(path, allow_native_duplicates=True)
            with self.assertRaises(ValueError): M.read_rows(path)
            result = W.read_native_rows(M, path, audit)
            self.assertEqual([r['image'] for r in result], [r['image'] for r in rows])
            self.assertEqual([r['label'] for r in result], [0, 0, 1])


if __name__ == '__main__':
    unittest.main()
