"""Allocation/UUID/eligibility regressions; standard library, no GPU work."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('allocation_guard', ROOT / 'allocated_mr.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Guards(unittest.TestCase):
    def test_require_running_gpu_allocation(self):
        for record in [{}, {'job_state': 'Q'}, {'job_state': 'R', 'Resource_List': {'ngpus': 0}}]:
            with self.assertRaises(ValueError):
                m.validate_allocation(record, 'gpu01')
        m.validate_allocation({'job_state': 'R', 'Resource_List': {'ngpus': 1, 'nodect': 1}, 'exec_host': 'gpu01/2*8'}, 'gpu01')

    def test_wrong_node_rejected(self):
        with self.assertRaises(ValueError):
            m.validate_allocation({'job_state': 'R', 'Resource_List': {'ngpus': 1}, 'exec_host': 'gpu02/0*8'}, 'gpu01')

    def test_busy_or_protected_card_rejected(self):
        row = ['3', 'GPU-6a002609-de85-4949-c707-ee9317b284f0', '81559', '70000', '0']
        self.assertFalse(m.eligible(row, {row[1]}, set()))
        self.assertFalse(m.eligible(row, set(), {row[1]}))
        self.assertTrue(m.eligible(row, set(), set()))

    def test_headroom_and_site_mapping(self):
        row = ['2', 'GPU-2544ad3e-a8af-7b58-55e6-c24c57497b01', '81559', '70000', '0']
        self.assertFalse(m.eligible(row, set(), set(), {'3'}))
        self.assertTrue(m.eligible(row, set(), set(), {'2'}))
        self.assertFalse(m.eligible([*row[:3], '20000', '0'], set(), set()))
        self.assertFalse(m.eligible([*row[:4], '90'], set(), set()))

    def test_strict_cuda_uuid_equality(self):
        value = '2544ad3e-a8af-7b58-55e6-c24c57497b01'
        self.assertEqual(m.normalized_uuid('GPU-' + value), m.normalized_uuid(value))
        self.assertNotEqual(m.normalized_uuid(value), m.normalized_uuid('6a002609-de85-4949-c707-ee9317b284f0'))
        with self.assertRaises(ValueError):
            m.normalized_uuid('invalid')


if __name__ == '__main__':
    unittest.main(verbosity=2)
