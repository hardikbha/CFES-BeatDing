"""Regression for the timing helper's compute-node import environment."""
import cProfile
import importlib.util
from pathlib import Path
import unittest
import torch


class ProfileEnvironmentTests(unittest.TestCase):
    def test_stdlib_cprofile_and_optimizer_import(self):
        self.assertTrue(callable(cProfile.runctx))
        self.assertFalse((Path(__file__).parent/'profile.py').exists())
        p=torch.nn.Parameter(torch.tensor([1.]))
        opt=torch.optim.AdamW([p],lr=.1)
        opt.zero_grad();p.square().sum().backward();opt.step()
        self.assertLess(float(p.detach()),1.)

    def test_timing_module_import_is_inert(self):
        path=Path(__file__).parent/'training_schedule_profile.py'
        spec=importlib.util.spec_from_file_location('safe_schedule_timing',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        self.assertEqual(m.WARMUP,40)
        self.assertEqual(m.MEASURE,80)
        self.assertTrue(callable(m.run))


if __name__=='__main__':unittest.main(verbosity=2)
