"""Unit checks for the FF2 real-FPR stretch trainer (no data, no GPU)."""
import importlib.util, math, unittest
from pathlib import Path
import torch
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / 'src'
spec = importlib.util.spec_from_file_location('rf', SRC / 'ces_fl_realfpr.py'); rf = importlib.util.module_from_spec(spec); spec.loader.exec_module(rf)


class RealFPRTests(unittest.TestCase):
    def test_jpeg_plan_mix_and_determinism(self):
        plans = [rf.jpeg_plan(17, 3, i) for i in range(20000)]
        self.assertEqual(plans, [rf.jpeg_plan(17, 3, i) for i in range(20000)])
        none = sum(p is None for p in plans) / len(plans); q75 = sum(p is not None and p[0] == 75 and p[1] == 2 for p in plans) / len(plans)
        self.assertAlmostEqual(none, .5, delta=.015); self.assertAlmostEqual(q75, .25 + .25 / 52, delta=.015)  # 25% fixed + 1/52 of the random branch
        self.assertTrue(all(p is None or (70 <= p[0] <= 95 and p[1] in (0, 2)) for p in plans))
        self.assertNotEqual(plans, [rf.jpeg_plan(17, 4, i) for i in range(20000)])

    def test_jpeg_plan_custom_probabilities(self):
        plans = [rf.jpeg_plan(1, 0, i, p_any=.2, p_q75=1.0) for i in range(20000)]
        none = sum(p is None for p in plans) / len(plans)
        self.assertAlmostEqual(none, .8, delta=.015)
        self.assertTrue(all(p is None or (p[0] == 75 and p[1] == 2) for p in plans))  # p_q75=1 -> only exact q75/420

    def test_jpeg_ramp(self):
        c = {'jpeg_p_any': .5, 'jpeg_ramp_epochs': 2}
        self.assertAlmostEqual(rf.jpeg_p_any_effective(c, 0), .25)
        self.assertAlmostEqual(rf.jpeg_p_any_effective(c, 1), .5)
        self.assertAlmostEqual(rf.jpeg_p_any_effective(c, 5), .5)
        c0 = {'jpeg_p_any': .3, 'jpeg_ramp_epochs': 0}
        self.assertAlmostEqual(rf.jpeg_p_any_effective(c0, 0), .3)  # no ramp: immediate full strength

    def test_jpeg_emulate_shape_and_change(self):
        im = Image.effect_noise((256, 256), 64).convert('RGB')
        out = rf.jpeg_emulate(im, 75, 2, 1.158)
        self.assertEqual(out.size, (256, 256)); self.assertNotEqual(list(out.getdata())[:50], list(im.getdata())[:50])

    def test_real_fpr_loss_pulls_high_group_down(self):
        z = torch.tensor([3., 3., 3., 3., -3., -3., -3., -3.], requires_grad=True); g = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loss, rates = rf.real_fpr_loss(z, g, torch.tensor([.5, .5]), 0., 1.)
        loss.backward()
        self.assertGreater(float(loss), 0); self.assertGreater(float(rates[0]), float(rates[1]))
        self.assertTrue((z.grad[:4] > 0).all() and (z.grad[4:] < 0).all())  # descent lowers group 0, raises group 1

    def test_equal_groups_zero_loss(self):
        z = torch.zeros(8); g = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        loss, _ = rf.real_fpr_loss(z, g, torch.full((4,), .25), 0., .5); self.assertAlmostEqual(float(loss), 0., places=7)

    def test_selection_rule(self):
        c = {'min_auc': .995, 'max_fnr': .005651}
        rep = lambda auc, fnr, f: {'overall': {'auc': auc, 'fnr': fnr}, 'axes': {'gender_race': {'F_FPR': f}}}
        ok, k1 = rf.feasible_key(rep(.998, .003, .30), c); self.assertTrue(ok)
        ok2, k2 = rf.feasible_key(rep(.998, .003, .20), c); self.assertLess(k2, k1)
        self.assertFalse(rf.feasible_key(rep(.990, .003, .05), c)[0]); self.assertFalse(rf.feasible_key(rep(.998, .01, .05), c)[0])
        _, k3 = rf.feasible_key(rep(.999, .003, .20), c); self.assertLess(k3, k2)  # tie on F_FPR broken by AUC

    def test_schedule(self):
        c = {'lambda_rf': .1, 'rf_ramp_epochs': 2, 'epochs': 10, 'tau_start': 1., 'tau_end': .25}
        self.assertEqual(rf.schedule(c, 0)[0], 0.); self.assertAlmostEqual(rf.schedule(c, 1)[0], .05); self.assertAlmostEqual(rf.schedule(c, 5)[0], .1)
        self.assertAlmostEqual(rf.schedule(c, 0)[1], 1.); self.assertAlmostEqual(rf.schedule(c, 9)[1], .25)


if __name__ == '__main__': unittest.main()
