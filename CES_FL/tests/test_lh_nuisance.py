import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

SRC = Path(__file__).resolve().parents[1] / 'src'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


new = module('nuisance_new', SRC/'ces_fl_nuisance_resnet50.py')
old = module('nuisance_ref', SRC/'ces_fl_head_ablation_resnet50.py')
torch.set_num_threads(2)


class NuisanceTests(unittest.TestCase):
    def test_native_model_loss_and_selection_math_unchanged(self):
        def definitions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                    if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        a, b = definitions(Path(old.__file__)), definitions(Path(new.__file__))
        for key in ['Detector','load_initial','normalize','intervention','hierarchical_real_error',
                    'response','sensitivity','response_alignment','train_step','build_optimizer',
                    'calibrate','metric_report','original_weights','predict']:
            self.assertEqual(a[key], b[key], key)

    def test_matching_cancels_zero_intervention_for_every_transform(self):
        torch.manual_seed(17); x = torch.rand(3, 75, 75)
        for op in new.OPERATORS:
            s, t, _, _ = new.intervention(x, op, 0., 17)
            for case in new.NUISANCE_BANK:
                _, ns, nt = new.common_nuisance(x, s, t, case)
                torch.testing.assert_close(ns, nt, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(ns).all())
                self.assertGreaterEqual(float(ns.min()), 0)
                self.assertLessEqual(float(ns.max()), 1)

    def test_case_is_independent_of_global_rng_and_metadata(self):
        c = new.defaults(); c['nuisance_mode'] = 'matched'
        expected = [new.nuisance_case(c, e, 'main', 'same/source') for e in range(4)]
        self.assertEqual(set(expected), set(new.NUISANCE_BANK))
        torch.manual_seed(987); np.random.seed(789)
        self.assertEqual(expected, [new.nuisance_case(c, e, 'main', 'same/source') for e in range(4)])
        self.assertEqual(new.nuisance_case({**c, 'nuisance_mode': 'none'}, 0, 'any'), 'identity')

    def test_originals_control_parity_eval_is_unaugmented_and_labels_cannot_change_x(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'im.png'; rng = np.random.default_rng(19)
            Image.fromarray(rng.integers(0, 256, (80, 83, 3), dtype=np.uint8)).save(p)
            rows = [{'image': str(p), 'label': 0}]
            c = new.defaults(); c['size'] = 75
            for epoch in range(4):
                baseline = old.Originals(rows, c, epoch, True)[0]
                control = new.Originals(rows, c, epoch, True, training=True)[0]
                for key in ('x','s','t'): torch.testing.assert_close(control[key], baseline[key], rtol=0, atol=0)
            c['nuisance_mode'] = 'matched'
            epoch = next(e for e in range(4) if new.nuisance_case(c, e, 'main', str(p)) == 'jpeg90')
            train = new.Originals(rows, c, epoch, True, training=True)[0]
            fake = new.Originals([{**rows[0], 'label': 1}], c, epoch, False, training=True)[0]
            torch.testing.assert_close(train['x'], fake['x'], rtol=0, atol=0)
            self.assertEqual(train['nuisance_index'], fake['nuisance_index'])
            evaluation = new.Originals(rows, c, epoch, False, training=False)[0]
            raw = old.normalize(old.load_pixels(rows[0], 75))
            torch.testing.assert_close(evaluation['x'], raw, rtol=0, atol=0)
            self.assertFalse(torch.equal(train['x'], evaluation['x']))

    def test_auxiliary_has_one_matching_transform_for_every_group_and_class(self):
        with tempfile.TemporaryDirectory() as d:
            rows = []
            for i in range(8):
                p = Path(d)/f'{i}.png'
                Image.fromarray(np.random.default_rng(i).integers(0,256,(75,75,3),dtype=np.uint8)).save(p)
                g, r = ('A' if i//2<2 else 'B'), ('X' if (i//2)%2==0 else 'Y')
                rows.append({'image':str(p),'label':i%2,'gender':g,'race':r,'gender_race':g+'-'+r,
                             'identity_ids':str(i),'source_ids':str(i)})
            c = {**new.defaults(), 'axes':['gender_race'], 'size':75, 'group_per_class':1,
                 'min_train_support':1,'nuisance_mode':'matched'}
            active, _ = new.group_inventory(rows, c)
            cases_seen = set()
            for epoch in range(4):
                old_aux = old.auxiliary(rows, active, c, 0, epoch)
                new_aux = new.auxiliary(rows, active, c, 0, epoch)
                case = new_aux['nuisance_case']; cases_seen.add(case)
                expected = {'x': [], 's': [], 't': []}
                for ix in old_aux['indices']:
                    raw = old.load_pixels(rows[ix], c['size'])
                    expected['x'].append(new.normalize(new.nuisance_view(raw, case)))
                    if rows[ix]['label'] == 0:
                        s,t,_,_=old.intervention(raw,old_aux['operator'],old_aux['strength'],
                            old.seed_for(c['seed'],epoch,0,'matched-nuisance'))
                        expected['s'].append(new.normalize(new.nuisance_view(s,case)))
                        expected['t'].append(new.normalize(new.nuisance_view(t,case)))
                for key in ('x','s','t'):
                    torch.testing.assert_close(new_aux[key],torch.stack(expected[key]),rtol=0,atol=0)
                self.assertEqual(old_aux['indices'],new_aux['indices'])
                torch.testing.assert_close(old_aux['y'],new_aux['y'])
            self.assertEqual(cases_seen,set(new.NUISANCE_BANK))

    def test_loss_shortcuts_are_explicit_not_scientific_success(self):
        a = new.sensitivity(new.response(torch.tensor([-1.]),torch.tensor([-.9])),.5)
        b = new.sensitivity(new.response(torch.tensor([-10.]),torch.tensor([-9.])),.5)
        self.assertGreater(float(a), .39); self.assertEqual(float(b), 0.)
        s = torch.tensor([-2., 2.]); r = new.response(s, s+.6)
        self.assertLess(float(new.response_alignment(r,torch.tensor([0,1]))),1e-6)
        self.assertNotEqual(int(s[0]>=0),int(s[1]>=0))


if __name__ == '__main__': unittest.main()
