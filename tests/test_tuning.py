import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import numpy as np
import torch
from PIL import Image

SOURCE=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('tuning_test',SOURCE/'schedule_tuning.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class TuningTests(unittest.TestCase):
    def config(self,ceiling=.1):
        return dict(suite_method='lh',response_schedule=dict(delay_epochs=2,ramp_epochs=6,ceiling=ceiling,shape='smoothstep'),
            lambda_sensitivity=.1,lambda_fair=.1,lambda_alignment=.1,sham_weight=.5,aux_every=4)

    def test_delay_ramp_and_plateau_are_fixed(self):
        for cap in (0.,.05,.1,.2):
            values=[m.response_factor(self.config(cap),e) for e in range(10)]
            self.assertEqual(values[:2],[0.,0.]);self.assertEqual(values[7:],[cap]*3)
            self.assertTrue(all(a<=b for a,b in zip(values,values[1:])))
            self.assertAlmostEqual(values[2],cap*2/27)
            self.assertAlmostEqual(values[4],cap*.5)

    def test_control_response_gradient_is_exactly_zero(self):
        x=torch.tensor(.7,requires_grad=True);loss=m.response_factor(self.config(0),9)*x.square()
        self.assertEqual(float(torch.autograd.grad(loss,x)[0]),0.)

    def test_lower_ceiling_scales_response_gradient(self):
        grads=[]
        for cap in (.05,.1,.2):
            x=torch.tensor(.7,requires_grad=True);loss=m.response_factor(self.config(cap),9)*x.square()
            grads.append(float(torch.autograd.grad(loss,x)[0]))
        np.testing.assert_allclose(grads,np.array([1.,2.,4.])*grads[0],rtol=1e-6)

    def test_full_loss_value_and_gradient_match_registered_formula(self):
        c=self.config();z=torch.tensor([-.6,.3,-.2,.8],requires_grad=True);y=torch.tensor([0.,1.,0.,1.]);g=torch.tensor([0,0,1,1])
        fair=z.square().mean();sens=torch.sigmoid(z).mean();sham=z.abs().mean();align=(z-z.mean()).abs().mean();zero=z.sum()*0
        with patch.object(m.base,'response_factor',m.response_factor):
            out=m.objective(c,z,y,g,fair,sens,sham,align,zero,True,7)
        expected=torch.nn.functional.binary_cross_entropy_with_logits(z,y)+.4*fair+.1*(.1*sens+.05*sham+.1*align)
        torch.testing.assert_close(out['total'],expected,rtol=0,atol=0)
        a=torch.autograd.grad(out['total'],z,retain_graph=True)[0];b=torch.autograd.grad(expected,z)[0]
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_component_logs_sum_to_total(self):
        c=self.config();z=torch.tensor([-.2,.8],requires_grad=True);y=torch.tensor([0.,1.]);g=torch.tensor([0,1]);x=z.square().mean()
        with patch.object(m.base,'response_factor',m.response_factor):out=m.objective(c,z,y,g,x,x,x,x,x,True,4)
        torch.testing.assert_close(out['total'],out['classification_objective']+out['weighted_fairness']+out['weighted_response'],rtol=0,atol=0)

    def test_no_response_phase_matches_fairness_only(self):
        z=torch.randn(8,requires_grad=True);y=torch.tensor([0.,1.]*4);g=torch.arange(8);x=z.square().mean()
        with patch.object(m.base,'response_factor',m.response_factor):
            losses=[m.objective(self.config(cap),z,y,g,x,x,x,x,x,True,1)['total'] for cap in (0.,.05,.1,.2)]
        self.assertTrue(all(torch.equal(losses[0],x) for x in losses[1:]))

    def test_original_recipe_and_parent_source_are_unmodified(self):
        self.assertEqual(m.core.sha(SOURCE/'src/core.py'),'8c25ba8cb844b059adbbce75b57785a156333182afa2d5d576985997f2d36733')
        self.assertEqual(m.core.sha(SOURCE/'src/shared.py'),'ff8b569ac606c9eefb736259c5a4524d10c504a908eb86f2b944c6493c78fd3a')

    def test_no_TEST_or_CAL_CLI_stage(self):
        text=(SOURCE/'schedule_tuning.py').read_text()
        self.assertIn("choices=('dry','train')",text)
        self.assertNotIn('base.evaluate(',text)
        self.assertNotIn('core.evaluate(',text)

    def test_native_training_records_delayed_response_and_exact_resume(self):
        class Tiny(torch.nn.Module):
            def __init__(self,*args):
                super().__init__();self.encoder=torch.nn.Conv2d(3,3,1);self.classifier=torch.nn.Linear(3,1)
            def forward(self,x):return {'logit':self.classifier(self.encoder(x).mean((2,3))).squeeze(1)}
            def counts(self):return dict(total=sum(p.numel() for p in self.parameters()),trainable=sum(p.numel() for p in self.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data={}
            for split in ('train','val'):
                rr=[]
                for i in range(8):
                    p=root/f'{split}_{i}.png';Image.fromarray(np.random.default_rng(i+(100 if split=='val' else 0)).integers(0,256,(75,75,3),dtype=np.uint8)).save(p)
                    gender='A' if i<4 else 'B';race='X' if i//2%2==0 else 'Y'
                    rr.append(dict(image=str(p),label=i%2,identity_ids=split+str(i),source_ids=split+str(i),family='fixture',
                        gender=gender,race=race,gender_race=gender+'-'+race,age_bin=str(i//4),skin_bin=str(i//4),ids8=str(i//2),image_sha256=m.core.sha(p)))
                data[split]=rr
            c=m.core.defaults();c.update(self.config());c.update(seed=17,epochs=3,batch_size=4,workers=0,size=75,amp=False,
                axes=['gender_race'],group_names=['A-X','A-Y','B-X','B-Y'],min_train_support=1,min_eval_support=1,
                group_per_class=1,aux_every=1,train_calibration_size=4,error_control='hierarchical_real',
                synthetic_fixture=True,max_gpu_hours=1.,source_sha256=m.core.sha(m.core.__file__))
            def initial(config):m.core.seed_all(config['seed']);return Tiny()
            m.configure()
            with patch.object(m.core,'Detector',Tiny),patch.object(m.core,'load_initial',side_effect=initial),\
                    patch.object(m.core,'checked_inputs',return_value=(data['train'],data['val'],dict(status='PASS'))):
                m.core.train(c,'fixture',root/'normal','D',torch.device('cpu'),fixture=True)
                m.core.train(c,'fixture',root/'resumed','D',torch.device('cpu'),fixture=True,stop_after=2)
                m.core.train(c,'fixture',root/'resumed','D',torch.device('cpu'),fixture=True,resume=True)
            a=torch.load(root/'normal/last.pt',weights_only=True);b=torch.load(root/'resumed/last.pt',weights_only=True)
            self.assertTrue(all(torch.equal(v,b['model'][k]) for k,v in a['model'].items()))
            history=a['history'];self.assertEqual([x['losses_step_mean']['weighted_response'] for x in history[:2]],[0.,0.])
            self.assertGreater(history[2]['losses_step_mean']['weighted_response'],0)
            self.assertAlmostEqual(history[2]['losses_step_mean']['response_multiplier'],m.response_factor(c,2),places=8)
            self.assertTrue(all(x['originals_once_each'] for x in history));self.assertEqual(a['validation'],b['validation'])


if __name__=='__main__':torch.set_num_threads(2);unittest.main(verbosity=2)
