import unittest
import numpy as np
from sklearn.metrics import roc_auc_score
from audit import source
from audit_stats import AUC,connected_components,run_statistics


class DiagnosticChecks(unittest.TestCase):
    def test_weighted_auc_ties_matches_explicit_resampling(self):
        y=np.array([0,1,0,1,0,1]);z=np.array([0.,0.,1.,1.,2.,2.]);comp=np.array([0,1,2,3,4,5])
        w=np.array([2,1,3,2,0,4]);plan=AUC(y,z,comp,np.arange(6))
        ix=np.repeat(np.arange(6),w)
        self.assertAlmostEqual(plan(w),roc_auc_score(y[ix],z[ix]),places=14)

    def test_component_union_uses_every_master_and_donor_link(self):
        rows=[{'image':'a','identity_ids':'person:a;person:donor','source_ids':'master:a'},
              {'image':'b','identity_ids':'person:b;person:donor','source_ids':'master:b'},
              {'image':'c','identity_ids':'person:c','source_ids':'master:b'},
              {'image':'d','identity_ids':'person:d','source_ids':'master:d'}]
        c=connected_components(rows)
        self.assertEqual(c[0],c[1]);self.assertEqual(c[1],c[2]);self.assertNotEqual(c[2],c[3])

    def test_source_mapping_is_documented_not_a_first_token_guess(self):
        root='/home/sachin.chaudhary/ICLR_2027/fairforge_scaleup/real/'
        self.assertEqual(source({'label':0,'source_ids':'ffhq:00001','image':root+'A_ffhq/cell/00001.jpg'}),'FFHQ')
        self.assertEqual(source({'label':0,'source_ids':'img:images/real/C_fairface/cell/x.jpg','image':root+'C_fairface/cell/x.jpg'}),'FairFace')
        self.assertEqual(source({'label':0,'source_ids':'img:images/real/B_rfw/cell/x.jpg','image':root+'B_rfw/cell/x.jpg'}),'UNRESOLVED')
        self.assertEqual(source({'label':0,'source_ids':'ffhq:00001','image':root+'A_ffhq/cell/00002.jpg'}),'UNRESOLVED')

    def fixture(self,missing=False):
        rows=[];src=[];z=[]
        for group,counts in [('Female-A',(90,10)),('Male-A',(10,90))]:
            for s,nreal,rate in [('FFHQ',counts[0],.1),('FairFace',counts[1],.5)]:
                if missing and group=='Male-A' and s=='FFHQ':continue
                for lab,n in [(0,nreal),(1,10)]:
                    for k in range(n):
                        i=len(rows);rows.append({'image':str(i),'label':lab,'gender_race':group,'gender':group.split('-')[0],
                            'race':'A','age_bin':'adult','skin_bin':'known','family':'real' if lab==0 else 'fake',
                            'identity_ids':'id:'+str(i),'source_ids':'master:'+str(i)})
                        src.append(s);z.append(1. if lab==1 or k<int(n*rate) else -1.)
        freeze={'groups':{'gender_race':['Female-A','Male-A']},'sources':['FFHQ','FairFace'],
                'source_weights':{'FFHQ':.5,'FairFace':.5},'bootstrap':{'replicates':20,'seed':17,'required_valid_fraction':.95}}
        names=['RF-S_s17','RF-S_s7','RF-S_s27','RF-SJg_s17','RF-SJg_s7','RF-SJg_s27']
        with np.errstate(invalid='ignore',divide='ignore'):
            r=run_statistics(rows,np.array(src),np.arange(len(rows)),np.tile(z,(6,1)),np.zeros(6),names,freeze)
        return r

    def test_known_source_mixture_contrast_and_all_pairs(self):
        r=self.fixture();k=r['keys'].index('native_minus_standardized/gender_race')
        self.assertAlmostEqual(r['point'][0,k],.32,places=13)
        self.assertAlmostEqual(r['point'][0,r['keys'].index('D_standardized/gender_race')],0.,places=13)
        for seed in [17,7,27]:
            p=next(x for x in r['intervals'] if x['model']=='SJg-minus-S_s'+str(seed) and x['metric']=='D_native/gender_race')
            self.assertEqual(p['estimate'],0.)
            self.assertEqual(p['ci95_low'],0.);self.assertEqual(p['ci95_high'],0.)

    def test_missing_required_cell_blocks_without_renormalization(self):
        r=self.fixture(True)
        self.assertTrue(np.isnan(r['point'][0,r['keys'].index('D_standardized/gender_race')]))
        self.assertTrue(np.isfinite(r['point'][0,r['keys'].index('D_native/gender_race')]))
        self.assertEqual(r['primary']['status'],'BLOCKED_UNDEFINED_ESTIMAND')
        self.assertIsNone(r['primary']['ci95_low'])
        self.assertEqual(r['coverage']['gender_race']['missing_real_cells'][0]['group'],'Male-A')


if __name__=='__main__':unittest.main(verbosity=2)
