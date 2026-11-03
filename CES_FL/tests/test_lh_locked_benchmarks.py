import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

PATH=Path(__file__).resolve().parents[1]/'scripts/lh_locked_benchmarks.py'
spec=importlib.util.spec_from_file_location('locked_lh',PATH)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def cache():
    rows=[dict(label=i%2,gender='g'+str(i//2%2),race='r'+str(i//4%2),gender_race='j'+str(i//2%4),
               age_bin='a'+str(i//2%2),skin_bin='s'+str(i//2%2),ids8='j'+str(i//2%4),family='fixture') for i in range(16)]
    return dict(rows=rows,logits=[(i%2)*.4+i*.03 for i in range(16)],threshold=.4,min_support=1,
                checkpoint_sha256='fixture-only',split='validation',test_read=False,manifest_sha256='fixture')


class LockedBenchmarkTests(unittest.TestCase):
    def test_threshold_scan_does_not_change_auc_or_choose_threshold(self):
        s=cache();before=copy.deepcopy(s);r=m.threshold_scan(s)
        self.assertEqual(s,before);self.assertFalse(r['selected_new_threshold'])
        self.assertEqual([x['target_validation_fpr'] for x in r['results']],[.01,.02,.05,.1])
        auc=[x['validation']['overall']['auc'] for x in r['results']]
        self.assertEqual(len(set(auc)),1)
        for axis in ('gender','race','gender_race'):
            self.assertEqual(len({x['validation']['axes'][axis]['es_AUC'] for x in r['results']}),1)

    def test_reject_test_and_unmarked_threshold_cache(self):
        for key,value in [('split','test'),('test_read',True),('split',None)]:
            s=cache();s[key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):m.threshold_scan(s)

    def test_reject_changed_plan_source(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan.json';p.write_text(json.dumps(dict(format='LH_LOCKED_TABLES12_v1',runner_sha256='wrong',source_sha256=m.SOURCE_SHA)))
            with self.assertRaises(ValueError):m.load_plan(p)

    def test_completion_selection_cache_and_tamper_guards(self):
        with tempfile.TemporaryDirectory() as d,patch.object(m.core,'check_config'),patch.object(m,'expected_config') as expected,patch.object(m.core,'read_manifest') as read:
            root=Path(d);s=cache();report=m.core.metric_report(s['rows'],s['logits'],s['threshold'],1)
            c=dict(head_type='linear',val_sha256='fixture',val='fixture.csv');expected.return_value=c;read.return_value=s['rows']
            history=[dict(epoch=i,validation=copy.deepcopy(report),calibration={'threshold_logit':.4}) for i in range(1,11)]
            # Tied maxima must keep the earliest epoch.
            ck=dict(format=m.core.VERSION,source_sha256=m.SOURCE_SHA,arm='D',synthetic_fixture=False,
                    config=c,epoch=1,validation=report,calibration={'threshold_logit':.4})
            torch.save(ck,root/'best.pt');s['checkpoint_sha256']=m.core.sha(root/'best.pt')
            (root/'config.json').write_text(json.dumps(c));(root/'history.json').write_text(json.dumps(history))
            (root/'validation_scores_PRIVATE.json').write_text(json.dumps(s))
            proof=dict(status='COMPLETED_VERIFIED',selected_checkpoint_sha256=s['checkpoint_sha256'],source_sha256=m.SOURCE_SHA)
            (root/'RUN_VERIFIED.json').write_text(json.dumps(proof));(root/'DONE').write_text('fixture')
            got,_=m.verify_selected(root,'L_H');self.assertEqual(got['epoch'],1)
            (root/'history.json').write_text(json.dumps(history[:9]))
            with self.assertRaisesRegex(ValueError,'ten'):m.verify_selected(root,'L_H')
            (root/'history.json').write_text(json.dumps(history))
            s['threshold']=9.;(root/'validation_scores_PRIVATE.json').write_text(json.dumps(s))
            with self.assertRaisesRegex(ValueError,'threshold'):m.verify_selected(root,'L_H')
            (root/'DONE').unlink()
            with self.assertRaisesRegex(ValueError,'complete'):m.verify_selected(root,'L_H')


if __name__=='__main__':unittest.main()
