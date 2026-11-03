"""One fixed-VAL analysis after native H completion; never trains or reads TEST."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def read(p):return json.loads(Path(p).read_text())
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser()
    for name in ('source','run','out','plotter','exporter'):p.add_argument('--'+name,required=True)
    a=p.parse_args();root=Path(a.run);run=root/'H_joint_real/training';out=Path(a.out)
    source=Path(a.source)/'CES_FL/src/ces_fl_forensic_response.py'
    m=module(source,'final_hier_cesfl')
    assert m.sha(source)=='2549cb1345d1e2ee617e8a77ca288a0acbbfd05cf480f7b91a46b8627fcb74b2'
    c=m.load_config(root/'H_joint_real/config.json');r=read(run/'RUN_VERIFIED.json');scores=read(run/'validation_scores_PRIVATE.json')
    assert m.sha(root/'H_joint_real/config.json')=='63baed480e251945dba5ebb360e9ac1c7ea9fbc1cebf02acfef6abd6299fb3fd'
    assert (run/'DONE').exists() and r['epochs_completed']==10 and r['status']=='COMPLETED_VERIFIED'
    assert r['source_sha256']==m.sha(source) and r['config_digest']==m.digest(c) and not r['test_evaluated']
    assert m.sha(run/'best.pt')==r['selected_checkpoint_sha256']==scores['checkpoint_sha256']
    assert scores['split']=='validation' and scores['test_read'] is False
    assert m.metric_report(scores['rows'],scores['logits'],scores['threshold'],scores['min_support'])==r['validation']
    oldroot=Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_overlapfix_20261103_r1_ffpp_s17/D')
    reference_path=Path('/home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/diagnostics/D_validation_scores_PRIVATE.json')
    d=read(oldroot/'RUN_VERIFIED.json');ds=read(reference_path)
    assert m.sha(reference_path)=='5be661e012c81759cb7eeaa36df565e6810db96c1218c5ae67ac52c9e02b227e'
    assert m.sha(oldroot/'best.pt')=='7d9409e2cb6aa00fdd61cd3a072bd5473848be6dd86aafcc3b7b740da9f25d9d'
    assert m.sha(oldroot/'best.pt')==ds['checkpoint_sha256']==d['selected_checkpoint_sha256']
    assert ds['threshold']==-4.397897720336913 and ds['split']=='validation' and ds['test_read'] is False
    assert m.metric_report(ds['rows'],ds['logits'],ds['threshold'],ds['min_support'])==d['validation']
    assert scores['rows']==ds['rows'] and scores['manifest_sha256']==ds['manifest_sha256']==c['val_sha256']
    protocol=read(root/'decision_protocol.json');out.mkdir(exist_ok=False)
    print(json.dumps(dict(event='PAIRED_BOOTSTRAP_STARTED',replicates=1000,seed=17,test_read=False)),flush=True)
    m.compare(run/'validation_scores_PRIVATE.json',reference_path,out/'H_vs_D.json',1000,17)
    paired=read(out/'H_vs_D.json');v=r['validation'];axes=v['axes']
    guards=dict(gender_20pct=axes['gender']['F_FPR']<=protocol['gender_F_FPR_max_inclusive'],
                retained_gender=axes['gender']['F_FPR']<=protocol['retained_E_F_gender_guard'],
                intersection=axes['gender_race']['F_FPR']<protocol['primary_F_FPR_max_exclusive'],
                auc=v['overall']['auc']>=protocol['auc_min_inclusive'],
                age=axes['age_bin']['F_FPR']<=protocol['age_F_FPR_max_inclusive'],
                pooled_fake_FNR=v['overall']['fnr']<=protocol['pooled_fake_FNR_max_inclusive'],
                **{'worst_'+axis:axes[axis]['worst_group_auc']>=value for axis,value in protocol['worst_group_auc_min_by_axis'].items()})
    interval=paired['intervals']['gender_race/F_FPR']
    usable=interval['valid_replicates']>=1000*protocol['bootstrap']['required_usable_primary_replicate_fraction']
    negative=interval['ci95'] is not None and interval['ci95'][1]<0
    decision='FAIL' if not all(guards.values()) else 'PASS' if usable and negative else 'INCONCLUSIVE'
    controls={'D':d['validation'],'H':v}
    for name in ['E_support','F_support_ramp3']:
        controls[name]=read(Path('/home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1')/name/'training/RUN_VERIFIED.json')['validation']
    plots=module(a.plotter,'hier_validation_plots');plots.render(controls,out/'heatmaps')
    subprocess.run([sys.executable,a.exporter,'--scores',str(run/'validation_scores_PRIVATE.json'),
                    '--checkpoint',str(run/'best.pt'),'--source',str(source),
                    '--out',str(root/'H_failure_audit'),'--kind','candidate'],check=True)
    ledger=read('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/budget_ledger.json')
    sessions=[s for s in ledger['gpu_sessions'] if s['id'].startswith('hier_')]
    assert all(s['state']=='COMPLETED' for s in sessions)
    gpu_hours=sum(s['elapsed_seconds']*s['ngpus']/3600 for s in sessions)
    assert gpu_hours<=4 and ledger['consumed_gpu_hours']<=24
    payload=dict(status='COMPLETED_VALIDATION_ANALYSIS',decision=decision,
                 observed_management_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 scientific_commit='3f5f3c16839bf4b96209bea3f968255b5f351ebb',source_sha256=m.sha(source),
                 config_file_sha256=m.sha(root/'H_joint_real/config.json'),config_digest=m.digest(c),
                 selected_checkpoint_sha256=r['selected_checkpoint_sha256'],selected_epoch=r['selected_epoch'],
                 D_checkpoint_sha256=d['selected_checkpoint_sha256'],D_threshold=ds['threshold'],H_threshold=scores['threshold'],
                 numerical_guards=guards,uncertainty_usable=usable,primary_ci_entirely_below_zero=negative,
                 primary_interval=interval,gender_interval=paired['intervals']['gender/F_FPR'],
                 D=d['validation'],H=v,fixed_probability_0_5=m.metric_report(scores['rows'],scores['logits'],0.,scores['min_support']),
                 epochs_completed=10,new_gpu_hours=gpu_hours,total_gpu_hours=ledger['consumed_gpu_hours'],
                 remaining_gpu_hours=24-ledger['consumed_gpu_hours'],gpu_sessions=sessions,
                 private_failure_root=str(root/'H_failure_audit'),test_read=False,training_seed_variability_measured=False,
                 automatic_retry_or_promotion=False,interpretation='One seed, fixed selected validation predictions; not independent TEST confirmation or a win over Ding',
                 limitations=['Inherited group labels and age/skin proxies are not human-verified truth',
                              'Recorded contributor components are not proven biological identities',
                              'Soft-mean risk can be equal despite unequal hard FPR tails',
                              'Existing >=90% usable primary rule is retained despite sparse-group support'])
    lines=['# H joint original-real risk: '+decision,'',payload['observed_management_utc']+' management UTC. FF++ validation, seed17 only; no TEST or new Ding computation.','',
           'Only the intersection-auxiliary real error summand changes. All fake error control, other-axis real penalties, sensitivity/alignment/control losses and training settings remain D. No E/F discount or ramp.','',
           '|Arm|Epochs|Selected epoch|AUC %|Intersection F_FPR fraction|Gender F_FPR fraction|Age F_FPR fraction|Achieved pooled FPR %|Pooled fake FNR %|',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,rr in [('D',d),('H',r)]:
        vv=rr['validation'];o=vv['overall'];aa=vv['axes'];lines.append(f'|{name}|10|{rr["selected_epoch"]}|{100*o["auc"]:.6f}|{aa["gender_race"]["F_FPR"]:.6f}|{aa["gender"]["F_FPR"]:.6f}|{aa["age_bin"]["F_FPR"]:.6f}|{100*o["fpr"]:.6f}|{100*o["fnr"]:.6f}|')
    lines+=['','Numerical guards: `'+json.dumps(guards,sort_keys=True)+'`. The20%gender ceiling is0.042798380714166834. All earlier safeguards remain. No different checkpoint is chosen for fairness.','',
            f'Primary H−D95%paired percentile interval: {interval["ci95"]}; usable{interval["valid_replicates"]}/1000, required900. Gender H−D interval: {payload["gender_interval"]["ci95"]}.',
            'Intervals resample recorded contributor components with fixed thresholds/eligible groups. They measure conditional dataset sampling uncertainty, not seed variability. Sparse/missing strata are not imputed or counted as independent frames. See H_vs_D.json for every group interval.','']
    for axis in ['gender','race','gender_race','age_bin','skin_bin']:
        lines+=['## '+axis,'','|Group|Real n|Fake n|D FP / FPR %|H FP / FPR %|D FN / FNR %|H FN / FNR %|D AUC %|H AUC %|','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for group,x in d['validation']['axes'][axis]['groups'].items():
            y=v['axes'][axis]['groups'][group];assert(x['n_real'],x['n_fake'])==(y['n_real'],y['n_fake'])
            vals=[f'{int(round(t[rate]*t[n]))} / {100*t[rate]:.4f}' if t[rate] is not None else 'undefined' for rate,n,t in [('fpr','n_real',x),('fpr','n_real',y),('fnr','n_fake',x),('fnr','n_fake',y)]]
            aucs=[f'{100*t["auc"]:.5f}' if t['auc'] is not None else 'undefined' for t in [x,y]]
            lines.append(f'|{group}|{x["n_real"]}|{x["n_fake"]}|'+'|'.join(vals+aucs)+'|')
        lines+=['']
    lines+=['## Provenance and compute','',f'Source `{payload["scientific_commit"]}`, SHA `{payload["source_sha256"]}`; config file SHA `{payload["config_file_sha256"]}`; semantic digest `{payload["config_digest"]}`; selected H checkpoint SHA `{payload["selected_checkpoint_sha256"]}`.',
            f'New GPU task-held time{gpu_hours:.6f}h; cumulative{ledger["consumed_gpu_hours"]:.6f}/24h. No budget reset/extension, automatic retry, tuning or TEST evaluation.',
            '', 'All supported-axis regressions, worst-group/FPR/FNR gaps, fake-family reports and fixed0.5 diagnostics are in DECISIONS.json. Heatmaps are aggregate rates; private contacts remain on cluster. Counterfactual/causal forensic validity is not established by image plausibility or software tests.','',
            'Decision applies only to the frozen validation criteria. A validation improvement is not evidence of having beaten Ding.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Completion is published last, after report/heatmaps/private diagnostics.
    m.write_json(out/'DECISIONS.json',payload)
    print(json.dumps(dict(status='COMPLETED_VALIDATION_ANALYSIS',decision=decision,numerical_guards=guards,new_gpu_hours=gpu_hours,test_read=False)),flush=True)


if __name__=='__main__':main()
