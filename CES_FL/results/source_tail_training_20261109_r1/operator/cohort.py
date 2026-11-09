"""Run the explicitly authorized cohort inside one externally guarded GPU session."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--root',required=True);a=p.parse_args()
src=Path(a.source);root=Path(a.root);py=sys.executable;prop=src/'CES_FL/proposals/source_tail_training_20261109_r1';runtime=root/'runtime'
def stage(name,args,required=True):
    begin=time.time();record={'stage':name,'start_node_unix':begin,'pid':None,'status':'RUNNING'}
    with (root/'logs'/f'{name}.log').open('x') as f:
        child=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,cwd=src);record['pid']=child.pid
        (root/'CURRENT_STAGE.json').write_text(json.dumps(record,indent=2)+'\n');code=child.wait()
    record.update(status='COMPLETED' if code==0 else 'FAILED',exit_code=code,end_node_unix=time.time(),argv=args)
    (root/f'{name}.RECEIPT.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
    if code and required:raise SystemExit(code)
stage('qualify_profile',[py,'-u',str(runtime/'qualify_profile.py'),'--source',str(src),'--root',str(root)])
profile=json.loads((root/'REAL_TRAIN_PROFILE.json').read_text());assert profile['status']=='PASS'
stage('prepare',[py,'-u',str(prop/'run.py'),'prepare','--teacher-run','/home/sachin.chaudhary/ces_fl_runs/ff2_lh_ps_20261106_r1/L_A',
    '--source-csv',str(root/'train_real_sources_PRIVATE.csv'),'--source-provenance',str(root/'source_provenance_PRIVATE.json'),'--out',str(root/'cache'),'--device','cuda'])
for arm in ('control','treatment'):
    stage('train_'+arm,[py,'-u',str(prop/'run.py'),'train','--cache',str(root/'cache'),'--arm',arm,'--out',str(root/arm),'--device','cuda'])
for arm in ('control','treatment'):
    result=json.loads((root/arm/'RUN_VERIFIED.json').read_text())
    if result['status']=='COMPLETED_VERIFIED':
        stage('export_'+arm,[py,'-u',str(prop/'run.py'),'export','--run',str(root/arm),'--cache',str(root/'cache'),
            '--teacher-run','/home/sachin.chaudhary/ces_fl_runs/ff2_lh_ps_20261106_r1/L_A','--out',str(root/(arm+'_student.pt')),'--device','cuda'],required=False)
    else:
        (root/('export_'+arm+'.RECEIPT.json')).write_text(json.dumps({'status':'BLOCKED_NO_FEASIBLE_CHECKPOINT','arm':arm,'native_status':result['status']},indent=2)+'\n')
(root/'COHORT_DONE.json').write_text(json.dumps({'status':'COHORT_FINISHED_INSPECT_RESULTS','control':json.loads((root/'control/RUN_VERIFIED.json').read_text())['status'],
    'treatment':json.loads((root/'treatment/RUN_VERIFIED.json').read_text())['status'],'test_cal_read':False,'automatic_extension':False},indent=2)+'\n')
