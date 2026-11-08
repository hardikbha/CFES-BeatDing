import fcntl, json, os, pathlib, sys, tempfile, time
LEDGER=pathlib.Path('/home/sachin.chaudhary/ces_fl_runs/ff2_baseline_suite_s17_20261103T110324Z/additional72_gpu_hours.json')
LOCK=pathlib.Path(str(LEDGER)+'.lock')
def write_update(fn):
    LOCK.parent.mkdir(parents=True,exist_ok=True)
    with LOCK.open('a+') as lf:
        fcntl.flock(lf,fcntl.LOCK_EX)
        d=json.loads(LEDGER.read_text()); fn(d)
        fd,tmp=tempfile.mkstemp(dir=LEDGER.parent,prefix='.gpu-ledger-')
        with os.fdopen(fd,'w') as f: json.dump(d,f,indent=1); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,LEDGER)
    return d

def reserve(run,hours):
    key='ff2_sjgentle_confirm_'+run
    def op(d):
        jobs=d.setdefault('jobs',{})
        if key in jobs and jobs[key].get('state') in ('RUNNING','RESERVED','FINISHED'):
            raise SystemExit('reservation already exists: '+key)
        used=sum(float(v.get('charged_gpu_hours',0)) for v in jobs.values())
        reserved=sum(float(v.get('reserved_gpu_hours',0)) for v in jobs.values() if v.get('state') in ('RUNNING','RESERVED'))
        cap=float(d.get('cap_gpu_hours',72))
        if used+reserved+hours>cap+1e-9: raise SystemExit(f'budget blocked: used={used:.4f} reserved={reserved:.4f} ask={hours:.4f} cap={cap:.2f}')
        jobs[key]={'state':'RESERVED','gpus':1,'reserved_gpu_hours':hours,'charged_gpu_hours':0.0,'start_unix':time.time(),
                   'proposal':'ff2_sjgentle_confirmation_20261108_r1','run':run,'reservation_owner':'Codex'}
        print(f'reserved {hours:.2f} GPU-h for {run}; used={used:.4f}, remaining after reservation={cap-used-reserved-hours:.4f}')
    write_update(op)

def finish(run,start,end,host,gpu,exit_code,commit):
    key='ff2_sjgentle_confirm_'+run
    def op(d):
        e=d.setdefault('jobs',{}).get(key)
        if e is None or e.get('state') not in ('RESERVED','RUNNING'): raise SystemExit('no active reservation: '+key)
        charged=max(0,int(end)-int(start))/3600
        reserved=float(e.get('reserved_gpu_hours',0))
        if charged>reserved+0.02: print(f'warning: actual {charged:.4f} exceeds reserved {reserved:.4f} GPU-h')
        e.update({'state':'FINISHED' if int(exit_code)==0 else 'FAILED','gpus':1,'reserved_gpu_hours':0.0,
                  'charged_gpu_hours':charged,'start_unix':float(start),'end_unix':float(end),
                  'allocation_evidence':f'direct-SSH shared-GPU lease {host} {gpu}','exit_code':int(exit_code),'commit':commit})
        used=sum(float(v.get('charged_gpu_hours',0)) for v in d['jobs'].values())
        print(f'finalized {run}: {charged:.4f} GPU-h, exit={exit_code}; total charged={used:.4f}/{d.get("cap_gpu_hours",72):.2f}')
    write_update(op)

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='reserve': reserve(sys.argv[2],float(sys.argv[3]))
    elif mode=='finish': finish(sys.argv[2],sys.argv[3],sys.argv[4],sys.argv[5],sys.argv[6],sys.argv[7],sys.argv[8])
    else: raise SystemExit('usage: ledger_update.py reserve RUN HOURS | finish RUN START END HOST GPU EXIT COMMIT')
