"""No production runner is launched: copy only take_lock and simulate its critical interleaving.
Report fixtures and every shell output live in review_scratch. No GPUs or PBS calls.
"""
import sys, os, subprocess, time, json, importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; S=ROOT/'review_scratch'
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
text=(ROOT/'run/cesel_run.sh').read_text()
fn=text[text.index('take_lock(){'):text.index('\ntrain(){')]
# Scheduling barrier after both contenders observed stale ownership, before either writes owner.
fn=fn.replace('  echo "$me" > $L/owner;', '  touch "$BARRIER/ready_$TAG"; while [ ! -e "$BARRIER/ready_A" ] || [ ! -e "$BARRIER/ready_B" ]; do sleep 0.01; done\n  echo "$me" > $L/owner;')
harness=S/'lock_harness.sh'
harness.write_text(fn+'\ntake_lock "$1"\necho ACQUIRED_$TAG\nsleep 0.2\n')
bar=S/'lock_barrier'; bar.mkdir(exist_ok=True)
for f in bar.glob('ready_*'): f.unlink()
lock=S/'fake_run.RUNLOCK'; lock.mkdir(exist_ok=True)
(lock/'owner').write_text('otherhost nojob 99999999\n'); (lock/'heartbeat').touch()
os.utime(lock/'heartbeat',(time.time()-1000,time.time()-1000))
ps=[]; handles=[]
for tag in ('A','B'):
    h=open(S/f'lock_{tag}.log','w'); handles.append(h)
    ps.append(subprocess.Popen(['bash',str(harness),str(S/'fake_run')],env=dict(os.environ,BARRIER=str(bar),TAG=tag),stdout=h,stderr=h))
for p in ps: p.wait(timeout=10)
for h in handles: h.close()
a=(S/'lock_A.log').read_text(); b=(S/'lock_B.log').read_text()
assert 'ACQUIRED_A' in a and 'ACQUIRED_B' in b
print('REPRO stale RUNLOCK race: both A and B acquired the same run after observing the same stale heartbeat')
# Fresh-lock negative control uses the unmodified function excerpt.
harness2=S/'fresh_lock_harness.sh'; harness2.write_text(text[text.index('take_lock(){'):text.index('\ntrain(){')]+'\ntake_lock "$1"\n')
lock.mkdir(exist_ok=True); (lock/'owner').write_text('otherhost nojob 99999999\n'); (lock/'heartbeat').touch()
r=subprocess.run(['bash',str(harness2),str(S/'fake_run')],capture_output=True,text=True,timeout=5)
assert r.returncode==4
print('PASS fresh RUNLOCK contender exits 4')
import cesel_report as R
fixture=S/'report_fixture'; cf=fixture/'cfes'; cf.mkdir(parents=True,exist_ok=True)
# No test-score files: report only reads the controlled TRAINED records.
vals={'B0':90,'sens':89,'inv':91,'floor':92,'full':95,'C_randcue':80,'C_shufgroup':80,'C_novalid':80,'B0_cfesD':90}
def trained(p,v):
    p.mkdir(parents=True,exist_ok=True); (p/'TRAINED').write_text(json.dumps({'best':v,'best_epoch':0,'epoch':5}))
trained(fixture/'runs/ffpp_B0_s7',90)
for arm,v in vals.items(): trained(fixture/f'runs/ffpp_x5_{arm}_s7',v)
for seed,v in ((7,87.62),(17,89.68),(27,89.28)): trained(cf/f'runs_rev4/clip_D_w0.3_s{seed}',v)
R.CF=str(cf); sys.argv=['cesel_report.py','--out_root',str(fixture)]
R.main(); dec=json.loads((fixture/'results/cesel_ffpp_screen.json').read_text())['decisions']
assert not dec['keep']['sens'] and dec['winner']=='full'
print('REPRO promotion: +sens fails B0-continued (89 < 90), but Full containing sens is promoted (95)')
# No spread references: completion still promotes, even though controls cannot be assessed.
for p in cf.glob('runs_rev4/*/TRAINED'): p.unlink()
R.main(); dec=json.loads((fixture/'results/cesel_ffpp_screen.json').read_text())['decisions']
assert dec['spread'] is None and dec['winner']=='full'
print('REPRO promotion: unavailable reference spread leaves drops pending but still promotes Full')
