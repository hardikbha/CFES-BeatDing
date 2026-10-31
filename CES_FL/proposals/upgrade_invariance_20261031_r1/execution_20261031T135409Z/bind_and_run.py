"""Bind one free physical device inside an existing PBS ngpus=1 allocation.

Operational helper only: research source/config remain unchanged. PBS at this
site reserves an ngpus count on a host, without a physical UUID mapping. Record
that limitation; coordinate our physical UUID using the existing project lock.
"""
import fcntl, hashlib, json, os, signal, socket, subprocess, sys, time
from pathlib import Path

ROOT = Path("/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_20261031T135409Z")
SOURCE = Path('/home/sachin.chaudhary/ces_fl_sources/upgrade_875bce025084871438ef799ca9086985c4375672')
LOCKS = Path('/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA/work/gpulocks')
EXPECTED_COMMIT = '875bce025084871438ef799ca9086985c4375672'
EXPECTED_LAUNCHER = 'c2723650c2839e625e286c93e15cd41d18bca5efdc1e3b8924d280e70f9321f3'

def capture(args):
    return subprocess.check_output(args, text=True)

def write(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)

def main():
    stage = os.environ['CES_UPGRADE_STAGE']
    assert stage in {'SMOKE_R1','R0_frozen_bce','R1_partial_bce','R2_partial_dro','RN_partial_dro_nochange_pilot','RD_partial_dro_age_pilot'}
    job = os.environ['PBS_JOBID']
    host = socket.gethostname().split('.')[0]
    assert host in ('gpu01','gpu02')
    assert os.environ['CES_SOURCE_DIR'] == str(SOURCE)
    assert capture(['git','-C',str(SOURCE),'rev-parse','HEAD']).strip() == EXPECTED_COMMIT
    assert not capture(['git','-C',str(SOURCE),'status','--porcelain','--','CES_FL']).strip()
    launcher = SOURCE/'CES_FL/cluster/upgrade_one_gpu.pbs'
    assert hashlib.sha256(launcher.read_bytes()).hexdigest() == EXPECTED_LAUNCHER
    output = Path(os.environ['CES_OUTPUT_DIR'])
    assert not output.exists()
    allocation = capture(['qstat','-f','-F','json',job])
    record = json.loads(allocation)['Jobs'][job]
    assert record['job_state'] == 'R'
    assert int(record['Resource_List']['ngpus']) == 1
    assert host in record['exec_host']
    (ROOT/'runtime'/f'{stage}.allocation.json').write_text(allocation)
    guard = (ROOT/'runtime'/f'{stage}.lock').open('a')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock = None
    child = None
    owner = ROOT.name+':'+stage+':'+job
    def stop(number, _):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, number)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        raise SystemExit(128+number)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        deadline = time.monotonic()+7200
        while lock is None:
            inherited = os.environ.get('CUDA_VISIBLE_DEVICES')
            permitted = set(inherited.split(',')) if inherited else None
            occupied = set(capture(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader']).splitlines())
            inventory = capture(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'])
            for line in inventory.splitlines():
                index,uuid,total,free,util = [part.strip() for part in line.split(',')]
                if permitted is not None and index not in permitted and uuid not in permitted:
                    continue
                if uuid in occupied or int(free)<24576 or int(util)>10:
                    continue
                candidate = LOCKS/(host+'_'+index)
                try:
                    candidate.mkdir()
                except FileExistsError:
                    continue
                (candidate/'by').write_text(owner+'\n')
                lock = candidate
                current = set(capture(['nvidia-smi','--query-compute-apps=gpu_uuid','--format=csv,noheader']).splitlines())
                if uuid in current:
                    (lock/'by').unlink(); lock.rmdir(); lock=None
                    continue
                break
            if lock is None:
                if time.monotonic()>deadline:
                    raise RuntimeError('No unoccupied project-lockable GPU within two hours; no unrelated job modified')
                write(ROOT/'runtime'/f'{stage}.binding.json', {'state':'WAITING_FOR_EMPTY_GPU','pbs_job_id':job,'host':host,'time':time.time()})
                time.sleep(20)
        binding={'state':'BOUND','pbs_job_id':job,'host':host,'gpu_uuid':uuid,'physical_index':int(index),
            'memory_total_mib':int(total),'memory_free_mib':int(free),'inventory':inventory,
            'binding_method':'PBS ngpus=1 allocation plus verified empty physical UUID and shared project lock; site provides host/count rather than UUID binding',
            'time':time.time(),'launcher_sha256':EXPECTED_LAUNCHER}
        write(ROOT/'runtime'/f'{stage}.binding.json',binding)
        environment={**os.environ,'CUDA_VISIBLE_DEVICES':uuid,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'}
        child=subprocess.Popen(['bash',str(launcher)],cwd=SOURCE,env=environment,start_new_session=True)
        code=child.wait()
        write(ROOT/'runtime'/f'{stage}.exit.json',{'exit_code':code,'pbs_job_id':job,'host':host,'gpu_uuid':uuid,'finished_at':time.time()})
        raise SystemExit(code)
    except BaseException as error:
        if not isinstance(error,SystemExit) or error.code:
            write(ROOT/'runtime'/f'{stage}.failure.json',{'error':repr(error),'pbs_job_id':job,'host':host,'time':time.time()})
        raise
    finally:
        if lock is not None and (lock/'by').exists() and (lock/'by').read_text().strip()==owner:
            (lock/'by').unlink();lock.rmdir()
if __name__=='__main__':
    main()
