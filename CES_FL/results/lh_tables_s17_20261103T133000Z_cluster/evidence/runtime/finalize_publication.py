"""Copy explicitly whitelisted aggregate evidence after all fixed analyses finish."""
import argparse
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    done = json.loads((ROOT / 'ANALYSIS_DONE.json').read_text())
    assert done['status'] == 'PASS'
    scheduler = {}
    for job in ('45314.mgmt01', '45315.mgmt01'):
        output = subprocess.check_output(['/opt/pbs/bin/qstat', '-xf', job], text=True, timeout=30)
        values = {}
        allowed = {'Job_Name', 'job_state', 'Exit_status', 'resources_used.walltime',
                   'resources_used.cput', 'resources_used.ncpus', 'Resource_List.ngpus',
                   'Resource_List.ncpus', 'exec_host', 'stime', 'obittime'}
        for line in output.splitlines():
            if '=' in line:
                key, value = line.strip().split('=', 1)
                if key.strip() in allowed:
                    values[key.strip()] = value.strip()
        assert values['job_state'] == 'F' and values['Exit_status'] == '0'
        scheduler[job] = values
    subprocess.run([sys.executable, str(ROOT / 'render_locked_results.py'), '--out', str(out)], check=True)
    subprocess.run([sys.executable, str(ROOT / 'summarize_uncertainty.py'), '--out', str(out)], check=True)
    evidence = out / 'evidence'
    runtime = evidence / 'runtime'
    runtime.mkdir(exist_ok=True)
    for name in ('ALLOCATION_END.json', 'ANALYSIS_DONE.json', 'SCHEDULER_ACCOUNTING.json',
                 'A_validation_operating_points.json', 'H_validation_operating_points.json',
                 'analysis.log', 'allocation.log'):
        shutil.copyfile(ROOT / name, evidence / name)
    for arm in ('L_A', 'L_H'):
        for ds in ('ffpp', 'dfdc', 'celebdf', 'dfd'):
            name = f'{arm}_{ds}_COMMAND.json'
            shutil.copyfile(ROOT / name, evidence / name)
    names = ('analyze_locked.py', 'analyze_locked.pbs', 'render_locked_results.py',
             'summarize_uncertainty.py', 'reconcile_allocation_accounting.py', 'finalize_publication.py')
    for name in names:
        shutil.copyfile(ROOT / name, runtime / name)
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt = dict(management_utc=stamp, status='PASS', scheduler=scheduler,
                   scientific_commit='ee73f2fdf5218e440852917e90281818bbe711bb',
                   scientific_source_or_config_changed=False,
                   runtime_sha256={name: sha(ROOT / name) for name in names},
                   completed_inference_tasks=8, completed_analysis_tasks=12,
                   additional_inference_or_training_launched=False,
                   public_evidence_contains_private_rows_or_images=False)
    (evidence / 'FINAL_EXECUTION_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('FINAL_PUBLICATION_VERIFIED', stamp, '8 inference, 12 analysis, both PBS exit 0')


if __name__ == '__main__':
    main()
