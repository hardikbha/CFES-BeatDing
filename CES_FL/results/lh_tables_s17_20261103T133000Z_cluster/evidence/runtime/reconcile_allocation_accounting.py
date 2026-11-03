"""Record scheduler bookend time without changing original task receipts."""
import datetime
import importlib.util
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODULE = Path('/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/allocated_stage.py')
spec = importlib.util.spec_from_file_location('budget_accounting', MODULE)
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def main():
    receipt = ROOT / 'SCHEDULER_ACCOUNTING.json'
    if receipt.exists():
        raise RuntimeError('Accounting already reconciled; preserve existing receipt')
    entries = []
    for job, session in [('45295.mgmt01', 'plain_head_0d699d2'),
                         ('45314.mgmt01', 'lh_tables_ee73f2f_eval')]:
        text = subprocess.check_output(['/opt/pbs/bin/qstat', '-xf', job], text=True, timeout=30)
        assert re.search(r'job_state\s*=\s*F\b', text)
        assert re.search(r'Exit_status\s*=\s*0\b', text)
        assert re.search(r'Resource_List.ngpus\s*=\s*1\b', text)
        match = re.search(r'resources_used.walltime\s*=\s*(\d+):(\d+):(\d+)', text)
        hours, minutes, seconds = map(int, match.groups())
        elapsed = 3600 * hours + 60 * minutes + seconds
        with budget.ledger() as data:
            original = next(s for s in data['gpu_sessions'] if s['id'] == session)
            assert original['state'] == 'COMPLETED'
            overhead = max(0., elapsed - original['elapsed_seconds'])
            name = session + '_pbs_bookend_overhead'
            assert not any(s['id'] == name for s in data['gpu_sessions'])
            data['gpu_sessions'].append(dict(
                id=name, state='COMPLETED', ngpus=1, reserved_gpu_hours=0.,
                elapsed_seconds=overhead, pbs_job_id=job,
                accounting='max(0, completed PBS walltime minus original task-held seconds)',
                original_session=session, original_receipt_preserved=True))
            entries.append(dict(pbs_job_id=job, session=session,
                                pbs_walltime_seconds=elapsed,
                                original_task_seconds=original['elapsed_seconds'],
                                extra_bookend_seconds=overhead))
    with budget.ledger() as data:
        used = sum(s.get('elapsed_seconds', 0) * s['ngpus'] / 3600 for s in data['gpu_sessions'])
        cap = data['initial_total_gpu_hours']
    budget.write(receipt, dict(management_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                              status='PASS', allocations=entries,
                              study_total_gpu_hours=used, study_cap_gpu_hours=cap,
                              remaining_unreserved_gpu_hours=cap - used,
                              ff2_additional72_budget_untouched=True))
    print(json.dumps(json.loads(receipt.read_text()), indent=2))


if __name__ == '__main__':
    main()
