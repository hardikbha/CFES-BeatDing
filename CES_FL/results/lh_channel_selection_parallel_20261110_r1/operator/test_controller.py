import unittest
import os,signal,subprocess,sys,threading,time,tempfile
from pathlib import Path
from controller import held_seconds,cost_gate,Operator,detached_head

class BudgetTests(unittest.TestCase):
    def test_detached_worktree_metadata_without_git_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);meta=p/'metadata';meta.mkdir();work=p/'work';work.mkdir()
            head='1a6d63db6baf7f6ad2a8e842a31341d2c128de25'
            (meta/'HEAD').write_text(head+'\n');(work/'.git').write_text('gitdir: '+str(meta)+'\n')
            self.assertEqual(detached_head(work),head)
            (meta/'HEAD').write_text('ref: refs/heads/main\n')
            with self.assertRaises(ValueError):detached_head(work)

    def test_parallel_devices_charge_sum_not_wall_time(self):
        leases=[dict(started=0,ended=None) for _ in range(3)]
        self.assertEqual(held_seconds(leases,600),1800)

    def test_late_acquisitions_charge_only_actual_hold(self):
        leases=[dict(started=0,ended=600),dict(started=400,ended=None),dict(started=400,ended=None)]
        self.assertEqual(held_seconds(leases,600),1000)

    def test_CPU_wait_before_lease_is_not_GPU_usage(self):
        self.assertEqual(held_seconds([],5000),0)

    def test_released_zero_time_lease_does_not_keep_accruing(self):
        self.assertEqual(held_seconds([dict(started=0,ended=0)],600),0)

    def test_each_arm_must_fit_even_if_parallel_wall_window_fits(self):
        self.assertFalse(cost_gate(dict(none=2000,random=1000,informed=1000),0))

    def test_setup_plus_all_arms_must_fit_aggregate(self):
        self.assertFalse(cost_gate(dict(none=1000,random=1000,informed=1000),4500))
        self.assertTrue(cost_gate(dict(none=1000,random=1000,informed=1000),500))

    def test_watchdog_stops_only_registered_process_groups(self):
        owned=subprocess.Popen([sys.executable,'-S','-c','import time;time.sleep(30)'],start_new_session=True)
        protected=subprocess.Popen([sys.executable,'-S','-c','import time;time.sleep(30)'],start_new_session=True)
        op=Operator.__new__(Operator);op.leases=[dict(started=time.monotonic()-7000,ended=None)]
        op.started=time.monotonic();op.abort=None;op.arm_starts={};op.watch_stop=threading.Event();op.children=[(owned,None)]
        watcher=threading.Thread(target=op.watchdog,daemon=True);watcher.start()
        try:
            owned.wait(timeout=4)
            self.assertIsNotNone(op.abort)
            self.assertIsNone(protected.poll())
        finally:
            op.watch_stop.set();watcher.join(timeout=3)
            for child in (owned,protected):
                if child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
                child.wait(timeout=3)

if __name__=='__main__':unittest.main(verbosity=2)
