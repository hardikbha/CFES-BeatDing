# User-authorized resource expansion — 3 November 2026

Job43937 was the idle CIRCE v3 scheduler on gpu01. The actual job requested8CPUs and zero ngpus; its process tree contained only the old shell/scheduler/sleep, with no CES-FL or Ding training descendants. The user explicitly requested stopping it. Actual command `qdel 43937.mgmt01` returned0; subsequent `qstat -xf` reports stateF and its old processes are absent. Its scheduler/task/log artifacts are preserved privately. No other job was stopped.

The freed workq slot is now actual PBS job **45202.mgmt01**, named cesfl_AB_s17, granted `(gpu01:ncpus=12:ngpus=2:mem=25165824kb)`, walltime04:00:00. A/B controls run in parallel with already-running C/D, following the user's request to use more GPUs. Scientific source, configuration digest, seed17, batch32, ten epochs, data exposure, optimizer, selection and calibration are unchanged. The coordinator sees their original expected launch records and will skip duplicate launches; exact native completion receipts remain required.

PBS provides a resource-count grant but does not expose the physical device mapping (`CUDA_VISIBLE_DEVICES` was empty). The previously explicit direct-SSH authorization applies to physical shared pool-v2 leasing; no exclusive GPU or scheduler physical mapping is claimed. Both bindings were verified in actual CUDA_BINDING logs and nvidia-smi process observations:

| Arm | Host | Physical GPU | Training PID | Route |
|---|---|---:|---:|---|
|A|gpu01|4|3054073|PBS ngpus2 count grant; explicit shared UUID lease|
|B|gpu01|3|3054081|PBS ngpus2 count grant; explicit shared UUID lease|
|C|gpu01|6|3009986|Existing authorized directSSH shared lease, preserved|
|D|gpu02|6|89578|Existing authorized directSSH shared lease, preserved|

A UUID GPU-9b669a06-9156-a98e-2194-36f2fc11d7ca; B UUID GPU-6a002609-de85-4949-c707-ee9317b284f0. CUDA device_count1, H10080GB, bf16 true, torch2.7.0+cu126 for both. This is independent-arm parallelism, not a change to DDP, global batch, or gradient reduction.

Source remains immutable commit c332fd34fb603b8919b15fee08e8d022fb9f57c8, Python SHA89530025a00b9b0405fba7026d3eed5f2749d86620174ad3ffde93a44f8b770b. Existing source-bound CPU/CUDA/backward receipts remain valid. Resource-only launcher guards: actual command in resource_guard_tests.log,5tests PASS0.002s; py_compile and bash-n syntax checks passed. A separate actual observer metadata check confirms job R,2GPU grant and distinct UUIDs. These checks are not scientific results.

The initial24TOTAL GPU-hour ceiling is unchanged. Per-arm task-held wall time counts oneGPU each, even shared. Any extra idle PBS GPU-slot time is added as `ffpp_AB_pbs_idle_overhead`: max(0,2*PBS-controller walltime - A/B task-held seconds). Overhead has its own conservative reservation. No test inference, new generation, review fabrication, architecture or loss change. FairForge-2 still requires its own preflight/review/CUDA gates.

The original publisher script/log are preserved. Observer version2 adds the actual PBS job/grant and physical-mapping limitation to LIVE_STATE.json; every10minutes until the original48hour deadline. It cannot launch/stop training. Training, old Ding job, and original coordinator were not restarted. See STATUS.md and LIVE_STATE.json for subsequent epochs and metrics.

Logs stay on cluster: `/home/sachin.chaudhary/ces_fl_runs/forensic_response_overlapfix_20261103_r1_ffpp_s17/ffpp_A_initial.log` and `ffpp_B_initial.log`; C/D retain `train_C.log`/`train_D.log` in the same root. PBS stdout may remain in the MOM spool while the job is running; inspect these direct arm logs for immediate progress.
