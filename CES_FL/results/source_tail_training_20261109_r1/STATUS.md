# Source-tail cohort — CUDA qualification observation

Observed on9November2026 after successful process checks: controller3056234, budget wrapper3056321, qualifier3056378 active on gpu01. One pool-v2 capacity lease binds physical GPU6 / UUID GPU-61f54ca7-3cee-1c93-def3-118523e95241. It is shared capacity, not an exclusive/PBS ownership claim. No other job was displaced.

CUDA qualification PASS for the pinned three-epoch synthetic lifecycle, finite head updates, frozen encoder and full224 feature/native-forward identity. These are software checks, not real benchmark results. Real-TRAIN throughput profiling is RUNNING; production cache/control/treatment/export remain PENDING until its budget gate passes.

The initial source-linkage guard correctly failed before allocation because its Git metadata was login-node-local; a new self-contained shared clone is detached at exactly9bb8e83 with identical scientific hashes. A second idle-only attempt found no idle card and also consumed zeroGPU time. The successful attempt uses the existing explicitly authorized shared-pool candidate/memory guard without changing the model recipe. Failure logs remain private and are not overwritten.

Scientific source: `/home/sachin.chaudhary/ces_fl_sources/cesfl-source-tail-shared-9bb8e83-20261109`. Private live logs: `/home/sachin.chaudhary/ces_fl_runs/source_tail_training_20261109_r1/logs/`; controller `lease_controller_attempt3.log`. All CUDA stages are inside the existing ledger wrapper with1.98h reservation and an outer2h cohort guard; actual lease startup/cleanup/idle is included. No automatic extension.

---

# Source-tail cohort — preparation observation, 9 November 2026

Scientific source is pinned in a clean detached worktree at `9bb8e8332b8a3aa78d83f74ef21bca0b7dcd6754`. No scientific configuration, loss, encoder/head architecture or historical guard was changed.

- Supplied CPU pipeline checks: PASS (synthetic only, not benchmark performance).
- Complete L_A teacher SHA `cce4476491829702d60a74ed7b2044aa8436768d6e08e0129da5a1fd84b0cb6d`: actual bytes/verified completion/config/TRAIN+SELECT fit inventory PASS. Selected epoch2; complete existing2049-parameter linear head is retained.
- Canonical real TRAIN source sidecar:16777 exact unique rows; FFHQ3176, FairFace13601; provenance/hash bound. Private sidecars are not published.
- Shared72 ledger reconciled conservatively: prior64.601102 charged plus3.227956 upper-bound historical TEST/sanity charge =67.829058; verified remaining lower bound4.170942 GPU-hours. Earlier CUDA-disabled CPU checks are not GPU work. The upper-bound charge covers ALL three historically observed cards over the entire first-to-last foreground command window including gaps; it is not claimed actual GPU compute. Existing manual charges retained unchanged, with missing state fields normalized for the strict ledger wrapper.
- New cohort cap:2 total GPU-hours, one leased GPU at a time, including qualification/profile/cache/heads/exports/held idle. No additional grant and no automatic extension.
- GPU qualification, timing profile, production cache, control, treatment and exports: **PENDING** at this observation. The established authorized pool-v2 capacity/lease guard and existing budget wrapper are required; idle utilization alone is not ownership.

Execution helpers in `operator/` only handle the bounded launch, timing and bookkeeping. Model stages call the exact pinned `run.py`. TRAIN source labels come from the previous recorded-key audit, not appearance. No original L_H probe/alignment losses, JPEG, new demographic predictions or synthetic pairs are part of these arms. No TEST/CAL data access or new Ding computation.

Private root/logs: `/home/sachin.chaudhary/ces_fl_runs/source_tail_training_20261109_r1/`. This is a dated preparation snapshot, not a claim that training is running.
