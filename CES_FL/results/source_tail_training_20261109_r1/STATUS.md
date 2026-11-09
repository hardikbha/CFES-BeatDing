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
