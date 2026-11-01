# Direct SSH recovery: actual FF++ counterfactual 50+50 execution

Observed UTC: 2026-11-01T07:45:24.998603+00:00. Scientific source remains `fc4e63b8811c5813afa0d298a8172ba9555b7e40` with unchanged source/configuration/input hashes.

## Actual route and launch evidence

The user explicitly requested direct SSH on gpu01/gpu02 and continuing GitHub updates. PBS 44948 was checked as QUEUED and cancelled before execution. Its records remain preserved in the earlier execution package. The user instruction supersedes the allocation route for this execution; **no PBS allocation or exclusive GPU ownership is claimed**. Existing training jobs were not terminated.

The initial direct worker stopped before model smoke because its UUID equality check compared NVIDIA's `GPU-...` prefix with PyTorch's bare UUID. Actual diagnosis on gpu02 confirmed these identify the same device. The failed wrapper, traceback and submission are in `preserved_uuid_failure/`. The new wrapper compares normalized complete UUIDs and passed three regression guards: bare/prefixed equality, different-device rejection and malformed-ID rejection. Scientific code was not changed. Prior output roots were preserved.

Actual current root: `/home/sachin.chaudhary/ces_fl_runs/cf5050_ssh_recovery_20261101T073243Z`. Actual preflight SSH worker PID 2850831 on gpu02. CUDA device binding and memory conditions are in [checks/preflight_BINDING.json](checks/preflight_BINDING.json). The PyTorch allocator is capped at 20% of one selected GPU for preflight; this does not cap every driver/library allocation or reserve that memory. Other processes may share the GPU.

## Verified checks and pending stages

**Seven CUDA smoke checks passed** on the real H100 under PyTorch 2.7.0+cu126, Python 3.12.11. Total/trainable parameters actually measured: 21,594,159 / 21,594,159. The exact source/configuration remain SHA256 `3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab` / `56e77211a966ae12a449c47cdaf8e3fc271f149c6031513747813a1d1abfaef0`.

Actual command and wall duration: [checks/smoke_cuda_EXIT.json](checks/smoke_cuda_EXIT.json), 56.8368 seconds. Complete actual output: [logs/smoke_cuda.log](logs/smoke_cuda.log); machine receipt: [checks/CUDA_SMOKE_RESULT.json](checks/CUDA_SMOKE_RESULT.json). These tests use synthetic fixtures, including tiny training/resume/evaluation checks; they are not evidence of real-data fairness improvement or completion of full pretraining.

At this observation, the complete 76,139-original byte-hash generation plan is still running. The next command is the real configured batch backward dry-run with zero optimizer steps. The two-source generator pilot starts only after both pass. Every original remains in scope; no silent subset replaces the protocol.

Pilot generation is a separate explicit SSH milestone, selecting one real and one fake original and generating the seven configured conditions independently from each original. It measures cached model binary hashes, runtime, peak Torch allocation and canonical storage. The nominal pilot memory policy is 55% allocator cap with at least 52GiB spare VRAM, with a fresh physical-device check. It waits or stops if safe headroom is unavailable; existing jobs are not evicted. The generation source, model revision, 40 steps, CFG and geometry remain unchanged.

No real Qwen result, quality acceptance, full 50-epoch P pretraining, full bank generation, checkpoint binding or B/N/C training has completed in this run. The exact pilot/quality/resource gates remain binding. Contact sheets and pending per-image review artifacts stay private; no approval is fabricated. Full B/N/C also requires a valid reviewed edit for every required attribute and an approved no-change control for every TRAIN original.

**FairForge-2: DO NOT RUN.** Its independent supervisor-review blockers remain unresolved, and no FF++ configuration is substituted. The earlier CESel-MA gate remains FAIL. No test tuning or real-test evaluation is part of this sequence.

## Publication and artifacts

[Actual current status](../../../status/cf5050_ssh_recovery_20261101T073243Z_live.md) and [aggregate receipts/progress](../../../results/cf5050_ssh_recovery_20261101T073243Z/status.json) update through finite read-only observer PID 1061367. It cannot launch/retry/cancel/approve/advance work. It stops at pilot success/failure or 24h. No indefinite uptime is guaranteed; always read the observation timestamp.

Immutable runtime source: `/home/sachin.chaudhary/ces_fl_sources/cesfl-cf5050-fc4e63b`. Only dated operational wrappers/profilers changed. Files here preserve the exact executed CUDA evidence and wrapper versions. Private source inventories, faces, binary weights and per-image review records are excluded from GitHub. Node wall clocks differ from management observations; actual subcommand durations use monotonic time.
