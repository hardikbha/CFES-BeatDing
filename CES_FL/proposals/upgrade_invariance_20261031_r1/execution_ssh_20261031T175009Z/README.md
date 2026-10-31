# CES-FL direct-SSH execution — 31 October 2026

## Actual running study

The latest user instruction requested execution through ssh gpu01 and ssh gpu02. This changes the execution route from the earlier PBS submission. **These are direct SSH processes, with no new PBS allocation or PBS job IDs claimed.** Model source, scientific configurations, input hashes, approvals, original sampling/update budgets, validation selection and calibration remain fixed.

Exact execution source: commit 875bce025084871438ef799ca9086985c4375672, detached clean worktree:

/home/sachin.chaudhary/ces_fl_sources/upgrade_875bce025084871438ef799ca9086985c4375672

Byte-identical copies of the upgrade module, tests, official launcher and proposal/configurations are published in main for review; published_running_source_hashes.json ties them to the actual execution commit. The upstream request.json/software_checks.json/README remain historical proposal receipts. Use this execution receipt and live observations for current state.

## Verified CUDA gate and launches

**Eight CUDA software tests passed on gpu01 and eight on gpu02**, using the unchanged official upgrade_one_gpu.pbs with CES_SMOKE_ONLY=1. Both smoke processes exited with status 0; real-input preflight passed and CUDA_SMOKE_PASS markers exist. CUDA_GATE_PASSED.json contains launch/config/source hashes, actual device UUIDs and log hashes. The synthetic fixture AUC values printed by the test suite are not research results.

The five exact configured arms were then launched with CES_SMOKE_ONLY=0, each in a separate new directory. The official launcher repeats preflight and CUDA smoke for every arm. All five entered train-config on CUDA; process identity was independently checked against nvidia-smi PIDs and /proc command lines, and training provenance exists.

| Arm | Node | GPU index | Model PID at verified observation | Worker PID |
|---|---|---:|---:|---:|
| R0_frozen_bce | gpu01 | 2 | 2095157 | 2088892 |
| R1_partial_bce | gpu02 | 6 | 3612314 | 3608346 |
| R2_partial_dro | gpu01 | 1 | 2095435 | 2089029 |
| RN_partial_dro_nochange_pilot | gpu02 | 5 | 3612318 | 3608456 |
| RD_partial_dro_age_pilot | gpu01 | 4 | 2094283 | 2089262 |

This table is a dated observation. Read the [live process/status page](../../../status/upgrade_invariance_20261031_r1_ssh_bounded_20261031T175009Z_live.md) for subsequent progress and failures, and [aggregate evidence](../../../results/upgrade_invariance_20261031_r1_ssh_bounded_20261031T175009Z/aggregate.json) for actual validation subgroup metrics and exposure.

Measured model parameter counts from saved production provenance: total 21,594,159 for every arm; trainable 787,207 for R0 and 7,575,591 for the partial-tuning arms.

## Operational policy

The nodes had active workloads and spare VRAM. Execution uses bounded shared GPUs: at least 24 GiB free at binding, one CES process per physical device coordinated with the existing project lock, and a 0.1 PyTorch caching-allocator fraction (about 8 GiB on 80 GB H100) via the separately hashed cuda_policy/sitecustomize.py hook. CUDA context/library allocations are additional. This is an execution-resource limit, not a model/loss modification. Shared-device contention affects elapsed time.

Existing jobs 44861 and44881 were left running. The duplicate queued PBS smoke 44906 was cancelled before execution, and its superseded submission/finite executor were recorded. An idle-device-only SSH waiter was also stopped before any CUDA test/model execution. All earlier failed/waiting artifacts remain preserved.

No source/configuration/input hash was substituted, no approved sampling/seed/selection/calibration value changed, and no batch-size reduction or automatic retry is enabled.

## Protocol and interpretation

TRAIN: 76,139; validation: 25,386, the exact pinned Ding manifests. The original protocol declares 163 overlapping identity/source identifiers; it is not an identity-disjoint benchmark. Checkpoint selection remains validation AUC only. Calibrated thresholds use selected-checkpoint validation-real scores only. No test set is automatically evaluated.

RN/RD use six reviewed sources and matched no-change/age variants. They are small-pool diagnostics, not the expanded 300-source study. There is no new generation, forensic objective, map consistency or Ding reproduction. The historical CESel-MA gate remains FAIL. No fairness improvement or win over Ding is established by launching these arms or passing synthetic tests.

## Artifacts and observation

Output root:
/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_ssh_bounded_20261031T175009Z

Logs:
/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_ssh_bounded_20261031T175009Z/logs/

Each arm's real training log is <root>/<arm>/training.log and epoch evidence is <root>/<arm>/training/history.jsonl. CUDA smoke is saved separately under each stage directory.

Observer PID 3109255 reads and publishes actual GPU processes, validation histories, group metrics, pair-exposure aggregates and checkpoint hashes for at most 24 hours. It never launches, retries or cancels experiments. A DONE marker is not treated as sufficient verification: selected checkpoints are strictly reloaded and compared with recorded source/config/input identity, validation-only selection and calibration. This checks saved metadata and reload, not an independent validation forward replay. Images, weights and private row-level records stay off GitHub.

Node clocks were measured about648seconds behind the management node; CLOCK_OBSERVATIONS.json preserves the readings. Cross-node wall-clock ordering is unreliable. Epoch counters, exit receipts, management observation times and per-process monotonic durations are used; clocks were not changed.
