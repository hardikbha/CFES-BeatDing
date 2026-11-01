# Manipulation-response seed17: direct SSH execution

Observed management UTC: **2026-11-01T17:05:33.870856+00:00**.

The user superseded PBS with SSH to gpu01/gpu02. Queued smoke44963 was cancelled **before execution**; its records are preserved. No PBS allocation or exclusive ownership is claimed. The existing Qwen pilot is untouched.

Exact scientific source: `58773bbef9d9fe79dd313610f694ddeea12c6037`. Architecture, loss, config, seed, selection and calibration are unchanged.

| Check | Actual state |
|---|---|
| New CUDA software suite | COMPLETED |
| Full original TRAIN/VAL decode/hash + configured backward | COMPLETED |

| Arm | Actual status | Recorded epochs |
|---|---|---|
| P | COMPLETED | 50/50 |
| bce | RUNNING | 2/50 |
| fair | RUNNING | 2/50 |
| nochange | BLOCKED_INPUTS | 0/50 |
| score | BLOCKED_INPUTS | 0/50 |
| response | BLOCKED_INPUTS | 0/50 |

P/BCE/fair need no quartet bank. All paired arms remain blocked by valid matched manipulations and independent review. No source-real substitution, Qwen-only quartet, synthetic-fixture bank or fabricated approval is accepted.

Each SSH milestone acquires the shared dispatcher exclusion and project lease, protects Qwen, requires **20GiB free headroom**, and uses a **15% Torch allocator cap** on one verified CUDA UUID. Existing processes continue. Both nodes were inspected; actual per-stage host, physical GPU and worker/child PIDs are in the machine record. Duplicate jobs are not launched.

The synthetic six-group CUDA suite is software evidence, not a demographic/forensic validity or fairness result. The original-only track uses all76,139 TRAIN originals, validation-only AUC selection and frozen validation-real empirical10% FPR calibration. Recorded validation metrics are not test results or a win over Ding. No FairForge-2 or test evaluation is attached.

Run root: `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z`. Actual logs: `logs/smoke.log`, `logs/real_preflight.log`, `logs/P.log` when those stages execute. Only explicitly requested single milestones run; this observer cannot advance them.

[Actual metrics, PIDs, bindings, hashes and failures](../results/manip_response_ssh_20261101T115720Z/status.json) · [SSH execution record](../proposals/manipulation_response_20261101_r1/execution_manip_response_ssh_20261101T115720Z/README.md) · [Matched quartet blockers](../proposals/manipulation_response_quartets_20261101_r1/README.md).

Finite observation lasts at most72h, normally publishing every ten minutes/on state changes. Read the observation timestamp; uninterrupted monitoring is not guaranteed.
