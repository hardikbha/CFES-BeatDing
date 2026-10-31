# Controlled CES-FL upgrade: actual execution status

Observed UTC: **2026-10-31T14:30:52.729716+00:00**. Queued/allocated is not proof of training.

Exact source commit: 875bce025084871438ef799ca9086985c4375672 from [the supplied fork](https://github.com/hardiksharma6555/CFES-BeatDing/tree/875bce025084871438ef799ca9086985c4375672).

CPU software suite: **19 tests passed**. External-input preflight: **five exact configurations passed**.
TRAIN: 76,139 records; validation: 25,386. The preserved protocol declares 163 overlapping contributor identifiers; this is not an identity-disjoint benchmark.

Initial smoke job: **44880.mgmt01**; CUDA gate: **PENDING**.

| Arm | PBS job | Observed status | Epochs recorded | Best observed validation AUC (%) |
|---|---|---|---:|---:|
| R0_frozen_bce | — | NOT_SUBMITTED | 0 | — |
| R1_partial_bce | — | NOT_SUBMITTED | 0 | — |
| R2_partial_dro | — | NOT_SUBMITTED | 0 | — |
| RN_partial_dro_nochange_pilot | — | NOT_SUBMITTED | 0 | — |
| RD_partial_dro_age_pilot | — | NOT_SUBMITTED | 0 | — |

**LAUNCH SEQUENCE STOPPED:** RuntimeError('CUDA smoke PBS job failed: {"Job_Name": "cesup_smoke", "job_state": "F", "queue": "gpu", "Exit_status": 1, "exec_host": "gpu01/2*8", "exec_vnode": "(gpu01:ncpus=8:ngpus=1:mem=50331648kb)", "Resource_List": {"mem": "48gb", "ncpus": 8, "ngpus": 1, "nodect": 1, "place": "free", "select": "1:ncpus=8:ngpus=1:mem=48gb", "walltime": "12:00:00"}, "resources_used": {"cpupercent": 0, "cput": "00:00:00", "mem": "0kb", "ncpus": 8, "vmem": "0kb", "walltime": "00:00:01"}, "comment": "Job run at Sat Oct 31 at 19:59 on (gpu01:ncpus=8:ngpus=1:mem=50331648kb) and failed"}')

[Aggregate evidence, validation subgroup metrics, exposure and software logs](../../CES_FL/results/upgrade_invariance_20261031_r1_20261031T135409Z/aggregate.json).

RN/RD are six-source small-pool diagnostics; repeated draws do not establish broader coverage. No expanded300-source study, new images, forensic loss, map consistency, test tuning or test evaluation is authorized here.

Selection is validation AUC only. Calibration remains the selected checkpoint’s validation-real empirical FPR budget. Observed metrics do not establish fairness improvement or a win over Ding.

GPU jobs request ngpus=1. The site exposes an allocated host/count rather than a UUID mapping; the operational wrapper verifies the allocation, binds a verified empty physical UUID under the existing project lock, then invokes the byte-identical approved upgrade_one_gpu.pbs. Allocation/binding evidence is retained.

The finite handoff executor submits only the five preauthorized arms after successful initial CUDA smoke, never retries failures, and observes/publishes for at most24h. It does not cancel other jobs or modify research code/configuration. Checkpoint review confirms strict reload and matching recorded selection/calibration; it is not an independent validation replay.

Cluster output root: /home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_20261031T135409Z.
Logs: /home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_20261031T135409Z/logs. Source: /home/sachin.chaudhary/ces_fl_sources/upgrade_875bce025084871438ef799ca9086985c4375672.
