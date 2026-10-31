# Upgrade handoff execution receipt — 31 October 2026

## Verified source and scope

Fetched https://github.com/hardiksharma6555/CFES-BeatDing.git branch upgrade/ces-fl-controlled-invariance-20261031. FETCH_HEAD exactly matched **875bce025084871438ef799ca9086985c4375672**. Created a new clean detached worktree:

/home/sachin.chaudhary/ces_fl_sources/upgrade_875bce025084871438ef799ca9086985c4375672

Read CES_FL/AGENTS.md, README.md, STATUS.md, WORKFLOW.md and the proposal README, CLUSTER_AGENT_PROMPT.md, request.json, software_checks.json, executable source and all five configurations. Existing source checkouts, checkpoints, outputs and jobs were preserved. This does not change the historical CESel-MA gate.

## Executed checks

- Exact external-input preflight passed for all five configurations: 76,139 TRAIN records, 25,386 validation records and 16 label × ids8 cells. RN/RD each have six eligible reviewed sources with both age directions. The existing protocol declares 163 overlapping identity/source identifiers between TRAIN and validation; it is preserved and not labeled identity-disjoint.
- Target-environment CPU suite: **19 tests passed in22.113s** using /home/sachin.chaudhary/.conda/envs/aiface/bin/python -m unittest discover -s CES_FL/tests -v.
- Seven additional operational tests passed with qsub mocked. They test smoke failure/pending guards, required CUDA evidence, five exact configurations/new outputs, no repeated submissions and duplicate/uncertain-attempt rejection. The 99001–99005 job IDs printed in that test log are **mock values, not PBS jobs**.
- GPU operational wrapper and both PBS scripts passed syntax checks. Model/loss/configuration files in the detached worktree were not edited.

Actual per-arm preflight commands and input/config/source SHA256 values are in handoff_verification.json, preflight_summary.json and the public aggregate evidence.

## Actual submitted job and current prerequisite

The initial smoke-only PBS job is **44880.mgmt01**, submitted with CES_SMOKE_ONLY=1 and R1_partial_bce.json. At the initial execution observation it was **QUEUED** because the user had reached the GPU queue's two-running-job limit. **CUDA has not yet been claimed to pass. No upgrade training arm had been submitted at that observation.**

The scheduler reserves ngpus=1 on a host but exposes no physical UUID mapping. allocated_upgrade.pbs requests ncpus=8, ngpus=1, mem=48gb and a12h limit. Its operational binder checks the live PBS allocation, verifies an unoccupied physical GPU, acquires the existing shared project lock, records the physical UUID and CUDA binding, then invokes the **unchanged** CES_FL/cluster/upgrade_one_gpu.pbs (SHA256 c2723650c2839e625e286c93e15cd41d18bca5efdc1e3b8924d280e70f9321f3). It does not evict any process.

## Finite authorized execution

The user explicitly authorized these exact configurations after passing input preflight and actual CUDA smoke:

1. R0_frozen_bce
2. R1_partial_bce
3. R2_partial_dro
4. RN_partial_dro_nochange_pilot
5. RD_partial_dro_age_pilot

The finite login-node handoff executor was started as PID **3531362**. It checks the actual smoke PBS exit status, launch/binding evidence, the eight-test CUDA suite output and CUDA_SMOKE_PASS before submitting those five separate jobs. Each uses CES_SMOKE_ONLY=0, the matching unchanged configuration and a new output directory. It checks pinned bytes and existing jobs, records each real qsub response, stops the submission sequence on failed checks, never retries uncertain/failed submissions, and never generates images or evaluates test sets.

The stock approved launcher repeats real-input preflight and CUDA smoke before each training run. No other experiments or configurations are authorized by this executor. It observes/publishes only this request, for at most24h. The operational scripts are included here for review, not a request to start another executor. Do not run a duplicate.

Root: /home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_20261031T135409Z

Executor log: /home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_20261031T135409Z/logs/handoff_executor.log

Smoke PBS log: /home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_20261031T135409Z/logs/SMOKE_R1.pbs.log

## Validation publication and review limits

[Live execution status](../../../status/upgrade_invariance_20261031_r1_20261031T135409Z_live.md) and [aggregate evidence](../../../results/upgrade_invariance_20261031_r1_20261031T135409Z/aggregate.json) are published to hardikbha/CFES-BeatDing main. They record real job IDs/states, allocation/binding, software logs, epoch validation/subgroup metrics, group-DRO weights, pair exposure, image-forward counts, elapsed time, checkpoint hashes and failures when available.

After DONE, checkpoint review strictly reloads the selected model and compares its source/config/input identity, selected epoch, validation report and calibration metadata against the recorded AUC-only selection history. This does not independently replay the validation forward pass and does not certify counterfactual validity; the receipt labels that limit.

RN/RD are six-source diagnostics, not the expanded300-source study. Selection/calibration, approvals and manifests remain fixed. There is no automatic test evaluation, test tuning, forensic loss, map consistency, new generation, model download, job cancellation or protocol change.
