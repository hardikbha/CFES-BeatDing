# CFES / Beat-Ding: fairness-aware deepfake detection vs Ding et al. (CVPR'26)

## Completed original-only comparison — 2 November 2026

[Verified P/BCE/fair completion and validation fairness comparison](CES_FL/results/manip_response_originals_s17_20261102/REPORT.md). Both downstream arms completed50epochs with native independent replay. BCE/fair validationAUC98.45/98.49%, but primaryF_FPR and FPRgap are worse for fair; no primary fairness win. Nochange/score/response remain blocked by reviewed manipulation quartets. Qwen224/224generated, no recorded pair approvals. No TEST or FairForge-2 run.

## P50 verified complete; matched original-only baselines launched

[Current per-arm state](CES_FL/status/manip_response_ssh_20261101T115720Z_live.md): P50 selected epoch29, validationAUC98.405989%, exact validation replay. BCE and fairness50epoch arms launched separately on gpu02 GPUs5/4 after actual CUDA backward PASS. Paired arms remain blocked by reviewed manipulation quartets. Qwen224/224generation complete, not independently validated. No test or FairForge-2 launch. This is a dated execution observation.

**[Shareable current supervisor handoff: progress, running jobs and complete code map](CES_FL/handoffs/SUPERVISOR_HANDOFF_2026-11-01_MANIPULATION_RESPONSE.md)** — verified snapshot 1 November 19:41 IST; refresh native/live-status evidence before acting.

## Direct SSH: new CUDA/input checks passed; P50 running

[Actual manipulation-response stage and arm status](CES_FL/status/manip_response_ssh_20261101T115720Z_live.md): all six CUDA software groups passed, all101,525TRAIN/VALimages decoded/hashed, configured backward PASS. Original-only P50 launched on gpu02GPU5 (training PID1393109); BCE/fair await P50, paired arms await reviewed quartets. Qwen continues separately on gpu01. No test or FairForge-2 launch. This is a dated observation.

## Priority manipulation-response experiment

[Actual per-arm status](CES_FL/status/manip_response_20261101T104925Z_live.md): exact source `58773bb`, new CUDA smoke **44963.mgmt01 queued with ngpus=1**; no new training yet. Original-only P/BCE/fair can proceed after their own checks. No-change/score/response are blocked by the matched manipulation bank. Existing Qwen32 continues separately. This is a timestamped launch observation; follow the linked status for updates.

## Authorized bounded Qwen quality pilot

[Executable 32-source quality diagnostic](CES_FL/proposals/qwen_bounded_quality_20261101_r2/README.md): six selection/isolation checks passed; 224 outputs planned, 4–6 generation hours after GPU binding. Training and full-bank generation remain stopped; no validity approvals are inferred. Actual launch and current state are recorded on its linked status page.

## Latest continuation: verified recheck, private review kit and bounded quality proposal

[New audit and proposed 32-source quality diagnostic](CES_FL/proposals/qwen_bounded_quality_20261101_r1/README.md). All 14 pilot outputs reverified; zero recorded approvals. No new training or generation launched. Proposed generation cost 4.33 GPU-hours before review/retries; this does not replace the full training bank. Observed UTC: 2026-11-01T09:43:09.620570+00:00.

## Shared progress and results

**Latest: [Qwen FF++ pilot completed—cost/validity report and scaling hold](CES_FL/results/cf5050_registry_recovery_20261101T081523Z/REPORT_QWEN_PILOT.md).** Fourteen outputs completed; full-bank projection ≈10,292 GPU-hours, no fairness gain established, no full training launched.

[New FF++ 50+50 gated execution](CES_FL/status/cf5050_20261101T065603Z_live.md) · [FairForge-2 supervisor verdict: DO NOT RUN](CES_FL/proposals/ff2_supervisor_review_20261101_r1/REVIEW.md)

**[Research updates and reviewer index](RESEARCH_UPDATES.md)** · [Current benchmark observations](CES_FL/status/benchmark_s17_20261031T112401_live.md) · [Verified FairForge-2 in-domain results](FAIRFORGE2_INDOMAIN_RESULTS_20261031.md)

[Completed upgrade: validation, 25 test evaluations and limitations](CES_FL/results/upgrade_tests_s17_20261101T040407Z/REPORT.md) · [Separate FairForge-2 public in-domain split and prepared plan](CES_FL/proposals/ff2_indomain_s17_20261101T042828Z/README.md)

As verified on 1 November 2026, all five upgrade arms completed ten epochs and all 25 authorized frozen-checkpoint evaluations completed. FairForge-2 in that evaluation batch measures FF++-trained transfer. The new FairForge-2 in-domain plan has its own verified public TRAIN/VAL/TEST manifests; training for that plan has not started. Read the report for mixed fairness findings, weak transfer performance and remaining control-validation requirements.

This repository is public. Reviewers can read the same versioned code, configurations, aggregate results and progress records without cluster access. Check each status file's observation time. The current benchmark has an existing finite publication monitor; other updates are published after verified milestones.

## Current supervisor entry point — CES-FL

**[Start with CES_FL/README.md](CES_FL/README.md)** · [Current observed status](CES_FL/STATUS.md) · [Supervisor prompt](CES_FL/SUPERVISOR_PROMPT.md) · [Review → push → cluster training workflow](CES_FL/WORKFLOW.md)

The standalone CES-FL model, actual completed seed 17 configurations, validation histories and verification receipts are versioned under `CES_FL/`. Use the dedicated benchmark observations linked above for that batch's evolving job states and test results; general status pages and handoffs are dated snapshots. Fairness findings remain exploratory. Read the recorded protocol and distinguish current CES-FL from the older archived methods before comparing results or proposing runs.

Historical comparison code plus selected aggregate progress/results; no binary checkpoints or raw images.
The archived agent paths mirror parts of the cluster. The current CES-FL working directory is outside this checkout;
new CES-FL runs should use the pinned repository source described below.

| Dir | What |
|---|---|
| `BEAT_DING/` | The "BD" recipe (Arm E + augmentation + score alignment + multi-layer evidence + group-AUC floor) aimed at beating Ding's published Tables 1-3; matched-operating-point eval, failure-case analysis, claim-fragility report. |
| `CESEL/`, `CESEL_MA/` | Counterfactual Evidence Selectivity model + multi-axis (gender/age/skin-tone) variant, and the priority task-queue runner that drives all GPU work across both nodes. |
| `CLIPFAIR/` | Matched-backbone baselines (DAW-FDD, DAG-FDD, PG-FDD, Ding SFD+GDA port, FairReL) trained on the same CLIP backbone as ours, for apples-to-apples comparison. |
| `DEMO_AXES/` | Demographic label pipeline: cleaner (AI-Face) vs noisy (Xu) labels, FairFace age/race/sex annotation, ITA skin-tone estimation, multi-axis re-scoring. |
| `DINGREPRO_LIN/` | Archived reproduction of Ding et al.'s official code (SFD+GDA); distinguish these local results from the paper's published numbers. |
| `FAIRREL_REPRO/` | Reproduction of FairReL (BMVC'26), a same-backbone baseline. |

See `decisions/*.md` for every pre-registered rule, and `STATUS_SNAPSHOT_27oct.md` for the project log as of this push.

## Running on the cluster
Compute nodes (gpu01/gpu02) have no internet; only the login node does. Workflow:
```
ssh <you>@mgmt01
cd ~/ICLR_2027/research/cvpr_readiness/r20261009/agents/<BEAT_DING|CESEL_MA|CLIPFAIR|...>
git pull origin main
qsub run/pbs_*.sh      # same launchers as before
```
**Caution:** several `run/*.sh` files are read live by running PBS jobs. Never let `git pull` edit one of those in
place while its job is running — check `qstat` first, or ask whoever manages the queue.
