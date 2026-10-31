# CES-FL upgrade recovery — 31 October 2026

## Verified status

**The five upgrade training arms have not started. There are no new validation or fairness results.**

The first smoke job, **44880.mgmt01**, obtained ngpus=1 on gpu01 and exited with status 1 before CUDA software tests. The finite executor stopped all five training submissions. Its last publication recorded the failure; it is no longer running.

Exact failure: FileNotFoundError: [Errno 2] No such file or directory: git.

Failing location: /home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_20261031T135409Z/runtime/bind_and_run.py:31, at the first subprocess Git commit check. This was an error in the operational wrapper supplied for the handoff.

The missing job log was recovered without deleting the original from gpu01:/var/spool/pbs/undelivered/44880.mgmt01.OU. The full traceback is preserved in failed_smoke_diagnosis.json and at /home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_20261031T135409Z/logs/SMOKE_R1.recovered.pbs.log.

## Correction and actual checks

Git is not installed at its usual compute-node path. A byte-identical copy of the management node's /usr/bin/git is now available at /home/sachin.chaudhary/ces_fl_tools/git_cluster_20261031/bin/git. Its SHA256 is d220991337f55201c733e9ad80c11bec384680ec7b927a341a8cf03bb62d19a4. No packages or model weights were downloaded and no system installation was changed.

The new separate PBS wrapper adds that tool directory and /opt/pbs/bin to PATH. It also captures stdout/stderr directly into a shared live log, so future startup failures remain visible even if PBS leaves output in undelivered storage.

Executed CPU checks on **both gpu01 and gpu02**, using a minimal environment:
- Git, qstat, bash, nvidia-smi and tee resolve.
- Git resolves the exact source commit 875bce025084871438ef799ca9086985c4375672.
- The source is detached and its tracked CES_FL files are clean.
- The exact R1 configuration is tracked.

All five real-input preflights were rerun successfully. Seven operational guard tests passed with qsub mocked; their printed 99001–99005 IDs are synthetic. The prior 19-test CPU receipt remains valid for the unchanged research source and is explicitly retained, not described as a newly executed suite.

The official launcher, model source, five configurations, input hashes, approvals, selection/calibration and sampling budgets remain unchanged. The official launcher SHA256 remains c2723650c2839e625e286c93e15cd41d18bca5efdc1e3b8924d280e70f9321f3.

## Replacement execution

Recovery smoke job: **44906.mgmt01**, CES_SMOKE_ONLY=1, R1_partial_bce.json, ngpus=1, eight CPUs, 48 GB RAM.

Observed state: **QUEUED**, with scheduler comment: Not Running: User has reached queue gpu running job limit. Current GPU slots are occupied by 44861 and44881; neither was modified or terminated. CUDA remains **UNVERIFIED** for this upgrade.

Recovery output root:
/home/sachin.chaudhary/ces_fl_pipeline/experiments/upgrade_invariance_20261031_r1_recovery_20261031T165803Z

Finite executor PID: **1343127**. It will submit only the five previously authorized exact arms after actual recovery CUDA smoke success. It records real job IDs, stops on failed checks, avoids duplicate/uncertain submissions, and publishes validation/subgroup/exposure/checkpoint evidence as available. It does not retry failures automatically. This was a manual operational recovery during the user's follow-up.

[Current recovery observations](../../../status/upgrade_invariance_20261031_r1_recovery_20261031T165803Z_live.md) · [Aggregate evidence](../../../results/upgrade_invariance_20261031_r1_recovery_20261031T165803Z/aggregate.json)

## Limits

RN/RD remain six-source diagnostics. The historical CESel-MA gate is unchanged. There is no new generation, architecture/loss/configuration change, test evaluation or test tuning. Existing experiments and the failed first attempt are preserved.
