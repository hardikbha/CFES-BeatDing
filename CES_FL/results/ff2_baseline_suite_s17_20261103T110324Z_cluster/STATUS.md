# FairForge-2 matched BCE/DAG-FDD/DAW-FDD wave

2026-11-03T16:56:35.705523+00:00 management UTC.

Scientific commit `815b4d4bc679c55d4ee6323cc7284335ce210568`. Dataset:218,266TRAIN /31,107VAL, seed17,50epochs,batch128,float32 full Xception/BN,SGD.

|Method|Actual status|Phase|Completed epochs|Selected epoch|Validation AUC %|Gender F_FPR|Race F_FPR|Joint F_FPR|
|---|---|---|---:|---:|---:|---:|---:|---:|
|bce|RUNNING|training|18|17|99.343882|0.017336|0.045409|0.114056|
|dag|RUNNING|training|21|19|99.645234|0.032590|0.135666|0.290917|
|daw|RUNNING|training|17|15|99.520773|0.025811|0.148606|0.340415|

Additional72h ledger: finished charge 0.107034GPUh; active reservations 68.000GPUh; active GPUs 3/3; uncommitted 3.892966GPUh. Active reservation is not consumed compute. Live held time and allocator bookends are preserved in session/allocation receipts; cleanup overhead is reconciled after allocation release.

Measured held time at the verified gpu01 clock: 12.833401GPUh including 12.726366h in active sessions. These are live elapsed durations, not reservation totals.

All7 native/parity tests and6 budget tests passed without skips. Each preflight independently hashed/decoded all249,373TRAIN/VAL images; native membership/group mapping, split provenance and byte-overlap checks passed. CUDA fixtures are software tests, not benchmark results.

Initial18h profiles passed software/real backward checks but returned BUDGET_PENDING before training. The documented24/21/23h resource continuation stays inside the same72h allowance with the scientific configs unchanged. Every continuation reruns the pinned checks/projection; no failed receipt is replaced.

PG-FDD and FairAdapter remain conditional on the specific gates in CONDITIONAL_METHODS_REVIEW.md. RSEF remains blocked. Existing jobs were preserved. No benchmark TEST access, new Ding training, generation, additional seed or hyperparameter search.

F_FPR is a fraction-unit sum of group-versus-pooled false-positive deviations. Overall FPR/FNR, all group supports/rates, age/skin axes and checkpoint hashes appear in LIVE_STATE when epochs/verified artifacts exist. Final CPU report uses1000 paired recorded-contributor bootstrap draws with fixed thresholds/eligibility. Single-seed intervals do not measure seed variability or prove identity independence.

This is a dated observation, not an unattended monitor. No new validation scores are claimed before a completed epoch.

Detailed snapshot in Ding's column layout: [report and subgroup diagnostics](ding_layout_20261103T165635Z/REPORT.md), [PDF](ding_layout_20261103T165635Z/DING_LAYOUT.pdf), [LaTeX](ding_layout_20261103T165635Z/DING_LAYOUT.tex). It preserves this18/21/17-epoch baseline observation. A separate read-only check at17:13UTC verified the existing Ding job on two gpu01 GPUs:33/50classification epochs completed, epoch34active; SFD and corrected GDA remain pending. See the report for stage provenance and protocol differences.
