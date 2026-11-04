# FairForge-2 matched BCE/DAG-FDD/DAW-FDD wave

2026-11-04T05:18:36.508444+00:00 management UTC.

Scientific commit `815b4d4bc679c55d4ee6323cc7284335ce210568`. Dataset:218,266TRAIN /31,107VAL, seed17,50epochs,batch128,float32 full Xception/BN,SGD.

|Method|Actual status|Phase|Completed epochs|Selected epoch|Validation AUC %|Gender F_FPR|Race F_FPR|Joint F_FPR|
|---|---|---|---:|---:|---:|---:|---:|---:|
|bce|COMPLETED_VERIFIED|selected checkpoint independently replayed on validation|50|40|99.804621|0.034285|0.195413|0.405237|
|dag|COMPLETED_VERIFIED|selected checkpoint independently replayed on validation|50|50|99.813559|0.042760|0.183245|0.378911|
|daw|COMPLETED_VERIFIED|selected checkpoint independently replayed on validation|50|41|99.836011|0.034285|0.128659|0.306953|

Additional72h ledger: finished charge 28.010872GPUh; active reservations 42.000GPUh; active GPUs 2/3; uncommitted 1.989128GPUh. Active reservation is not consumed compute. Live held time and allocator bookends are preserved in session/allocation receipts; cleanup overhead is reconciled after allocation release.

Measured held time at the verified gpu01 clock: 28.414414GPUh including 0.403541h in active sessions. These are live elapsed durations, not reservation totals.

All7 native/parity tests and6 budget tests passed without skips. Each preflight independently hashed/decoded all249,373TRAIN/VAL images; native membership/group mapping, split provenance and byte-overlap checks passed. CUDA fixtures are software tests, not benchmark results.

Initial18h profiles passed software/real backward checks but returned BUDGET_PENDING before training. The documented24/21/23h resource continuation stays inside the same72h allowance with the scientific configs unchanged. Every continuation reruns the pinned checks/projection; no failed receipt is replaced.

PG-FDD and FairAdapter remain conditional on the specific gates in CONDITIONAL_METHODS_REVIEW.md. RSEF remains blocked. Existing jobs were preserved. No benchmark TEST access, new Ding training, generation, additional seed or hyperparameter search.

F_FPR is a fraction-unit sum of group-versus-pooled false-positive deviations. Overall FPR/FNR, all group supports/rates, age/skin axes and checkpoint hashes appear in LIVE_STATE when epochs/verified artifacts exist. Final CPU report uses1000 paired recorded-contributor bootstrap draws with fixed thresholds/eligibility. Single-seed intervals do not measure seed variability or prove identity independence.

This is a dated observation, not an unattended monitor. No new validation scores are claimed before a completed epoch.
