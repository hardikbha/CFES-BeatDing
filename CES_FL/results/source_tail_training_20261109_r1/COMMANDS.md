# Actual commands, versions and checks

Scientific commit: `9bb8e8332b8a3aa78d83f74ef21bca0b7dcd6754`, fetched from `hardiksharma6555/CFES-BeatDing`. Initial detached linked worktree was created at `/home/sachin.chaudhary/ces_fl_sources/cesfl-source-tail-9bb8e83-20261109`. Its administrative .git metadata referenced login-local `/tmp`, so the first remote guard refused it before allocation. A self-contained detached shared clone at `/home/sachin.chaudhary/ces_fl_sources/cesfl-source-tail-shared-9bb8e83-20261109` uses the exact same commit/files/config and passed remote Git/source checks. This operational recovery did not bypass hashes or change the scientific implementation.

All stage argv, node-local timestamps, exit codes and private output paths are preserved in `*.RECEIPT.json` and `STAGE_COMMANDS.log`. Node clocks differ from login, so stage durations/compute use monotonic or same-node elapsed times rather than uncorrected cross-host timestamps.

## CPU qualification and inputs

```bash
export ST_PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
export ST_SOURCE=/home/sachin.chaudhary/ces_fl_sources/cesfl-source-tail-shared-9bb8e83-20261109
export ST_PROP=$ST_SOURCE/CES_FL/proposals/source_tail_training_20261109_r1
export ST_ROOT=/home/sachin.chaudhary/ces_fl_runs/source_tail_training_20261109_r1
export ST_TEACHER=/home/sachin.chaudhary/ces_fl_runs/ff2_lh_ps_20261106_r1/L_A
CUDA_VISIBLE_DEVICES='' "$ST_PY" "$ST_PROP/check_pipeline.py"
```

Actual output: supplied `SOFTWARE_CHECKS_PASS`; production/fixture separation, exact resume, coefficient-dependent parameters, encoder-feature/native-forward identity, teacher detachment, fake/source gradient directions and cache-tamper rejection passed. Synthetic utility numbers are not real benchmark results.

Read-only CPU input preparation verified checkpoint SHA cce4476491829702d60a74ed7b2044aa8436768d6e08e0129da5a1fd84b0cb6d, DONE/RUN_VERIFIED, selected epoch2, linear-head shape1×2048, existing head weights, fit membership249373 and exact TRAIN/SELECT manifest hashes. The prior audited canonical-key mapping produced an exact unique16777-row private TRAIN-real source sidecar (FFHQ3176/FairFace13601) and bound provenance. `CPU_INPUT_BINDINGS.json` records the aggregate outcome.

Historical inference accounting was reconstructed from local execution command timestamps, without rerunning or evaluating any historical model. Nineteen foreground GPU calls involved at most three observed cards between17:09:52Z and18:14:25Z on8November; a conservative3.227956GPU-hour charge counts all three cards throughout that entire window, including gaps. Earlier explicit CUDA-disabled CPU attempts were excluded. This is an upper bound, not asserted measured actual GPU compute. All older measured manual charges were preserved; missing ledger `state` fields were normalized to FINISHED for strict wrapper compatibility. `operator/reconcile_budget.py` and `BUDGET_RECONCILIATION.json` preserve the logic. This provided4.170942 conservative hours before the new2-hour cap; no budget was created/transferred.

## Actual guarded GPU cohort

The established prior user-authorized direct-SSH shared-pool-v2 capacity guard acquired one UUID lease on gpu01 physical GPU6. It honors live task reservations/protected jobs and does not infer exclusivity from idle utilization. One idle-only preallocation attempt failed without taking a lease; the successful run used the previously authorized shared route and available reserved memory. No existing job was displaced.

```bash
ssh gpu01 '/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u \
 /home/sachin.chaudhary/ces_fl_runs/source_tail_training_20261109_r1/runtime/lease_cohort.py \
 --source /home/sachin.chaudhary/ces_fl_sources/cesfl-source-tail-shared-9bb8e83-20261109 \
 --root /home/sachin.chaudhary/ces_fl_runs/source_tail_training_20261109_r1'
```

The actual launch used nohup/log redirection and retained PID3056234; inner budget wrapper3056321. All new CUDA operations ran with one visible UUID inside the exact `with_gpu_budget.py` wrapper,1.98h reservation plus an outer2h total lease deadline and measured startup/cleanup charge. `ALLOCATION.json` explicitly describes shared capacity, not PBS/exclusive ownership. Actual context: H10080GB, Torch2.7.0+cu126. No lingering source-tail lease/process remains.

The runtime qualification executed both supplied synthetic three-epoch arms on CUDA plus finite-head gradients/frozen encoder and full224 feature/native-forward identity. A fixed seeded1024-image real-TRAIN timing sample covered78real/946fake and all7families. Exact preprocessing/hash validation/batch32/workers4 with I/O yielded200.42images/s; margin projection0.7985GPU-h passed the frozen1.7h profile gate. Profiling disposable head updates did not alter production teacher/init.

Pinned stages called by the guarded cohort, verbatim scientific configuration:

```bash
"$ST_PY" "$ST_PROP/run.py" prepare --teacher-run "$ST_TEACHER" \
 --source-csv "$ST_ROOT/train_real_sources_PRIVATE.csv" \
 --source-provenance "$ST_ROOT/source_provenance_PRIVATE.json" --out "$ST_ROOT/cache" --device cuda

"$ST_PY" "$ST_PROP/run.py" train --cache "$ST_ROOT/cache" --arm control --out "$ST_ROOT/control" --device cuda
"$ST_PY" "$ST_PROP/run.py" train --cache "$ST_ROOT/cache" --arm treatment --out "$ST_ROOT/treatment" --device cuda

"$ST_PY" "$ST_PROP/run.py" export --run "$ST_ROOT/control" --cache "$ST_ROOT/cache" \
 --teacher-run "$ST_TEACHER" --out "$ST_ROOT/control_student.pt" --device cuda
"$ST_PY" "$ST_PROP/run.py" export --run "$ST_ROOT/treatment" --cache "$ST_ROOT/cache" \
 --teacher-run "$ST_TEACHER" --out "$ST_ROOT/treatment_student.pt" --device cuda
```

All five production stages exited0. Shared cache exact teacher replay PASS; both three-epoch arms COMPLETED_VERIFIED and selected epoch1; both native full-student exports PASS with exact validation-report equality. All image bytes were checked against checkpoint-bound hashes by the pinned reader during the authorized cache/exports. No partial cache or tolerance relaxation was accepted.

One-GPU held time0.401479714h, including profiling/checks/cache/head-training/exports/I/O/bookends. Existing shared ledger now68.230537680/72 charged,3.769462320 remaining; no automatic extension or use of that balance. GPU overhead and old conservative charges are distinguished explicitly.

## CPU final reporting, no additional encoder passes

From the results worktree, with CUDA visibility empty and numerical CPU thread counts2:

```bash
ST_OUT=/home/sachin.chaudhary/ces_fl_sources/cesfl-source-tail-results-20261109/CES_FL/results/source_tail_training_20261109_r1
CUDA_VISIBLE_DEVICES='' "$ST_PY" "$ST_OUT/operator/report.py" --source "$ST_SOURCE" --root "$ST_ROOT" --out "$ST_OUT"
CUDA_VISIBLE_DEVICES='' "$ST_PY" "$ST_OUT/operator/extended_metrics.py" --root "$ST_ROOT" --out "$ST_OUT"
CUDA_VISIBLE_DEVICES='' "$ST_PY" "$ST_OUT/operator/verify_outcomes.py" --source "$ST_SOURCE" --root "$ST_ROOT" --out "$ST_OUT"
/usr/bin/python3 "$ST_OUT/operator/build_summary.py"
```

`report.py` reads saved SELECT predictions and uses CPU linear-head arithmetic on existing TRAIN features solely for full-TRAIN descriptive source/fake-guard diagnostics; it does not rerun the encoder. Paired recorded-contributor bootstrap uses2000draws/seed17, fixed eligible groups/thresholds and1900-valid minimum. Primary joint CI unavailable at999valid; other reported intervals include zero. No imputed source cells or frame-i.i.d. uncertainty.

Independent extended aggregation checks teacher/control/treatment metrics against native reports at1e-12. A teacher-only float32 scalar comparison would round a nextafter float64 threshold and change one boundary decision; this reporting issue was detected and fixed before publication by explicit float64 decisions, matching the unchanged evaluator. Original experimental metrics/thresholds were not modified. Private initial presentation attempt retained.

`verify_outcomes.py` actually passed nine integrity/behavior categories: all nine cache hashes; exact complete-teacher head; TRAIN-only reference recomputation; actual steps/coverage/fixed selection; both exports and274 unchanged encoder/BN tensors; source coefficient changes trained head; identical auxiliary draws; independent presentation/native equality; actual charge inside both caps. It reconstructs unique auxiliary coverage from the fixed schedule without resampling a new training run.

Final report is a validation-only negative/inconclusive result for meaningful fairness gain. No TEST/CAL raw data, new Ding computation, threshold tuning, model search, image edits or original L_H intervention losses. Full binaries/features/face rows/source sidecars are private. The entire original source and result history is preserved.
