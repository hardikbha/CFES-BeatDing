# Verified FairForge-2 launch record

Observation:3November2026,12:46UTC management time. This records actual execution; later liveness requires a fresh process/receipt check.

Scientific commit `815b4d4bc679c55d4ee6323cc7284335ce210568`, fetched from `hardiksharma6555/CFES-BeatDing`, clean detached checkout `/home/sachin.chaudhary/ces_fl_sources/cesfl-ff2-suite-815b4d4`. Main adapter SHA256 `f14b818cad66c92a07f09e0c06767a2980a0b8644e8bb74b33710e142b971904`. All four source hashes and exact three config hashes/digests are in `evidence/SOURCE_AND_CONFIG.json`. No scientific source/config edits were made.

Private work root: `/home/sachin.chaudhary/ces_fl_runs/ff2_baseline_suite_s17_20261103T110324Z`. Upstream source-only trees: `/home/sachin.chaudhary/ces_fl_dependencies/ff2_baselines_pinned_20261103`. The exact fetch verified87DAG/DAW files,43PG files and97FairAdapter files by pinned Git blob; no weights were fetched. A working shared Git2.45.2 binary was verified on GPU nodes and added only to each worker's PATH. No fake Git commands or mutable source checkout were used.

## Actual CPU and input checks

Commands from the pinned checkout, with `/home/sachin.chaudhary/.conda/envs/aiface/bin/python`:

```
python CES_FL/proposals/ff2_baseline_suite_s17_20261103T110324Z/fetch_sources.py --root /home/sachin.chaudhary/ces_fl_dependencies/ff2_baselines_pinned_20261103
FF2_BASELINE_UPSTREAM=/home/sachin.chaudhary/ces_fl_dependencies/ff2_baselines_pinned_20261103/dag_daw python -m unittest discover -s CES_FL/tests -p test_ff2_baseline_suite.py -v
python -m unittest discover -s CES_FL/tests -p test_ff2_budget.py -v
python CES_FL/src/ff2_fairness_baselines.py preflight --config CES_FL/proposals/ff2_baseline_suite_s17_20261103T110324Z/configs/METHOD.json --out WORK/preflight_METHOD
```

`METHOD` above means the three separately executed commands for bce,dag,daw; `WORK` is the exact private root above. Seven native/official-loss/gradient tests passed in1.585s, zero skips; six budget tests passed in0.004s. Logs are published. Each independent preflight passed:218,266TRAIN /31,107VAL, all249,373image hashes/decodes, native/canonical membership and eight-cell mapping, no forbidden TRAIN/VAL image/path/recorded-contributor/byte overlap. All three inventories hash to `9f5d4b587d5dd87fe04a7db9b353c3aabc043b892c931dfd7f2b059112276621`. Actual TRAIN has16,777real/201,489fake; VAL2,438real/28,669fake. Aggregate demographic/method support and label provenance are in GROUP_SUPPORT.json. Recorded separation does not prove biological-person independence.

## Initial sessions preserved and reconciled

The three actual18h sessions `ff2_METHOD_gpu01_20261103T123610Z` all passed fresh CUDA DAW integration (native two-epoch training, bitwise mid-epoch resume, selected reload, coverage and frozen calibration), then20production TRAIN backward steps at full batch128. Each projected cost plus the fixed40% margin exceeded its initial session allowance, so all returned BUDGET_PENDING before benchmark training. The projected hours were BCE24.79695, DAG18.60845, DAW20.52174. Their logs, receipts and resource-gate results were preserved.

Actual initial task-held cost including reconciled allocation startup/cleanup:0.1070343257GPUh. No initial training directory/checkpoint existed. Resource plan `RESOURCE_RECONCILIATION_20261103T124200Z.json` was written before continuation, using the authorized remaining72h allowance. New reservations are24hBCE,21hDAG,23hDAW (68h total), leaving3.8929656743h uncommitted after the initial charge. The pinned worker permits up to24h/session. Its cumulative `--budget-hours` is an execution allowance; seed17,50epochs,batch128,optimizer,losses,selection and calibration are unchanged. No time was reset and no scientific candidate was retried.

The original18h site wrapper was preserved. A separate private `launch_pool_method_v2.py` adds explicit resource hours and a verified UUID filter only. It invokes the exact pinned `run_one.py`; it does not replace its CUDA/projection gates. Runtime wrapper hashes and commands are in launch receipts.

## Actual active training sessions

|Method|Session ID|Host/GPU|Controller PID|Pinned worker PID|New profile train-only h|With40% margin h|Reservation h|
|---|---|---|---:|---:|---:|---:|---:|
|BCE|ff2_bce_gpu01_20261103T124210Z|gpu01/6|3683232|3683278|6.33161|8.86426|24|
|DAG-FDD|ff2_dag_gpu01_20261103T124210Z|gpu01/4|3683248|3683309|6.48591|9.08027|21|
|DAW-FDD|ff2_daw_gpu01_20261103T124210Z|gpu01/5|3683267|3683293|6.67571|9.34599|23|

New sessions each reran the CUDA integration and20TRAIN backward checks. All passed, and all new projections fit their reservations. Native training subprocesses were observed alive under their corresponding workers. No full epoch had completed at this observation, so no benchmark validation score is reported yet. Estimates reflect current shared-node contention and input caching; they are not runtime guarantees or measured architecture improvements.

All three use actual, mutex-protected shared pool-v2 UUID leases under the user's previously explicit direct-SSH authorization. These are not claimed PBS jobs or exclusive physical GPUs. UUIDs are in each ALLOCATION.json; existing lease reservations and memory use were accounted before claiming. Existing Ding devices and the running plain-head task were excluded. Each new wave method occupies a distinct card; the shared72h ledger enforces at most3active GPUs and the cumulative cap.

Exact native invocation is recorded in each allocation COMMAND.json and session command files:

```
python -u PINNED_PROPOSAL/run_one.py --method METHOD --work-root WORK --sources SOURCES --ledger WORK/additional72_gpu_hours.json --allocation-record ACTUAL_ALLOCATION_JSON --session-id ACTUAL_SESSION_ID --hours HOURS
```

This is a readable summary of executed argv, not a placeholder allocation to submit. The published command receipts contain actual paths, session IDs and24/21/23hours. Float32 SGD0.001,momentum0.9,decay0.005,all original TRAIN rows once per epoch,full encoder/BN,identical ImageNet encoder and seeded head,earliest strict maximum VAL AUC,one global VAL-real empirical10%FPR threshold and fixed-logit0 diagnostics remain the accepted recipe. Each method has20,809,001total/trainable parameters. The20profiling updates were discarded; training freshly initializes from the pinned ImageNet weights.

## Logs and continuation

For each METHOD, the live native log is `WORK/METHOD/sessions/ff2_METHOD_gpu01_20261103T124210Z/train.log`; the session LAUNCH,PROJECTION and command JSONs bind it to the source/config/allocation. Native training outputs are `WORK/METHOD/training/`. Histories and selected checkpoint hashes appear there as actual epochs complete. Root controller logs are `METHOD_controller_budget24.log` forBCE, `dag_controller_budget21.log`, and `daw_controller_budget23.log`.

At completion verify DONE,RUN_VERIFIED,selected checkpoint hash and exact validation replay, then release devices before the pinned CPU1000draw report. Inspect an interruption before any exact-compatible resume. Do not reset source/config/cursors or erase failed evidence. Reconcile each allocation's recorded startup/cleanup bookends into the same72h ledger after worker release; actual held times and the reconciliation-required flag are preserved. No automatic retry or subsequent experiment is installed.

PG-FDD/FairAdapter remain blocked on the specific adapter/metadata/entrypoint gates in CONDITIONAL_METHODS_REVIEW.md. RSEF remains BLOCKED_SOURCE/DATA. No conditional GPU run was launched. No new Ding computation, TEST access, image generation, additional seed or hyperparameter search occurred. Existing Ding and plain-head jobs/results were left untouched.

The authenticated cluster account has verified write/admin access to `hardikbha/CFES-BeatDing`; the preparation environment's old fork-only permission statement is a historical snapshot. Aggregate evidence is published on upstream branch `results/ff2-baselines-815b4d4-20261103` in a separate publication checkout. Private image inventories, row-level scores, review faces and checkpoints remain on the cluster.
