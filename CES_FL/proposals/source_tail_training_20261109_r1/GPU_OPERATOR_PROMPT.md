# Run the source-tail control/treatment pilot

Fetch the supplied exact commit from https://github.com/hardiksharma6555/CFES-BeatDing.git into a NEW detached worktree. Read CES_FL/AGENTS.md,WORKFLOW.md,this README/config and the actual run.py/source_tail_objective.py. Preserve existing jobs/results. This is an original-image source-tail experiment; do not mix it with original L_H probe/alignment losses or call it the full counterfactual method.

Run the CPU check first:
```bash
export ST_PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
export ST_PROP=CES_FL/proposals/source_tail_training_20261109_r1
export ST_ROOT=/home/sachin.chaudhary/ces_fl_runs/source_tail_training_20261109_r1
export ST_TEACHER=/home/sachin.chaudhary/ces_fl_runs/ff2_lh_ps_20261106_r1/L_A
"$ST_PY" "$ST_PROP/check_pipeline.py"
```

The pinned teacher SHA is cce4476491829702d60a74ed7b2044aa8436768d6e08e0129da5a1fd84b0cb6d. Verify actual paths/DONE/RUN_VERIFIED. The frozen teacher-core copy is a compatibility reference; do not execute its historical trainer as this new method. Reuse any already completed identical source-tail run; do not duplicate it.

Prepare a PRIVATE CSV with columns `image,source`, covering exactly all16,777 real TRAIN images and no others. Source values must be FFHQ or FairFace from the previously audited canonical TRAIN acquisition keys. Use the source-stratified audit's documented metadata mapping, not face appearance, predictions or guessed demographic labels. Preserve unknown/unresolved source evidence; incomplete mapping blocks this run. A PRIVATE provenance JSON must contain `source_csv_sha256`, `train_manifest_sha256` (4f83e7dc8042533329d7047fefd0f30cbd96f45d42bde2666dd0edc7d58f29cc), and `source_reference` identifying the verified derivation/code/source inventory. Report source counts3176FFHQ/13601FairFace and any mismatch before proceeding. Do not read mixed TEST/CAL master inventories to construct these TRAIN labels.

Refresh the existing FF2 ledger. Last paper snapshot64.6011/72charged is incomplete until historical TEST/sanity inference charges are reconciled. This request grants no extra hours. Use at most one actually allocated/bound GPU, cap ALL new GPU stages at2aggregate GPU-hours within verified available budget, and count held idle/check/export time. Do not displace other jobs or transfer original L_H funds. Use the established ledger/pool guard and scheduler/external deadline around EVERY CUDA command. The stage program does not itself reserve GPUs. The existing `with_gpu_budget.py` wrapper can be used if its detached-checkout/site binding requirements are met; never create another72hour ledger or disable a missing-binding check.

CPU-verify source mapping and metadata before holding a GPU. Estimate cache extraction on representative TRAIN batches with the pinned model and exact preprocessing; include image I/O and full SELECT replay. Preserve profiling evidence. If the full preparation/cohort/export cannot fit, report BUDGET_PENDING before launching it.

In the verified allocation, with exactly one visible GPU and private sidecars bound:
```bash
"$ST_PY" "$ST_PROP/run.py" prepare --teacher-run "$ST_TEACHER" \
  --source-csv "$ST_ROOT/train_real_sources_PRIVATE.csv" \
  --source-provenance "$ST_ROOT/source_provenance_PRIVATE.json" \
  --out "$ST_ROOT/cache" --device cuda
```

This preserves the complete teacher classifier and performs one shared encoder pass over TRAIN/SELECT. It uses checkpoint-bound image hashes and validates each actual image read. All cache blobs must finish and teacher validation must replay exactly before CACHE_VERIFIED is written. Do not turn an incomplete cache into a PASS by editing metadata; preserve failures. Allow severalGB of private disk for float32 features and checkpoint exports.

Run BOTH matched arms, with the same complete teacher initialization and cached features:
```bash
"$ST_PY" "$ST_PROP/run.py" train --cache "$ST_ROOT/cache" --arm control \
  --out "$ST_ROOT/control" --device cuda
"$ST_PY" "$ST_PROP/run.py" train --cache "$ST_ROOT/cache" --arm treatment \
  --out "$ST_ROOT/treatment" --device cuda
```

CPU head training is an option if it saves allocatedGPU time; record device/precision and keep the full native export replay gate. Both runs are exactly3epochs,seed17,batch32,AdamW headLR1e-4,weight_decay1e-4,clip5. The encoder is frozen throughout. Both use weighted BCE+fake guard(lambda_fake1); only lambda_source differs0versus0.1. Source references are fixed TRAIN-only85/90/95quantiles,IQR scale and pooled teacher soft risks. No JPEG, synthetic pairs, demographic inference or new coefficient/threshold sweep. Source auxiliary batches use16reals/source/update. Record all losses and source×joint exposure.

A stopped head run can resume with the SAME command plus `--resume`, after inspecting its own last.pt and accounting actual time. Do not restart with a new seed/config. An infeasible completed arm stays COMPLETED_NO_FEASIBLE_CHECKPOINT. Do not substitute an unqualified epoch or relax AUC99.5%/FNR0.5651%guards.

For each COMPLETED_VERIFIED arm, reconstruct and verify the full student:
```bash
"$ST_PY" "$ST_PROP/run.py" export --run "$ST_ROOT/control" --cache "$ST_ROOT/cache" \
  --teacher-run "$ST_TEACHER" --out "$ST_ROOT/control_student.pt" --device cuda
"$ST_PY" "$ST_PROP/run.py" export --run "$ST_ROOT/treatment" --cache "$ST_ROOT/cache" \
  --teacher-run "$ST_TEACHER" --out "$ST_ROOT/treatment_student.pt" --device cuda
```

Exact native validation metrics must match the selected cached-head metrics; report any floating-point/processing mismatch instead of declaring deployment PASS. No TEST or CAL inference is authorized by this pilot. No source labels or teacher branch are used by the exported student at inference.

Publish code/config/source/cache/checkpoint hashes, CPU/CUDA evidence, per-epoch logs, timing/ledger and both final outcomes. Use saved SELECT scores plus the existing verified SELECT source mapping for source-specific diagnostics; it is evaluation metadata only. Report all demographic supports/FPR/FNR/AUC/F_FPR and pooled utility, with fixed0.5 and stored VAL10%views separately. Use the audited contributor/master graph and fixed support rules for paired intervals; otherwise label point-only/inconclusive. The empty FFHQ×Others cells remain empty; no full standardized-race/joint estimate is manufactured.

Keep faces, source sidecars, features, row scores and model binaries private. Do not use the already-exposed TEST to choose a candidate or claim success from synthetic checks. Report whether source risk decreased, whether fake detection was retained, whether demographic disparities improved, and every regression. Stop after this cohort; no automatic scale-up or further tuning.
