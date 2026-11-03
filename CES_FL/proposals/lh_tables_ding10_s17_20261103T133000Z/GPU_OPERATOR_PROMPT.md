# Operator prompt: current L_H evaluation first; separate training work orders

Fetch exact supervisor commit on branch `experiment/lh-tables-ding10-s17-20261103T133000Z` from `https://github.com/hardiksharma6555/CFES-BeatDing.git` into a NEW detached worktree. Read CES_FL/AGENTS.md, WORKFLOW.md and this proposal's README, protocol and actual helper. Do not pull over active sources. Preserve all jobs and previous results. Current L_H and L_A science remains commit0d699d241e27f758ec0990fbe58493a15bdb0ff9/source08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177.

First inspect the existing `/home/sachin.chaudhary/ces_fl_runs/plain_head_s17_20261103T101418Z/L_A` and `L_H` native runs. The last GitHub observation has H9/10. Do not restart or force an epoch1checkpoint. Wait for actual10epochs/DONE/RUN_VERIFIED/native validation replay and keep the checkpoint selected by the original earliest maximum-AUC rule. Publish actual completion and the planned CPU analysis; keep private rows/images/checkpoints off GitHub.

The user clarified that a new Ding10epoch adaptation comparison is authorized. The earlier no-new-Ding instruction is superseded only for that comparison. Do not launch archived GDA with a changed epoch flag: the fairness gradient issue remains. The README's Ding/ResNet/schedule work orders require implemented and smoke-tested new adapters/configs before launch. They are not ready-to-run trainers in this commit. Record the exact inputs, differences and costs first. Existing shared budgets and allocations must be checked; do not spend FF2's72hours here.

For frozen Tables1/2 evaluation:

```bash
export CES_PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
export CES_EVAL=CES_FL/scripts/lh_locked_benchmarks.py
export CES_EVAL_PROP=CES_FL/proposals/lh_tables_ding10_s17_20261103T133000Z
export CES_EVAL_ROOT=/home/sachin.chaudhary/ces_fl_runs/lh_tables_s17_20261103T133000Z
"$CES_PY" -m unittest discover -s CES_FL/tests -p test_lh_locked_benchmarks.py -v
```

Create the fresh private output root. `runs.json` maps L_A/L_H to their actual completed native directories. `audits.json` maps ffpp/dfdc/celebdf/dfd to actual private JSON inventories containing `declared_overlap_values` with `identity_ids:fit:test` and `source_ids:fit:test`, plus explicit `declared_content_overlap_sha256` list. fit means TRAIN+VAL. Derive FF++ contributor declarations from the existing pinned TRAIN/TEST and VAL/TEST inventories; retain their provenance. Cross-domain audits must match documented benchmark inventories. Do not invent or silently broaden overlap exceptions. If the prior provenance is missing, publish that blocker. The evaluation checks actual recorded contributors and image bytes against these declarations; true biological-identity independence is not implied.

Before any TEST scores are produced, bind both checkpoints and every manifest/overlap hash:

```bash
"$CES_PY" "$CES_EVAL" bind --runs-json "$CES_EVAL_ROOT/runs.json" --audits-json "$CES_EVAL_ROOT/audits.json" --out "$CES_EVAL_ROOT/LOCKED_PLAN.json"
"$CES_PY" "$CES_EVAL" threshold-scan --scores /home/sachin.chaudhary/ces_fl_runs/plain_head_s17_20261103T101418Z/L_H/validation_scores_PRIVATE.json --out "$CES_EVAL_ROOT/H_validation_operating_points.json"
"$CES_PY" "$CES_EVAL" threshold-scan --scores /home/sachin.chaudhary/ces_fl_runs/plain_head_s17_20261103T101418Z/L_A/validation_scores_PRIVATE.json --out "$CES_EVAL_ROOT/A_validation_operating_points.json"
```

Report the full validation curves, not a cherry-picked target. Do not feed a preferred curve point into the locked TEST plan; native10%calibration and fixed.5 stay unchanged. Freeze all new convergence-training designs before reading these TEST outcomes, and never retune based on them.

Within a legitimate allocated GPU (ngpus=1, actual CUDA UUID/binding recorded), run the following ONCE for each arm L_A/L_H and dataset ffpp/dfdc/celebdf/dfd, sequentially if necessary. These variables deliberately enumerate the complete approved evaluation set:

```bash
for CES_ARM in L_A L_H; do
  for CES_DATASET in ffpp dfdc celebdf dfd; do
    "$CES_PY" "$CES_EVAL" infer --plan "$CES_EVAL_ROOT/LOCKED_PLAN.json" --arm "$CES_ARM" --dataset "$CES_DATASET" --out "$CES_EVAL_ROOT/inference/$CES_ARM/$CES_DATASET" --device cuda
  done
done
```

The helper refuses existing output paths. On interruption inspect each actual receipt, preserve partial failures, and resume only missing work in a newly recorded destination; do not rerun successful TEST inference or erase evidence. Release GPU reservations before CPU bootstrap. Then:

```bash
for CES_ARM in L_A L_H; do
  for CES_DATASET in ffpp dfdc celebdf dfd; do
    "$CES_PY" "$CES_EVAL" analyze --plan "$CES_EVAL_ROOT/LOCKED_PLAN.json" --arm "$CES_ARM" --dataset "$CES_DATASET" --inference-dir "$CES_EVAL_ROOT/inference/$CES_ARM/$CES_DATASET" --out "$CES_EVAL_ROOT/analysis/${CES_ARM}_${CES_DATASET}.json"
  done
done
```

Add a paired H-minus-A connected-contributor bootstrap using the two native scores_PRIVATE.json files for each dataset, with their stored thresholds. Do not compare unpaired seeds or change eligibility. Report1000draws, actual usable draws,900minimum and all sparse/missing cells. Compute fixed.5 paired contrasts separately using copies of threshold metadata in memory, never altering native score files. No extra inference is needed.

Publish Table1FF++ and Table2DFDC/Celeb-DF/DFD in Ding's layout: gender/race/joint each F_FPR↓,F_DP↓,es-AUC↑, then overall AUC↑; percentage units. Clearly separate paper-published Ding, archived reproduced Ding, new Ding10(if eventually completed), L_A and current L_H. Use exact method/backbone/seed/selection/threshold labels. Include fixed.5 and frozen-source10%views separately. Table3 L_H remains PENDING until the new ResNet study actually completes; never substitute older ResNet A/B/C rows. Publish aggregate metrics and code/config/checkpoint/source hashes in a new dated directory, plus execution/resource receipts and failures. Do not publish private rows, face images, weights or credentials.
