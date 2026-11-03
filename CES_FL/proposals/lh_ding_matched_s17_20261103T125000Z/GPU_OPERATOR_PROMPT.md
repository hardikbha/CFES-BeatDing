# GPU operator: CES-FL matched training-recipe preparation and conditional run

**Priority update:** this recipe is prepared and CPU-tested, but deferred while the current L_H result is finalized and diagnosed. See [LH_NEXT_STEPS_REVIEW.md](LH_NEXT_STEPS_REVIEW.md). Do not interpret these commands as a new automatic launch.

Continue from the exact supervisor commit supplied with this prompt on branch `experiment/lh-ding-matched-s17-20261103T125000Z`, published to `https://github.com/hardiksharma6555/CFES-BeatDing.git`. This package trains CES-FL only. Do not run, reevaluate, repair or alter Ding on FF++; the user has not answered the scope clarification authorizing one corrected Ding baseline. Do not touch existing jobs, source, datasets, results or preregistration. No TEST command is included.

Read CES_FL/AGENTS.md, WORKFLOW.md and this proposal's README, PROTOCOL_AUDIT, CHECKS, request, three configs and the actual adapter plus pinned science source. Preserve ongoing ten-epoch L_H and external FairForge-2 baselines. A Git push is not a job launch.

Create a NEW detached source worktree at the exact supplied SHA, not moving branch HEAD. Record the source/config hashes, package versions and scheduler state. Verify the new source manifest. Do not use an older P checkpoint: it has a different optimizer/sampling/head/selection recipe.

```bash
git fetch https://github.com/hardiksharma6555/CFES-BeatDing.git experiment/lh-ding-matched-s17-20261103T125000Z
CES_COMMIT=<exact supervisor SHA>
CES_SOURCE_DIR=/home/sachin.chaudhary/ces_fl_sources/lh_ding_matched_${CES_COMMIT}
git worktree add --detach "$CES_SOURCE_DIR" "$CES_COMMIT"
cd "$CES_SOURCE_DIR"
export CES_PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
export CES_MATCHED_SRC=CES_FL/src/ces_fl_matched_recipe.py
export CES_MATCHED_PROP=CES_FL/proposals/lh_ding_matched_s17_20261103T125000Z
export CES_MATCHED_RUN=/home/sachin.chaudhary/ces_fl_runs/lh_ding_matched_s17_20261103T125000Z
"$CES_PY" -m unittest discover -s CES_FL/tests -p test_matched_recipe.py -v
"$CES_PY" -m unittest discover -s CES_FL/tests -p test_head_ablation.py -v
"$CES_PY" "$CES_MATCHED_SRC" preflight --config "$CES_MATCHED_PROP/P_config.json" --out "$CES_MATCHED_RUN/pre_P"
```

Preflight is CPU-only and hashes/decodes fixed TRAIN/VAL, checks all six declared overlap keys compatibly, verifies supplied manifests/weights, and inventories actual steps and batch units. It does not read TEST. Preserve failure receipts; never edit hashes/splits to bypass a failure. Confirm against actual archived launch-source evidence whether Ding classification used the inspected224px import; the older report's256px claim conflicts with code.

Before reserving a GPU, record the explicitly remaining FF++ allowance. The separate72GPU-hours for FairForge-2 baselines are NOT assigned here. This proposal has no silently added total budget. If insufficient/unassigned, report that concrete blocker with the audit; do not launch full training. Acquire a legitimate scheduler allocation with ngpus=1 and verified GPU UUID/CUDA binding. One GPU is enough for P; at most two separate allocated GPUs may run A and H after P if the approved total ledger permits. Do not borrow another job's device. Record reserved idle time in the aggregate ledger.

Within the available profile allowance, use a fresh discarded24step model on the full batch. Set CES_PROFILE_HOURS and later cumulative per-run CES_P_HOURS/CES_A_HOURS/CES_H_HOURS from that explicitly approved ledger; these are execution allowances, not hyperparameters. The trainer checks budgets at epoch boundaries, so enforce hard time limits using the scheduler/ledger as well.

```bash
"$CES_PY" "$CES_MATCHED_SRC" profile --config "$CES_MATCHED_PROP/P_config.json" --preflight "$CES_MATCHED_RUN/pre_P" --out "$CES_MATCHED_RUN/profile_P" --device cuda --steps 24 --budget-hours "$CES_PROFILE_HOURS"
```

Report measured peak memory and estimate with40%margin before committing to the full study. No lowering batch, AMP switch, schedule shortening, optimizer substitution, seed sweep or automatic OOM retries. A low profile estimate is not an allocation or permission to consume another experiment's budget. Run P only after the explicit allowance covers it and downstream projected work has been accounted for:

```bash
"$CES_PY" "$CES_MATCHED_SRC" train --config "$CES_MATCHED_PROP/P_config.json" --preflight "$CES_MATCHED_RUN/pre_P" --profile-receipt "$CES_MATCHED_RUN/profile_P/PROFILE.json" --out "$CES_MATCHED_RUN/P" --device cuda --budget-hours "$CES_P_HOURS"
"$CES_PY" "$CES_MATCHED_SRC" bind-parent --template "$CES_MATCHED_PROP/A_template.json" --parent-run "$CES_MATCHED_RUN/P" --out "$CES_MATCHED_RUN/A_config.json"
"$CES_PY" "$CES_MATCHED_SRC" bind-parent --template "$CES_MATCHED_PROP/H_template.json" --parent-run "$CES_MATCHED_RUN/P" --out "$CES_MATCHED_RUN/H_config.json"
"$CES_PY" "$CES_MATCHED_SRC" preflight --config "$CES_MATCHED_RUN/A_config.json" --out "$CES_MATCHED_RUN/pre_A"
"$CES_PY" "$CES_MATCHED_SRC" preflight --config "$CES_MATCHED_RUN/H_config.json" --out "$CES_MATCHED_RUN/pre_H"
"$CES_PY" "$CES_MATCHED_SRC" profile --config "$CES_MATCHED_RUN/A_config.json" --preflight "$CES_MATCHED_RUN/pre_A" --out "$CES_MATCHED_RUN/profile_A" --device cuda --steps 24 --budget-hours "$CES_PROFILE_HOURS"
"$CES_PY" "$CES_MATCHED_SRC" profile --config "$CES_MATCHED_RUN/H_config.json" --preflight "$CES_MATCHED_RUN/pre_H" --out "$CES_MATCHED_RUN/profile_H" --device cuda --steps 24 --budget-hours "$CES_PROFILE_HOURS"
"$CES_PY" "$CES_MATCHED_SRC" preview --config "$CES_MATCHED_RUN/H_config.json" --preflight "$CES_MATCHED_RUN/pre_H" --out "$CES_MATCHED_RUN/preview_H"
```

The H profile must exercise every axis, including a finite nonzero hierarchical fairness gradient. It discards its weights and does not qualify procedural validity. H needs a source/config-bound named review of the preview. Prior review can be reused only if case membership/settings/image/sheet bytes match exactly using the existing audited reuse helper; do not transfer approval flags blindly. Otherwise inspect the images and complete `review_PENDING_PRIVATE.json` truthfully. Existing AI review must stay labeled AI. No fake reviewers or causal-validity claim. Then:

```bash
"$CES_PY" "$CES_MATCHED_SRC" qualify --preview "$CES_MATCHED_RUN/preview_H" --review /actual/completed/private/review.json --out "$CES_MATCHED_RUN/qualification_H.json"
"$CES_PY" "$CES_MATCHED_SRC" train --config "$CES_MATCHED_RUN/A_config.json" --preflight "$CES_MATCHED_RUN/pre_A" --profile-receipt "$CES_MATCHED_RUN/profile_A/PROFILE.json" --out "$CES_MATCHED_RUN/A" --device cuda --budget-hours "$CES_A_HOURS"
"$CES_PY" "$CES_MATCHED_SRC" train --config "$CES_MATCHED_RUN/H_config.json" --preflight "$CES_MATCHED_RUN/pre_H" --profile-receipt "$CES_MATCHED_RUN/profile_H/PROFILE.json" --qualification "$CES_MATCHED_RUN/qualification_H.json" --out "$CES_MATCHED_RUN/H" --device cuda --budget-hours "$CES_H_HOURS"
```

For a legitimate interrupted epoch-boundary run, inspect config/source/checkpoint/history and use the SAME command/output with `--resume`. Budget-hours is cumulative, not a reset. Partial-epoch crash outputs must be preserved and audited; never claim the attempted extra compute cost disappeared. No branching off a favorable intermediate epoch. Final epoch50 is selected prospectively, even if validation AUC was better earlier.

Require50epochs, DONE, RUN_VERIFIED and final checkpoint replay for each stage. Export all gender/race/joint/age/skin counts, AUC, FPR, FNR, F_FPR, FDP, es-AUC, gaps and worst-group AUC. Report fixed.5 and validation10%FPR separately, and show nonempty-group versus20/class-supported-group results explicitly. Compare A/H using paired connected-contributor bootstrap1000draws, fixed eligibility and their frozen thresholds. Existing900usable minimum is retained; report insufficiency, not fabricated significance. Core `compare` can consume the matching PRIVATE validation score files for each threshold view.

Publish aggregate status/metrics, epoch histories, source/config/checkpoint hashes, actual exposure/forward counts, compute ledger, scheduler evidence and failure receipts under a NEW dated result directory. Keep binary checkpoints, face images/review sheets, row-level scores and credentials private. Update project memory with the measured result. There is no TEST/evaluation launch in this package; submit a frozen common-evaluator handoff after training. Do not call the result an apple-to-apple win over published Ding while the documented baseline/seed/BN/evaluation issues remain.
