# GPU operator prompt: plain Xception head, matched BCE and H

Carry out the user-authorized head ablation from the exact commit supplied with this prompt on branch `experiment/cesfl-plain-head-s17-20261103T101418Z`. Do not substitute `main` or an unpinned latest branch. This is two CES-FL FF++ runs, L_A and L_H, seed17. It is not a request to rerun Ding, generate edits, tune hyperparameters or access TEST.

Read `CES_FL/AGENTS.md`, `WORKFLOW.md`, `PROJECT_MEMORY.md`, this proposal's README/CHECKS, actual source and both configs. Inspect existing jobs/results first. Preserve the current gated H run, all previous source/checkpoints/results and other users' jobs. A stale GitHub RUNNING status is not live verification.

## 1. Pin source and check the resource ledger

Fetch the provided exact commit into a NEW detached worktree. Never pull over active source. From a clean Git checkout:

```bash
CES_COMMIT=<exact SHA supplied by supervisor>
git fetch origin experiment/cesfl-plain-head-s17-20261103T101418Z
CES_SOURCE_DIR=/home/sachin.chaudhary/ces_fl_sources/plain_head_s17_${CES_COMMIT}
git worktree add --detach "$CES_SOURCE_DIR" "$CES_COMMIT"
cd "$CES_SOURCE_DIR"
git rev-parse HEAD
git status --short
export CES_PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
export CES_RUN_ROOT=/home/sachin.chaudhary/ces_fl_runs/plain_head_s17_20261103T101418Z
CES_COMMANDS=CES_FL/proposals/plain_head_s17_20261103T101418Z/commands.sh
```

Use a genuinely allocated H100 and record scheduler allocation/GPU UUID/CUDA binding. An idle GPU is not an allocation. Request ngpus under the site's scheduler when obtaining a new job. Do not stop/restart existing jobs, borrow Ding's device or invent a binding. One GPU is enough; avoid holding an unused second allocation.

Inspect the actual remaining original24aggregateGPU-hour ledger. This proposal allows at most6additional aggregateGPU-hours including CUDA smoke, dry runs, training and VAL replay, and never exceeds the remaining ledger. Last observed H publication had10.5006hours remaining at2026-11-03T10:19:59Z; refresh that. Record all task-held GPU time, including idle reserved time. CPU preflight/preview/bootstrap need no GPU reservation when run separately. commands.sh is an explicit command convenience, not a scheduler or total-budget enforcer; you must enforce both caps and log each stage.

## 2. Check code and inputs before training

Source must hash to `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`. Verify the two config hashes in `evidence/config_comparison.json`. Do not edit configs to bypass input errors. Record Python/PyTorch/CUDA versions, commit, source/config/initial-checkpoint hashes and actual commands.

Run the11new focused tests and relevant regression tests from CHECKS.md. CUDA/runtime dependencies must already exist; no automatic model download. Ensure enough private disk space for synthetic/checkpoint artifacts. Keep a failed receipt if a check fails.

Within the legitimate allocation, for a fresh root:

```bash
bash "$CES_COMMANDS" prepare
```

This hashes/decodes both fixed TRAIN/VAL inputs, checks the unchanged six-key split audit, executes fresh linear CUDA smoke, and performs one discarded real-input optimizer step for each config using the production TRAIN-real calibration rule. It does not train a benchmark epoch. For CPU preflight before allocation, use the exact `preflight` commands inside the script, then invoke its remaining commands individually instead of rerunning `prepare` over existing outputs. A partially failed preparation is inspected and continued stage by stage in fresh destinations; do not erase it.

Required PASS: correct2048→1 head;20,809,001total/6,790,433trainable; finite head/exit gradients; frozen BN; source/head/config/device-bound smoke/backward receipts; correct original membership, prior checkpoint and declared overlaps. L_H dry run must show a finite nonzero changed fairness gradient. A source/device mismatch or missing data is a blocker, not permission to fabricate receipts.

## 3. Obtain L_H procedural qualification without new fabricated approvals

```bash
bash "$CES_COMMANDS" preview-h
bash "$CES_COMMANDS" reuse-review-h /actual/private/path/to/original_H/qualification.json
```

Locate the original H qualification from native launch/provenance evidence; do not guess a file. The helper permits reuse only if all case IDs, image/sheet bytes, settings and metadata match the old pinned H review and fresh new-head preview. It preserves every reviewer name, flag and evidence note and records that this is reuse, not new review. Existing AI review stays labeled AI and has only procedural-recipe scope.

If it rejects, preserve the failure. Actually inspect the fresh preview sheets and record a substantive named review using `review_PENDING_PRIVATE.json` as the schema, or leave L_H blocked. Never auto-fill approval. Run the source's `qualify --preview ... --review ... --out "$CES_RUN_ROOT/review_H/qualification.json"` only after a valid completed review exists, using a fresh review directory if a failed attempt already created one. No source drops, recipe changes or reduced coverage are authorized. L_A can proceed independently.

These probes are local procedural mismatches with matched shams, NOT age/gender/skin edits or validated real-world forgery removal. Keep that scope in reports.

## 4. Train only the two plain-head arms

```bash
bash "$CES_COMMANDS" train-a
bash "$CES_COMMANDS" train-h
```

L_A uses native armA and BCE only. L_H uses native armD with `error_control=hierarchical_real`: the SAME H objective, not older D's legacy error control. Both use the SAME P50 encoder, fresh seed17 linear head, ten epochs, batch32, exit-block tuning, frozen BN, same optimizer rates, all original exposure and unchanged auxiliary schedule. No additional pretraining, extra epoch/seed, full-encoder switch, group-specific threshold, lambda change or fairness-selected checkpoint.

Native per-arm allowance is2hours at epoch boundaries. If interrupted, inspect last.pt/history/source/config/qualification and use `resume-a` or `resume-h` for the SAME run; a compatible resume does not create another scientific candidate. A documented cumulative `--budget-hours` extension to finish the same ten epochs is allowed only within the6hcombined/remaining24hledger caps. Never reset elapsed accounting. If unable to finish within caps, report incomplete rather than quietly changing epochs or budget.

Publish aggregate progress after meaningful milestones: exact epoch, selected-so-far AUC/fairness, elapsed compute, receipts and failures. Do not call interim results final. No unattended service or new automatic runner is requested.

## 5. Verify and compare fixed validation results

Require ten complete epochs, DONE, RUN_VERIFIED.json and exact native selected-checkpoint replay for each new arm. Selection remains earliest strict highest VAL AUC. Each selected checkpoint retains its own global validation-real10%FPR threshold. Inspect original and real-pair exposure plus group support. Do not invoke `evaluate`: that command is for external/TEST evaluation and is outside this request.

Wait for actual completed/verified gated H before the final H head comparison. Gated A is already completed; locate its native run via provenance. Existing correct private VAL scores may be reused after checking checkpoint/manifest/row hashes. If they are absent or need verification, these helpers replay VAL only and never recalibrate:

```bash
bash "$CES_COMMANDS" replay-a /actual/private/completed/gated_A_run
bash "$CES_COMMANDS" replay-h /actual/private/completed/gated_H_run
bash "$CES_COMMANDS" compare
```

Reference replays demand exact old config/source,10epoch history, correct selected checkpoint and exact stored validation metrics. If they disagree, preserve the failure and diagnose; do not rewrite historical scores or reselect a checkpoint. Where a verified existing score file is reused, place a provenance-recorded copy in the reference_A/reference_H PRIVATE locations expected by commands.sh and preserve its original hash.

Create one aggregate table with four rows: gated A, linear A, gated H, linear H. Show selected epoch, VAL AUC, overall FPR/FNR, each gender/race/joint/age/skin F_FPR, gaps and worst-group AUC, plus actual total/trainable parameters and runtime. Include full group denominators/FPR/FNR/AUC and fixed0.5 diagnostics separately. Report linear-minus-gated deltas under each objective and descriptive H-minus-A deltas within each head; do not cherry-pick axes.

Use1000paired connected-contributor bootstrap draws with seed17, frozen eligible groups and thresholds. Report usable counts for every interval. Existing FF++ primary support gives746/1000usable, below inherited900minimum; keep that limitation. A nonsignificant difference is not proof of equivalence. One training seed cannot establish seed stability. VAL-only findings cannot claim beating Ding's TEST tables or a paper-ready SOTA result.

## 6. Publish truthful evidence and stop

Push only code-free aggregate result summaries, run/provenance hashes, receipts without private rows, compute ledger and a final status under a NEW dated result directory. Keep all checkpoint binaries, images/reviews and full row-level scores private. Do not overwrite old results/preregistration or scientific source. Record the actual final commit and source SHA. Update PROJECT_MEMORY with evidence-linked outcomes and remaining blockers; no automatic next experiment or TEST run.

Finish by stating whether the plain head helped, hurt or remains inconclusive under BCE and H separately, why the evidence supports that interpretation, and whether Part C can reasonably be simplified for the next proposal. No guaranteed-win language.
