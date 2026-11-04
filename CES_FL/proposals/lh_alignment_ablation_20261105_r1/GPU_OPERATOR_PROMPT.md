# Operator: repeated alignment audit, then one conditional C1 run

Fetch the supplied exact commit from `https://github.com/hardiksharma6555/CFES-BeatDing.git`, branch `experiment/lh-alignment-ablation-20261105-r1`, into a new detached worktree. Read CES_FL/AGENTS.md, WORKFLOW.md, this proposal's README/protocol/CHECKS. Preserve existing jobs/results/reviews. The scientific trainer and H1 config are unchanged; the candidate uses native armC to omit response alignment only.

This is FF++ TRAIN/validation work, not the blocked FairForge-2 study. Do not modify FF2reviews, claim new human approval, access benchmark TEST, generate new intervention images or train another candidate/seed.

## 1. Bind existing evidence on CPU

Set a new private root and copy `references.template.json` there. Fill its five paths using actual prior native receipts: completed H0/H1/A1 run directories, H1 preflight directory and H1 qualification JSON. Do not guess paths or edit any existing artifact. The exact expected source/config/checkpoint hashes are in protocol.json.

```bash
export CES_PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
export CES_C1_ROOT=/home/sachin.chaudhary/ces_fl_runs/lh_alignment_ablation_20261105_r1
export CES_C1=CES_FL/scripts/lh_alignment_followup.py
"$CES_PY" -m unittest discover -s CES_FL/tests -p test_lh_alignment_followup.py -v
"$CES_PY" "$CES_C1" bind --references "$CES_C1_ROOT/references.json" --out "$CES_C1_ROOT/BINDING_PRIVATE.json"
```

Binding verifies the existing352-case H1 review against canonical TRAIN selection and all hashes. It does not regenerate sheets, invent approvals or certify forensic validity. If anything is absent, stale, rejected or inconsistent, preserve the failure and stop. Do not rerun prepare/preview/qualify or transplant a FairForge-2 review.

## 2. Reserve and verify one GPU; run the fixed diagnostic

Refresh the actual shared study ledger and existing allocations. The entire follow-up has a maximum1aggregate GPU-hour, including loading/checks/idle/training/replay; no extra total budget is granted. Native C1training has a0.75hourcap. Use an external deadline/ledger guard, reserve enough for the complete remainder, and preserve all other jobs. Do not use FF2baseline hours.

Set CUDA_VISIBLE_DEVICES to the actual allocated/authorized UUID BEFORE Python imports PyTorch. Record the scheduler/lease evidence and actual process-device mapping in CES_GPU_BINDING_RECEIPT. An idle device or a receipt filename is not ownership proof. Use the original H1 aiface environment; the helper checks the recorded PyTorch/CUDA version.

```bash
"$CES_PY" "$CES_C1" diagnose --binding "$CES_C1_ROOT/BINDING_PRIVATE.json" --device cuda --out "$CES_C1_ROOT/diagnostic"
```

This audits H0/H1 at20prespecified TRAIN events under identity/resize90, with zero optimizer updates. Publish all aggregate gradients/responses, including null or contradictory findings; keep TRAIN_PLAN_PRIVATE.json private. Treat pixel attenuation and head-normalized responses as diagnostics, not proof of preserved forensic cues.

Read DIAGNOSTIC.json. If `decision.run_C1` is false, stop and report; no alternative diagnostic sample, threshold, condition or candidate is authorized. If true, it only supports doing the ablation—not a claim of causal harm or statistical significance.

## 3. Fresh C backward check and three-epoch ablation

```bash
"$CES_PY" "$CES_C1" cuda-check --binding "$CES_C1_ROOT/BINDING_PRIVATE.json" --diagnostic "$CES_C1_ROOT/diagnostic/DIAGNOSTIC.json" --out "$CES_C1_ROOT/dry_C1"
```

Locate the ORIGINAL full H1 native CUDA software-smoke receipt with matching source/head/nuisance-mode/runtime. A summarized public receipt that omits fields is not a substitute. It can be reused because the scientific source/config/views are unchanged and the local C-path tests plus fresh native C backward check specifically cover the new arm. If missing or stale, run the unchanged trainer's matched-mode CUDA smoke in a new output directory only if it fits the same cohort cap. Never modify an old receipt.

Compute remaining allowance after actual diagnostic/check time and projected three-epoch training/replay. Set CES_C1_REMAINING_HOURS to a verified positive value no greater than0.75and the remaining cohort/ledger allowance. Run once from the same P50/fresh seed17head:

```text
"$CES_PY" "$CES_C1" train --binding "$CES_C1_ROOT/BINDING_PRIVATE.json" --diagnostic "$CES_C1_ROOT/diagnostic/DIAGNOSTIC.json" --dry-run-receipt "$CES_C1_ROOT/dry_C1/DRY_RUN.json" --software-receipt /actual/prior/H1/smoke/SMOKE_RESULT.json --budget-hours "$CES_C1_REMAINING_HOURS" --out "$CES_C1_ROOT/C1"
```

Keep all settings identical to H1. The CLI arm is C; do not set lambda_alignment=0 in the config, because the pinned validator requires positive coefficients and armC already omits alignment. Do not initialize from trained H1, select a fairness-best epoch or extend beyond3epochs. Preserve any incomplete result; no automatic restart/resume.

## 4. Release GPU, compare validation on CPU and publish

Require three epochs, DONE, RUN_VERIFIED and native validation replay. Then:

```bash
"$CES_PY" "$CES_C1" report --binding "$CES_C1_ROOT/BINDING_PRIVATE.json" --candidate "$CES_C1_ROOT/C1" --out "$CES_C1_ROOT/comparison"
```

The report verifies identical source/augmentation exposure and zero alignment loss, compares C1−H1/C1−A1, and retains1000draw/900usable uncertainty requirements. Report all group metrics, AUC/FNR and regressions. A favorable point result with inadequate primary uncertainty remains inconclusive. No TEST inference follows this pilot.

Publish only aggregate diagnostics/results, exact source/config/checkpoint/launch hashes, real receipts, progress/failures and final GPU accounting in a new dated directory. Keep private paths/images/review rows/caches/weights private. Return the exact commit and whether the diagnostic warranted C1, whether C1 helped under the fixed primary/utility guards, and remaining limitations. No SOTA, disentanglement or real-forgery validity claim follows automatically.
