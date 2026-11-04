# Run the bounded ResNet nuisance pilot

Fetch the exact supplied commit from `hardiksharma6555/CFES-BeatDing`, branch `experiment/lh-nuisance-pilot-20261104-r1`, into a new detached worktree. Read CES_FL/AGENTS.md, WORKFLOW.md and this proposal's README, request and CHECKS. Preserve all current jobs, original models and results. This package has runnable code/configs; no new pretraining is needed.

First refresh scheduler/direct-lease state and the correct shared study ledger. Reserve at most3aggregate GPU-hours for this WHOLE pilot, including diagnostics/checks/idle/replay, and at most2authorized devices. Do not take FF2baseline budget or assume free VRAM means ownership. Bind CUDA_VISIBLE_DEVICES to the actual reserved UUID BEFORE importing PyTorch, verify the process/device mapping, and record CES_GPU_BINDING_RECEIPT. Enforce both the global ledger/deadline and native per-run cap; abort incomplete at a safe epoch boundary if projected work no longer fits. No automatic extension or additional candidate.

## 1. CPU preparation and source diagnostics

Set a NEW private run root and use the cluster Python environment with existing packages:

```bash
export CES_PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
export CES_RUN_ROOT=/home/sachin.chaudhary/ces_fl_runs/lh_nuisance_pilot_20261104_r1
export CES_STAGES=CES_FL/proposals/lh_nuisance_pilot_20261104_r1/commands.sh
"$CES_PY" -m unittest discover -s CES_FL/tests -p test_lh_nuisance.py -v
bash "$CES_STAGES" preflight
bash "$CES_STAGES" preview H0
bash "$CES_STAGES" preview H1
```

The configs pin ResNet P50 SHA `c28bbb9cfcd06a11894027dc9369c5fd674f640de6c182c28b6d6f0aa8d77f55` and the original TRAIN/VAL hashes. Verify them exactly. The native fixture smoke uses synthetic files labeled train/val/test for software isolation only; never substitute benchmark TEST files.

Inspect the emitted private previews and complete genuine named reviews. H1 covers all four common-processing cases; source-bound qualification must reflect those cases. Do not use the old hard-coded reuse script on changed preview bytes, edit hashes or auto-fill approvals. Record AI reviewer limitations honestly. If processing erases the intended probe or introduces confounds, record the failure and stop that dependent arm without dropping sources. Bind completed reviews:

```text
bash "$CES_STAGES" qualify H0 /actual/completed/H0_review_PRIVATE.json
bash "$CES_STAGES" qualify H1 /actual/completed/H1_review_PRIVATE.json
```

On a verified device, first run the zero-update diagnostic on the ORIGINAL selected ResNet L_H checkpoint, locating its actual path through the prior native run receipts:

```text
"$CES_PY" CES_FL/scripts/diagnose_lh_nuisance.py --mode checkpoint --model-family original --backbone resnet50 --checkpoint /actual/original/ResNet_L_H/best.pt --checkpoint-sha256 b08c2b5180d999f81e044872a8228f7e876d0250a6cb29034d59b11b1a897f10 --device cuda --out "$CES_RUN_ROOT/diagnostic_original_resnet_H"
```

Optionally run the same diagnostic for original Xception L_H, checkpoint `6ef2caad3979d101fd94d7350aa3c1a50f95ee8307673cfaae63ac9d89af36f0`, only if it fits the same3hourcohort cap. No new Xception training. These read TRAIN only and make zero optimizer updates. Publish aggregate response/nuisance/gradient summaries; keep rows_PRIVATE.json and faces private. Actual model-level behavior is unverified until these commands complete.

## 2. Four matched short runs

All arms use seed17, the SAME P50 encoder, identical fresh linear heads,3epochs,batch32,layer4training,frozenBN,headLR1e-3,encoderLR1e-5 and unchanged original loss coefficients. Do not initialize from trained L_H or adopt SHAM-FIX/NO-ALIGN. A0/H0 are original processing; A1/H1 use the fixed common nuisance bank. Every original is visited once/epoch. A1/H1 augmentation draws match per image and epoch; auxiliary processing is identical across each event's groups/classes.

After verified device allocation and setting CES_GPU_BINDING_RECEIPT, for each ARM in A0,H0,A1,H1:

```text
bash "$CES_STAGES" checks ARM
bash "$CES_STAGES" train ARM
```

Replace ARM literally with the intended name. Profile CUDA smoke and the real-data dry-run first; ensure the complete remaining cohort fits. At most two arms may run concurrently on separately verified UUIDs. No source mutation during a run. Source-specific CUDA smoke and configuration-specific real backward receipts must pass. Keep failures and all source/config/checkpoint/lease hashes. Publish meaningful epoch progress; do not launch an unattended retry loop.

## 3. Compare and diagnose the selected models

Require all3epochs, DONE, RUN_VERIFIED and native validation replay per completed arm. Preserve earliest maximum clean-validation-AUC selection and the original threshold policy; no fairness-based checkpoint replacement. Run CPU comparison after releasing GPUs:

```bash
bash "$CES_STAGES" compare
```

The report checks matched configs, validation membership, source/weight hashes, exposure and main augmentation counts. Report A1−A0,H1−H0,H1−A1 and the descriptive factorial interaction, every demographic metric, AUC/FPR/FNR, worst groups, all supports and regressions. Preserve900/1000usable paired-bootstrap support; no significance or equivalence claim from unsupported intervals. The interaction helper provides a point estimate only.

If remaining cohort time permits, repeat the checkpoint diagnostic for H0 and H1 using `--model-family pilot`, their newly verified checkpoint hashes and fresh outputs. At least report those runs' actual score-scale/response/gradient diagnostic availability; do not substitute original-model diagnostics for candidate evidence. Check whether nuisance sensitivity decreases while procedural responses remain measurable, and whether a larger score scale explains apparent sensitivity gains. Pixel difference retention is a diagnostic, not a validity certificate.

No benchmark TEST reads, Celeb-DF-specific tuning, extra epoch/seed/augmentation strength, new loss or automatic follow-up. Existing Celeb-DF TEST tables remain frozen; this pilot is source-domain development after prior TEST exposure. It does not establish cross-domain generalization or feature disentanglement by itself.

Push code/config/source provenance, actual CPU/CUDA receipts, aggregate diagnostics, full validation histories, comparison and GPU ledger to a new dated results directory. Keep private data/checkpoints off GitHub. Return the exact publication commit and helpful/harmful/mixed/inconclusive assessment. A failure to improve is a result; do not modify settings until a separate request.
