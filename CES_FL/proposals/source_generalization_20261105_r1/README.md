# TRAIN-only source generalization proposal — 5 November 2026

## Hypothesis

CES-FL's validation performance may depend on having seen the same manipulation family during original-only pretraining. Test this with one fixed leave-one-family-out experiment on FF++ TRAIN metadata. This is a bounded robustness diagnostic; it is not a fairness intervention and it does not establish broad cross-dataset generalization.

## Fold rule

Input is only `ces_fl_pipeline/manifests/ding_ffpp/train.csv`, SHA256 `6ee9326fe8361dc3aee0f803c49b5eada169be5cf3bc3cb44ad3d934ba5ca657`. Rows are connected by namespaced recorded `identity_ids`, `source_ids`, or image identity; this graph is a provenance proxy and is not verified ground-truth person identity. The lexicographically first supported known fake family is selected using the fixed rule of at least 20 fake rows in at least two recorded components. The rule selects **Deepfakes** (12,504 TRAIN fakes in 270 recorded components).

All components containing a Deepfakes fake are removed from fitting. Among the remaining 201 components, 41 (20%, rounded up) are deterministically selected for development by a seed-17 SHA256 rank. Fit uses the other 160 components. Development contains all Deepfakes fakes and only real images from the selected 41 development components. All remaining rows are unused; no row is reassigned to improve a group count.

Verified fold counts: 76,139 TRAIN rows / 471 recorded components; fit 3,745 rows (842 real, 2,903 fake); development 12,773 rows (269 real across 41 components, 12,504 held-out-family fake across 270 components); 59,621 unused rows. The eight ids8 fit cells retain both classes (smallest cell: 10 real and 60 fake). The development set is group-sparse: some gender×race, race, age and ids8 real cells have fewer than 20 examples or zero. Therefore this fold cannot support a full-axis fairness claim; report only supported cells and confidence limits.

## Two matched configurations

Both start independently from the same verified local ImageNet ResNet-50 weights SHA256 `d9ad71c77abfb482d7d9fc338f4536beda71d6ef4b495b7f60aad8f689832d11` and fresh seed-17 heads. Neither may initialize from the full-split P50 checkpoint, which has seen all five FF++ manipulation families. Both use the exact native P50 original-only pretraining recipe: ResNet-50 plus CES global/local heads and learned gates; size 224; batch 16; 50 complete epochs; four workers; ids8 inverse label×group weighting; AdamW; head lr `0.001`; encoder lr `0.0001`; weight decay `0.0001`; all encoder parameters trainable; BatchNorm updates; validation-AUC-only selection with earliest-maximum tie rule; validation-real empirical 10% threshold calibration.

- **S0_original:** original TRAIN images, weighted BCE only.
- **S1_nuisance_bank:** same training rows, updates and objective. Each original is presented once per epoch. A seed-17 hash schedule selects one of identity, JPEG quality 90, 90%-resize round trip or Gaussian blur radius 0.35 after the fixed 224 resize, equally independent of class, group and model score. Validation remains clean/original.

The bank is a processing robustness probe, not a validated forensic cue. This experiment does not use demographic edits, synthetic counterfactuals, or TEST.

## Launch gate and compute

The total cap is 5 aggregate GPU-hours including fold/config/input checks, CUDA smoke, real-input backward/profile and both full 50-epoch arms. The full schedules must fit the cap before either training arm starts; no shortened schedule or automatic extension is allowed. Use only the configured local ImageNet weights and verified GPU bindings. Publish the code/config hashes and checks before launch. Keep the fold manifests, images, row scores and checkpoints private.

Success is not a prespecified win. Report development AUC and its contributor-component uncertainty, thresholded FPR/FNR, worst-group results where support permits, every group count, transform exposure and runtime. The 269 real development frames come from only 41 recorded components, so intervals and any fairness subgroup inference will be limited. The component graph does not verify true-person independence.

## Current fold artifact

The deterministic fold generator is `make_train_fold.py`. The private generated files are stored under `$CES_RUN_ROOT/source_generalization/fold_s17/`; the committed repository contains no row-level manifest. The adapter uses the pinned native P50 architecture and source hash; its code and private configs are separately hash-bound.
