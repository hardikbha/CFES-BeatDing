# CES-FL ResNet-50 nuisance pilot — seed 17

**Status: all four arms completed; FF++ validation only. No benchmark TEST was read.**

## Protocol and provenance

- Scientific source commit: `84ad9838ac14f9e252d292146df5b1e1b1121d7c`; source SHA256 `1997eebf093aa3e37de1cbbe58302a1539ac2e517f966a4607aae89dc695a8e0`.
- Backbone/initialization: the same verified ImageNet-initialized FF++ P50 ResNet-50 checkpoint (`c28bbb9cfcd06a11894027dc9369c5fd674f640de6c182c28b6d6f0aa8d77f55`), fresh seed-17 linear head per arm; batch 32, layer4 fine-tuning, frozen BatchNorm, 3 epochs.
- A0/H0 use original processing; A1/H1 use the fixed four-case common nuisance bank. H0/H1 preserve the existing L_H objective. Checkpoint selection: earliest maximum clean-validation AUC. Threshold: per-arm FF++ validation-real target FPR 10%; fixed probability 0.5 is also reported.
- TRAIN 76,139 / VAL 25,386 images; all input hashes and pixels passed preflight. Image-byte overlap is zero. Manifests declare 163 TRAIN/VAL identity/source contributor overlaps; true-person separation is not established. TEST-related declarations remain deferred.
- H0 preview 88 cases; H1 preview 352 cases (88×four common-processing cases). Review was AI-only (`Codex`); no human review. It screened processing/crop/local-effect fidelity, not real-forgery or causal validity. These are procedural probes, not validated forensic interventions.

## Selected validation results

| Arm | Selected epoch | AUC % | FPR % | FNR % | Gender F_FPR % | Race F_FPR % | Gender×race F_FPR % | Age F_FPR % | Skin F_FPR % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 3/3 | 96.6217 | 9.9951 | 9.6036 | 6.0357 | 12.3531 | 40.5914 | 12.9181 | 7.7712 |
| H0 | 1/3 | 96.4214 | 9.9951 | 9.9934 | 5.6438 | 9.9922 | 33.3735 | 12.2680 | 5.0148 |
| A1 | 1/3 | 96.4903 | 9.9951 | 9.3407 | 6.6237 | 11.0651 | 37.6674 | 6.7056 | 8.6143 |
| H1 | 1/3 | 96.6281 | 9.9951 | 9.3547 | 5.6438 | 11.4295 | 35.6454 | 4.6997 | 7.3456 |

All F_FPR values are fractions converted to percent. Full group-wise AUC/FPR/FNR/support at both threshold views are in `GROUP_METRICS_ALL_ARMS.csv`; full selected-checkpoint validation reports and histories are in the accompanying JSON files.

## Matched contrasts (percentage-point changes; first minus second)

| Contrast | Δ AUC | Δ FNR | Δ gender F_FPR | Δ race F_FPR | Δ gender×race F_FPR | Δ age F_FPR | Δ skin F_FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1−A0 | -0.1314 | -0.2630 | +0.5879 | -1.2880 | -2.9240 | -6.2124 | +0.8431 |
| H1−H0 | +0.2067 | -0.6387 | +0.0000 | +1.4373 | +2.2719 | -7.5682 | +2.3308 |
| H1−A1 | +0.1378 | +0.0141 | -0.9799 | +0.3644 | -2.0220 | -2.0059 | -1.2687 |
| Descriptive interaction (H1−A1)−(H0−A0) | +0.3381 | -0.3757 | -0.5879 | +2.7253 | +5.1959 | -1.3558 | +1.4877 |

H1 vs A1 point estimates improve AUC, gender, gender×race, age and skin F_FPR; race F_FPR and FNR are slightly worse. H1 vs H0 improves AUC/FNR and age F_FPR, but race, gender×race and skin F_FPR worsen. The descriptive interaction is not causal and has no interval.

## Paired contributor-bootstrap uncertainty

The H1−A1 and H1−H0 comparisons use 1,000 paired connected-recorded-contributor resamples (seed 17). They represent validation sampling uncertainty for one training seed, not training-seed uncertainty or independent-person confirmation. The resampling graph has 209 components, largest component 360 rows; person independence is unverified.

Both intervals meet the 900-draw floor for overall AUC/FPR/FNR and most marginal axes. Gender×race fairness intervals do not: F_FPR/F_DP/es_AUC/FPR-gap/FNR-gap have 746 valid draws; worst-group AUC has 804. Female–Black subgroup intervals are also under the 900 floor (773–862). Treat these point estimates as unsupported for inferential claims. Full intervals and valid-draw counts are in `BOOTSTRAP_H1_A1.json` and `BOOTSTRAP_H1_H0.json`.

## Nuisance and response diagnostics

Original L_H and H0 diagnostics are numerically identical on the fixed 128-image TRAIN diagnostic sample, consistent with the shared seed/epoch-1 solution. H1 reduces median absolute original-logit drift for resize90 from 6.281 to 2.588 and p90 from 14.268 to 6.741. Median absolute procedural-response change under resize90 falls from 0.963 (H0) to 0.424 (H1); JPEG90 response drift is slightly larger for H1, while blur035 remains small. Identity-case response medians remain measurable for both warp and resample probes. These are deterministic procedural diagnostics, not evidence that a real forensic cue was preserved.

H0/H1 gradient norms on one fixed clean TRAIN batch are finite and nonzero. H1 alignment gradient norm is 0.3573 with cosine −0.600 to BCE; H0 is 0.2599 with cosine −0.498. This shows local gradient conflict on this batch, not population-level conflict. Full aggregate diagnostics are in `diagnostics/`; private row-level diagnostic records remain off GitHub.

## Interpretation and limitations

Result: **mixed / inconclusive**. The strongest mechanistic evidence is reduced score/response sensitivity to resize under H1. Detection and several marginal fairness metrics improve for H1 over the augmented BCE control, but race F_FPR worsens slightly, FNR is not improved, and the primary gender×race bootstrap is unsupported. The interaction indicates that augmentation weakens H’s joint fairness gain relative to H0 versus A0. One seed, a three-epoch pilot, automated demographic proxies, declared contributor overlap and the AI-only processing review limit conclusions.
Do not claim statistical significance, feature disentanglement, real-forgery selectivity, cross-domain transfer, or a win over Ding. No test evaluation or automatic extension was run.

## Compute and device

The pilot used 1.1951 of 3.0000 GPU-hours, leaving 1.8049. All GPU sessions ran sequentially on shared gpu02 H100 UUID `GPU-bff97893-385b-6f3f-831b-113284c2e1f1` under pool-v2 leases and 16% process memory cap. No job was preempted; no PBS allocation was claimed. FairForge-2 budget was untouched.

## Files

- `VALIDATION_RESULTS.json`: selected validation metrics, all axis/group metrics, contrasts and interaction.
- `GROUP_METRICS_ALL_ARMS.csv`: per-group support, AUC, FPR and FNR at source-validation and fixed-0.5 thresholds.
- `histories/`: all four epoch histories, including source exposure and pair counts.
- `BOOTSTRAP_H1_A1.json`, `BOOTSTRAP_H1_H0.json`: paired intervals and usable-draw counts.
- `diagnostics/`: aggregate original/H0/H1 score, response and gradient diagnostics.
- `PREFLIGHT_SUMMARIES.json`, `GPU_LEDGER_AGGREGATE.json`, `ARTIFACT_HASHES.json`, `CHECKS_SUMMARY.json`: source/input/check/device provenance.


## Fixed probability 0.5 validation diagnostics

| Arm | AUC % | FPR % | FNR % | Gender F_FPR % | Race F_FPR % | Gender×race F_FPR % | Age F_FPR % | Skin F_FPR % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 96.6217 | 14.7849 | 6.4854 | 6.9017 | 15.3319 | 54.0652 | 13.3304 | 11.9991 |
| H0 | 96.4214 | 13.0743 | 8.0070 | 5.8505 | 9.2403 | 38.2081 | 12.5884 | 10.0478 |
| A1 | 96.4903 | 8.6510 | 10.6791 | 6.1477 | 12.8549 | 37.0486 | 7.0978 | 7.0393 |
| H1 | 96.6281 | 7.4291 | 11.6418 | 5.2428 | 10.3533 | 34.6888 | 3.5069 | 3.5144 |

This fixed-cutoff view is diagnostic only; it does not replace the stored validation-real 10% FPR operating point and was not tuned.

## H0 versus A0 (unaugmented objective contrast)

| Δ AUC pp | Δ FNR pp | Δ gender F_FPR pp | Δ race F_FPR pp | Δ gender×race F_FPR pp | Δ age F_FPR pp | Δ skin F_FPR pp |
|---:|---:|---:|---:|---:|---:|---:|
| -0.2003 | +0.3898 | -0.3920 | -2.3609 | -7.2179 | -0.6501 | -2.7564 |

## Paired connected-contributor bootstrap (H1 contrasts)

| Metric | H1−A1 estimate (95% CI; valid) | H1−H0 estimate (95% CI; valid) |
|---|---:|---:|
| AUC (pp) | +0.1378 [-0.0387, +0.3140] (1000/1000) | +0.2067 [-0.0016, +0.4459] (1000/1000) |
| FNR (pp) | +0.0141 [-0.6747, +1.1531] (1000/1000) | -0.6387 [-1.1195, -0.1705] (1000/1000) |
| Gender F_FPR (pp) | -0.9799 [-3.2943, +0.7530] (1000/1000) | +0.0000 [-1.8326, +2.0418] (1000/1000) |
| Race F_FPR (pp) | +0.3644 [-8.8454, +8.9235] (998/1000) | +1.4373 [-7.0532, +12.0491] (998/1000) |
| Gender×race F_FPR (pp) | -2.0220; CI unsupported (746/1000) | +2.2719; CI unsupported (746/1000) |
| Age F_FPR (pp) | -2.0059 [-13.5189, +6.8212] (976/1000) | -7.5682 [-15.6124, +1.8822] (976/1000) |
| Skin F_FPR (pp) | -1.2687 [-5.8407, +2.3680] (1000/1000) | +2.3308 [-4.0083, +5.7967] (1000/1000) |

Intervals are in percentage points. The bootstrap uses 1,000 paired recorded-contributor draws (seed 17). The primary gender×race F_FPR interval is unsupported (746/1000); do not infer significance from its point estimate. The full interaction has no interval.

## Candidate score-scale, response, and gradient diagnostics

Zero-update checkpoint diagnostics used 128 TRAIN images (8 per joint group/class cell), kept BatchNorm buffers unchanged, and applied no optimizer step. They test the trained checkpoints on artificial procedural interventions and processing cases only.

### Original logit and procedural-response drift by nuisance (median absolute change; p90 in parentheses)

| Checkpoint | Nuisance | Original logit | Procedural response |
|---|---|---:|---:|
| Original L_H | jpeg90 | 0.788 (10.893) | 0.279 (0.859) |
| Original L_H | resize90 | 6.281 (14.268) | 0.963 (1.924) |
| Original L_H | blur035 | 0.117 (1.284) | 0.058 (0.206) |
| H0 | jpeg90 | 0.788 (10.893) | 0.279 (0.859) |
| H0 | resize90 | 6.281 (14.268) | 0.963 (1.924) |
| H0 | blur035 | 0.117 (1.284) | 0.058 (0.206) |
| H1 | jpeg90 | 0.666 (4.253) | 0.308 (0.701) |
| H1 | resize90 | 2.588 (6.741) | 0.424 (1.140) |
| H1 | blur035 | 0.108 (0.723) | 0.061 (0.194) |

H0 matches the original L_H diagnostic on this sample. H1 reduces resize90 logit drift and procedural-response drift; JPEG90 response drift is slightly larger than H0. The absolute-logit statistics are measured on a small stratified TRAIN diagnostic sample, not an evaluation set.

### Identity-case response remains measurable

| Operation | Strength | H0 identity median response | H1 identity median response | H0 resize90 median |H1 resize90 median |
|---|---:|---:|---:|---:|---:|
| local_warp | 0.5 | 0.555 | 0.552 | 0.831 | 0.301 |
| local_warp | 1.0 | 0.650 | 0.719 | 0.645 | 0.456 |
| local_resample | 0.5 | 1.049 | 1.093 | 0.827 | 0.290 |
| local_resample | 1.0 | 1.128 | 1.128 | 1.340 | 0.688 |

Responses are `asinh(z_treated−z_sham)` for synthetic local warp/resample probes; these are not validated real forgery cues.

### Weighted component-gradient norms and cosine to BCE

| Checkpoint | Component | Weighted gradient norm | Cosine to BCE |
|---|---|---:|---:|
| Original L_H | BCE | 0.2871 | 1.000 |
| Original L_H | fairness | 0.7540 | 0.986 |
| Original L_H | sensitivity | 0.2635 | 0.659 |
| Original L_H | sham_control | 0.0377 | 0.207 |
| Original L_H | alignment | 0.2599 | -0.498 |
| H0 | BCE | 0.2871 | 1.000 |
| H0 | fairness | 0.7540 | 0.986 |
| H0 | sensitivity | 0.2635 | 0.659 |
| H0 | sham_control | 0.0377 | 0.207 |
| H0 | alignment | 0.2599 | -0.498 |
| H1 | BCE | 0.3156 | 1.000 |
| H1 | fairness | 0.6564 | 0.983 |
| H1 | sensitivity | 0.2860 | 0.664 |
| H1 | sham_control | 0.0457 | 0.646 |
| H1 | alignment | 0.3573 | -0.600 |

Each gradient report is from one fixed clean TRAIN batch and no parameter update. The alignment gradient opposes BCE on this batch in H0 and H1; this local observation does not establish population-wide conflict.


All secondary axis metrics (F_DP, es_AUC, worst-group AUC, FPR/FNR gaps) at both thresholds are in `FAIRNESS_AXIS_METRICS_ALL_ARMS.csv`. `CONTRASTS_ALL_METRICS.csv` contains every arm contrast and group-level AUC/FPR/FNR change, including regressions; `GROUP_METRICS_ALL_ARMS.csv` lists every group support and point metric.

## Provenance and coverage detail

The TRAIN manifest contains 76,139 frames and 773 distinct recorded identity/source tokens (these are manifest provenance tokens, not verified unique people). Recorded contributor support is uneven: the TRAIN Male–Black intersection has 22 tokens, Female–Black 25, and age 60+ 33. The VAL Female–Black cell has 30 real and 120 fake images. The validation group metric CSV includes every group’s exact real/fake row support. Skin-bin unknown labels exist in TRAIN; no demographic labels were inferred during this pilot.

Every original TRAIN row was visited once in each epoch. A0/A1 had no procedural pairs; H0/H1 each exposed 12,926 real-source procedural pairs per epoch, with per-axis group exposure and nuisance-case counts preserved in the aggregate training histories.

## Review and check receipts

The source-bound H0 preview had 88 cases and H1 had 352 (the same 88 source/operator/strength cases under four common-processing conditions). Both reviews were performed by an AI visual reviewer only; no human approvals are claimed. The review accepts the processing behavior for this bounded experiment, while the generated local warp/resample probes remain unvalidated as real forensic evidence.

CPU unit checks: 6/6 passed. All four full TRAIN/VAL preflights passed (101,525 images decoded per arm; no image-byte overlap). All four CUDA smoke and config-specific real-input backward checks passed on one verified H100 UUID. Full process-level mappings, source/config hashes and aggregate GPU ledger are in the receipt files.

Files also include `RECORDED_TRAIN_COVERAGE.json`, `REVIEW_SUMMARY.json` and `CUDA_BINDINGS_AGGREGATE.json`. Per-image inventories, contact sheets, review rows, raw prediction rows and checkpoints remain private on the cluster.
