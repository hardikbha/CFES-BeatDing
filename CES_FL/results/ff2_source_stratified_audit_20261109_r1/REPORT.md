# FairForge-2 source-stratified frozen-score audit

**Final stop status: BLOCKED for the primary gender×race source-standardization estimand; INCONCLUSIVE for the relevant fairness intervals. CPU execution completed. STOP—no training or data revision follows.**

This is exploratory SELECT analysis of already selected, unchanged-image RF-S/RF-SJg exports. It is not a new model, an improved decision rule, or independent confirmation. All matched seeds 17/7/27 are reported; no best-cell selection.

## Primary result and support

The fixed TRAIN source weights are FFHQ=0.18930678905644632 and FairFace=0.8106932109435536, from 3,176/13,601 real TRAIN images. SELECT has 242/2,196 reals, respectively. They were frozen before private prediction outcomes were parsed. Full race and gender×race standardization is not identified: SELECT has zero FFHQ-real examples labelled Others, including both Female-Others and Male-Others. The same cells are absent in TRAIN. No source/group was discarded, imputed or renormalized away. The missing counterfactual source cells cannot be learned from existing scores.

Therefore the primary RF-S seed 17 contrast D_native−D_standardized has **no point estimate or confidence interval**. This remains blocked for both methods at all three seeds. The full-group within-FFHQ race/joint estimands are likewise unavailable; observed cell rates remain visible.

## Source and input verification

Documentation handoff: `76dcb54e240dcf56475dbf587836cdadcdad5aab`, documentation-only over reviewed scientific commit `e7fe0ee03e6cfacca81c558bbf632d2f72e80bc0`. A separate clean detached source checkout was used. Scientific source/configuration/guard files were unchanged.

All six actual checkpoint byte hashes, actual config dictionaries/digests, source/core hashes, preflight inventory bindings, selected epochs and CPU-loaded checkpoint thresholds passed. RF-S seed 17 was verified against its historical source 0a135a0/SHA5ee7c8b, rather than forced through the later schema; the other runs match ac6cba4/SHA2d99295. Cache rows match the full 31,107-row SELECT manifest in order, labels, demographic fields, method/family, complete recorded identity/source-ID sets and source-bound cached image hashes. Every stored global threshold matched the selected checkpoint exactly. Native metrics replay from cached scores matched the saved reports to 1e-12.

Acquisition labels come from documented canonical real-source keys and exact source-key/own-image matches, not appearance. Master/person/donor identifiers are recorded metadata, not independently verified biological identities. Only TRAIN/SELECT manifests, their cached inventories, SELECT exports and relevant builder source code were used. Mixed-split source-master manifests, raw image pixels, TEST/CAL data and Ding models were not accessed. Individual crop/config/master-byte histories are not present in these permitted split caches: a finer processing-stratified or causal history decomposition is **BLOCKED**, not silently inferred from the dataset name. Fake source breakdowns below refer to recorded contributing-reference acquisition, not proof of pixel/evidence origin.

## Native unchanged-image utility

Supplementary age/skin utility rows use inherited automatic estimates. Unknown skin values remain explicit; no demographic labels were generated or inferred from appearance. Intervals condition on recorded labels and do not account for annotation error. The full metadata check also matched every one of the16 CSV fields against all six exports, including label provenance and proxy values.

Markdown tables multiply rates/AUC/disparities by 100; raw machine-readable CSV estimates and intervals remain fractions. `100D` is the sum of absolute group FPR deviations, not an overall FPR or a percentage explained. AUC/FNR intervals below are recorded-component, conditional on the already selected checkpoints; they do not include selection or encoder-pretraining uncertainty.

| Model | AUC | FPR | FNR | Gender×race 100D | AUC95% CI | FNR95% CI |
|---|---|---|---|---|---|---|
| RF-S_s17 | 99.8032 | 9.9672 | 0.2930 | 34.1222 | [99.5467, 99.9059] | [0.2318, 0.3816] |
| RF-S_s7 | 99.7993 | 9.9672 | 0.2756 | 34.4818 | [99.5372, 99.9047] | [0.2236, 0.3495] |
| RF-S_s27 | 99.8079 | 9.9672 | 0.3000 | 34.1222 | [99.5575, 99.9072] | [0.2308, 0.4030] |
| RF-SJg_s17 | 99.7549 | 9.9672 | 0.4500 | 27.5259 | [99.4913, 99.8663] | [0.2902, 0.6916] |
| RF-SJg_s7 | 99.7603 | 9.9672 | 0.4465 | 30.8859 | [99.4988, 99.8671] | [0.2991, 0.6709] |
| RF-SJg_s27 | 99.7514 | 9.9672 | 0.5058 | 28.6686 | [99.4849, 99.8643] | [0.3319, 0.7794] |

## Identified gender-only standardization, with inconclusive intervals

Gender has both acquisition sources in both groups, so its point estimand is identified. Source weights remain fixed from TRAIN; the same SELECT real-group proportions pool native/standardized rates. These are additional exploratory estimates, not replacements for native model rows.

| Model | Native gender100D | Standardized gender100D | Native−standardized | 95% CI/status |
|---|---|---|---|---|
| RF-S_s17 | 4.1065 | 3.1963 | 0.9102 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (1871/2000) |
| RF-S_s7 | 4.1065 | 2.9596 | 1.1469 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (1871/2000) |
| RF-S_s27 | 4.1065 | 3.1360 | 0.9704 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (1871/2000) |
| RF-SJg_s17 | 3.2590 | 2.4098 | 0.8492 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (1871/2000) |
| RF-SJg_s7 | 3.5980 | 2.7219 | 0.8761 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (1871/2000) |
| RF-SJg_s27 | 3.4285 | 3.0758 | 0.3527 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (1871/2000) |

Only 1,871/2,000 component replicates defined all required gender/source cells, below the fixed 1,900 requirement. The apparent point reductions therefore do not support a statistically resolved source-composition association. The threshold was not retuned.

## Source rates and residual within-source disparity

| Model | FFHQ real FPR% | FairFace real FPR% | Within-FairFace joint100D | Joint residual CI/status |
|---|---|---|---|---|
| RF-S_s17 | 64.4628 | 3.9617 | 5.2312 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (202/2000) |
| RF-S_s7 | 66.1157 | 3.7796 | 5.5657 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (202/2000) |
| RF-S_s27 | 64.0496 | 4.0073 | 4.5733 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (202/2000) |
| RF-SJg_s17 | 51.2397 | 5.4189 | 6.3973 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (202/2000) |
| RF-SJg_s7 | 54.5455 | 5.0546 | 4.6034 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (202/2000) |
| RF-SJg_s27 | 47.1074 | 5.8743 | 7.8937 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (202/2000) |

FairFace has all eight joint groups and a computable residual disparity. Its full joint interval only has 202/2,000 valid draws, so residual differences are inconclusive. FFHQ lacks the two Others cells: its full eight-group residual is blocked rather than recomputed on a favorable subset. The augmentation lowers FFHQ false alarms but raises FairFace false alarms in these native point estimates. Source sensitivity and this observational association do not establish that compression alone causes demographic disparity or quantify a causal fraction explained.

[Gender/source support and rates](SOURCE_GENDER_FPR.md) · [Race/source](SOURCE_RACE_FPR.md) · [Gender×race/source](SOURCE_GENDER_RACE_FPR.md). These preserve all empty cells. Native per-group/family/source AUC/FNR, fake support, component counts and interval status are in GROUP_FAMILY_SOURCE_UTILITY.csv. Source/family cells without both classes remain undefined; no fake-source standardized aggregate is manufactured.

## Exact paired SJg-minus-S contrasts, every seed

| Seed | Native joint Δ100D | Joint Δ CI/status | ΔAUC pp | AUC Δ95% CI | ΔFNR pp | FNR Δ95% CI |
|---|---|---|---|---|---|---|
| 17 | -6.5963 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (999/2000) | -0.0482 | [-0.0738, -0.0265] | 0.1570 | [0.0454, 0.3251] |
| 7 | -3.5958 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (999/2000) | -0.0390 | [-0.0615, -0.0191] | 0.1709 | [0.0589, 0.3432] |
| 27 | -5.4536 | INCONCLUSIVE_BOOTSTRAP_SUPPORT (999/2000) | -0.0564 | [-0.0877, -0.0351] | 0.2058 | [0.0867, 0.4005] |

Native joint point differences favor SJg at all three seeds, but their intervals only have 999/2,000 valid draws. Utility AUC/FNR differences have 2,000 valid draws and indicate a detection cost descriptively. All additional intervals are unadjusted/exploratory; no significance or method selection is inferred from them. Standardized race/joint paired contrasts are blocked by the same missing cells. Standardized gender paired contrasts are reported with their inconclusive interval status in PAIRED_CONTRASTS.csv.

## Why image counts overstate uncertainty support

The all-contributor/master dependency graph projects to 45 SELECT-bearing components, largest 2,787 rows. These are **dependency components, not 45 biological people**. Many real source×joint cells have only one or two recorded components because fakes join multiple contributors. Resampling every frame independently would discard that dependence. Recorded TRAIN/SELECT identity/source-token overlap counts were both zero; this does not prove biological-person separation.

The prespecified 2,000-draw bootstrap uses analysis seed 17, whole recorded connected components, the same draws for all six models and both estimands, fixed decisions and fixed eligible groups. Acquisition remains an analytic stratum rather than two bootstrap units. Whole components are resampled globally, preserving any cross-stratum links; this choice was explicitly frozen before private outcomes. TRAIN source weights remain fixed; SELECT group proportions are re-estimated together in both views within each replicate. No component, group or difficult draw was removed to meet the 95% rule.

Primary race/joint standardization is structurally unidentified even before bootstrapping. Other intervals are withheld when fewer than 1,900 draws retain the necessary denominators. Missing support and interval failures must not be converted into a declaration that source composition dominates, that within-source bias is absent, or that we improved fairness.

## Explicit stop and files

**STOP at this report.** No new predictions, labels, images, GPU/CUDA operation, threshold/group calibration, scientific code/configuration/approval/guard change, new Ding computation, or TEST/CAL access. No trained arm was added. A training or data-revision proposal requires separate review.

- FREEZE.json records source definitions, TRAIN weights/support, contrast/bootstrap/multiplicity rules and the pre-score timestamp.
- BINDINGS.json and ALIGNMENT.json record the six actual artifact checks and input SHA256s.
- METRICS_AND_INTERVALS.csv contains all native/standardized/residual/group/family metrics and validity denominators; derivatives are provided as smaller CSVs.
- COVERAGE.json and COHORT_SUPPORT.json document missing cells, source provenance and recorded-component support.
- PRIMARY_RESULT.json and STOP_STATUS.json are the machine-readable blocked/inconclusive conclusion.
- audit.py, audit_stats.py, test_audit.py and CPU_CHECKS.log preserve the scoped helper and meaningful checks. No trainer is modified.
- Private bootstrap arrays and opening-event metadata remain under `/home/sachin.chaudhary/ces_fl_runs/ff2_source_stratified_audit_20261109_r1/`; no private rows, faces, prediction vectors or checkpoints are published.

This analysis cannot certify independent person-level links or processing histories absent from the permitted metadata. It conditions on previously selected checkpoints and shared pretraining, uses exposed SELECT for exploration, and supplies no independent confirmation or overall superiority claim.
