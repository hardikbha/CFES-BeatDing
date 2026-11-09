# Completed source-tail cohort — 9 November 2026

[Full results and regressions](REPORT.md) · [Decision](DECISION.json) · [Independent verification](FINAL_VERIFICATION.json) · [Epoch summary](EPOCH_SUMMARY.csv) · [All validation metrics](EXTENDED_VALIDATION_METRICS.csv) · [Paired contributor intervals](PAIRED_CONTRIBUTOR_INTERVALS.csv) · [Commands](COMMANDS.md)

Both three-epoch seed17 arms and their full native exports completed. The scientific source is pinned to `9bb8e8332b8a3aa78d83f74ef21bca0b7dcd6754`; its configuration was not changed.

| Model | Selected epoch | SELECT AUC% | SELECT FNR% | Gender×race100F_FPR |
|---|---:|---:|---:|---:|
| Existing complete L_A teacher | 2 | 99.8057 | 0.2825 | 37.3715 |
| Control: weighted BCE + fake-score guard | 1 | 99.8073 | 0.2895 | 34.8501 |
| Treatment: control + source-tail penalty | 1 | 99.8069 | 0.2895 | 34.8206 |

**Execution/export PASS; no meaningful fairness benefit established. Primary uncertainty INCONCLUSIVE. STOP after this cohort.** The treatment's primary point difference is only −0.02954 disparity points, with an unavailable valid contributor-component interval (999/2000 draws,1900 required). Gender/age disparity is unchanged; skin/other intersections regress. The TRAIN source surrogate is lower than control but higher than teacher. This is original-image head adaptation, not full original L_H or validated counterfactual training, and not evidence of beating Ding.

All stages together held one shared H100 lease for **0.401480 GPU-hours (24.089minutes)**, including loading/checks/cache/training/export/idle/bookends. The two-hour cap was respected. The shared72 ledger is conservatively reconciled at68.230538 charged,3.769462 remaining. No automatic use of the remaining balance is authorized.

## Public outputs

- `*_HISTORY.json`, `EPOCH_SUMMARY.csv`: every epoch's losses, validation/selection and source×joint exposure.
- `VALIDATION_METRICS.csv`, `VALIDATION_GROUP_FAMILY_METRICS.csv`: native core metrics and per-group/family support/errors in fractions.
- `EXTENDED_VALIDATION_METRICS.csv`, `EXTENDED_GROUP_METRICS.csv`: independent metrics including age×skin/coarse eight-group views, max/mean EO definitions separately, accuracy gap and worst-group AUC.
- `SOURCE_PARENT_JOINT_AND_FAMILY_UTILITY.csv`, `VALIDATION_SOURCE_REAL_FPR.csv`: source/group/family native utility, preserving empty cells. Acquisition for fakes denotes contributing-reference metadata, not proven pixel origin.
- `FULL_TRAIN_SOURCE_RISKS.csv`, `FULL_TRAIN_OBJECTIVE_DIAGNOSTIC.json`: descriptive selected-head TRAIN surrogate diagnostics, not new model selection.
- `AUXILIARY_SOURCE_JOINT_EXPOSURE.csv`, `ACTUAL_AUXILIARY_COVERAGE.csv`: identical auxiliary schedule counts, unique real-row coverage and reuse.
- `PAIRED_CONTRIBUTOR_INTERVALS.csv`, `UNCERTAINTY_SCOPE.json`: paired 2000-draw recorded-component analysis with fixed1900-valid rule; unsupported CIs withheld.
- Native qualification/cache/selection/export receipts, hashes and ledger evidence. Full faces, sidecars, features, row predictions and model binaries remain private.

Raw metric CSVs contain fractions; report tables multiply by100. Summed F_FPR can exceed1 before scaling. Unknown demographic labels are retained in native totals and reported; age/skin/ids8 are estimates. `ids8` differs from the explicit coarse sex×age×race grouping. All fixed0.5 metrics are kept separate from the validation-calibrated view. No standardized race/joint statistic is fabricated for missing FFHQ Others cells.

Private root: `/home/sachin.chaudhary/ces_fl_runs/source_tail_training_20261109_r1/`. Logs and checkpoints are preserved. No source-tail training remains active; generic scheduler jobs were left untouched. No TEST/CAL evaluation or new Ding computation occurred. A new model/data study requires its own review.
