# Exact dataset splits and existing FairForge-2 fairness results

Reporting snapshot, 2November2026. No new training, inference, calibration or test tuning.

| Current dataset | TRAIN images | VAL images | TEST images |
|---|---:|---:|---:|
| Ding FF++ | 76,139 | 25,386 | 25,401 |
| FairForge-2 public | 218,266 | 31,107 | 87,564 |

These are image/frame rows, not video or unique-person counts. Native Ding manifests were independently counted; real/fake counts, exact paths and hashes are in SPLITS.json and the TeX report.

Current FairForge-2 public original-only Xception P, seed17,10epochs, selected epoch9: **VAL AUC99.8105%, FPR9.9672%, FNR0.3802%**. No TEST result exists for this new in-domain model. These measured fairness outcomes do not establish improvement from counterfactual training.

| Axis | F_FPR x100 (lower better) | FPR gap (pp) | FNR gap (pp) | Worst-group AUC (%) |
|---|---:|---:|---:|---:|
| gender | 3.5980 | 3.5980 | 0.1180 | 99.8080 |
| race | 12.1565 | 7.5679 | 0.1260 | 99.6840 |
| gender_race | 28.3857 | 11.8517 | 0.2848 | 99.5827 |
| age_bin | 9.2329 | 7.4848 | 0.1737 | 99.6982 |
| skin_bin | 11.6827 | 7.9134 | 0.2700 | 99.6977 |
| ids8 | 34.8666 | 18.1073 | 0.5583 | 99.5209 |

F_FPR is a summed disparity, distinct from the max-min gap and an overall FPR. Age and skin labels are automatic proxies. All six recorded axes, every group support/error row and stored conditional transfer CIs are included. Unknown/ineligible groups are retained visibly.

The standalone TeX/PDF also includes all five existing FF++-trained public-test transfer arms at stored validation and fixed0.5 thresholds; historical public CFES-D at three recorded operating modes; and distinct archived Ding FF2 subset/internal records. **They are separate studies, not a matched leaderboard.** FF++ to public FF2 transfer AUC is38.8312-42.8294%; this weak result is retained. Legacy CFES-D public-test AUC98.2598% is a different CLIP model. The old Ding in-domain99.3573% belongs to the internal182,884-row test build, not public87,564. No matching public in-domain Ding result or trained CDF result is invented.

## Files

- `ding_and_fairforge2_tables.tex`: standalone LaTeX, with all tables and source records.
- `ding_and_fairforge2_tables.pdf`: compiled report, 22pages.
- `SPLITS.json`: fresh metadata counts and hashes, including native Ding cross-check.
- `SNAPSHOT.json`: actual aggregate results and source hashes.
- `AXIS_METRICS.csv`, `GROUP_METRICS.csv`, `OVERALL_METRICS.csv`, `TRANSFER_EXISTING_CI.csv`: normalized current-study metrics.
- `BUILD_VERIFICATION.json`: actual compiler command/exit, PDF/count checks and hashes.
- `build_tables.py`: reporting-only reproducible exporter; no model inference or training.

LaTeX compile exited0. No overfull boxes remain. The local pdfTeX engine emits longtable glue diagnostics; representative pages were inspected and all selected counts/metrics reconciled. No data or methodological threshold was changed to address layout.

Absolute source: `/home/sachin.chaudhary/cesel_ma_audit_logs/dataset_fairness_tables_20261102T104319Z/ding_and_fairforge2_tables.tex`.

Absolute PDF: `/home/sachin.chaudhary/cesel_ma_audit_logs/dataset_fairness_tables_20261102T104319Z/ding_and_fairforge2_tables.pdf`.
