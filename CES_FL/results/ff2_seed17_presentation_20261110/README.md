# FF2 seed 17 fixed for the single-seed presentation

Selected at management UTC: 2026-11-10T12:26:54.707445+00:00, following the author's instruction to fix seed 17.

All seven methods use the same seed 17. Existing checkpoints, global thresholds and metrics are unchanged. Full L_H, ramp and low-response remain separate model rows; no best-cell mixture is created. The selection is retrospective, after TEST results for all three seeds were available. The [complete three-seed results](../ff2_fully_matched_final_20261110/README.md) and [all-seed workbook](../ff2_complete_metric_tables_20261110/README.md) remain available.

[Seed17 Excel workbook](FF2_SEED17_RESULTS.xlsx) · [complete metric tables](FULL_METRICS.md) · [editable LaTeX table](FF2_SEED17_TABLE.tex) · [selection record](SEED_SELECTION.json).

## Overall detection and intersectional disparity

AUC/FPR/FNR are percentages. Joint 100F_FPR is summed group false-positive disparity, not an error rate. Lower FPR/FNR/disparity is better; higher AUC is better.

| Method | AUC | FPR | FNR | Joint 100F_FPR |
|---|---:|---:|---:|---:|
| BCE | 99.0146 | 15.4200 | 1.2664 | 67.9232 |
| DAG-FDD | 99.0065 | 15.0891 | 1.3809 | 65.4379 |
| DAW-FDD | 99.0083 | 14.9237 | 1.4381 | 64.3660 |
| Ding adaptation | 99.0169 | 15.2178 | 1.2786 | 66.3557 |
| Our full L_H | 98.8270 | 13.7842 | 2.5547 | 67.4903 |
| Our ramp | 99.0114 | 14.8870 | 1.4734 | 63.7480 |
| Our low response | 99.0114 | 14.8870 | 1.4734 | 63.7480 |

Full L_H leads age and skin-tone FPR disparity within this seed, with lower AUC and higher FNR. Ramp/low lead race, joint and IDS8 FPR disparity; their selected epoch 2 precedes response activation, so those results cannot establish a response-learning benefit. No overall superiority claim is supported. Age/skin axes are automatic estimates; Ding is the controlled adaptation under the shared recipe.

The LaTeX file is prepared for review. This task does not edit Overleaf or relaunch training. All CSV/workbook numbers are exact filtered copies of the verified public results. [Verification](VERIFICATION.json), [artifact hashes](SHA256.json).
