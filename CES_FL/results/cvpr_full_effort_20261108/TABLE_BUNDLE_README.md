# Ding Tables 1–3 and FairForge-2 comparison bundle

Combined PDF: `COMBINED_DING123_FF2_TABLES.pdf` (13 pages).

## Contents

- Ding Table 1, Xception, FF++.
- Ding Table 2, Xception, DFDC, Celeb-DF, and DFD.
- Ding Table 3, ResNet-50, FF++, DFDC, Celeb-DF, and DFD.
- The current FairForge-2 detailed draft, appended unchanged from `ff2_fairness_tables_20261105/pdf10/FF2_TABLES_DRAFT.pdf`.

`X` labels the Xception runs. `R` labels our ResNet-50 runs. `LA`/`LH` identify the arm; suffixes identify seeds. Each “Best ours, cellwise” row selects a different run/seed independently for each metric where appropriate. It is an oracle summary, not a model/checkpoint result.

## Counts in the cover page

- Ding Table 1: cellwise best Xception values beat Ding's published row on 9/10 cells and the best published method on 7/10.
- Ding Table 2: 20/30 vs Ding's published row and 19/30 vs the best published method.
- Ding Table 3: cellwise best R50 values beat Ding's published row on 13/40 cells and the best published method on 8/40.
- FairForge-2: the current table has 27 metric columns; best currently reported own values strictly beat the best reported baseline on 9/27. These are seed-17 SELECT/validation results. The full CES-FL objective has not been run on FF2.

## Important limitations

The Xception best-per-cell selection uses benchmark test metrics and is strictly descriptive. It must not be used to select seeds, checkpoints, or methods or to claim confirmatory superiority. The ResNet run is seed 17, 50-epoch original-only pretraining plus 10 downstream epochs; it is not Ding's training recipe. Its FF++ audit records 231 fit/test overlaps in identity/source IDs, and true-person independence is unverified. Threshold/protocol alignment with the published values is not fully established. FairForge-2 results are validation results, not test results.

## Inputs

- Xception table source: `/home/sachin.chaudhary/ces_fl_runs/xception_seeds_7_27_20261107/TABLES12_3seed.md`
- ResNet receipts: `/home/sachin.chaudhary/ces_fl_runs/lh_resnet50_table3_s17_20261104_cudaenv3/table3_eval/LOCKED_PLAN.json` and the `analysis/` directory.
- FairForge-2 table source: `/home/sachin.chaudhary/ces_fl_runs/ff2_fairness_tables_20261105/pdf10/FF2_TABLES_DRAFT.pdf`
- Published source: https://arxiv.org/html/2511.10150

See `BEST_PER_METRIC_DING123.csv`, `BEST_PER_METRIC_FF2.csv`, and `BUILD_SUMMARY.json` for selected cell values and input hashes.
