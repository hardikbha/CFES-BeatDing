# CES-FL: current research and paper handoff

Observation: 9 November 2026, 01:43 IST on mgmt01; both GPU clocks reported 01:32. This is a timestamped snapshot, not an automatic monitor.

## What is complete

- Three downstream RF-S seeds and three RF-SJg seeds, each ten epochs, with selected-checkpoint replay. All share one seed17 BCE50 encoder.
- Prior frozen-list TEST exports for ten checkpoints / six model families. Existing singleton comparators: corrected Ding-style GDA, DAW-FDD, CES-FL P, L_A. BCE/DAG and other validation-only methods must not acquire invented TEST entries.
- New CPU-only audit of actual checkpoint-byte hashes and receipts, RF selection/threshold bindings, 80 aggregate summary rows and 540 subgroup rows. Independent arithmetic checks pass; model code is unchanged.
- An actual seven-page working paper, LaTeX source, references, four tables, three numeric figures, public aggregates and explicit limitations. Start at [paper README](../paper/README.md) and [PDF](../paper/main.pdf).

## Results and interpretation

| Family | Downstream seeds | TEST AUC (%) | TEST gender×race 100F_FPR |
|---|---:|---:|---:|
| Corrected local Ding-style GDA | 1 | 93.08 | 9.47 |
| DAW-FDD adaptation | 1 | 98.87 | 45.35 |
| CES-FL P | 1 | 98.78 | 42.23 |
| L_A plain-head BCE | 1 | 99.00 | 63.67 |
| RF-S | 3 | 98.98±0.01 | 60.95±1.92 |
| RF-SJg | 3 | 98.60±0.06 | 47.69±3.91 |

F_FPR is a sum, multiplied by100 here; it is not overall FPR. ± is sample SD over downstream seeds, not a confidence interval. RF-SJg improves this disparity relative to RF-S in all three paired downstream seeds, but loses detection performance, gender/age fairness, and worst-group AUC. It does not dominate the local GDA or DAW comparator. Different baseline training/selection budgets remain a limitation. These are public FF2 in-domain results, not Ding's original benchmark or its published Table1–3 values.

## Code map

| Role | File |
|---|---|
| Actually reported RF-S/RF-SJg trainer/loss/augmentation/selection | `CES_FL/src/ces_fl_realfpr.py` |
| Shared Xception model, data, calibration, prediction, parameter policy | `CES_FL/src/ces_fl_head_ablation.py` |
| Historical full global/local/gated CES-FL prototype | `CES_FL/src/ces_fl_diagram_model_v021.py` |
| Historical procedural forensic-response study | `CES_FL/src/ces_fl_forensic_response.py` |
| Exact RF configs and exploratory history | `CES_FL/proposals/ff2_realfpr_stretch_20261107_r1/` |
| Existing TEST freeze/export protocol and aggregate results | `CES_FL/proposals/ff2_test_eval_20261108_r1/` |
| Original separate confirmation source/operator, now completed/superseded | `CES_FL/proposals/ff2_sjgentle_confirmation_20261108_r1/` |
| Manuscript generation and independent aggregate checks | `CES_FL/paper/tools/` |

The current RF study has one GAP+linear head, not the full gated/counterfactual architecture. No generated edits, forensic-response loss, map-consistency loss or demographic counterfactual training is active. Keep those scientific hypotheses distinct.

## Provenance and operational state

Reviewed upstream commit `30a6b263aaa63677cf8d3d9f4ee51bdb472ae284`. RF/core source hashes, every config/selected checkpoint, fixed TEST thresholds and cluster artifact paths are in [evidence map](../paper/EVIDENCE_MAP.md). Immutable running-job sources and old outputs were preserved.

Two seed7 executions exist. The existing TEST freeze's RFA-tree checkpoint is used consistently; the original confirmation run's identical selected VAL metrics are not counted as an extra seed. All actual compute stays charged. Ledger64.6011/72GPU-h has zero reservations, with TEST/sanity duration reconciliation still pending. No new GPU process was started. No live RF trainer was found; unrelated scheduler jobs are not CES-FL training evidence.

The current paper branch also preserves the earlier [Ding1–3 + FF2 descriptive table bundle](../results/cvpr_full_effort_20261108/TABLE_BUNDLE_README.md). Those best-cell oracle tables are internal exploratory summaries, not a deployable model result or the manuscript's primary evidence.

## Immediate writing priorities

Review [claims and gaps](../paper/CLAIMS_AND_GAPS.md) and [writing plan / supervisor prompt](../paper/WRITING_PLAN.md). Confirm author/dataset text, investigate freeze chronology and numerical replay, complete contributor-aware uncertainty and label/source provenance. Do not silently change a frozen protocol, launch more jobs or tune against the exposed TEST set. No CVPR acceptance or overall SOTA result is promised.
