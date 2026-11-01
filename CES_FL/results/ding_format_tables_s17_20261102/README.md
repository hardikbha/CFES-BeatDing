# CES-FL compared with Ding: Tables 1, 2 and 3

Prepared 2 November 2026 from existing, completed evaluations. This is a reporting update: no training, inference, checkpoint selection or threshold calibration was performed.

| File | Ding's layout and scope |
|---|---|
| [Table 1](TABLE_1.md) | FF++ intra-domain; vertical Gender, Race and Intersection blocks; F_FPR, F_DP, es-AUC and AUC. |
| [Table 2](TABLE_2.md) | DFDC, Celeb-DF and DFD; grouped Gender, Race and Intersection columns, followed by overall AUC. |
| [Table 3](TABLE_3.md) | ResNet-50 on FF++, DFDC, Celeb-DF and DFD, with the same grouped columns. |
| [Combined PDF](TABLES_1_2_3.pdf) | Printable Tables 1–3. Blue rows are our measured single-seed results. |
| [Combined HTML](TABLES_1_2_3.html) | Download and open locally for grouped, browser-printable tables. |
| [Latest validation-only comparison](LATEST_VALIDATION_ONLY.md) | The newer 50+50 original-only experiment, source `58773bb`; not TEST. |
| [Thresholds and support](THRESHOLDS_AND_SUPPORT.md) | Requested and achieved validation FPR, measured test FPR and eligible-group counts. |
| [Our metrics CSV](OUR_TEST_METRICS.csv) | Both previously stored threshold reports, with no threshold chosen using test performance. |
| [Numeric snapshot and provenance](source_snapshot.json) | Exact fractions, subgroup counts, stored calibration, checkpoint/config/manifest hashes, and source paths. |
| [Formatting verification](VERIFY.json) | Actual checks on 44 archived test evaluations and 88 saved score reports. |

Each attribute/dataset block retains **every published method first**, then adds all applicable completed CES-FL arms. The paper's “Ours” is explicitly renamed **Ding et al. (published)**. Published baseline names, backbones and numbers are preserved. We have not rerun Ding or relabelled paper results as a reproduction.

All values in the tables are percentages, including summed F_FPR disparity multiplied by 100. That quantity can exceed 100; it is not a probability or the max–min FPR gap. Lower F_FPR/F_DP is better; higher es-AUC/AUC is better. Intersection means gender × race. Missing/undefined values render as a dash.

## Which of our experiments are included?

All our rows are **seed 17**, with FF++ original-image training. There is no multi-seed mean or significance claim.

| Label | Actual experiment | Interpretation |
|---|---|---|
| A | ImageNet encoder frozen; CES-FL heads/gates trained with BCE | Encoder-strength baseline. |
| B | Detection-pretrained encoder frozen; heads/gates trained with BCE | Compare A with B for detection pretraining. |
| C | Same detection-pretrained encoder as B, frozen; demographic invariance | Invariance-only pilot: 36 reviewed variants of six TRAIN originals; no forensic-sensitivity training. |
| Upgrade R0 | Frozen encoder, BCE | Separate archived upgrade record; its reported test scores repeat B's, so do not count it as an independent seed or corroborating replicate. |
| Upgrade R1 | Partial encoder training, BCE | Encoder-policy control. |
| Upgrade R2 | Partial encoder training, group DRO | Fairness baseline. |
| Upgrade RN | R2-style training with editor no-change consistency | Six-source processing-control diagnostic. |
| Upgrade RD | R2-style training with age-edit consistency | Six-source age-invariance diagnostic; not the proposed expanded source study. |

A/B/C are included for both Xception and ResNet-50. The later upgrade has measured Xception results only; no ResNet upgrade numbers are invented. The saved downstream budgets are ten epochs. This is distinct from the newer 50-epoch pretraining plus 50-epoch downstream study. The latest manipulation-response arm has no measured TEST row and is not represented by C or RD.

The benchmark execution source is `a11927affe394daf877e37234c21fea88cc171be`; the archived model code hash is `0e001dd1251570ce43fb82bde307c519dea4045d9734506e1ce90d2ed014f8de`. The original Xception artifacts and configs are preserved under the earlier encoder-controls audit; the benchmark source performed their later testing. The upgrade source is `875bce025084871438ef799ca9086985c4375672`. Exact per-arm configs, selected epochs and checkpoint hashes are in the snapshot.

## Protocol and comparability

Our main tables use **each selected checkpoint's already stored validation-real empirical FPR10% threshold**. Selection was validation AUC only, taking the earliest highest AUC; thresholds were frozen before TEST. Validation calibration achieved 409/4,092 false positives (9.995112%), not exactly 10%. Actual test FPR varies and is reported separately. The CSV also retains the pre-existing fixed-probability-0.5 report as a separate diagnostic. We did not select between the reports using test outcomes.

The metric equations in the stored evaluator are:

- F_FPR = sum over eligible groups of |group FPR − pooled FPR|.
- F_DP = max group positive rate − min group positive rate.
- es-AUC = pooled AUC / (1 + sum over eligible groups of |group AUC − pooled AUC|).

Our existing eligibility rule requires at least 20 real and 20 fake rows in a group. This rule is preserved, including exclusions on Celeb-DF and DFD. Native DFDC has 22,857 rows but 21,381 unique images: its 1,476 duplicate excess rows retain their original weight. These choices, selected operating points, optimization budgets, architecture heads and subgroup handling must be disclosed in comparisons with published numbers. Exact archived membership is not independent verification of the authors' unpublished run protocol or contributor independence.

Some rows exceed Ding's published detection AUC, while other datasets/metrics are worse. **These tables do not establish an overall fairness win, a direct matched-protocol win, or superiority of the current unrun manipulation-response method.** RN/RD and C do not provide broad counterfactual coverage. Learned gates are weights; signed local contributions are not validated forgery masks. Existing preregistration/gate outcomes have not been changed.

FairForge-2 is outside Ding's Tables 1–3. Its existing transfer results are preserved in the linked benchmark/upgrade reports; they are not an in-domain FairForge-2 training comparison. No new FairForge-2 study was launched here.

## Evidence and verification scope

- [Ding paper](https://arxiv.org/html/2511.10150v5); the rows here were parsed from the archived CVF accepted-paper text [Ding_CVPR2026.txt](../../../BEAT_DING/refs/Ding_CVPR2026.txt), not inferred from previous chat summaries.
- [Completed original A/B/C benchmark](../../status/benchmark_s17_20261031T112401_live.md).
- [Completed five-arm upgrade and uncertainty report](../upgrade_tests_s17_20261101T040407Z/REPORT.md), with [full metrics and confidence intervals](../upgrade_tests_s17_20261101T040407Z/aggregate.json).
- [Latest 50+50 original-only completion and fairness report](../manip_response_originals_s17_20261102/REPORT.md).

This reporting task read actual native `metrics.json`, `EVALUATION_VERIFIED.json`, training configs/receipts, pinned manifests and the published upgrade aggregate. It hashed 74 input files, checked checkpoint/manifest references against receipts, and recomputed the stored fairness aggregates from stored group summaries. It did **not** reload or rehash large checkpoint binaries, rerun validation/test inference, validate new counterfactuals, or establish fresh software/CUDA correctness. Those earlier receipts remain historical evidence, not newly executed tests. Point-estimate tables do not replace the existing confidence intervals or measure seed variability.

`build_tables.py` rebuilds the three tables from the committed snapshot without data/GPU access. It refuses to overwrite output files:

```bash
python build_tables.py --snapshot source_snapshot.json --out /new/output/directory
# Optional PDF: add --pdf in an environment with ReportLab, working Pillow and DejaVu Sans.
```

The actual reporting commands, environment issue and successful recovery are recorded in [COMMANDS.md](COMMANDS.md). Private image files, predictions, review sheets and checkpoint weights remain outside GitHub.
