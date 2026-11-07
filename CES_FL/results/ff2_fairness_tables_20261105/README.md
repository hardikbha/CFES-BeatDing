# FairForge-2 fairness tables and failure diagnosis (5 Nov 2026)

Aggregate-only package. No training, inference, threshold tuning or TEST access was performed for it. The tables are
built by `build_tables.py` from existing files; the one new computation (`diag_source.py`) re-reads the private VAL
score files already used for the 5 Nov seven-run report and writes counts only. All detailed tables are in
[TABLES.md](TABLES.md) (CSV for every table; `tables.tex` for T5 and T9).

## 1. Status update

**Ding's dataset (FF++ train → FF++/DFDC/Celeb-DF/DFD test, Xception, seed 17, against Ding's published Tables 1–2).**
CES-FL L_A wins 23/40 cells (FF++ 7/10, DFDC 9/10, Celeb-DF 1/10, DFD 6/10); L_H wins 21/40. Previous best was BD0 at
17/40. The pre-registered "beats Ding" rule (≥25/40, AUC and all es-AUC on all four sets, intersection F_FPR on ≥3 sets)
is **not** met: AUC is above Ding only on FF++, and Celeb-DF is the weak set. ResNet-50 Table 3: L_A 9/40, L_H 12/40.
These L_A/L_H rows are the plain-head ablation, not the full CES-FL diagram model.

**FairForge-2 (in-domain, VAL only, seed 17).** Seven verified runs. AUC is 99.81–99.84 for every run except the
Ding-style GDA run (99.06). On F_FPR, Ding-style GDA is best on gender, race, gender×race and age; DAW-FDD is best on skin.
No CES-FL run (P/A/B) is best on any F_FPR axis (T5). F3Net and CORE trained 50 epochs but their post-hoc verification
failed three times, so they are not tabled.

## 2. Why our model is failing on FF2 fairness

1. **The "race" and "age" gaps on FF2 are mostly a source-dataset gap, and none of our losses targets it (T9).**
   FFHQ real photos are called fake far more often than FairFace real photos, at each model's own 10% VAL-real threshold:

   | | BCE | DAG-FDD | DAW-FDD | Ding-style GDA | CES-FL P | CES-FL A | CES-FL B |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | FFHQ reals (n=242) FPR % | 72.73 | 59.92 | 47.11 | 14.88 | 44.21 | 45.04 | 45.04 |
   | FairFace reals (n=2,196) FPR % | 3.05 | 4.46 | 5.87 | 9.43 | 6.19 | 6.10 | 6.10 |

   "Others" is 100% FairFace, while Asian/Black/White include FFHQ reals, so Others gets a low FPR and the other three a
   high one. Inside FairFace alone, the race FPR spread is small: BCE 2.12–3.51% (1.4 pts) versus 12.5 pts overall;
   CES-FL P 5.88–7.14% (1.3 pts) versus 7.6 pts overall. FFHQ reals under 30 reach 79% FPR for BCE, which drives the
   `<30` age-bin gap. Ding-style GDA "wins" F_FPR mainly because it pulls FFHQ FPR down to 15% (and pushes FairFace up to
   9.4%), at the cost of AUC (99.06) and a 3–5× higher miss rate (FNR 1.07% vs 0.22–0.38%).

   Candidate causes (not yet tested): FFHQ fakes per real are higher (≈17 vs ≈13); seven generators were trained on FFHQ
   (StyleGAN2, EG3D, StyleSwin, Projected GAN, CodeFormer, GFPGAN, DifFace) and all eight TEST-only labels are FFHQ-sourced;
   FFHQ and FairFace use different canonicaliser profiles (v1 vs v2) and different master resolutions (1024 vs 448 px).
   Any of these can make "looks like FFHQ" a fake cue.

2. **The errors are false positives on reals, not missed fakes.** Per-group FNR is 0.05–0.52% for every non-GDA run, and
   per-group AUC is 99.4–99.95%. Fairness is therefore set almost entirely by where each group's real-score distribution
   sits relative to one global threshold (T8).

3. **P/A/B are not the full CES-FL method.** P is BCE on originals, A is weighted BCE, B adds a soft error-disparity term.
   The counterfactual sensitivity/response terms have never run on FF2 (blocked on intervention review). B's selected
   epoch is 1, and B is worse than A on four of five F_FPR axes.

4. **The fairness term is weak.** It is computed per batch of 16; in a 20k-batch simulation it is active in 67% of batches,
   usually comparing 2–3 of 8 groups with 1–2 real images each. It also uses demographic groups, not source.

5. **Small real support and one seed.** VAL has 2,438 reals (194 Female-Black); one image moves a group FPR by ~0.5 pt.
   The primary bootstrap has 490 valid draws, below the pre-registered 900 floor.

## 3. Suggested next steps (not launched; each needs a dated proposal)

- Report FF2 group metrics **within source** (FairFace-only and FFHQ-only) alongside pooled ones; add source as a
  stratification variable in the datasheet and paper.
- Test the candidate causes on saved scores first (CPU): FPR by source × canonicaliser profile, by generator family, and
  by skin bin within source.
- If the source cue is confirmed, a source-aware real-calibration or source-balanced sampling arm is the direct fix;
  demographic-only fairness losses will not remove it.
- Keep TEST sealed until the model, threshold and decision rule are frozen.

## 4. Table index

| Table | Content | Split |
|---|---|---|
| T1 | Real/fake counts per gender×race group, split and source | composition |
| T2 | 52 non-degenerate off-the-shelf detectors (of 80): AUC, i-F_FPR, i-F_MEO, i-F_DP, i-F_OAE, i-es-AUC | public TEST |
| T3a/b | Ding format (gender/race/intersection × F_FPR/F_DP/es-AUC + AUC), FF++-trained → FF2; all / unseen models | public TEST |
| T4 | AUC per fake family, FF++-trained | public TEST |
| T5 | FF2-trained: F_FPR on gender, race, gender×race, age, skin + AUC/FPR/FNR | VAL |
| T5-axis | Per axis: F_FPR, F_DP, es-AUC, FPR gap, TPR gap, F_MEO, F_OAE, worst-group AUC | VAL |
| T6 | FairForensics-style 8 groups (sex × age × White/Non-White) | VAL |
| T7 | Age × skin 16-cell FPR and FNR | VAL |
| T8 | Per-group FPR and FNR for every axis | VAL |
| T9 | Real FPR by source dataset × race | VAL |

Caveats: age and skin bins are machine estimates (InsightFace age, ITA skin tone), not human labels. The FF++-trained
rows in T3/T4 are older (Oct) results recomputed on the frozen public TEST; Ding rows there are our reproduction, not
published numbers. The FF2-full-trained TEST rows in the source CSV were trained on the internal set (incl. RFW) and are
excluded here.

## 5. Inputs

- `../ff2_multimodel_validation_20261105/METRICS_LONG_7RUNS.csv`, `GROUP_SUPPORT_7RUNS.csv` (branch
  `results/ff2-multimodel-findings-20261105`, commit `6eef208`).
- Frozen public-release stats and benchmarks: `release_ff2_public/stats/` (`panel_public.csv`, `ding_format_public.csv`,
  `family_auc_public.csv`, `groups_*_cellvalid.csv`), frozen 25 Oct 2026.
- `diag_source.py` → private `diag_source_fpr.json` (counts only; VAL manifest sha256 `aeda0512…`, 31,107 rows per model).

## 6. Final paper table format (added 6 Nov 2026)

[FF2_TABLES_DRAFT.pdf](FF2_TABLES_DRAFT.pdf) is the current filled draft (SELECT/validation numbers, seed 17; built by `build_paper_pdf.py`). [PAPER_TABLE_TEMPLATE.tex](PAPER_TABLE_TEMPLATE.tex) holds the two FF2 tables planned for the paper; every cell is `--`
until a one-time TEST evaluation of frozen models.

- **Table A (`tab:ff2main`), Ding format.** Rows: Ori (BCE), DAG-FDD, DAW-FDD, PG-FDD, F3Net, CORE, Ding et al. (our
  run), ours; all Xception, trained on FF2 public TRAIN, SELECT-chosen checkpoint and 10%-FPR threshold, mean ± std of
  3 seeds. Columns: gender / race / gender×race × F_FPR, F_DP, es-AUC, plus AUC. Two blocks: all TEST (87,564) and
  unseen models (16,953).
- **Table B (`tab:ff2axes`), FF2-only axes.** Age and skin (F_FPR, es-AUC, worst-group AUC), race F_FPR within FairFace
  and within FFHQ (because of the source effect in T9), overall FPR and FNR.

Filling status: VAL-only results exist for BCE, DAG-FDD, DAW-FDD, Ding-style GDA and CES-FL P/A/B (seed 17).
F3Net/CORE need their verification fixed; PG-FDD and the full CES-FL method have no FF2 run; seeds 2–3 and TEST are
pending. The FF2 rows of the older `tab:ff2ding` (internal build incl. RFW, CFES ViT-L/14) must not be reused.

## 7. Metrics where CES-FL surpasses the best baseline on FF2 (added 6 Nov 2026)

Counted on the 27 metrics of `FF2_TABLES_DRAFT.pdf` (Tables 2–4), VAL, seed 17, two-decimal values. "Best baseline" =
best of BCE, DAG-FDD, DAW-FDD and Ding-style GDA per metric; "ours" = best of CES-FL P/A/B. Row-level detail:
[SOTA_COUNT_VAL.csv](SOTA_COUNT_VAL.csv).

| Table | Metrics | Ours better than every baseline | Tie | Ours better than Ding-style GDA alone |
|---|---:|---:|---:|---:|
| 2: Ding format (g/r/i × F_FPR, F_DP, es-AUC; AUC) | 10 | 4 (r F_DP, r es-AUC, i F_DP, i es-AUC) | 1 (g es-AUC) | 7 |
| 3: age, skin, within-source race, FNR | 9 | 2 (age es-AUC, race F_FPR within FairFace) | 1 (skin es-AUC) | 6 |
| 4: per-group FPR (8 groups) | 8 | 0 | 1 (Male-Black) | 2 |
| **Total** | **27** | **6** | **3** | **15** |

Ours is not best on any F_FPR axis (gender, race, gender×race, age, skin) nor on AUC. Margins of the six wins are small
(es-AUC +0.02–0.23 pt, F_DP −0.23–0.24 pt) on one seed and one VAL split. CES-FL P, which supplies most wins, is
original-only BCE, not the CES-FL fairness objective. This is not a state-of-the-art claim.

## 8. L_A added (6 Nov 2026)

L_A, the matched BCE control for L_H-PS (proposal branch `proposal/ff2-lh-padding-safe-20261106`), finished 10/10 epochs,
`COMPLETED_VERIFIED`, selected epoch 2, 2.55 GPU-hours (seed 17, plain linear head, encoder from FF2 BCE50 epoch 40).
Its VAL aggregates (same `aggregate_metrics.py` code, reproduces the run's own verified metrics on 5 axes) are in
`l_a_ps_20261106/`, and it is a new row in [FF2_TABLES_DRAFT.pdf](FF2_TABLES_DRAFT.pdf).

| | AUC | g F_FPR | r F_FPR | g×r F_FPR | age F_FPR | skin F_FPR | FNR | FFHQ real FPR | FairFace real FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L_A | 99.81 | 4.11 | 17.54 | 37.37 | 9.58 | 14.23 | 0.28 | 67.77 | 3.60 |

L_A behaves like the Xception BCE baseline (g×r F_FPR 40.52), with the same FFHQ-vs-FairFace source gap. L_H-PS is
pending the human review of its probe.

## 9. FairForensics-style cross-dataset tables (added 6 Nov 2026)

[CROSSDATASET_TABLES_DRAFT.pdf](CROSSDATASET_TABLES_DRAFT.pdf) uses the layout of FairForensics (arXiv 2608.01661) Table III:
per test set, F_FPR, F_MEO, F_DP, F_OAE over their 8 sex × age × White/Non-White groups (FairFace-classifier labels) and AUC.

- **Table 1 (filled from existing saved scores):** trained on FF++ (Ding protocol), tested on FF++, Celeb-DF, DFDC, DFD and
  FairForge-2 TEST; Ding reproductions and the CFES (CLIP) lineage, seed means at each model's FF++-validation threshold.
- **Table 2 (empty):** the exact FairForensics analogue, trained on FF2 TRAIN and tested on external datasets. No FF2-trained
  checkpoint has been evaluated there; that one-time inference needs approval.

## 10. Cause of the FFHQ-vs-FairFace gap found (6 Nov 2026)

[source_diagnosis_20261106/REPORT.md](source_diagnosis_20261106/REPORT.md): the FFHQ-real false alarms come from the sources'
**compression history**. FairFace sources are 448 px q75 4:2:0 JPEGs whose trace survives the v2 crop; FFHQ masters are
lossless. Re-making the 242 FFHQ VAL reals with FairFace-like geometry plus a q75 4:2:0 round-trip drops L_A's flag rate from
67.8% to 2.1%, and the same treatment lets 94% of FFHQ-sourced fakes pass. Every FF2 fairness number above partly measures this
shortcut. The dataset needs a common compression history across sources before release and retraining.

## 11. L_A/L_H added to the cross-dataset table (7 Nov 2026)

CES-FL L_A and L_H (Xception plain head, FF++-trained, seed 17) are now rows of Table 1 in
[CROSSDATASET_TABLES_DRAFT.pdf](CROSSDATASET_TABLES_DRAFT.pdf), rescored on the FairForensics 8 groups from their saved 3 Nov
scores (no inference; 100% label join; AUCs reproduce the published ones within 0.0005). A1 thresholds re-derived on FF++
validation only (L_A 18.314, L_H 5.077). No FairForge-2 TEST scores exist for them. Single seed vs 3-seed means elsewhere;
at A1 their DFDC TPR is ~10%. Details: `lh_ids8_rescore/`.

## 12. L_H-PS result on FF2 (7 Nov 2026)

L_H-PS (our fairness method: hierarchical-real error control + padding-safe probe sensitivity/control + response alignment;
plain-head Xception, encoder from FF2 BCE50, 10 epochs, seed 17) finished 10/10 epochs, `COMPLETED_VERIFIED`, selected epoch
10, 1.49 GPU-hours. Human review of the probe: blanket approval by Ashish, relayed by the user (see the proposal log).
VAL aggregates in `l_h_ps_20261107/`; row added to [FF2_TABLES_DRAFT.pdf](FF2_TABLES_DRAFT.pdf).

| | AUC | g F_FPR | r F_FPR | g×r F_FPR | age F_FPR | skin F_FPR | FNR | FFHQ real FPR | FairFace real FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L_A (control) | 99.81 | 4.11 | 17.54 | 37.37 | 9.58 | 14.23 | 0.28 | 67.77 | 3.60 |
| L_H-PS (ours) | 99.75 | 4.28 | 17.55 | 36.23 | 11.90 | 13.35 | 0.38 | 66.94 | 3.69 |

**Pre-declared reading: not met.** g×r F_FPR fell by 1.14 points (rule: ≥ 2.0); AUC −0.06 (≤ 0.5, ok); FNR 1.33× (≤ 2×, ok).
As predicted before training, L_H-PS leaves the FFHQ-vs-FairFace source gap essentially unchanged (66.9% vs 3.7%).

## 13. Stretch experiment: real-only FPR equalisation + JPEG augmentation (8 Nov 2026)

Pre-registered on `experiment/ff2-realfpr-stretch-20261107` (Codex gpt-6-astra review, gate Q-J passed). **Target not met.**
Only S (real-only false-alarm equalisation on group-balanced reals, no JPEG) has a checkpoint inside AUC ≥ 99.5 / FNR ≤ 0.5651%:
row **RF-S** in [FF2_TABLES_DRAFT.pdf](FF2_TABLES_DRAFT.pdf): AUC 99.80, FNR 0.29%, g×r F_FPR 34.12 (L_A 37.37); FFHQ real
FPR 64.5% vs FairFace 4.0%. The JPEG-augmented runs (J, SJ, SJF) reached g×r F_FPR 21–28 only at AUC 99.1–99.6 with FNR
0.6–1.6%, so none has a feasible checkpoint; Ding-style GDA (12.44) remains the fairest FF2 run. Aggregates: `rf_s_20261108/`.
