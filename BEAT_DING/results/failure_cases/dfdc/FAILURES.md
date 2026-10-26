# BD0 seed 7: failure cases on DFDC (cross-domain)

Model: BD0 s7 (CLIP ViT-L/14 + rank-1 adapters, CFES-D + Arm E + ESA). Scores: `runs/BD0_s7/eval/dfdc_scores.npz`.
Comparison: Ding reproduction (official code, Xception, s7, ep49) at its 0.5 rule (logit >= 0).
Every number below comes from `analyze_dfdc_failures.py` in this folder (raw output: `failures_dfdc.json`, `run.log`;
full tables: `tables.md`, appended at the end of this file).

**Data notes.**
- 22,857 protocol rows (21,381 unique images). 1,476 paths appear twice in Lin's DFDC test CSV. All counts are over rows, as the metric counts them.
- Reals: 14,699 rows, 473 videos. Fakes: 8,158 rows, 284 videos.
- **DFDC fakes carry no manipulation-method label** (method = `unknown` for all fakes). No per-method or per-family breakdown is possible on this protocol.
- Asian fakes exist in only 4 videos (120 rows).

## Operating points

| op | logit t | FP | FPR % | FN | FNR % | TPR % | real videos with >= 1 FP | fake videos with >= 1 miss |
|---|---|---|---|---|---|---|---|---|
| **matched** (primary) | -73.76 | 4,885 | **33.23** | 2,246 | 27.53 | 72.47 | 380 / 473 | 197 / 284 |
| native | -65.97 | 2,857 | 19.44 | 3,324 | 40.75 | 59.25 | 321 / 473 | 225 / 284 |
| A1 | -19.13 | 280 | 1.90 | 5,942 | 72.84 | 27.16 | 74 / 473 | 261 / 284 |
| Ding repro 0.5 | 0 | 7,221 | 49.13 | 3,622 | 44.40 | 55.60 | 385 / 473 | 229 / 284 |

**Score shift.** The same matched threshold gives FPR 15.81% on FF++ val but 33.23% on DFDC reals. DFDC real logits sit higher than FF++ val reals:

| | p10 | median | p90 |
|---|---|---|---|
| FF++ val reals | -108.15 | -92.22 | -67.54 |
| DFDC reals | -97.21 | -80.86 | -57.65 |
| DFDC fakes | -85.94 | -59.45 | 30.73 |

40.86% of DFDC real rows and 30.74% of fake rows lie within ±10 logits of t_matched.

## Failure modes (plain words)

1. **The threshold does not transfer, so false positives are everywhere rather than in a few videos.**
   - At the matched threshold, 33.23% of real DFDC frames are flagged, about twice the FF++ val FPR (15.81%).
   - 380 of 473 real videos have at least one FP frame. 142 videos are flagged on half or more of their frames, and 49 on 90% or more.
   - FPs are spread out: 79 videos are needed to cover half of all FP frames, and the top 15 videos hold only 14.70% of them.
2. **Dark, low-contrast and blurry footage is read as fake.** This holds within every race group, so it is not just skin tone.
   - Real FPR by brightness quartile: Q1 (darkest) 42.91% vs Q4 26.63%. By contrast: Q1 43.28% vs Q4 23.90%. Lowest sharpness quartile: 47.35%.
   - The brightness effect within each cleaner-label race group (FPR, darkest vs brightest quartile):

     | race | brightness Q1 | brightness Q4 |
     |---|---|---|
     | white | 43.1% | 26.8% |
     | Black | 46.5% | 28.5% |
     | Asian | 54.3% | 20.3% |

   - Black subjects' real frames are darker on average: median luminance 71.5 vs 100.8 for white subjects. Lighting therefore feeds a skin-tone gap:
     - ITA Q1 (darkest, FF++ edges) FPR 36.35% vs Q4 23.82%.
     - MST 7-10 FPR 36.78% vs MST 4-6 24.64%.
     - Cleaner-label Black 39.31% vs white 30.76%.
   - The same bias helps on fakes: dark fakes are rarely missed (FNR 16.67% in brightness Q1 vs 34.44% in Q3).
3. **Using the Xu labels the metric relies on, the intersectional FPR gap looks no larger than chance across videos. With cleaner labels it is a real gap that falls on women of colour.**
   - At matched, i-F_FPR = 32.23. It is driven by F-Black (+12.24 pp above the pooled FPR) and M-Asian (-10.74 pp).
   - A permutation null that shuffles group labels across real videos gives a median of 34.18 (5-95%: 19.82-52.80, p = 0.57). On Xu labels, the disparity cannot be told apart from noise in which videos land in which group.
   - It is also fragile:
     - Greedily dropping 9 real videos halves it.
     - Random label assignments need a median of 7 drops (range 4-14), so this fragility is generic.
     - Dropping the top 10 of 68 F-Black videos takes F-Black FPR from 45.48% to 36.17%.
   - The labels themselves are noisy:
     - The Xu and cleaner video-level labels put 76 of 473 real videos in different intersectional groups.
     - 6 of the 15 top-FP videos are mislabelled (e.g. 0_6461, 0_6305 and 0_5143 are labelled male but are female under the cleaner labels).
   - With cleaner video-level labels, i-F_FPR at matched is **68.79**, above its own permutation null (median 36.25, p95 54.70, p = 0.0025). Group FPRs:
     - F-Black 45.47%, F-Asian 43.85%
     - M-Asian 14.86%, M-Others 17.03%
   - Ding's reproduction scores 62.52 on the same cleaner labels.
4. **Older faces are flagged more often, but only under one age estimator.**
   - InsightFace age 60+: FPR 49.20% (437 rows, only 15 videos) vs 27.81% for under-30s.
   - AI-Face age does not agree: 65+ at 31.70% vs 36.02% for ages 25-44. Treat this as weak evidence.
5. **Missed fakes are mostly clean, well-lit, frontal clips of white women.**
   - F-White alone accounts for 49.82% of all FN frames (1,119 / 2,246) while holding 37.7% of fake rows. Its FNR is 36.39%, vs 16.92% for M-White.
   - Women are missed more often than men: Xu labels 31.90% vs 21.00%, cleaner labels 30.59% vs 22.00%.
   - 11 of the 15 most-missed fake videos are F-White. 20 of the 24 videos on the FN contact sheet are white women under the cleaner labels.
   - Misses cluster by video:
     - 72 of 284 fake videos are missed on half or more of their frames, and 8 are missed on every frame.
     - 43 videos hold half of all FN frames.
   - Ding's reproduction is worse here: 128 fake videos are missed on half or more of their frames.
6. **Asian fakes are almost absent, so their numbers are unreliable.**
   - There are 4 Asian fake videos (120 rows), with FNR 57.50%. F-Asian is a single video with 30 rows.
   - Any Asian TPR, AUC or F_MEO figure on DFDC rests on 1-3 identities.
7. **Our failures overlap with Ding's only a little more than chance, but the most confident failures are shared.**
   - At matched, 54.78% of our FP frames are also Ding FPs (chance would be Ding's FPR of 49.13%), and 54.32% of our FN frames are also Ding misses (chance 44.40%).
   - The two models' logits barely agree in ranking: Spearman 0.246 on reals and 0.287 on fakes.
   - Errors only we make: 2,209 FP and 1,026 FN frames. Errors only Ding makes: 4,545 FP and 2,402 FN frames.
   - At video level:
     - Our FPR exceeds Ding's by >= 50 pp on 46 real videos; the reverse holds on 125.
     - Our miss rate exceeds Ding's by >= 50 pp on 22 fake videos; the reverse holds on 81.
   - On the contact sheets, 22 of the 24 top FP videos and 23 of the 24 top FN videos are also wrong for Ding. The hardest cases are dataset-level, not specific to BD0.
8. **The most confident errors come from a handful of videos and include possibly corrupted "real" clips.** This is a visual observation, not quantified.
   - The 12 highest-logit FPs come from only 4 videos (0_7434 F-Black contributes 6). The 12 lowest-logit FNs come from 6 videos (0_4680 F-White contributes 6).
   - On the FP sheet, several "real" frames show compositing or overlay artefacts that would legitimately look fake:
     - 0_7406: a pasted rectangular face patch
     - 0_4516: magenta glitch bands
     - 0_4543: blue line overlays
   - These may be DFDC test-set distractors or augmentations. They are worth a manual check before treating them as model errors.

## Key tables

### FPR / FNR by Xu intersectional group (the metric's groups)

| group | real n (videos) | FPR % matched | FPR % native | FPR % A1 | FPR % Ding | fake n (videos) | FNR % matched | FNR % native | FNR % A1 | FNR % Ding |
|---|---|---|---|---|---|---|---|---|---|---|
| M-Asian | 458 (14) | 22.49 | 14.41 | 0.66 | 51.75 | 90 (3) | 57.78 | 74.44 | 98.89 | 54.44 |
| M-White | 4190 (138) | 32.39 | 18.90 | 1.67 | 43.22 | 2270 (79) | 16.92 | 29.30 | 67.75 | 46.78 |
| M-Black | 2008 (64) | 30.43 | 17.98 | 0.75 | 62.25 | 507 (17) | 24.85 | 40.63 | 92.50 | 55.82 |
| M-Others | 1244 (35) | 33.60 | 21.54 | 2.57 | 37.86 | 404 (15) | 30.94 | 43.07 | 67.82 | 29.46 |
| F-Asian | 628 (20) | 31.37 | 16.40 | 0.96 | 47.13 | 30 (1) | 56.67 | 60.00 | 90.00 | 23.33 |
| F-White | 3366 (110) | 29.98 | 16.79 | 1.52 | 48.63 | 3075 (107) | 36.39 | 50.28 | 74.93 | 49.50 |
| F-Black | 2100 (68) | **45.48** | 28.29 | 4.14 | 55.95 | 1285 (45) | 21.32 | 35.72 | 73.46 | 35.56 |
| F-Others | 705 (24) | 33.33 | 15.32 | 2.27 | 48.79 | 497 (17) | 29.98 | 38.03 | 59.76 | 24.75 |

### Intersectional F_FPR: how much is real?

| op | i-F_FPR (Xu) | perm-null median [5-95%] | p | greedy video drops to halve | i-F_FPR (cleaner video labels) |
|---|---|---|---|---|---|
| matched | 32.23 | 34.18 [19.82, 52.80] | 0.57 | 9 (random labels: median 7) | 68.79 (null 36.25 [20.24, 54.70], p = 0.0025) |
| native | 27.78 | 26.81 [15.24, 41.29] | 0.44 | 9 | 42.97 |
| A1 | 7.25 | 7.49 [4.60, 11.92] | 0.54 | 7 | 9.50 |
| Ding 0.5 | 42.56 | – | – | – | 62.52 |

### Real FPR (matched) by attribute: selected rows

| axis | bin | n | FPR % |
|---|---|---|---|
| cleaner gender (frame) | female / male | 8090 / 6609 | 37.56 / 27.93 |
| cleaner race (video) | Black / Asian / white / others | 4363 / 1525 / 7707 / 1104 | 38.99 / 35.41 / 30.92 / 23.64 |
| FairFace race7 | SE Asian / Black / White / E Asian | 356 / 4158 / 6770 / 916 | 54.21 / 40.14 / 30.90 / 22.05 |
| ITA (FF++ edges) | Q1 darkest / Q4 lightest | 10645 / 1553 | 36.35 / 23.82 |
| ITA (DFDC-local edges) | Q1-Q3 / Q4 lightest | 3465, 3397, 3673 / 3450 | 34.57, 37.65, 36.46 / 25.28 |
| MST (video) | 7-10 / 4-6 | 10457 / 4212 | 36.78 / 24.64 |
| age (InsightFace) | 60+ / <30 | 437 / 5984 | 49.20 / 27.81 |
| brightness | Q1 / Q4 | 3526 / 3699 | 42.91 / 26.63 |
| sharpness | Q1 / Q2 / Q3 / Q4 | 3164 / 3559 / 3652 / 4324 | 47.35 / 28.60 / 23.25 / 35.15 |
| grayscale frames | gray / colour | 146 / 14553 | 57.53 / 32.99 |

### Top-15 real videos by FP frames (matched)

| video | Xu group | cleaner gender / race | frames | FP | FPR % | Ding FPR % | ITA (video) | median brightness |
|---|---|---|---|---|---|---|---|---|
| 0_4069 | M-Black | male / black | 60 | 60 | 100.0 | 96.7 | -45.2 | 96.7 |
| 0_6461 | M-Others | **female** / others | 60 | 60 | 100.0 | 0.0 | -51.4 | 64.1 |
| 0_6104 | M-Others | male / others | 60 | 58 | 96.7 | 100.0 | 5.7 | 85.7 |
| 0_6305 | M-Asian | **female** / asian | 60 | 58 | 96.7 | 0.0 | -71.8 | 41.2 |
| 0_7138 | F-White | female / white | 60 | 58 | 96.7 | 0.0 | -83.5 | 47.9 |
| 0_5432 | F-Black | female / black | 60 | 56 | 93.3 | 90.0 | -65.8 | 33.4 |
| 0_5536 | M-Black | **female** / black | 60 | 56 | 93.3 | 90.0 | -29.4 | 67.9 |
| 0_7509 | F-White | female / **asian** | 60 | 52 | 86.7 | 0.0 | -89.1 | 40.7 |
| 0_5143 | M-Others | **female / asian** | 54 | 50 | 92.6 | 0.0 | n/a | 18.3 |
| 0_6112 | M-White | male / white | 60 | 48 | 80.0 | 0.0 | -11.8 | 112.0 |
| 0_4763 | F-Asian | female / asian | 60 | 38 | 63.3 | 30.0 | 26.6 | 123.6 |
| 0_6247 | M-White | male / white | 60 | 32 | 53.3 | 53.3 | 8.5 | 102.0 |
| 0_7068 | M-Others | **female / black** | 60 | 32 | 53.3 | 53.3 | -23.1 | 68.1 |
| 0_4421 | F-Black | female / black | 30 | 30 | 100.0 | 100.0 | -68.2 | 42.4 |
| 0_4459 | M-White | male / white | 30 | 30 | 100.0 | 80.0 | -56.7 | 80.6 |

(bold = cleaner label disagrees with the Xu label)

### Top-15 fake videos by missed frames (matched)

| video | Xu group | frames | FN | FNR % | Ding FNR % | median brightness |
|---|---|---|---|---|---|---|
| 0_4203 | F-White | 60 | 56 | 93.3 | 13.3 | 105.4 |
| 0_4367 | F-Others | 60 | 32 | 53.3 | 63.3 | 96.4 |
| 0_4680 | F-White | 30 | 30 | 100.0 | 100.0 | 177.5 |
| 0_4734 | F-White | 30 | 30 | 100.0 | 90.0 | 109.6 |
| 0_4776 | F-White | 30 | 30 | 100.0 | 100.0 | 98.4 |
| 0_6009 | F-White | 30 | 30 | 100.0 | 53.3 | 105.5 |
| 0_4093 | F-White | 30 | 29 | 96.7 | 33.3 | 96.4 |
| 0_4414 | M-Others | 30 | 29 | 96.7 | 100.0 | 14.8 |
| 0_4799 | M-White | 30 | 29 | 96.7 | 23.3 | 95.2 |
| 0_6894 | F-White | 30 | 29 | 96.7 | 100.0 | 124.9 |
| 0_7411 | F-White | 30 | 29 | 96.7 | 80.0 | 105.1 |
| 0_5695 | F-Black (cleaner: white) | 30 | 28 | 93.3 | 16.7 | 134.2 |
| 0_6231 | F-White | 30 | 28 | 93.3 | 33.3 | 52.9 |
| 0_7113 | F-White | 30 | 28 | 93.3 | 66.7 | 111.2 |
| 0_7129 | F-White | 30 | 28 | 93.3 | 86.7 | 126.5 |

### Shared vs unique with Ding (frame rows)

| op | our FP also Ding FP % | Ding FP also ours % | FP only ours / only Ding | our FN also Ding FN % | Ding FN also ours % | FN only ours / only Ding |
|---|---|---|---|---|---|---|
| matched | 54.78 | 37.06 | 2209 / 4545 | 54.32 | 33.68 | 1026 / 2402 |
| native | 58.07 | 22.97 | 1198 / 5562 | 52.26 | 47.96 | 1587 / 1885 |
| A1 | 86.43 | 3.35 | 38 / 6979 | 50.07 | 82.14 | 2967 / 647 |

- **We fail, Ding is right.**
  - Real videos where our FPR is 100% and Ding's is 0%: 0_7834, 0_6461, 0_7703, 0_7459, 0_4859, 0_4343, 0_4996.
  - Fakes we miss and Ding catches: 0_6912 (86.7% vs 0%), 0_4203 (93.3% vs 13.3%), 0_4401 (76.7% vs 0%).
- **Ding fails, we are right.**
  - Reals Ding flags on every frame and we flag on none: 0_7747, 0_7741, 0_4409, 0_5781, 0_6887.
  - Fakes Ding misses on every frame and we catch on every frame: 0_7755, 0_7230, 0_6643, 0_7206, 0_4254.
- Frame-level top-10 lists for both directions are in `tables.md` §3.

### Most confident errors

**Top-12 FP** (highest logit among reals):

| video | frames in top 12 | logits | Ding |
|---|---|---|---|
| 0_6104 | 71, 51 | 67.54, 49.88 | FP |
| 0_7434 | 171, 221, 121, 51, 41, 231 | 61.53 to 47.22 | FP |
| 0_7964 | 51 | 55.01 | FP |
| 0_5633 | 241, 291, 121 | 50.49 to 45.29 | FP |

**Top-12 FN** (lowest logit among fakes):

| video | frames in top 12 | logits | Ding |
|---|---|---|---|
| 0_5573 | 191 | -122.63 | missed |
| 0_4680 | 251, 261, 91, 241, 201, 181 | -119.21 to -111.50 | missed |
| 0_7028 | 191, 41 | -118.14, -111.89 | missed frame 191, caught frame 41 |
| 0_7113 | 251 | -112.92 | missed |
| 0_4367 | 91 | -112.59 | missed |
| 0_4776 | 1 | -110.39 | missed |

All frames are under `EXACTMATCH_plan/work/extracted/dfdc/<video>_<frame>.png`. Full paths are in `tables.md` §4.

### Contact sheets

- `fp_sheet.png` and `fn_sheet.png`: 24 tiles each, one frame per video (the highest-confidence error in that video). Each caption gives video id, Xu group, method, logit and the Ding decision with its logit.
- The tile lists, with paths, are in `fp_sheet_items.csv` and `fn_sheet_items.csv`.

## Files

| file | contents |
|---|---|
| `analyze_dfdc_failures.py` | the analysis (CPU; aiface env) |
| `failures_dfdc.json` | every number |
| `run.log` | stdout of the last run |
| `tables.md` | all breakdowns at matched, native and A1 plus Ding |
| `fp_videos_matched.csv`, `fn_videos_matched.csv` | per-video error tables |
| `fp_sheet.png`, `fn_sheet.png` | contact sheets |
| `fp_sheet_items.csv`, `fn_sheet_items.csv` | contact-sheet tile lists |
| `image_quality.csv` | per-image brightness, contrast, sharpness (Laplacian variance) and saturation |

---

# DFDC failure tables (generated by analyze_dfdc_failures.py)

Rows 22857 (21381 unique images; 1476 rows are repeated paths, counted as the metric counts them). Reals 14699 rows / 473 videos; fakes 8158 rows / 284 videos. DFDC fakes carry NO method label in the protocol CSV (all 'unknown').

## Operating points

| op | logit t | FP | FPR% | FN | FNR% | TPR% | FP videos (>=1 FP frame) | FN videos (>=1 miss) |
|---|---|---|---|---|---|---|---|---|
| matched | -73.76 | 4885 | 33.23 | 2246 | 27.53 | 72.47 | 380/473 | 197/284 |
| native | -65.97 | 2857 | 19.44 | 3324 | 40.75 | 59.25 | 321/473 | 225/284 |
| A1 | -19.13 | 280 | 1.9 | 5942 | 72.84 | 27.16 | 74/473 | 261/284 |
| Ding repro 0.5 | None | 7221 | 49.13 | 3622 | 44.4 | 55.6 | 385/473 | 229/284 |

## 1. False positives (reals flagged fake)

**by grp**

| grp | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| F-Asian | 628 | 20 | 197 | 31.37 | 103 | 16.40 | 6 | 0.96 | 47.13 |
| F-Black | 2100 | 68 | 955 | 45.48 | 594 | 28.29 | 87 | 4.14 | 55.95 |
| F-Others | 705 | 24 | 235 | 33.33 | 108 | 15.32 | 16 | 2.27 | 48.79 |
| F-White | 3366 | 110 | 1009 | 29.98 | 565 | 16.79 | 51 | 1.52 | 48.63 |
| M-Asian | 458 | 14 | 103 | 22.49 | 66 | 14.41 | 3 | 0.66 | 51.75 |
| M-Black | 2008 | 64 | 611 | 30.43 | 361 | 17.98 | 15 | 0.75 | 62.25 |
| M-Others | 1244 | 35 | 418 | 33.60 | 268 | 21.54 | 32 | 2.57 | 37.86 |
| M-White | 4190 | 138 | 1357 | 32.39 | 792 | 18.90 | 70 | 1.67 | 43.22 |

**by xu_gender**

| xu_gender | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| female | 6799 | 222 | 2396 | 35.24 | 1370 | 20.15 | 160 | 2.35 | 50.77 |
| male | 7900 | 251 | 2489 | 31.51 | 1487 | 18.82 | 120 | 1.52 | 47.71 |

**by xu_race**

| xu_race | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| asian | 1086 | 34 | 300 | 27.62 | 169 | 15.56 | 9 | 0.83 | 49.08 |
| black | 4108 | 132 | 1566 | 38.12 | 955 | 23.25 | 102 | 2.48 | 59.03 |
| others | 1949 | 59 | 653 | 33.50 | 376 | 19.29 | 48 | 2.46 | 41.82 |
| white | 7556 | 248 | 2366 | 31.31 | 1357 | 17.96 | 121 | 1.60 | 45.63 |

**by cl_gender**

| cl_gender | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| female | 8090 | 308 | 3039 | 37.56 | 1809 | 22.36 | 173 | 2.14 | 50.56 |
| male | 6609 | 272 | 1846 | 27.93 | 1048 | 15.86 | 107 | 1.62 | 47.37 |

**by cl_race**

| cl_race | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| asian | 1569 | 106 | 562 | 35.82 | 306 | 19.50 | 19 | 1.21 | 42.83 |
| black | 4363 | 190 | 1715 | 39.31 | 1047 | 24.00 | 103 | 2.36 | 59.55 |
| others | 1199 | 103 | 280 | 23.35 | 177 | 14.76 | 35 | 2.92 | 41.78 |
| white | 7568 | 300 | 2328 | 30.76 | 1327 | 17.53 | 123 | 1.63 | 45.59 |

**by cl_race_vid**

| cl_race_vid | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| asian | 1525 | 47 | 540 | 35.41 | 287 | 18.82 | 10 | 0.66 | 42.69 |
| black | 4363 | 139 | 1701 | 38.99 | 1037 | 23.77 | 103 | 2.36 | 59.29 |
| others | 1104 | 34 | 261 | 23.64 | 166 | 15.04 | 42 | 3.80 | 45.65 |
| white | 7707 | 253 | 2383 | 30.92 | 1367 | 17.74 | 125 | 1.62 | 45.14 |

**by ff_race7**

| ff_race7 | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| Black | 4158 | 186 | 1669 | 40.14 | 1032 | 24.82 | 102 | 2.45 | 58.92 |
| East Asian | 916 | 87 | 202 | 22.05 | 85 | 9.28 | 10 | 1.09 | 46.29 |
| Indian | 149 | 37 | 46 | 30.87 | 26 | 17.45 | 7 | 4.70 | 32.21 |
| Latino_Hispanic | 763 | 110 | 260 | 34.08 | 144 | 18.87 | 9 | 1.18 | 43.12 |
| Middle Eastern | 1587 | 177 | 423 | 26.65 | 247 | 15.56 | 46 | 2.90 | 42.72 |
| Southeast Asian | 356 | 52 | 193 | 54.21 | 137 | 38.48 | 20 | 5.62 | 47.75 |
| White | 6770 | 310 | 2092 | 30.90 | 1186 | 17.52 | 86 | 1.27 | 46.12 |

**by ff_vid_sex**

| ff_vid_sex | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| Female | 8439 | 270 | 3128 | 37.07 | 1838 | 21.78 | 175 | 2.07 | 51.18 |
| Male | 6260 | 203 | 1757 | 28.07 | 1019 | 16.28 | 105 | 1.68 | 46.36 |

**by age_if**

| age_if | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| 30-44 | 6826 | 215 | 2486 | 36.42 | 1468 | 21.51 | 101 | 1.48 | 50.59 |
| 45-59 | 1452 | 49 | 520 | 35.81 | 328 | 22.59 | 6 | 0.41 | 41.05 |
| 60+ | 437 | 15 | 215 | 49.20 | 142 | 32.49 | 0 | 0.00 | 35.93 |
| <30 | 5984 | 194 | 1664 | 27.81 | 919 | 15.36 | 173 | 2.89 | 50.38 |

**by age_aif**

| age_aif | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| 25-44 | 9474 | 303 | 3413 | 36.02 | 1960 | 20.69 | 206 | 2.17 | 52.70 |
| 45-64 | 2656 | 85 | 737 | 27.75 | 429 | 16.15 | 14 | 0.53 | 38.06 |
| 65+ | 896 | 32 | 284 | 31.70 | 177 | 19.75 | 3 | 0.33 | 45.76 |
| <25 | 1673 | 53 | 451 | 26.96 | 291 | 17.39 | 57 | 3.41 | 48.24 |

**by ff_vid_age4**

| ff_vid_age4 | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| 30-49 | 6000 | 190 | 2187 | 36.45 | 1251 | 20.85 | 107 | 1.78 | 44.45 |
| 50-69 | 1098 | 39 | 240 | 21.86 | 154 | 14.03 | 6 | 0.55 | 53.92 |
| 70+ | 30 | 1 | 5 | 16.67 | 4 | 13.33 | 0 | 0.00 | 50.00 |
| <30 | 7571 | 243 | 2453 | 32.40 | 1448 | 19.13 | 167 | 2.21 | 52.13 |

**by ita4_ffpp**

| ita4_ffpp | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| Q1 darkest | 10645 | 341 | 3869 | 36.35 | 2268 | 21.31 | 233 | 2.19 | 55.50 |
| Q2 | 1043 | 35 | 256 | 24.54 | 157 | 15.05 | 20 | 1.92 | 37.87 |
| Q3 | 744 | 24 | 193 | 25.94 | 101 | 13.58 | 2 | 0.27 | 40.05 |
| Q4 lightest | 1553 | 50 | 370 | 23.82 | 215 | 13.84 | 20 | 1.29 | 27.75 |
| n/a | 714 | 23 | 197 | 27.59 | 116 | 16.25 | 5 | 0.70 | 26.47 |

**by ita4_loc**

| ita4_loc | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| Q1 darkest | 3465 | 113 | 1198 | 34.57 | 681 | 19.65 | 41 | 1.18 | 66.96 |
| Q2 | 3397 | 112 | 1279 | 37.65 | 820 | 24.14 | 83 | 2.44 | 56.08 |
| Q3 | 3673 | 112 | 1339 | 36.46 | 726 | 19.77 | 109 | 2.97 | 45.03 |
| Q4 lightest | 3450 | 113 | 872 | 25.28 | 514 | 14.90 | 42 | 1.22 | 33.42 |
| n/a | 714 | 23 | 197 | 27.59 | 116 | 16.25 | 5 | 0.70 | 26.47 |

**by mst3**

| mst3 | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| MST 1-3 | 30 | 1 | 1 | 3.33 | 1 | 3.33 | 0 | 0.00 | 0.00 |
| MST 4-6 | 4212 | 136 | 1038 | 24.64 | 620 | 14.72 | 77 | 1.83 | 35.66 |
| MST 7-10 | 10457 | 336 | 3846 | 36.78 | 2236 | 21.38 | 203 | 1.94 | 54.69 |

**by gray_s**

| gray_s | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| colour | 14553 | 470 | 4801 | 32.99 | 2817 | 19.36 | 280 | 1.92 | 49.40 |
| grayscale | 146 | 14 | 84 | 57.53 | 40 | 27.40 | 0 | 0.00 | 21.92 |

**by det_q**

| det_q | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| det Q1 (lowest) | 3838 | 338 | 1455 | 37.91 | 846 | 22.04 | 40 | 1.04 | 53.18 |
| det Q2 | 3718 | 410 | 1193 | 32.09 | 666 | 17.91 | 49 | 1.32 | 50.05 |
| det Q3 | 3747 | 425 | 1123 | 29.97 | 634 | 16.92 | 79 | 2.11 | 46.41 |
| det Q4 | 3382 | 336 | 1100 | 32.53 | 701 | 20.73 | 112 | 3.31 | 46.72 |
| n/a | 14 | 5 | 14 | 100.00 | 10 | 71.43 | 0 | 0.00 | 0.00 |

**by lum_mean_q**

| lum_mean_q | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| brightness Q1 (low) | 3526 | 174 | 1513 | 42.91 | 959 | 27.20 | 94 | 2.67 | 61.74 |
| brightness Q2 | 3902 | 227 | 1352 | 34.65 | 768 | 19.68 | 96 | 2.46 | 56.07 |
| brightness Q3 | 3572 | 224 | 1035 | 28.98 | 569 | 15.93 | 50 | 1.40 | 41.38 |
| brightness Q4 (high) | 3699 | 204 | 985 | 26.63 | 561 | 15.17 | 40 | 1.08 | 37.25 |

**by lum_std_q**

| lum_std_q | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| contrast Q1 (low) | 3348 | 201 | 1449 | 43.28 | 906 | 27.06 | 100 | 2.99 | 59.08 |
| contrast Q2 | 3494 | 247 | 1141 | 32.66 | 646 | 18.49 | 70 | 2.00 | 51.37 |
| contrast Q3 | 3756 | 259 | 1315 | 35.01 | 798 | 21.25 | 85 | 2.26 | 50.27 |
| contrast Q4 (high) | 4101 | 221 | 980 | 23.90 | 507 | 12.36 | 25 | 0.61 | 38.04 |

**by sharp_lapvar_q**

| sharp_lapvar_q | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| sharpness Q1 (low) | 3164 | 173 | 1498 | 47.35 | 896 | 28.32 | 109 | 3.45 | 69.72 |
| sharpness Q2 | 3559 | 228 | 1018 | 28.60 | 555 | 15.59 | 101 | 2.84 | 59.45 |
| sharpness Q3 | 3652 | 258 | 849 | 23.25 | 484 | 13.25 | 56 | 1.53 | 46.28 |
| sharpness Q4 (high) | 4324 | 222 | 1520 | 35.15 | 922 | 21.32 | 14 | 0.32 | 27.96 |

**by sat_mean_q**

| sat_mean_q | n | videos | FP@matched | FPR%@matched | FP@native | FPR%@native | FP@A1 | FPR%@A1 | FPR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| saturation Q1 (low) | 3594 | 184 | 1114 | 31.00 | 620 | 17.25 | 20 | 0.56 | 42.10 |
| saturation Q2 | 3586 | 223 | 1117 | 31.15 | 615 | 17.15 | 102 | 2.84 | 42.58 |
| saturation Q3 | 3496 | 226 | 1174 | 33.58 | 660 | 18.88 | 89 | 2.55 | 53.52 |
| saturation Q4 (high) | 4023 | 202 | 1480 | 36.79 | 962 | 23.91 | 69 | 1.72 | 57.42 |

### FP concentration by video (matched)

- top5_share%: 6.02
- top10_share%: 11.38
- top15_share%: 14.7
- top25_share%: 20.84
- top50_share%: 35.56
- top100_share%: 59.24
- videos_for_50pct_of_FP: 79
- videos_for_80pct_of_FP: 166
- videos_with_any_FP: 380
- videos_with_FPR>=50%: 142
- videos_with_FPR>=90%: 49
- video_fpr_median%: 20.0

**Top-15 real videos by #FP frames (matched)**

| video | grp | n | fp | fpr% | ding_fpr% | cl_gender_vid | cl_race_vid | ff_race7 | age_vid_if | ita_vid | ita4 | lum | sharp | max_logit | med_logit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0_4069 | M-Black | 60 | 60 | 100.00 | 96.67 | male | black | Black | 39.00 | -45.18 | Q1 darkest | 96.68 | 1.58 | -45.92 | -58.24 |
| 0_6461 | M-Others | 60 | 60 | 100.00 | 0.00 | female | others | White | 37.00 | -51.39 | Q1 darkest | 64.10 | 280.67 | -33.10 | -58.29 |
| 0_6104 | M-Others | 60 | 58 | 96.67 | 100.00 | male | others | Southeast Asian | 26.00 | 5.67 | Q1 darkest | 85.65 | 1.47 | 67.54 | -25.33 |
| 0_6305 | M-Asian | 60 | 58 | 96.67 | 0.00 | female | asian | Southeast Asian | 39.00 | -71.82 | Q1 darkest | 41.19 | 1257.73 | -48.92 | -62.58 |
| 0_7138 | F-White | 60 | 58 | 96.67 | 0.00 | female | white | White | 37.00 | -83.52 | Q1 darkest | 47.93 | 611.34 | -25.00 | -59.83 |
| 0_5432 | F-Black | 60 | 56 | 93.33 | 90.00 | female | black | Black | 26.00 | -65.84 | Q1 darkest | 33.36 | 18.05 | 1.21 | -61.64 |
| 0_5536 | M-Black | 60 | 56 | 93.33 | 90.00 | female | black | Black | 44.00 | -29.41 | Q1 darkest | 67.88 | 8.59 | -39.76 | -61.87 |
| 0_7509 | F-White | 60 | 52 | 86.67 | 0.00 | female | asian | Southeast Asian | 63.00 | -89.13 | Q1 darkest | 40.71 | 1.33 | -54.87 | -67.23 |
| 0_5143 | M-Others | 54 | 50 | 92.59 | 0.00 | female | asian | Latino_Hispanic | 39.00 | nan | nan | 18.30 | 1257.72 | -34.61 | -56.57 |
| 0_6112 | M-White | 60 | 48 | 80.00 | 0.00 | male | white | White | 25.50 | -11.78 | Q1 darkest | 111.96 | 2.83 | -63.31 | -69.68 |
| 0_4763 | F-Asian | 60 | 38 | 63.33 | 30.00 | female | asian | Middle Eastern | 33.00 | 26.56 | Q2 | 123.57 | 5.70 | -52.63 | -69.24 |
| 0_6247 | M-White | 60 | 32 | 53.33 | 53.33 | male | white | Middle Eastern | 23.00 | 8.45 | Q1 darkest | 102.02 | 9.54 | -38.92 | -71.83 |
| 0_7068 | M-Others | 60 | 32 | 53.33 | 53.33 | female | black | Black | 27.50 | -23.06 | Q1 darkest | 68.09 | 18.59 | -43.73 | -73.62 |
| 0_4421 | F-Black | 30 | 30 | 100.00 | 100.00 | female | black | Black | 37.00 | -68.22 | Q1 darkest | 42.37 | 1.38 | 6.17 | -49.26 |
| 0_4459 | M-White | 30 | 30 | 100.00 | 80.00 | male | white | Middle Eastern | 30.00 | -56.69 | Q1 darkest | 80.55 | 2.52 | 31.86 | 6.39 |

### How few videos drive intersectional F_FPR

```
{
 "matched": {
  "iF_FPR": 32.2283,
  "group_FPR_minus_pooled_pp": {
   "M-Asian": -10.74,
   "M-White": -0.85,
   "M-Black": -2.81,
   "M-Others": 0.37,
   "F-Asian": -1.86,
   "F-White": -3.26,
   "F-Black": 12.24,
   "F-Others": 0.1
  },
  "greedy_drop_path_first10": [
   32.228,
   29.183,
   27.209,
   25.236,
   22.889,
   20.284,
   18.84,
   17.876,
   17.016,
   16.078,
   15.192
  ],
  "greedy_dropped_videos_first10": [
   "0_6756",
   "0_5432",
   "0_4074",
   "0_4221",
   "0_4567",
   "0_7002",
   "0_4421",
   "0_7256",
   "0_6887",
   "0_7434"
  ],
  "greedy_videos_to_halve_iF_FPR": 9,
  "greedy_iF_FPR_after_5_10_20_drops": [
   20.284,
   15.192,
   7.594
  ],
  "loo_best_single_video": "0_6756",
  "loo_best_iF_FPR": 29.183,
  "perm_null_median": 34.18,
  "perm_null_p5_p95": [
   19.82,
   52.8
  ],
  "perm_p_value_one_sided": 0.5675
 },
 "native": {
  "iF_FPR": 27.7793,
  "group_FPR_minus_pooled_pp": {
   "M-Asian": -5.03,
   "M-White": -0.53,
   "M-Black": -1.46,
   "M-Others": 2.11,
   "F-Asian": -3.04,
   "F-White": -2.65,
   "F-Black": 8.85,
   "F-Others": -4.12
  },
  "greedy_drop_path_first10": [
   27.779,
   24.741,
   22.887,
   21.029,
   19.368,
   17.95,
   16.67,
   15.347,
   14.072,
   12.848,
   11.728
  ],
  "greedy_dropped_videos_first10": [
   "0_5143",
   "0_7434",
   "0_6756",
   "0_4421",
   "0_5277",
   "0_5432",
   "0_7942",
   "0_4074",
   "0_6574",
   "0_7256"
  ],
  "greedy_videos_to_halve_iF_FPR": 9,
  "greedy_iF_FPR_after_5_10_20_drops": [
   17.95,
   11.728,
   2.979
  ],
  "loo_best_single_video": "0_5143",
  "loo_best_iF_FPR": 24.741,
  "perm_null_median": 26.81,
  "perm_null_p5_p95": [
   15.24,
   41.29
  ],
  "perm_p_value_one_sided": 0.444
 },
 "A1": {
  "iF_FPR": 7.2512,
  "group_FPR_minus_pooled_pp": {
   "M-Asian": -1.25,
   "M-White": -0.23,
   "M-Black": -1.16,
   "M-Others": 0.67,
   "F-Asian": -0.95,
   "F-White": -0.39,
   "F-Black": 2.24,
   "F-Others": 0.36
  },
  "greedy_drop_path_first10": [
   7.251,
   6.328,
   5.696,
   5.197,
   4.677,
   4.211,
   3.862,
   3.605,
   3.464,
   3.353,
   3.254
  ],
  "greedy_dropped_videos_first10": [
   "0_7434",
   "0_5277",
   "0_6554",
   "0_6208",
   "0_6593",
   "0_6574",
   "0_5432",
   "0_6012",
   "0_4763",
   "0_6305"
  ],
  "greedy_videos_to_halve_iF_FPR": 7,
  "greedy_iF_FPR_after_5_10_20_drops": [
   4.211,
   3.254,
   2.233
  ],
  "loo_best_single_video": "0_7434",
  "loo_best_iF_FPR": 6.328,
  "perm_null_median": 7.49,
  "perm_null_p5_p95": [
   4.6,
   11.92
  ],
  "perm_p_value_one_sided": 0.5405
 }
}
```

## 2. False negatives (fakes missed)

DFDC fakes have no method label in the protocol release (method = 'unknown' for all 8,158 fake rows), so no per-method breakdown is possible.

**by grp**

| grp | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| F-Asian | 30 | 1 | 17 | 56.67 | 18 | 60.00 | 27 | 90.00 | 23.33 |
| F-Black | 1285 | 45 | 274 | 21.32 | 459 | 35.72 | 944 | 73.46 | 35.56 |
| F-Others | 497 | 17 | 149 | 29.98 | 189 | 38.03 | 297 | 59.76 | 24.75 |
| F-White | 3075 | 107 | 1119 | 36.39 | 1546 | 50.28 | 2304 | 74.93 | 49.50 |
| M-Asian | 90 | 3 | 52 | 57.78 | 67 | 74.44 | 89 | 98.89 | 54.44 |
| M-Black | 507 | 17 | 126 | 24.85 | 206 | 40.63 | 469 | 92.50 | 55.82 |
| M-Others | 404 | 15 | 125 | 30.94 | 174 | 43.07 | 274 | 67.82 | 29.46 |
| M-White | 2270 | 79 | 384 | 16.92 | 665 | 29.30 | 1538 | 67.75 | 46.78 |

**by xu_gender**

| xu_gender | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| female | 4887 | 170 | 1559 | 31.90 | 2212 | 45.26 | 3572 | 73.09 | 43.16 |
| male | 3271 | 114 | 687 | 21.00 | 1112 | 34.00 | 2370 | 72.45 | 46.25 |

**by xu_race**

| xu_race | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| asian | 120 | 4 | 69 | 57.50 | 85 | 70.83 | 116 | 96.67 | 46.67 |
| black | 1792 | 62 | 400 | 22.32 | 665 | 37.11 | 1413 | 78.85 | 41.29 |
| others | 901 | 32 | 274 | 30.41 | 363 | 40.29 | 571 | 63.37 | 26.86 |
| white | 5345 | 186 | 1503 | 28.12 | 2211 | 41.37 | 3842 | 71.88 | 48.34 |

**by cl_gender**

| cl_gender | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| female | 5253 | 207 | 1607 | 30.59 | 2286 | 43.52 | 3882 | 73.90 | 43.99 |
| male | 2905 | 135 | 639 | 22.00 | 1038 | 35.73 | 2060 | 70.91 | 45.13 |

**by cl_race_vid**

| cl_race_vid | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| asian | 158 | 6 | 68 | 43.04 | 90 | 56.96 | 130 | 82.28 | 46.84 |
| black | 1763 | 61 | 332 | 18.83 | 581 | 32.96 | 1319 | 74.82 | 41.01 |
| others | 367 | 13 | 72 | 19.62 | 123 | 33.51 | 206 | 56.13 | 17.98 |
| white | 5870 | 204 | 1774 | 30.22 | 2530 | 43.10 | 4287 | 73.03 | 47.00 |

**by ff_race7**

| ff_race7 | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| Black | 1680 | 85 | 368 | 21.90 | 622 | 37.02 | 1336 | 79.52 | 43.33 |
| East Asian | 138 | 29 | 62 | 44.93 | 88 | 63.77 | 120 | 86.96 | 46.38 |
| Indian | 101 | 23 | 19 | 18.81 | 36 | 35.64 | 59 | 58.42 | 16.83 |
| Latino_Hispanic | 264 | 49 | 35 | 13.26 | 57 | 21.59 | 112 | 42.42 | 24.62 |
| Middle Eastern | 901 | 110 | 169 | 18.76 | 288 | 31.96 | 583 | 64.71 | 44.28 |
| Southeast Asian | 105 | 28 | 24 | 22.86 | 43 | 40.95 | 82 | 78.10 | 29.52 |
| White | 4969 | 221 | 1569 | 31.58 | 2190 | 44.07 | 3650 | 73.46 | 46.65 |

**by age_if**

| age_if | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| 30-44 | 3702 | 130 | 940 | 25.39 | 1482 | 40.03 | 2814 | 76.01 | 45.87 |
| 45-59 | 901 | 31 | 193 | 21.42 | 290 | 32.19 | 666 | 73.92 | 51.61 |
| 60+ | 175 | 6 | 25 | 14.29 | 39 | 22.29 | 94 | 53.71 | 72.00 |
| <30 | 3380 | 117 | 1088 | 32.19 | 1513 | 44.76 | 2368 | 70.06 | 39.44 |

**by age_aif**

| age_aif | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| 25-44 | 5102 | 177 | 1476 | 28.93 | 2203 | 43.18 | 3880 | 76.05 | 45.63 |
| 45-64 | 2270 | 78 | 574 | 25.29 | 810 | 35.68 | 1475 | 64.98 | 41.72 |
| 65+ | 426 | 16 | 153 | 35.92 | 213 | 50.00 | 363 | 85.21 | 41.55 |
| <25 | 360 | 13 | 43 | 11.94 | 98 | 27.22 | 224 | 62.22 | 47.22 |

**by ita4_ffpp**

| ita4_ffpp | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| Q1 darkest | 5565 | 194 | 1475 | 26.50 | 2252 | 40.47 | 4167 | 74.88 | 38.51 |
| Q2 | 452 | 16 | 118 | 26.11 | 173 | 38.27 | 280 | 61.95 | 40.27 |
| Q3 | 480 | 15 | 148 | 30.83 | 208 | 43.33 | 321 | 66.88 | 38.54 |
| Q4 lightest | 1168 | 40 | 374 | 32.02 | 474 | 40.58 | 771 | 66.01 | 61.22 |
| n/a | 493 | 19 | 131 | 26.57 | 217 | 44.02 | 403 | 81.74 | 80.53 |

**by ita4_loc**

| ita4_loc | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| Q1 darkest | 2054 | 70 | 428 | 20.84 | 693 | 33.74 | 1483 | 72.20 | 33.93 |
| Q2 | 2307 | 80 | 719 | 31.17 | 1107 | 47.98 | 1821 | 78.93 | 34.68 |
| Q3 | 1134 | 41 | 323 | 28.48 | 442 | 38.98 | 801 | 70.63 | 52.12 |
| Q4 lightest | 2170 | 74 | 645 | 29.72 | 865 | 39.86 | 1434 | 66.08 | 52.40 |
| n/a | 493 | 19 | 131 | 26.57 | 217 | 44.02 | 403 | 81.74 | 80.53 |

**by mst3**

| mst3 | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| MST 4-6 | 2487 | 85 | 815 | 32.77 | 1018 | 40.93 | 1563 | 62.85 | 51.19 |
| MST 7-10 | 5671 | 199 | 1431 | 25.23 | 2306 | 40.66 | 4379 | 77.22 | 41.42 |

**by gray_s**

| gray_s | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| colour | 7972 | 279 | 2234 | 28.02 | 3268 | 40.99 | 5765 | 72.32 | 43.78 |
| grayscale | 186 | 11 | 12 | 6.45 | 56 | 30.11 | 177 | 95.16 | 70.97 |

**by det_q**

| det_q | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| det Q1 (lowest) | 1874 | 192 | 528 | 28.18 | 883 | 47.12 | 1616 | 86.23 | 51.65 |
| det Q2 | 1983 | 227 | 486 | 24.51 | 740 | 37.32 | 1419 | 71.56 | 44.88 |
| det Q3 | 1981 | 240 | 546 | 27.56 | 784 | 39.58 | 1354 | 68.35 | 40.74 |
| det Q4 | 2301 | 201 | 685 | 29.77 | 915 | 39.77 | 1534 | 66.67 | 40.76 |
| n/a | 19 | 7 | 1 | 5.26 | 2 | 10.53 | 19 | 100.00 | 100.00 |

**by lum_mean_q**

| lum_mean_q | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| brightness Q1 (low) | 2189 | 101 | 365 | 16.67 | 681 | 31.11 | 1603 | 73.23 | 34.76 |
| brightness Q2 | 1812 | 113 | 502 | 27.70 | 815 | 44.98 | 1419 | 78.31 | 39.40 |
| brightness Q3 | 2143 | 128 | 738 | 34.44 | 983 | 45.87 | 1527 | 71.26 | 49.14 |
| brightness Q4 (high) | 2014 | 115 | 641 | 31.83 | 845 | 41.96 | 1393 | 69.17 | 54.32 |

**by lum_std_q**

| lum_std_q | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| contrast Q1 (low) | 2367 | 133 | 606 | 25.60 | 1001 | 42.29 | 1914 | 80.86 | 39.25 |
| contrast Q2 | 2220 | 148 | 612 | 27.57 | 895 | 40.32 | 1542 | 69.46 | 43.83 |
| contrast Q3 | 1958 | 141 | 585 | 29.88 | 803 | 41.01 | 1347 | 68.79 | 48.06 |
| contrast Q4 (high) | 1613 | 104 | 443 | 27.46 | 625 | 38.75 | 1139 | 70.61 | 48.30 |

**by sharp_lapvar_q**

| sharp_lapvar_q | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| sharpness Q1 (low) | 2551 | 135 | 656 | 25.72 | 1156 | 45.32 | 2168 | 84.99 | 31.99 |
| sharpness Q2 | 2156 | 151 | 644 | 29.87 | 841 | 39.01 | 1371 | 63.59 | 36.09 |
| sharpness Q3 | 2061 | 139 | 604 | 29.31 | 812 | 39.40 | 1286 | 62.40 | 49.73 |
| sharpness Q4 (high) | 1390 | 91 | 342 | 24.60 | 515 | 37.05 | 1117 | 80.36 | 72.16 |

**by sat_mean_q**

| sat_mean_q | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| saturation Q1 (low) | 2121 | 103 | 668 | 31.49 | 929 | 43.80 | 1503 | 70.86 | 50.68 |
| saturation Q2 | 2128 | 124 | 633 | 29.75 | 903 | 42.43 | 1536 | 72.18 | 41.40 |
| saturation Q3 | 2218 | 132 | 614 | 27.68 | 942 | 42.47 | 1688 | 76.10 | 43.28 |
| saturation Q4 (high) | 1691 | 102 | 331 | 19.57 | 550 | 32.53 | 1215 | 71.85 | 41.75 |

**by method**

| method | n | videos | FN@matched | FNR%@matched | FN@native | FNR%@native | FN@A1 | FNR%@A1 | FNR%@Ding |
|---|---|---|---|---|---|---|---|---|---|
| unknown | 8158 | 284 | 2246 | 27.53 | 3324 | 40.75 | 5942 | 72.84 | 44.40 |

### FN concentration by video (matched)

- top5_share%: 7.93
- top10_share%: 14.43
- top15_share%: 20.7
- top25_share%: 32.59
- top50_share%: 56.54
- videos_for_50pct_of_FN: 43
- videos_with_any_FN: 197
- videos_with_FNR>=50%: 72
- videos_with_FNR=100%: 8
- ding_videos_with_FNR>=50%: 128

**Top-15 fake videos by #missed frames (matched)**

| video | grp | n | fn | fnr% | ding_fnr% | cl_gender_vid | cl_race_vid | age_vid_if | ita_vid | lum | sharp | min_logit | med_logit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0_4203 | F-White | 60 | 56 | 93.33 | 13.33 | female | white | 29.50 | -42.33 | 105.42 | 4.67 | -94.49 | -85.92 |
| 0_4367 | F-Others | 60 | 32 | 53.33 | 63.33 | female | white | 30.00 | -46.85 | 96.38 | 30.75 | -112.59 | -75.56 |
| 0_4680 | F-White | 30 | 30 | 100.00 | 100.00 | female | white | 31.00 | 84.69 | 177.46 | 24.14 | -119.21 | -107.34 |
| 0_4734 | F-White | 30 | 30 | 100.00 | 90.00 | female | white | 28.00 | -41.71 | 109.60 | 17.23 | -103.08 | -87.56 |
| 0_4776 | F-White | 30 | 30 | 100.00 | 100.00 | female | white | 36.50 | nan | 98.35 | 19.82 | -110.39 | -96.54 |
| 0_6009 | F-White | 30 | 30 | 100.00 | 53.33 | female | white | 33.00 | -35.88 | 105.53 | 5.08 | -96.93 | -88.58 |
| 0_4093 | F-White | 30 | 29 | 96.67 | 33.33 | female | white | 27.50 | nan | 96.35 | 1.99 | -99.17 | -89.27 |
| 0_4414 | M-Others | 30 | 29 | 96.67 | 100.00 | male | others | 47.50 | nan | 14.78 | 0.87 | -89.67 | -82.28 |
| 0_4799 | M-White | 30 | 29 | 96.67 | 23.33 | male | white | 47.00 | -88.20 | 95.18 | 4.39 | -94.72 | -85.61 |
| 0_6894 | F-White | 30 | 29 | 96.67 | 100.00 | female | white | 31.00 | 59.41 | 124.87 | 9.32 | -107.67 | -100.87 |
| 0_7411 | F-White | 30 | 29 | 96.67 | 80.00 | female | white | 26.00 | -28.10 | 105.14 | 4.56 | -104.75 | -85.86 |
| 0_5695 | F-Black | 30 | 28 | 93.33 | 16.67 | female | white | 50.00 | 43.69 | 134.20 | 7.48 | -94.47 | -84.44 |
| 0_6231 | F-White | 30 | 28 | 93.33 | 33.33 | female | white | 27.00 | -85.83 | 52.94 | 6.34 | -103.50 | -85.09 |
| 0_7113 | F-White | 30 | 28 | 93.33 | 66.67 | female | white | 28.00 | -40.26 | 111.17 | 11.15 | -112.92 | -95.17 |
| 0_7129 | F-White | 30 | 28 | 93.33 | 86.67 | female | white | 30.00 | 47.26 | 126.48 | 3.68 | -103.07 | -88.95 |

## 3. Shared vs unique with the Ding reproduction (0.5 rule)

```
{
 "matched": {
  "ours_FP": 4885,
  "ours_FP_also_Ding%": 54.78,
  "Ding_FP": 7221,
  "Ding_FP_also_ours%": 37.06,
  "FP_only_ours": 2209,
  "FP_only_Ding": 4545,
  "ours_FN": 2246,
  "ours_FN_also_Ding%": 54.32,
  "Ding_FN": 3622,
  "Ding_FN_also_ours%": 33.68,
  "FN_only_ours": 1026,
  "FN_only_Ding": 2402,
  "chance_overlap_FP%(=Ding FPR)": 49.13,
  "chance_overlap_FN%(=Ding FNR)": 44.4
 },
 "native": {
  "ours_FP": 2857,
  "ours_FP_also_Ding%": 58.07,
  "Ding_FP": 7221,
  "Ding_FP_also_ours%": 22.97,
  "FP_only_ours": 1198,
  "FP_only_Ding": 5562,
  "ours_FN": 3324,
  "ours_FN_also_Ding%": 52.26,
  "Ding_FN": 3622,
  "Ding_FN_also_ours%": 47.96,
  "FN_only_ours": 1587,
  "FN_only_Ding": 1885,
  "chance_overlap_FP%(=Ding FPR)": 49.13,
  "chance_overlap_FN%(=Ding FNR)": 44.4
 },
 "A1": {
  "ours_FP": 280,
  "ours_FP_also_Ding%": 86.43,
  "Ding_FP": 7221,
  "Ding_FP_also_ours%": 3.35,
  "FP_only_ours": 38,
  "FP_only_Ding": 6979,
  "ours_FN": 5942,
  "ours_FN_also_Ding%": 50.07,
  "Ding_FN": 3622,
  "Ding_FN_also_ours%": 82.14,
  "FN_only_ours": 2967,
  "FN_only_Ding": 647,
  "chance_overlap_FP%(=Ding FPR)": 49.13,
  "chance_overlap_FN%(=Ding FNR)": 44.4
 },
 "spearman_logit_real_frames": 0.246,
 "spearman_logit_fake_frames": 0.287,
 "spearman_video_median_real": 0.234,
 "spearman_video_median_fake": 0.287
}
```

**FP: we flag, Ding does not (top-10 frames, one per video, by our logit)**

| video | grp | logit | ding_logit | cl_race_vid | age_vid_if | ita_vid | img_path |
|---|---|---|---|---|---|---|---|
| 0_6208 | F-Black | 34.87 | -1.58 | black | 29.00 | -68.90 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_6208_131.png |
| 0_6463 | F-White | 19.45 | -1.93 | white | 22.50 | 27.93 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_6463_201.png |
| 0_6554 | F-Others | 17.72 | -4.84 | others | 23.00 | 78.51 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_6554_61.png |
| 0_6219 | M-White | 14.73 | -1.66 | white | 36.00 | -3.48 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_6219_111.png |
| 0_5055 | F-Others | 13.81 | -0.17 | others | 24.00 | -23.29 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_5055_201.png |
| 0_7085 | F-Black | 11.20 | -1.46 | black | 23.00 | -20.95 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7085_181.png |
| 0_4348 | F-Others | 11.00 | -1.89 | others | 24.00 | 67.21 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4348_31.png |
| 0_6678 | F-White | 8.94 | -3.76 | white | 36.00 | 67.55 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_6678_281.png |
| 0_5804 | M-White | 7.76 | -0.11 | white | 45.00 | nan | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_5804_71.png |
| 0_4954 | F-Black | 4.19 | -1.52 | black | 21.00 | -68.99 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4954_91.png |

**FP: Ding flags, we do not (top-10, by Ding logit)**

| video | grp | logit | ding_logit | cl_race_vid | age_vid_if | ita_vid | img_path |
|---|---|---|---|---|---|---|---|
| 0_7510 | F-White | -78.11 | 5.74 | white | 25.00 | -65.40 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7510_1.png |
| 0_7522 | F-Black | -84.59 | 5.66 | asian | 26.00 | -12.36 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7522_211.png |
| 0_5015 | M-Others | -83.97 | 5.65 | white | 44.00 | -55.59 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_5015_231.png |
| 0_4954 | F-Black | -94.69 | 5.65 | black | 21.00 | -68.99 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4954_131.png |
| 0_4010 | M-White | -102.85 | 5.63 | white | 27.00 | -64.42 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4010_191.png |
| 0_7736 | F-White | -89.78 | 5.62 | white | 29.00 | 11.89 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7736_41.png |
| 0_5781 | M-White | -83.60 | 5.62 | white | 46.00 | -79.25 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_5781_71.png |
| 0_4350 | M-White | -79.10 | 5.60 | white | 31.00 | -86.56 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4350_71.png |
| 0_7076 | F-Black | -78.75 | 5.60 | black | 26.00 | -67.27 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7076_161.png |
| 0_7257 | M-White | -93.67 | 5.59 | white | 30.00 | -67.70 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7257_11.png |

**FN: we miss, Ding catches (top-10, lowest our logit)**

| video | grp | logit | ding_logit | cl_race_vid | age_vid_if | ita_vid | img_path |
|---|---|---|---|---|---|---|---|
| 0_7028 | F-Others | -111.89 | 1.62 | white | 31.00 | 37.46 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7028_41.png |
| 0_5849 | F-White | -103.40 | 3.15 | white | 31.50 | 21.30 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_5849_1.png |
| 0_5877 | F-White | -102.50 | 2.07 | white | 32.00 | -11.60 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_5877_161.png |
| 0_7984 | F-Black | -100.71 | 3.35 | black | 26.00 | -53.24 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7984_21.png |
| 0_7176 | M-Others | -99.64 | 0.51 | asian | 33.00 | 21.30 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7176_271.png |
| 0_7411 | F-White | -98.75 | 1.87 | white | 26.00 | -28.10 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7411_131.png |
| 0_6321 | F-White | -98.71 | 4.10 | white | 25.00 | -38.75 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_6321_41.png |
| 0_7811 | M-White | -98.54 | 3.76 | white | 29.00 | 86.80 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7811_261.png |
| 0_4434 | F-White | -98.36 | 0.04 | white | 28.00 | -40.70 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4434_1.png |
| 0_6544 | M-White | -97.99 | 0.57 | white | 44.50 | -22.73 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_6544_1.png |

**FN: Ding misses, we catch (top-10, lowest Ding logit)**

| video | grp | logit | ding_logit | cl_race_vid | age_vid_if | ita_vid | img_path |
|---|---|---|---|---|---|---|---|
| 0_7230 | M-White | -30.60 | -5.87 | white | 50.00 | 13.37 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7230_41.png |
| 0_6474 | F-White | -0.39 | -5.80 | white | 41.50 | 12.79 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_6474_81.png |
| 0_7493 | F-White | -56.34 | -5.75 | white | 31.00 | 75.74 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7493_161.png |
| 0_7724 | M-White | -37.75 | -5.74 | white | 45.00 | 0.78 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7724_161.png |
| 0_7220 | M-White | 48.86 | -5.73 | white | 27.00 | 86.25 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7220_51.png |
| 0_6647 | F-Black | -63.21 | -5.73 | white | 28.00 | -42.98 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_6647_291.png |
| 0_7734 | M-White | 39.19 | -5.71 | white | 24.00 | -30.96 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7734_141.png |
| 0_6182 | F-White | -65.32 | -5.70 | white | 38.00 | 29.70 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_6182_221.png |
| 0_6536 | F-White | -25.87 | -5.70 | white | 34.00 | 83.81 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_6536_111.png |
| 0_4188 | F-White | 27.59 | -5.69 | white | 46.00 | nan | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4188_31.png |

**Real videos where our FPR exceeds Ding's most (matched)**

| video | grp | n | fpr% | ding_fpr% | diff | cl_race_vid | age_vid_if | ita_vid |
|---|---|---|---|---|---|---|---|---|
| 0_7834 | M-White | 30 | 100.00 | 0.00 | 100.00 | white | 70.00 | 41.42 |
| 0_6461 | M-Others | 60 | 100.00 | 0.00 | 100.00 | others | 37.00 | -51.39 |
| 0_7703 | F-White | 30 | 100.00 | 0.00 | 100.00 | white | 40.00 | -86.68 |
| 0_7459 | M-White | 30 | 100.00 | 0.00 | 100.00 | white | 71.00 | 17.43 |
| 0_4859 | M-White | 30 | 100.00 | 0.00 | 100.00 | white | 81.50 | 69.60 |
| 0_4343 | F-Black | 20 | 100.00 | 0.00 | 100.00 | black | 38.00 | -82.60 |
| 0_4996 | M-Black | 9 | 100.00 | 0.00 | 100.00 | black | 49.00 | -51.59 |
| 0_6305 | M-Asian | 60 | 96.67 | 0.00 | 96.67 | asian | 39.00 | -71.82 |
| 0_7138 | F-White | 60 | 96.67 | 0.00 | 96.67 | white | 37.00 | -83.52 |
| 0_7707 | M-White | 30 | 96.67 | 0.00 | 96.67 | white | 54.50 | -32.27 |

**Real videos where Ding's FPR exceeds ours most**

| video | grp | n | fpr% | ding_fpr% | diff | cl_race_vid | age_vid_if | ita_vid |
|---|---|---|---|---|---|---|---|---|
| 0_7747 | M-White | 30 | 0.00 | 100.00 | -100.00 | white | 49.00 | 43.25 |
| 0_7741 | M-Black | 30 | 0.00 | 100.00 | -100.00 | black | 34.00 | -80.55 |
| 0_4409 | F-White | 30 | 0.00 | 100.00 | -100.00 | white | 32.50 | 13.87 |
| 0_5781 | M-White | 30 | 0.00 | 100.00 | -100.00 | white | 46.00 | -79.25 |
| 0_5000 | M-White | 15 | 0.00 | 100.00 | -100.00 | white | 52.00 | 36.10 |
| 0_5824 | F-White | 20 | 0.00 | 100.00 | -100.00 | white | 32.00 | -54.92 |
| 0_4106 | M-Others | 30 | 0.00 | 100.00 | -100.00 | others | 28.50 | -15.40 |
| 0_6887 | M-Black | 60 | 0.00 | 100.00 | -100.00 | black | 29.50 | -61.18 |
| 0_6197 | M-White | 30 | 0.00 | 100.00 | -100.00 | white | 46.00 | nan |
| 0_6524 | F-White | 30 | 0.00 | 100.00 | -100.00 | white | 30.00 | nan |

**Fake videos where our miss rate exceeds Ding's most (matched)**

| video | grp | n | fnr% | ding_fnr% | diff | cl_race_vid | age_vid_if | ita_vid |
|---|---|---|---|---|---|---|---|---|
| 0_6912 | M-Black | 30 | 86.67 | 0.00 | 86.67 | white | 38.00 | 48.55 |
| 0_4203 | F-White | 60 | 93.33 | 13.33 | 80.00 | white | 29.50 | -42.33 |
| 0_4401 | F-Black | 30 | 76.67 | 0.00 | 76.67 | black | 24.00 | -76.52 |
| 0_5695 | F-Black | 30 | 93.33 | 16.67 | 76.67 | white | 50.00 | 43.69 |
| 0_6711 | M-White | 21 | 76.19 | 0.00 | 76.19 | white | 50.00 | -82.94 |
| 0_4799 | M-White | 30 | 96.67 | 23.33 | 73.33 | white | 47.00 | -88.20 |
| 0_6321 | F-White | 30 | 80.00 | 6.67 | 73.33 | white | 25.00 | -38.75 |
| 0_6995 | M-White | 30 | 90.00 | 16.67 | 73.33 | white | 24.00 | 29.53 |
| 0_4434 | F-White | 15 | 100.00 | 26.67 | 73.33 | white | 28.00 | -40.70 |
| 0_5421 | F-Others | 30 | 83.33 | 16.67 | 66.67 | white | 26.50 | 76.54 |

**Fake videos where Ding's miss rate exceeds ours most**

| video | grp | n | fnr% | ding_fnr% | diff | cl_race_vid | age_vid_if | ita_vid |
|---|---|---|---|---|---|---|---|---|
| 0_7755 | M-Black | 15 | 0.00 | 100.00 | -100.00 | black | 42.00 | nan |
| 0_7230 | M-White | 30 | 0.00 | 100.00 | -100.00 | white | 50.00 | 13.37 |
| 0_6643 | M-Black | 30 | 0.00 | 100.00 | -100.00 | black | 41.00 | -83.82 |
| 0_7206 | M-White | 30 | 0.00 | 100.00 | -100.00 | white | 46.00 | -0.13 |
| 0_4254 | M-White | 30 | 0.00 | 100.00 | -100.00 | white | 50.50 | 82.07 |
| 0_4211 | F-White | 23 | 0.00 | 100.00 | -100.00 | white | 41.00 | nan |
| 0_4200 | M-White | 30 | 0.00 | 100.00 | -100.00 | white | 62.00 | 64.93 |
| 0_4177 | F-White | 30 | 0.00 | 100.00 | -100.00 | white | 34.00 | -36.04 |
| 0_4332 | F-White | 30 | 0.00 | 100.00 | -100.00 | white | 35.00 | -87.83 |
| 0_4963 | M-White | 30 | 0.00 | 100.00 | -100.00 | white | 66.00 | -37.68 |

```
{
 "real_videos_ours_fpr_gt_ding_by_50pp": 46,
 "real_videos_ding_fpr_gt_ours_by_50pp": 125,
 "fake_videos_ours_fnr_gt_ding_by_50pp": 22,
 "fake_videos_ding_fnr_gt_ours_by_50pp": 81
}
```

## 4. Most confident errors

**Top-12 FP (highest logit among reals; unique images)**

| video | grp | logit | ding_logit | ding_pred | cl_gender_vid | cl_race_vid | age_vid_if | ita_vid | img_path |
|---|---|---|---|---|---|---|---|---|---|
| 0_6104 | M-Others | 67.54 | 4.23 | True | male | others | 26.00 | 5.67 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_6104_71.png |
| 0_7434 | F-Black | 61.53 | 2.10 | True | female | black | 31.00 | -58.76 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7434_171.png |
| 0_7434 | F-Black | 60.60 | 4.98 | True | female | black | 31.00 | -58.76 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7434_221.png |
| 0_7434 | F-Black | 58.65 | 2.99 | True | female | black | 31.00 | -58.76 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7434_121.png |
| 0_7964 | F-White | 55.01 | 4.94 | True | female | white | 42.50 | -40.20 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7964_51.png |
| 0_7434 | F-Black | 54.98 | 3.43 | True | female | black | 31.00 | -58.76 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7434_51.png |
| 0_7434 | F-Black | 53.39 | 3.63 | True | female | black | 31.00 | -58.76 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7434_41.png |
| 0_5633 | M-White | 50.49 | 5.07 | True | male | white | 22.00 | -17.73 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_5633_241.png |
| 0_5633 | M-White | 50.47 | 5.17 | True | male | white | 22.00 | -17.73 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_5633_291.png |
| 0_6104 | M-Others | 49.88 | 4.88 | True | male | others | 26.00 | 5.67 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_6104_51.png |
| 0_7434 | F-Black | 47.22 | 4.72 | True | female | black | 31.00 | -58.76 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7434_231.png |
| 0_5633 | M-White | 45.29 | 5.37 | True | male | white | 22.00 | -17.73 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_5633_121.png |

**Top-12 FN (lowest logit among fakes)**

| video | grp | logit | ding_logit | ding_pred | cl_gender_vid | cl_race_vid | age_vid_if | ita_vid | img_path |
|---|---|---|---|---|---|---|---|---|---|
| 0_5573 | M-Others | -122.63 | -5.06 | False | female | white | 28.50 | -43.71 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_5573_191.png |
| 0_4680 | F-White | -119.21 | -5.70 | False | female | white | 31.00 | 84.69 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4680_251.png |
| 0_7028 | F-Others | -118.14 | -4.31 | False | female | white | 31.00 | 37.46 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7028_191.png |
| 0_4680 | F-White | -114.53 | -5.50 | False | female | white | 31.00 | 84.69 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4680_261.png |
| 0_7113 | F-White | -112.92 | -5.49 | False | female | white | 28.00 | -40.26 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7113_251.png |
| 0_4680 | F-White | -112.89 | -5.70 | False | female | white | 31.00 | 84.69 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4680_91.png |
| 0_4367 | F-Others | -112.59 | -1.76 | False | female | white | 30.00 | -46.85 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4367_91.png |
| 0_4680 | F-White | -112.57 | -5.74 | False | female | white | 31.00 | 84.69 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4680_241.png |
| 0_7028 | F-Others | -111.89 | 1.62 | True | female | white | 31.00 | 37.46 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_7028_41.png |
| 0_4680 | F-White | -111.57 | -3.69 | False | female | white | 31.00 | 84.69 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4680_201.png |
| 0_4680 | F-White | -111.50 | -5.69 | False | female | white | 31.00 | 84.69 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4680_181.png |
| 0_4776 | F-White | -110.39 | -5.64 | False | female | white | 36.50 | nan | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/dfdc/0_4776_1.png |

## Extras

```
{
 "real_videos_xu_vs_cleaner_group_disagree": 76,
 "real_videos_xu_vs_cleaner_gender_disagree": 46,
 "top15_fp_videos_xu_vs_cleaner_group_disagree": 6,
 "fake_videos_xu_vs_cleaner_group_disagree": 60,
 "n_real_videos": 473,
 "n_fake_videos": 284,
 "iF_FPR_cleaner_vid_labels": {
  "matched": 68.79,
  "native": 42.97,
  "A1": 9.5,
  "Ding repro 0.5": 62.52
 },
 "cleaner_perm_null_matched": {
  "obs": 68.79,
  "median": 36.25,
  "p5_p95": [
   20.24,
   54.7
  ],
  "p_one_sided": 0.0025
 },
 "F_Black_FPR_after_dropping_top_k_FBlack_videos": {
  "n_videos": 68,
  "drop0": 45.48,
  "drop3": 42.37,
  "drop5": 40.57,
  "drop10": 36.17,
  "pooled_after_drop10": 31.81,
  "top10_FBlack_videos": [
   "0_5432",
   "0_4421",
   "0_7256",
   "0_7434",
   "0_7942",
   "0_5277",
   "0_6574",
   "0_7081",
   "0_6961",
   "0_6290"
  ]
 },
 "cleaner_group_FPR_matched%": {
  "M-Asian": 14.86,
  "M-White": 32.82,
  "M-Black": 27.23,
  "M-Others": 17.03,
  "F-Asian": 43.85,
  "F-White": 28.96,
  "F-Black": 45.47,
  "F-Others": 32.55
 },
 "fp_rate_by_race_x_brightness_matched": {
  "asian": {
   "brightness Q1 (low)": "54.3 (n=479)",
   "brightness Q2": "31.7 (n=309)",
   "brightness Q3": "28.0 (n=422)",
   "brightness Q4 (high)": "20.3 (n=315)"
  },
  "black": {
   "brightness Q1 (low)": "46.5 (n=1580)",
   "brightness Q2": "37.7 (n=1571)",
   "brightness Q3": "32.4 (n=738)",
   "brightness Q4 (high)": "28.5 (n=474)"
  },
  "others": {
   "brightness Q1 (low)": "12.4 (n=370)",
   "brightness Q2": "36.4 (n=308)",
   "brightness Q3": "20.7 (n=304)",
   "brightness Q4 (high)": "32.8 (n=122)"
  },
  "white": {
   "brightness Q1 (low)": "43.1 (n=1097)",
   "brightness Q2": "32.0 (n=1714)",
   "brightness Q3": "29.2 (n=2108)",
   "brightness Q4 (high)": "26.8 (n=2788)"
  }
 },
 "fn_rate_by_race_x_brightness_matched": {
  "asian": {
   "brightness Q1 (low)": "32.1 (n=28)",
   "brightness Q2": "66.7 (n=6)",
   "brightness Q3": "35.2 (n=91)",
   "brightness Q4 (high)": "69.7 (n=33)"
  },
  "black": {
   "brightness Q1 (low)": "16.2 (n=1047)",
   "brightness Q2": "24.3 (n=580)",
   "brightness Q3": "12.8 (n=117)",
   "brightness Q4 (high)": "31.6 (n=19)"
  },
  "others": {
   "brightness Q1 (low)": "26.3 (n=137)",
   "brightness Q2": "19.4 (n=175)",
   "brightness Q3": "6.9 (n=29)",
   "brightness Q4 (high)": "0.0 (n=26)"
  },
  "white": {
   "brightness Q1 (low)": "15.4 (n=977)",
   "brightness Q2": "30.7 (n=1051)",
   "brightness Q3": "36.1 (n=1906)",
   "brightness Q4 (high)": "31.6 (n=1936)"
  }
 },
 "real_median_brightness_by_race": {
  "asian": 86.0,
  "black": 71.5,
  "others": 70.2,
  "white": 100.8
 },
 "spearman_logit_lum_mean_real": -0.173,
 "spearman_logit_lum_mean_fake": -0.056,
 "spearman_logit_lum_std_real": -0.157,
 "spearman_logit_lum_std_fake": 0.047,
 "spearman_logit_sharp_lapvar_real": -0.101,
 "spearman_logit_sharp_lapvar_fake": 0.073,
 "perm_greedy_videos_to_halve_matched": {
  "values": [
   7,
   8,
   4,
   8,
   14,
   5,
   8,
   10,
   6,
   8,
   11,
   6,
   4,
   5,
   7,
   4,
   10,
   5,
   7,
   9
  ],
  "median": 7.0
 },
 "fn_sheet_group_counts": {
  "F-White": 16,
  "M-Others": 2,
  "F-Others": 2,
  "M-White": 2,
  "F-Black": 1,
  "M-Black": 1
 },
 "fp_sheet_group_counts": {
  "F-Black": 7,
  "F-White": 7,
  "M-White": 4,
  "M-Others": 2,
  "M-Black": 2,
  "F-Asian": 1,
  "F-Others": 1
 },
 "top15_fn_videos_group_counts": {
  "F-White": 11,
  "F-Others": 1,
  "M-Others": 1,
  "M-White": 1,
  "F-Black": 1
 },
 "fn_sheet_cleaner_white_female": 20,
 "fake_rows_within_10_logits_of_t_matched%": 30.74,
 "real_rows_within_10_logits_of_t_matched%": 40.86
}
```
