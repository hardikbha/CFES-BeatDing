# Multi-axis (gender / race / age / skin tone) evaluation of every saved model

Agent DEMO_AXES, 26 Oct 2026. CPU only: saved per-image logits, no retraining, no GPU, no existing result modified. Code `agents/DEMO_AXES/code/`, labels `labels/`, numbers `results/` (README.md there). Pre-registration: `decisions/2026-10-26-final-model-spec.md`.

> **Status of every number here: EXPLORATORY / DESCRIPTIVE.** H1-H4 are the confirmatory family of the final-model spec and are *not tested* in this document: the final model (CESel-MA, or CFES-D on fallback) and the matched-CLIP baselines do not exist yet. Section 7 computes the H1-H4 *quantities* for existing models as descriptive values only (no Holm, no seed resampling, no NI decision). Intervals are video-cluster (FF2: person-cluster) percentile bootstraps with thresholds and models fixed (test-sample uncertainty only).

## 0. Summary (descriptive; read with the TPR/FPR columns of the tables below)

1. **Label coverage.** AI-Face cleaner gender/race: 100 % of FF++ train/test, DFDC, Celeb-DF, DFD; **0 % of FF++ val** (the release has no val file), so val falls back to Xu labels (flagged `cl_src`). AI-Face age and MST: 100 % of the protocol sets. Our InsightFace age: >= 99.9 %. Frame ITA valid: 89-99 % of frames; video-level ITA: 94.7-100 % of rows; FF2-public TEST ITA: 94.3 % of images. FF2 age = the manifest `if_age` (100 %).
2. **ITA bins and stability.** Declared edges (TRAIN quartiles, before any score): FF++ protocol 19.2 / 35.2 / 53.4 (582 real TRAIN videos); FF2 -36.6 / 2.9 / 38.5 (13,055 real TRAIN persons). Within a video, ITA is stable (ICC 0.77-0.95, median within-video SD 3-11 degrees; the frame-majority bin equals the median bin in 92-99 % of videos). But 36-69 % of videos have frames in more than one bin. As a skin-tone measure, ITA is weak: White-vs-Black AUC is 0.72-0.96, and 31 % of FF2 persons labelled Black fall in the two lighter bins. DFDC footage is dark, so 75 % of DFDC videos land in Q1 under the FF++ edges; a within-set-quartile sensitivity is in Section 4b. **No skin-tone claim should be made before the human check.** The 300-item stratified rater list is ready; no study has been run.
3. **Label unit changes the DFD conclusion.** The cleaner labels are assigned per frame and change inside a video. They split 2-36 % of real videos (up to 40 % of fake videos) across gender x race groups; the Xu labels split none. On DFD, the frame-level labels create two thin "Others" cells out of stray frames. At A1, the CFES-D - Ding difference in DFD gender x race F_FPR is +12.4 with frame labels, -2.8 with video-majority labels and -6.2 with Xu labels. At the deployed 0.5 rule it is +81.0 / +31.7 / +9.1. **Most of the 26-Oct "DFD becomes a resolved loss under cleaner labels" comes from the label unit.** Recommendation: fix video-majority labels as the H4 unit (spec amendment) before the final model is scored.
4. **Age.** The spec's bins (<30/30-44/45-59/60+) cannot be formed from AI-Face's 5-class age. Primary = AI-Face collapsed to 4 bins; the spec's bins are reported with our InsightFace age (the FF2 estimator). In FF++ test, the oldest cell (65+) has only 5 real videos and 34 frames. Of these frames, CESel B0 flags 82 % as fake, and that one cell drives B0's age F_FPR of 79.
5. **In-domain (FF++ test, A1).** Disparities are small for every CLIP model except B0 (age). Ding repro and arm E are lowest: gender x race F_FPR 6.4 (Ding, TPR 83.4) and 7.3 (E w3, TPR 80.4) vs 14.4 for CFES-D (TPR 79.7). Skin FPR gap: 1.2-3.1 for these models.
6. **Cross-domain (DFDC / Celeb-DF / DFD, A1).** On skin tone, CFES-D's FPR gap is 5.0 / 5.8 / 4.8 vs Ding's 8.1 / 10.8 / 17.0, and CFES-D also has higher TPR (38.6 / 72.9 / 75.6 vs 14.5 / 58.2 / 63.7). On age the comparison is mixed: F_FPR 6.8 / 10.4 / 3.2 vs 4.1 / 10.5 / 11.4. Arm E has the smallest FPR disparities on most axes, but on DFDC and DFD it does so at lower TPR (28 / 59 vs CFES-D 39 / 76).
7. **FF2-public TEST.** FF++-trained detectors are near-broken here (AUC 36-79). At A1, CFES-D's FPR is 46 % and Ding's AUC is 36. Descriptive CFES-D - Ding differences: H1 -25.7 [-36.6, -14.7]; H2 (age) +5.6 [+0.2, +8.3], so CFES-D is worse on age; H3 (skin) -5.5 [-9.1, -1.1]. The two models' TPRs differ by 48 points, so these are not comparable operating points. FF2-trained models (0.5 rule; trained on FF2-full including RFW) have small age (<= 2.4) and skin (<= 4.5) FPR gaps. For them the lightest ITA bin has the highest FPR (FF2-trained CFES 5.6 % vs 2.6-2.9 %).
8. **H4 (descriptive, mean over DFDC / CDF / DFD).** CFES-D 36.4 vs Ding 45.3: -9.0 [-23.9, +11.0] with frame labels, -12.7 [-24.0, +6.9] with video-majority labels. B0: -26.7 [-38.9, -9.1], with +13.7 TPR. None of this is the confirmatory test.
9. **Mechanism (B0, 4 test sets pooled).** On the real side, the group's total cue deficit tracks its FPR excess: Spearman +0.45 [+0.14, +0.70] for gender x race (video majority), +0.79 [+0.26, +0.94] for age (13 points) and +0.82 [+0.42, +0.89] for skin (16 points). But the group's mean margin deficit tracks FPR at least as well (+0.64 to +0.85), so most of this link is arithmetic. Beyond the margin, the total deficit keeps a partial rho of +0.30 to +0.73 (all CIs above 0), but no single cue carries it consistently: texture for skin (+0.58), boundary for age (+0.71), semantic for gender x race (+0.50 clean frame-level, +0.34 n.s. video majority), frequency never. On the fake side there is no relation beyond the margin.
10. **CESel p1 arms (Amendment-5 screen, seed 7).** Full, sens and inv had a finished eval and are included; floor, the controls and CLIPFAIR had not finished. Their stored A1 thresholds were used and re-derived exactly. Descriptive H4 (frame / video-majority labels): Full 44.5 / 30.4, sens 31.6 / 20.4, inv 30.2 / 17.8, vs B0 18.7 / 15.8; cross-domain TPR is 62.8 / 62.9 / 58.7 vs B0 59.1. The p1 screen has not been decided and is gated on validation; **these test numbers must not inform it.**
11. **FairFace-classifier age (0-29 / 30+, from the FairForensics-protocol agent's `labels_fairface/`)** is included as a further age sensitivity axis (`age_fairface2`). With only two bins, age F_FPR is small for every model (0.1-13.4).

## 1. Labels

One CSV per split, keyed exactly like the score npz `img_path` (`labels/labels_<set>.csv`; column dictionary in `code/build_labels.py` docstring).

| axis | source used here (primary) | sensitivity versions |
|---|---|---|
| gender, race | AI-Face cleaner labels (Lin et al. robust-fairness release, v2), frame level; Xu labels where missing (`cl_src = xu_fallback`: all of FF++ val). Celeb-DF/DFD: Asian -> Others (Ding six-group code book) | Xu labels (= protocol) |
| age | AI-Face v1 `Predicted Age`, video majority, 4 bins <25 / 25-44 / 45-64 / 65+ (child+youth / adult / middle / senior). FF2: manifest `if_age` (InsightFace, per image), bins <30 / 30-44 / 45-59 / 60+ | InsightFace buffalo_l age (ours, per-video median, spec bins <30/30-44/45-59/60+); AI-Face v2 young/middle/senior |
| skin tone | ITA of face-skin pixels (cheek + forehead ROI from SCRFD landmarks, YCrCb skin mask, 10-90 % L* trim), gray-world white balance estimated on the crop surround (chromatic only), per-video median; 4 bins at TRAIN quartiles | ITA without white balance; ITA with exposure normalisation; AI-Face MST 3-group (video majority) |

**Deviation from the spec (flag).** The spec asks for age bins <30/30-44/45-59/60+ *from AI-Face predicted age*, but AI-Face v1 only provides a 5-class age (0-14/15-24/25-44/45-64/65+), so those edges cannot be formed from it. Primary on FF++/DFDC/CDF/DFD = AI-Face classes collapsed to 4 bins; the spec's numeric bins are reported with our InsightFace buffalo_l estimate (the same estimator that produced FF2's `if_age`). The coordinator should choose which becomes the selection/training age axis before 2 Nov.

Coverage (fraction of rows with a value):

| set | rows | cl_src==aiface_v2|ff2 | aif_age5 | age4_aif | if_age | age4_if | aif_mst | ita_gw | ita4 |
|---|---|---|---|---|---|---|---|---|---|
| ffpp_train | 76139 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 96.8 | 98.4 |
| ffpp_val | 25386 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 96.5 | 98.5 |
| ffpp_test | 25401 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 96.2 | 98.5 |
| dfdc | 22857 | 100.0 | 100.0 | 100.0 | 99.9 | 100.0 | 100.0 | 89.0 | 94.7 |
| celebdf | 28458 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 98.6 | 99.6 |
| dfd | 9385 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 96.9 | 100.0 |
| ff2_test | 93543 | 100.0 | - | - | 100.0 | 100.0 | - | 94.3 | 94.3 |

`cl_src==aiface_v2|ff2` = share of rows with the cleaner (or FF2) gender/race label; the rest fall back to Xu. `ita_gw` = frames with a valid frame ITA; `ita4` = rows whose video has a valid median ITA.

### ITA bin edges (declared from TRAIN quartiles before any score was read)

| protocol | variant | q25 | q50 | q75 | TRAIN units |
|---|---|---|---|---|---|
| ffpp_protocol | ita_gw (primary) | 19.2 | 35.2 | 53.4 | 582 |
| ffpp_protocol | ita_raw | 18.8 | 32.9 | 46.8 | 588 |
| ffpp_protocol | ita_gwexp | 25.6 | 42.9 | 60.2 | 582 |
| ff2 | ita_gw (primary) | -36.6 | 2.9 | 38.5 | 13055 |
| ff2 | ita_raw | -31.6 | 3.8 | 34.4 | 13422 |
| ff2 | ita_gwexp | -39.8 | 1.9 | 41.9 | 13059 |

Bins: Q1 = ITA < q25 (darkest) ... Q4 = ITA >= q75 (lightest). FF++/DFDC/CDF/DFD use the FF++ TRAIN edges; FF2 uses the FF2-public TRAIN edges.

### ITA stability within a video (frame ITA, primary variant)

| set | videos (valid) | valid frames | frames/video | within-video SD (median) | within-video IQR (median) | between-video SD | ICC(1) | videos whose frames span >1 bin | frame-majority bin = median bin |
|---|---|---|---|---|---|---|---|---|---|
| ffpp_train | 2722 (2670) | 96.8% | 30 | 3.1 | 3.7 | 29.0 | 0.95 | 48% | 99% |
| ffpp_val | 963 (948) | 96.5% | 30 | 3.6 | 3.7 | 29.5 | 0.95 | 48% | 98% |
| ffpp_test | 965 (948) | 96.2% | 30 | 3.5 | 3.7 | 30.7 | 0.92 | 49% | 99% |
| dfdc | 757 (715) | 89.0% | 30 | 7.1 | 7.0 | 51.5 | 0.87 | 36% | 96% |
| celebdf | 956 (952) | 98.6% | 30 | 6.1 | 7.3 | 39.2 | 0.94 | 55% | 96% |
| dfd | 319 (319) | 96.9% | 30 | 10.6 | 9.9 | 40.1 | 0.77 | 69% | 92% |

### ITA construct checks (label-only; NOT validation: race is not skin tone)

| set | ITA variant | Spearman(ITA, -AI-Face MST) | median ITA White | Black | Asian | Others | AUC White-vs-Black |
|---|---|---|---|---|---|---|---|
| ffpp_train | ita_vid | 0.72 | 35.5 | -5.5 | 36.4 | 30.1 | 0.808 |
| ffpp_train | itavid_raw | 0.74 | 33.0 | -3.3 | 33.3 | 33.1 | 0.802 |
| ffpp_train | itavid_gwexp | 0.51 | 44.7 | 17.9 | 47.4 | 40.5 | 0.750 |
| ffpp_val | ita_vid | 0.75 | 32.1 | 0.9 | 36.0 | 27.2 | 0.781 |
| ffpp_val | itavid_raw | 0.77 | 27.7 | -0.6 | 31.9 | 27.6 | 0.791 |
| ffpp_val | itavid_gwexp | 0.53 | 41.7 | -8.6 | 39.0 | 39.5 | 0.862 |
| ffpp_test | ita_vid | 0.75 | 34.3 | 7.8 | 43.8 | 29.6 | 0.719 |
| ffpp_test | itavid_raw | 0.74 | 30.8 | 11.7 | 40.2 | 32.1 | 0.705 |
| ffpp_test | itavid_gwexp | 0.53 | 41.0 | 24.8 | 52.1 | 39.8 | 0.707 |
| dfdc | ita_vid | 0.76 | -15.4 | -64.3 | -11.5 | -40.5 | 0.735 |
| dfdc | itavid_raw | 0.78 | -7.4 | -56.4 | -2.0 | -29.8 | 0.769 |
| dfdc | itavid_gwexp | 0.55 | -14.3 | -54.1 | -16.2 | -25.6 | 0.674 |
| celebdf | ita_vid | 0.75 | 26.6 | -54.1 | 1.0 | - | 0.956 |
| celebdf | itavid_raw | 0.78 | 21.2 | -47.1 | 1.0 | - | 0.975 |
| celebdf | itavid_gwexp | 0.49 | 50.2 | -23.2 | 25.2 | - | 0.971 |
| dfd | ita_vid | 0.71 | 37.1 | -2.1 | 32.0 | 10.1 | 0.757 |
| dfd | itavid_raw | 0.75 | 27.4 | -4.0 | 22.3 | 11.1 | 0.748 |
| dfd | itavid_gwexp | 0.58 | 34.4 | -16.0 | 64.4 | -9.9 | 0.734 |
| ff2_test | ita_vid | - | 39.4 | -8.9 | 29.8 | -0.9 | 0.775 |
| ff2_test | itavid_raw | - | 33.2 | -7.4 | 25.2 | -0.3 | 0.782 |
| ff2_test | itavid_gwexp | - | 34.5 | -2.8 | 29.8 | -2.5 | 0.679 |

Human skin-tone check sample: `labels/human_check_sample.csv`, 300 reals (one frame per video/person, the frame closest to the video median ITA), stratified dataset x ITA bin x gender (seed 20261026). To be added to `release_ff2_public/docs/HUMAN_VALIDATION_PROTOCOL.md` as an MST-10 swatch question with an 'unsure' option; no study has been run.

| dataset | Q1 | Q2 | Q3 | Q4 | male | non-male |
|---|---|---|---|---|---|---|
| celebdf | 25 | 11 | 12 | 12 | 24 | 36 |
| dfd | 25 | 12 | 9 | 14 | 22 | 38 |
| dfdc | 21 | 14 | 11 | 14 | 25 | 35 |
| ff2_test | 18 | 14 | 14 | 14 | 32 | 28 |
| ffpp_test | 14 | 14 | 14 | 18 | 28 | 32 |

## 2. Metrics module

`code/multiaxis_metrics.py` (definitions = `agents/CFES/cfes/metrics.py`, on any partition). Per axis: F_FPR (Ding Eq. 8), F_DP, es-AUC, max-min FPR gap, TPR gap, equalised odds (EO_gap = max of the two gaps; EO_maxdev), worst-group FPR/TPR, with **FPR-specific eligibility** (a group enters FPR terms if it has >= 3 real videos, TPR terms if >= 3 fake videos, AUC/DP terms if both; FF2: persons) and per-group frame / video / person counts (`results/rescore_groups.csv`). Validation selection uses >= 2 videos (`MIN_CLUSTERS_VAL`). `dc_*` columns = the historical rule (groups with both classes, no minimum) for reproduction checks.

Reproduction check: 72/72 checks pass (`results/sanity_checks.csv`): stored A1 thresholds re-derived from saved val logits, and Xu-label `dc_` gender/race/intersection F_FPR and pooled TPR at A1 equal `CFES/results/fair_oppoint/fair_oppoint.json` for CFES-D s7/17/27 and the Ding reproduction on all four sets.

## 3. Models and thresholds

| model | seeds | t05 rule | A1 source | val TPR / FPR at A1 (mean) |
|---|---|---|---|---|
| CFES-D | 7,17,27 | logit + b >= 0 (deployed intercept) | stored | 80.1 / 3.9 |
| arm C | 7,17,27 | logit >= 0 | rule re-run on saved val logits (Xu labels) | 80.9 / 2.9 |
| arm E w1 | 7 | logit >= 0 | rule re-run on saved val logits (Xu labels) | 81.8 / 2.2 |
| arm E w3 | 7 | logit >= 0 | rule re-run on saved val logits (Xu labels) | 80.0 / 1.8 |
| arm E w10 | 7 | logit >= 0 | rule re-run on saved val logits (Xu labels) | 80.0 / 1.7 |
| CFES A (rev4) | 7,17,27 | logit >= 0 | rule re-run on saved val logits (Xu labels) | 81.5 / 9.2 |
| control (det-only) | 7,17,27 | logit >= 0 | rule re-run on saved val logits (Xu labels) | 80.2 / 10.1 |
| CFES-ResNet | 7,17,27 | logit >= 0 | rule re-run on saved val logits (Xu labels) | 80.3 / 9.3 |
| CESel B0 | 7 | logit >= 0 | stored | 80.2 / 5.4 |
| CESel p1 full | 7 | logit >= 0 | stored | 80.1 / 5.8 |
| CESel p1 sens | 7 | logit >= 0 | stored | 80.0 / 5.2 |
| CESel p1 inv | 7 | logit >= 0 | stored | 80.0 / 5.5 |
| Ding repro (SFD+GDA) | 7 | logit >= 0 | stored | 83.4 / 2.0 |
| Ding Ori repro (SFD) | 7,17,27 | logit >= 0 | rule re-run on saved val logits (Xu labels) | 82.9 / 7.3 |
| FF2-trained phase 1 | 7,17,27 | logit >= 0 | none (no FF++ val scores) | - / - |
| FF2-trained control | 7,17 | logit >= 0 | none (no FF++ val scores) | - / - |
| FF2-trained CFES | 7 | logit >= 0 | none (no FF++ val scores) | - / - |

Note: `CFES A (rev4)` and `control (det-only)` at seed 17 both kept their start checkpoint (TRAINED best_epoch = -1), i.e. both are the phase-1 model `runs/clip_s17_p1`; their logits agree to 1.5e-5. Their 3-seed means therefore share one seed.

CESel p1 arms (Amendment-5 screen) with a finished eval when this ran: CESel p1 full, CESel p1 inv, CESel p1 sens; the other arms and the CLIPFAIR baselines had no eval on disk (checked `CESEL/runs/ffpp_p1_*/eval`, `CLIPFAIR/runs/`). Rerun `code/rescore.py` + `code/make_report.py` when they finish (picked up automatically). **The p1 screen is validation-gated and not decided; the test-set numbers of p1 arms here must not inform the gate.** FF2-trained models were trained on FF2-full TRAIN (incl. RFW), so their public-TEST numbers are 'trained on full, tested on public'.

## 4. Results at A1 (per test set and axis; seed means)

Columns: pooled AUC / TPR / FPR (at the threshold), then for the axis: F_FPR [95% CI], FPR gap [CI], TPR gap [CI], EO gap, F_DP, es-AUC, eligible groups FPR/TPR (of all groups). All in points. A1 is not defined for FF2-trained models.

Reading the intervals: F_FPR and the gaps are sums / ranges of absolute deviations, so resampling noise inflates them (upward bias of the bootstrap distribution). The percentile interval can therefore sit above a small point estimate (e.g. a CI that excludes its own point). Use the intervals as a statement of width (how unstable the number is), not as tests; no model comparison here is a test. A pooled TPR far below another model's at the same rule usually explains a lower FPR disparity (fewer positives, fewer false positives): read every fairness number next to the TPR / FPR columns.

### FF++ test

**gender (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 95.6 | 79.7 | 3.9 | 1.3 [0.4, 3.7] | 1.3 [0.4, 3.7] | 8.6 [4.0, 13.0] | 8.6 | 6.5 | 94.0 | 2/2 (2) |
| arm C (mean of 3) | 95.8 | 79.7 | 3.2 | 1.8 [1.0, 3.8] | 1.8 [1.0, 3.8] | 12.3 [7.8, 16.5] | 12.3 | 9.6 | 93.8 | 2/2 (2) |
| arm E w1 | 97.0 | 81.6 | 2.1 | 0.9 [0.0, 2.4] | 0.9 [0.0, 2.4] | 11.2 [7.0, 15.4] | 11.2 | 8.5 | 95.3 | 2/2 (2) |
| arm E w3 | 97.1 | 80.4 | 1.6 | 0.6 [0.0, 1.8] | 0.6 [0.0, 1.8] | 12.1 [7.7, 16.1] | 12.1 | 9.2 | 95.3 | 2/2 (2) |
| arm E w10 | 96.7 | 80.6 | 2.0 | 0.8 [0.0, 2.3] | 0.8 [0.0, 2.3] | 9.9 [5.2, 14.2] | 9.9 | 7.5 | 94.8 | 2/2 (2) |
| CFES A (rev4) (mean of 3) | 93.0 | 79.0 | 8.7 | 4.2 [1.0, 8.7] | 4.2 [1.0, 8.7] | 13.5 [9.9, 17.3] | 13.5 | 11.2 | 90.7 | 2/2 (2) |
| control (det-only) (mean of 3) | 91.5 | 76.8 | 10.0 | 1.3 [1.0, 6.1] | 1.3 [1.0, 6.1] | 10.6 [6.6, 14.3] | 10.6 | 8.2 | 88.9 | 2/2 (2) |
| CFES-ResNet (mean of 3) | 92.5 | 81.1 | 10.5 | 1.9 [0.9, 6.0] | 1.9 [0.9, 6.0] | 8.7 [5.4, 12.1] | 8.7 | 6.4 | 90.3 | 2/2 (2) |
| CESel B0 | 93.3 | 76.8 | 7.3 | 3.4 [0.2, 8.5] | 3.4 [0.2, 8.5] | 9.4 [4.8, 13.7] | 9.4 | 7.6 | 91.9 | 2/2 (2) |
| CESel p1 full | 93.0 | 79.3 | 7.3 | 1.8 [0.1, 7.0] | 1.8 [0.1, 7.0] | 5.0 [0.8, 9.4] | 5.0 | 3.7 | 91.0 | 2/2 (2) |
| CESel p1 sens | 93.4 | 78.7 | 8.7 | 3.4 [0.2, 8.6] | 3.4 [0.2, 8.6] | 8.2 [3.9, 12.5] | 8.2 | 6.6 | 91.6 | 2/2 (2) |
| CESel p1 inv | 93.7 | 78.6 | 8.3 | 2.0 [0.1, 7.1] | 2.0 [0.1, 7.1] | 6.2 [2.0, 10.5] | 6.2 | 4.7 | 91.7 | 2/2 (2) |
| Ding repro (SFD+GDA) | 97.4 | 83.4 | 1.9 | 0.6 [0.0, 2.0] | 0.6 [0.0, 2.0] | 6.1 [1.9, 10.4] | 6.1 | 4.3 | 96.5 | 2/2 (2) |
| Ding Ori repro (SFD) (mean of 3) | 94.1 | 82.6 | 6.8 | 3.2 [0.9, 5.9] | 3.2 [0.9, 5.9] | 7.6 [4.5, 10.6] | 8.8 | 6.0 | 92.2 | 2/2 (2) |

**race (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 95.6 | 79.7 | 3.9 | 4.0 [2.8, 11.1] | 2.9 [2.0, 8.6] | 12.1 [6.4, 24.2] | 12.1 | 10.6 | 89.4 | 4/4 (4) |
| arm C (mean of 3) | 95.8 | 79.7 | 3.2 | 6.3 [3.2, 14.2] | 5.0 [2.2, 11.3] | 11.1 [6.1, 23.1] | 11.1 | 9.4 | 88.5 | 4/4 (4) |
| arm E w1 | 97.0 | 81.6 | 2.1 | 3.9 [1.8, 7.6] | 2.7 [1.2, 5.7] | 16.6 [8.7, 33.7] | 16.6 | 15.5 | 90.5 | 4/4 (4) |
| arm E w3 | 97.1 | 80.4 | 1.6 | 3.0 [1.4, 5.7] | 1.9 [1.0, 4.2] | 11.8 [5.7, 22.0] | 11.8 | 11.1 | 90.8 | 4/4 (4) |
| arm E w10 | 96.7 | 80.6 | 2.0 | 2.3 [1.6, 9.0] | 1.9 [1.3, 8.0] | 15.0 [7.0, 25.6] | 15.0 | 13.0 | 90.4 | 4/4 (4) |
| CFES A (rev4) (mean of 3) | 93.0 | 79.0 | 8.7 | 13.3 [8.7, 28.2] | 7.9 [6.1, 20.0] | 11.1 [7.7, 21.2] | 12.1 | 9.3 | 83.8 | 4/4 (4) |
| control (det-only) (mean of 3) | 91.5 | 76.8 | 10.0 | 10.7 [7.6, 26.7] | 8.2 [5.9, 20.9] | 10.4 [6.4, 21.6] | 11.4 | 9.9 | 83.6 | 4/4 (4) |
| CFES-ResNet (mean of 3) | 92.5 | 81.1 | 10.5 | 19.8 [10.7, 34.9] | 14.6 [7.7, 26.7] | 8.9 [5.1, 19.8] | 14.6 | 9.2 | 85.0 | 4/4 (4) |
| CESel B0 | 93.3 | 76.8 | 7.3 | 11.8 [4.4, 29.5] | 8.7 [3.1, 22.2] | 12.9 [5.1, 26.6] | 12.9 | 10.0 | 84.8 | 4/4 (4) |
| CESel p1 full | 93.0 | 79.3 | 7.3 | 13.8 [5.5, 39.3] | 7.8 [4.0, 30.7] | 8.6 [3.1, 23.0] | 8.6 | 6.2 | 85.5 | 4/4 (4) |
| CESel p1 sens | 93.4 | 78.7 | 8.7 | 12.1 [7.7, 32.5] | 11.2 [5.4, 25.2] | 10.0 [4.2, 23.7] | 11.2 | 8.3 | 85.4 | 4/4 (4) |
| CESel p1 inv | 93.7 | 78.6 | 8.3 | 11.7 [5.4, 25.5] | 8.2 [3.8, 19.6] | 11.9 [4.4, 24.9] | 11.9 | 8.8 | 87.1 | 4/4 (4) |
| Ding repro (SFD+GDA) | 97.4 | 83.4 | 1.9 | 2.1 [1.3, 6.8] | 1.9 [0.9, 5.4] | 9.3 [3.7, 19.4] | 9.3 | 9.3 | 92.8 | 4/4 (4) |
| Ding Ori repro (SFD) (mean of 3) | 94.1 | 82.6 | 6.8 | 7.2 [4.4, 18.8] | 6.4 [3.4, 16.1] | 7.1 [5.3, 14.6] | 9.4 | 7.0 | 87.8 | 4/4 (4) |

**gender x race (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 95.6 | 79.7 | 3.9 | 14.4 [12.8, 34.3] | 7.7 [4.8, 16.6] | 34.6 [25.0, 54.1] | 34.6 | 28.3 | 83.0 | 8/8 (8) |
| arm C (mean of 3) | 95.8 | 79.7 | 3.2 | 17.6 [11.8, 39.8] | 9.6 [4.3, 19.5] | 39.5 [28.6, 57.3] | 39.5 | 32.0 | 82.2 | 8/8 (8) |
| arm E w1 | 97.0 | 81.6 | 2.1 | 10.5 [7.1, 21.9] | 5.0 [2.4, 10.3] | 31.7 [24.2, 57.6] | 31.7 | 28.1 | 84.4 | 8/8 (8) |
| arm E w3 | 97.1 | 80.4 | 1.6 | 7.3 [5.0, 14.4] | 3.4 [1.6, 6.9] | 31.7 [21.9, 49.4] | 31.7 | 28.6 | 85.1 | 8/8 (8) |
| arm E w10 | 96.7 | 80.6 | 2.0 | 10.6 [7.2, 23.8] | 4.3 [2.2, 16.7] | 34.1 [22.0, 50.6] | 34.1 | 28.3 | 84.2 | 8/8 (8) |
| CFES A (rev4) (mean of 3) | 93.0 | 79.0 | 8.7 | 39.7 [31.7, 76.1] | 20.0 [13.7, 39.2] | 37.1 [28.5, 52.6] | 37.1 | 32.2 | 77.3 | 8/8 (8) |
| control (det-only) (mean of 3) | 91.5 | 76.8 | 10.0 | 39.0 [32.1, 76.5] | 19.2 [14.5, 43.2] | 36.3 [27.0, 54.0] | 36.3 | 31.6 | 77.4 | 8/8 (8) |
| CFES-ResNet (mean of 3) | 92.5 | 81.1 | 10.5 | 41.5 [33.2, 76.7] | 19.9 [14.5, 37.8] | 22.8 [18.4, 40.5] | 24.0 | 22.4 | 75.1 | 8/8 (8) |
| CESel B0 | 93.3 | 76.8 | 7.3 | 36.7 [25.1, 73.9] | 17.8 [10.3, 40.7] | 39.9 [30.1, 59.8] | 39.9 | 35.2 | 77.1 | 8/8 (8) |
| CESel p1 full | 93.0 | 79.3 | 7.3 | 38.9 [21.9, 98.0] | 19.3 [9.2, 69.2] | 22.9 [13.4, 44.5] | 22.9 | 21.3 | 76.4 | 8/8 (8) |
| CESel p1 sens | 93.4 | 78.7 | 8.7 | 41.0 [28.8, 83.4] | 19.7 [10.9, 48.1] | 34.7 [24.4, 53.1] | 34.7 | 31.5 | 77.8 | 8/8 (8) |
| CESel p1 inv | 93.7 | 78.6 | 8.3 | 30.8 [22.9, 64.8] | 16.5 [10.0, 31.5] | 32.2 [22.5, 50.3] | 32.2 | 29.2 | 79.8 | 8/8 (8) |
| Ding repro (SFD+GDA) | 97.4 | 83.4 | 1.9 | 6.4 [5.4, 16.8] | 3.3 [2.0, 9.7] | 26.1 [16.7, 42.0] | 26.1 | 25.1 | 87.2 | 8/8 (8) |
| Ding Ori repro (SFD) (mean of 3) | 94.1 | 82.6 | 6.8 | 24.5 [20.9, 53.8] | 11.3 [9.5, 28.3] | 19.2 [16.6, 34.1] | 22.6 | 20.1 | 78.7 | 8/8 (8) |

**age (AI-Face 4-bin; FF2: InsightFace <30/30-44/45-59/60+)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 95.6 | 79.7 | 3.9 | 4.7 [3.5, 10.9] | 3.6 [2.6, 7.6] | 3.9 [2.2, 9.7] | 5.0 | 5.6 | 93.3 | 4/3 (4) |
| arm C (mean of 3) | 95.8 | 79.7 | 3.2 | 6.5 [3.5, 13.5] | 4.9 [2.7, 9.1] | 5.8 [3.0, 11.2] | 6.5 | 6.0 | 93.6 | 4/3 (4) |
| arm E w1 | 97.0 | 81.6 | 2.1 | 4.8 [2.4, 9.2] | 3.7 [1.9, 6.3] | 5.0 [1.3, 10.9] | 5.0 | 3.8 | 93.8 | 4/3 (4) |
| arm E w3 | 97.1 | 80.4 | 1.6 | 3.1 [1.6, 6.0] | 2.4 [1.3, 4.2] | 7.9 [2.8, 14.1] | 7.9 | 6.1 | 94.2 | 4/3 (4) |
| arm E w10 | 96.7 | 80.6 | 2.0 | 3.7 [2.1, 6.8] | 2.9 [1.8, 4.8] | 5.8 [1.9, 11.9] | 5.8 | 5.5 | 94.0 | 4/3 (4) |
| CFES A (rev4) (mean of 3) | 93.0 | 79.0 | 8.7 | 12.4 [9.8, 21.2] | 10.2 [7.7, 15.9] | 6.6 [4.2, 11.5] | 11.6 | 6.4 | 90.0 | 4/3 (4) |
| control (det-only) (mean of 3) | 91.5 | 76.8 | 10.0 | 15.4 [12.6, 31.3] | 13.3 [10.2, 26.9] | 6.4 [3.9, 11.4] | 13.3 | 6.9 | 89.2 | 4/3 (4) |
| CFES-ResNet (mean of 3) | 92.5 | 81.1 | 10.5 | 12.6 [10.2, 23.4] | 9.5 [8.0, 17.7] | 8.6 [4.7, 13.2] | 11.5 | 4.6 | 88.4 | 4/3 (4) |
| CESel B0 | 93.3 | 76.8 | 7.3 | 79.2 [6.9, 92.5] | 77.0 [5.8, 87.6] | 6.7 [2.6, 13.1] | 77.0 | 7.5 | 90.1 | 4/3 (4) |
| CESel p1 full | 93.0 | 79.3 | 7.3 | 6.7 [2.4, 21.9] | 4.3 [1.7, 15.8] | 4.3 [1.1, 10.4] | 4.3 | 4.1 | 88.1 | 4/3 (4) |
| CESel p1 sens | 93.4 | 78.7 | 8.7 | 22.3 [9.1, 31.9] | 19.3 [7.8, 25.0] | 8.7 [3.8, 14.4] | 19.3 | 7.5 | 89.1 | 4/3 (4) |
| CESel p1 inv | 93.7 | 78.6 | 8.3 | 57.2 [8.7, 68.9] | 55.3 [7.2, 64.3] | 7.4 [3.1, 13.1] | 55.3 | 7.5 | 89.5 | 4/3 (4) |
| Ding repro (SFD+GDA) | 97.4 | 83.4 | 1.9 | 3.3 [1.8, 5.8] | 2.3 [1.5, 3.9] | 4.2 [1.0, 9.4] | 4.2 | 2.0 | 96.0 | 4/3 (4) |
| Ding Ori repro (SFD) (mean of 3) | 94.1 | 82.6 | 6.8 | 9.2 [7.4, 14.4] | 8.0 [6.5, 11.1] | 8.1 [4.7, 12.0] | 12.4 | 3.9 | 91.0 | 4/3 (4) |

**skin tone (ITA, 4 bins)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 95.6 | 79.7 | 3.9 | 4.1 [2.7, 10.6] | 3.1 [2.1, 7.7] | 4.4 [2.8, 11.7] | 4.4 | 4.0 | 93.9 | 4/4 (4) |
| arm C (mean of 3) | 95.8 | 79.7 | 3.2 | 3.3 [2.0, 11.4] | 2.4 [1.5, 8.4] | 7.4 [4.0, 14.2] | 7.4 | 6.1 | 94.1 | 4/4 (4) |
| arm E w1 | 97.0 | 81.6 | 2.1 | 1.4 [0.7, 5.6] | 0.8 [0.5, 4.2] | 2.8 [1.5, 10.2] | 2.8 | 1.0 | 96.0 | 4/4 (4) |
| arm E w3 | 97.1 | 80.4 | 1.6 | 1.9 [0.9, 5.1] | 1.7 [0.7, 4.0] | 3.9 [2.0, 11.1] | 3.9 | 2.9 | 95.6 | 4/4 (4) |
| arm E w10 | 96.7 | 80.6 | 2.0 | 1.5 [0.7, 5.3] | 1.3 [0.6, 3.9] | 3.5 [1.6, 11.0] | 3.5 | 3.2 | 94.9 | 4/4 (4) |
| CFES A (rev4) (mean of 3) | 93.0 | 79.0 | 8.7 | 5.5 [4.8, 18.0] | 4.2 [3.6, 13.2] | 8.3 [5.9, 14.2] | 8.3 | 7.0 | 90.9 | 4/4 (4) |
| control (det-only) (mean of 3) | 91.5 | 76.8 | 10.0 | 3.2 [4.3, 15.8] | 2.4 [3.1, 11.9] | 7.5 [4.4, 14.1] | 7.5 | 6.1 | 88.5 | 4/4 (4) |
| CFES-ResNet (mean of 3) | 92.5 | 81.1 | 10.5 | 4.5 [5.3, 17.4] | 3.3 [3.9, 12.7] | 5.2 [3.6, 10.9] | 5.2 | 4.3 | 90.7 | 4/4 (4) |
| CESel B0 | 93.3 | 76.8 | 7.3 | 6.6 [2.3, 20.7] | 4.8 [1.7, 14.4] | 5.2 [2.5, 13.5] | 5.2 | 4.8 | 90.3 | 4/4 (4) |
| CESel p1 full | 93.0 | 79.3 | 7.3 | 7.0 [2.9, 20.6] | 5.0 [2.2, 15.4] | 8.3 [3.8, 16.0] | 8.3 | 7.0 | 88.2 | 4/4 (4) |
| CESel p1 sens | 93.4 | 78.7 | 8.7 | 3.0 [2.6, 18.3] | 2.4 [1.9, 13.2] | 5.6 [2.3, 12.6] | 5.6 | 4.5 | 90.1 | 4/4 (4) |
| CESel p1 inv | 93.7 | 78.6 | 8.3 | 5.1 [2.7, 19.0] | 3.4 [1.9, 13.7] | 5.8 [2.7, 13.1] | 5.8 | 5.5 | 90.6 | 4/4 (4) |
| Ding repro (SFD+GDA) | 97.4 | 83.4 | 1.9 | 1.5 [0.9, 4.8] | 1.2 [0.6, 3.6] | 4.1 [1.9, 10.9] | 4.1 | 2.9 | 95.0 | 4/4 (4) |
| Ding Ori repro (SFD) (mean of 3) | 94.1 | 82.6 | 6.8 | 3.9 [2.8, 9.9] | 2.9 [2.1, 7.5] | 6.9 [4.6, 11.3] | 6.9 | 5.0 | 90.0 | 4/4 (4) |

### DFDC

**gender (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 76.9 | 38.6 | 6.7 | 0.7 [0.3, 2.9] | 0.7 [0.3, 2.9] | 6.4 [2.2, 14.3] | 6.4 | 1.3 | 69.8 | 2/2 (2) |
| arm C (mean of 3) | 75.1 | 37.2 | 5.7 | 3.1 [1.0, 5.5] | 3.1 [1.0, 5.5] | 3.5 [1.1, 11.9] | 4.9 | 3.6 | 67.0 | 2/2 (2) |
| arm E w1 | 79.2 | 28.0 | 2.8 | 0.9 [0.1, 2.9] | 0.9 [0.1, 2.9] | 0.8 [0.1, 9.4] | 0.9 | 2.5 | 71.5 | 2/2 (2) |
| arm E w3 | 77.9 | 28.3 | 3.2 | 1.0 [0.0, 3.1] | 1.0 [0.0, 3.1] | 2.9 [0.2, 11.0] | 2.9 | 1.9 | 70.3 | 2/2 (2) |
| arm E w10 | 76.4 | 32.9 | 3.5 | 1.7 [0.1, 3.7] | 1.7 [0.1, 3.7] | 3.0 [0.2, 11.4] | 3.0 | 2.7 | 69.0 | 2/2 (2) |
| CFES A (rev4) (mean of 3) | 73.2 | 48.2 | 16.4 | 8.4 [6.1, 12.3] | 8.4 [6.1, 12.3] | 6.7 [3.8, 11.9] | 12.5 | 7.6 | 65.7 | 2/2 (2) |
| control (det-only) (mean of 3) | 71.7 | 51.4 | 21.7 | 7.3 [4.6, 11.5] | 7.3 [4.6, 11.5] | 8.3 [3.1, 15.4] | 11.4 | 5.3 | 64.5 | 2/2 (2) |
| CFES-ResNet (mean of 3) | 58.7 | 29.9 | 19.5 | 5.5 [2.1, 9.1] | 5.5 [2.1, 9.1] | 5.1 [2.7, 9.7] | 6.3 | 5.1 | 55.4 | 2/2 (2) |
| CESel B0 | 76.5 | 35.5 | 7.2 | 2.5 [0.2, 5.1] | 2.5 [0.2, 5.1] | 6.0 [0.3, 14.8] | 6.0 | 2.1 | 69.8 | 2/2 (2) |
| CESel p1 full | 73.2 | 27.2 | 5.7 | 0.0 [0.0, 2.9] | 0.0 [0.0, 2.9] | 3.9 [0.2, 11.9] | 3.9 | 0.6 | 68.2 | 2/2 (2) |
| CESel p1 sens | 78.1 | 33.0 | 5.1 | 0.3 [0.0, 2.7] | 0.3 [0.0, 2.7] | 14.9 [6.3, 23.2] | 14.9 | 2.7 | 69.5 | 2/2 (2) |
| CESel p1 inv | 77.9 | 30.9 | 4.1 | 0.9 [0.1, 3.2] | 0.9 [0.1, 3.2] | 11.3 [2.9, 19.9] | 11.3 | 0.8 | 70.4 | 2/2 (2) |
| Ding repro (SFD+GDA) | 56.0 | 14.5 | 7.7 | 2.0 [0.1, 5.2] | 2.0 [0.1, 5.2] | 1.8 [0.1, 7.7] | 2.0 | 2.5 | 54.2 | 2/2 (2) |
| Ding Ori repro (SFD) (mean of 3) | 54.7 | 23.0 | 17.1 | 6.6 [3.4, 10.1] | 6.6 [3.4, 10.1] | 3.0 [1.4, 8.6] | 6.6 | 5.7 | 52.4 | 2/2 (2) |

**race (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 76.9 | 38.6 | 6.7 | 4.8 [3.4, 9.8] | 4.1 [2.5, 7.6] | 17.6 [7.8, 37.9] | 17.6 | 9.1 | 67.0 | 4/4 (4) |
| arm C (mean of 3) | 75.1 | 37.2 | 5.7 | 8.1 [5.8, 14.6] | 5.9 [3.9, 10.2] | 20.7 [9.2, 41.2] | 20.7 | 8.7 | 65.3 | 4/4 (4) |
| arm E w1 | 79.2 | 28.0 | 2.8 | 1.4 [1.1, 8.8] | 1.2 [0.8, 6.9] | 17.9 [6.0, 41.1] | 17.9 | 8.9 | 69.5 | 4/4 (4) |
| arm E w3 | 77.9 | 28.3 | 3.2 | 1.8 [1.3, 9.4] | 1.3 [0.9, 7.2] | 17.6 [5.9, 41.9] | 17.6 | 8.6 | 68.3 | 4/4 (4) |
| arm E w10 | 76.4 | 32.9 | 3.5 | 4.0 [1.8, 9.3] | 2.2 [1.3, 7.4] | 21.2 [6.4, 42.6] | 21.2 | 10.0 | 66.5 | 4/4 (4) |
| CFES A (rev4) (mean of 3) | 73.2 | 48.2 | 16.4 | 14.3 [10.4, 24.5] | 10.2 [7.3, 16.8] | 18.6 [11.8, 36.9] | 18.6 | 11.4 | 63.2 | 4/4 (4) |
| control (det-only) (mean of 3) | 71.7 | 51.4 | 21.7 | 17.7 [13.6, 27.3] | 12.4 [9.9, 18.4] | 15.7 [11.1, 34.9] | 18.8 | 10.6 | 62.4 | 4/4 (4) |
| CFES-ResNet (mean of 3) | 58.7 | 29.9 | 19.5 | 12.4 [7.7, 21.7] | 9.8 [5.7, 18.1] | 18.9 [11.3, 36.0] | 18.9 | 5.2 | 47.4 | 4/4 (4) |
| CESel B0 | 76.5 | 35.5 | 7.2 | 5.1 [2.0, 12.1] | 3.4 [1.4, 8.8] | 17.9 [6.3, 38.9] | 17.9 | 8.2 | 67.8 | 4/4 (4) |
| CESel p1 full | 73.2 | 27.2 | 5.7 | 4.2 [2.4, 11.5] | 3.0 [1.7, 8.7] | 20.3 [6.2, 40.7] | 20.3 | 7.0 | 61.8 | 4/4 (4) |
| CESel p1 sens | 78.1 | 33.0 | 5.1 | 4.2 [1.6, 11.5] | 2.7 [1.2, 8.5] | 23.3 [11.7, 43.7] | 23.3 | 9.7 | 64.6 | 4/4 (4) |
| CESel p1 inv | 77.9 | 30.9 | 4.1 | 3.1 [1.4, 9.8] | 1.9 [1.0, 7.2] | 16.4 [4.6, 38.3] | 16.4 | 7.0 | 67.3 | 4/4 (4) |
| Ding repro (SFD+GDA) | 56.0 | 14.5 | 7.7 | 7.5 [3.0, 14.2] | 5.6 [2.1, 9.4] | 11.6 [7.7, 26.1] | 11.6 | 5.8 | 45.7 | 4/4 (4) |
| Ding Ori repro (SFD) (mean of 3) | 54.7 | 23.0 | 17.1 | 11.8 [8.9, 20.1] | 8.2 [6.3, 14.3] | 20.2 [12.8, 31.7] | 20.2 | 5.4 | 45.6 | 4/4 (4) |

**gender x race (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 76.9 | 38.6 | 6.7 | 14.6 [13.1, 27.6] | 7.5 [6.0, 13.1] | 38.8 [33.9, 64.0] | 38.8 | 15.9 | 58.3 | 8/8 (8) |
| arm C (mean of 3) | 75.1 | 37.2 | 5.7 | 20.3 [15.7, 34.9] | 9.3 [7.2, 16.0] | 46.8 [40.2, 69.1] | 46.8 | 19.3 | 54.4 | 8/8 (8) |
| arm E w1 | 79.2 | 28.0 | 2.8 | 5.6 [5.9, 22.5] | 3.2 [2.9, 12.2] | 39.0 [30.7, 65.1] | 39.0 | 12.3 | 59.2 | 8/8 (8) |
| arm E w3 | 77.9 | 28.3 | 3.2 | 5.9 [6.1, 24.8] | 3.2 [3.0, 12.8] | 38.5 [32.0, 65.7] | 38.5 | 11.8 | 57.3 | 8/8 (8) |
| arm E w10 | 76.4 | 32.9 | 3.5 | 10.2 [7.6, 25.1] | 4.6 [3.6, 12.4] | 41.9 [33.3, 65.1] | 41.9 | 14.5 | 54.2 | 8/8 (8) |
| CFES A (rev4) (mean of 3) | 73.2 | 48.2 | 16.4 | 43.9 [35.6, 63.2] | 19.9 [16.7, 27.4] | 40.3 [29.6, 60.8] | 40.3 | 26.4 | 50.1 | 8/8 (8) |
| control (det-only) (mean of 3) | 71.7 | 51.4 | 21.7 | 55.2 [45.7, 74.0] | 24.2 [20.4, 33.8] | 45.4 [32.8, 64.3] | 46.1 | 30.4 | 51.0 | 8/8 (8) |
| CFES-ResNet (mean of 3) | 58.7 | 29.9 | 19.5 | 38.9 [32.7, 60.1] | 17.4 [15.4, 27.9] | 32.0 [23.9, 58.3] | 32.0 | 11.3 | 36.4 | 8/8 (8) |
| CESel B0 | 76.5 | 35.5 | 7.2 | 11.0 [9.5, 28.6] | 6.0 [4.4, 14.0] | 36.9 [28.8, 64.5] | 36.9 | 14.6 | 55.8 | 8/8 (8) |
| CESel p1 full | 73.2 | 27.2 | 5.7 | 11.3 [9.3, 30.2] | 6.2 [4.3, 16.1] | 36.0 [25.7, 62.2] | 36.0 | 12.0 | 53.8 | 8/8 (8) |
| CESel p1 sens | 78.1 | 33.0 | 5.1 | 12.6 [7.4, 31.6] | 6.0 [3.4, 15.9] | 37.5 [30.2, 63.5] | 37.5 | 16.1 | 52.9 | 8/8 (8) |
| CESel p1 inv | 77.9 | 30.9 | 4.1 | 9.5 [6.1, 26.3] | 3.8 [2.8, 13.4] | 33.5 [26.5, 60.0] | 33.5 | 12.5 | 54.3 | 8/8 (8) |
| Ding repro (SFD+GDA) | 56.0 | 14.5 | 7.7 | 18.7 [12.9, 33.4] | 8.3 [6.3, 13.3] | 16.7 [12.8, 42.0] | 16.7 | 9.4 | 33.1 | 8/8 (8) |
| Ding Ori repro (SFD) (mean of 3) | 54.7 | 23.0 | 17.1 | 37.7 [31.8, 55.4] | 17.4 [14.5, 26.7] | 32.7 [23.7, 53.7] | 32.7 | 16.3 | 35.7 | 8/8 (8) |

**age (AI-Face 4-bin; FF2: InsightFace <30/30-44/45-59/60+)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 76.9 | 38.6 | 6.7 | 6.8 [5.1, 12.6] | 5.1 [3.6, 10.2] | 19.9 [13.4, 47.0] | 19.9 | 13.2 | 63.7 | 4/4 (4) |
| arm C (mean of 3) | 75.1 | 37.2 | 5.7 | 8.6 [5.6, 13.4] | 4.9 [3.7, 7.5] | 23.3 [14.0, 50.4] | 23.3 | 13.2 | 59.8 | 4/4 (4) |
| arm E w1 | 79.2 | 28.0 | 2.8 | 4.4 [2.4, 8.6] | 3.2 [1.6, 6.2] | 29.7 [14.4, 56.2] | 29.7 | 13.6 | 67.4 | 4/4 (4) |
| arm E w3 | 77.9 | 28.3 | 3.2 | 4.7 [2.8, 9.3] | 3.3 [2.1, 6.5] | 31.5 [14.1, 58.4] | 31.5 | 13.3 | 66.8 | 4/4 (4) |
| arm E w10 | 76.4 | 32.9 | 3.5 | 5.6 [2.6, 9.5] | 3.7 [1.7, 7.1] | 28.5 [12.8, 57.6] | 28.5 | 13.6 | 66.8 | 4/4 (4) |
| CFES A (rev4) (mean of 3) | 73.2 | 48.2 | 16.4 | 16.6 [11.3, 24.6] | 9.9 [7.7, 15.2] | 21.4 [14.2, 42.9] | 21.4 | 14.5 | 58.6 | 4/4 (4) |
| control (det-only) (mean of 3) | 71.7 | 51.4 | 21.7 | 18.3 [14.3, 27.9] | 11.3 [9.4, 18.8] | 17.2 [10.9, 40.7] | 17.2 | 12.9 | 59.1 | 4/4 (4) |
| CFES-ResNet (mean of 3) | 58.7 | 29.9 | 19.5 | 13.0 [9.0, 22.1] | 8.4 [6.8, 15.8] | 21.2 [12.0, 42.2] | 21.2 | 8.8 | 49.1 | 4/4 (4) |
| CESel B0 | 76.5 | 35.5 | 7.2 | 9.5 [5.3, 15.9] | 5.0 [3.2, 8.4] | 19.4 [9.1, 49.6] | 19.4 | 11.1 | 64.0 | 4/4 (4) |
| CESel p1 full | 73.2 | 27.2 | 5.7 | 4.3 [2.0, 10.7] | 2.2 [1.4, 7.5] | 18.7 [7.4, 45.4] | 18.7 | 9.8 | 62.1 | 4/4 (4) |
| CESel p1 sens | 78.1 | 33.0 | 5.1 | 8.1 [5.2, 13.0] | 5.2 [3.6, 7.4] | 19.4 [10.6, 50.4] | 19.4 | 12.2 | 63.8 | 4/4 (4) |
| CESel p1 inv | 77.9 | 30.9 | 4.1 | 8.6 [4.8, 12.9] | 4.2 [2.8, 6.3] | 18.9 [7.3, 47.9] | 18.9 | 9.6 | 64.6 | 4/4 (4) |
| Ding repro (SFD+GDA) | 56.0 | 14.5 | 7.7 | 4.1 [1.9, 11.6] | 2.6 [1.3, 8.7] | 5.9 [4.0, 27.7] | 5.9 | 4.8 | 48.8 | 4/4 (4) |
| Ding Ori repro (SFD) (mean of 3) | 54.7 | 23.0 | 17.1 | 13.1 [10.5, 22.7] | 8.9 [6.6, 15.4] | 14.1 [10.4, 31.1] | 14.1 | 7.6 | 44.7 | 4/4 (4) |

**skin tone (ITA, 4 bins)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 76.9 | 38.6 | 6.7 | 8.7 [5.6, 14.1] | 5.0 [3.7, 7.7] | 13.4 [7.7, 39.6] | 13.4 | 4.0 | 67.8 | 4/4 (4) |
| arm C (mean of 3) | 75.1 | 37.2 | 5.7 | 7.8 [5.1, 13.1] | 5.1 [3.3, 9.4] | 11.2 [7.0, 36.7] | 11.2 | 3.8 | 67.6 | 4/4 (4) |
| arm E w1 | 79.2 | 28.0 | 2.8 | 2.0 [1.1, 8.1] | 1.1 [0.8, 6.6] | 15.0 [8.3, 36.6] | 15.0 | 6.7 | 69.8 | 4/4 (4) |
| arm E w3 | 77.9 | 28.3 | 3.2 | 1.6 [1.2, 9.5] | 0.9 [0.9, 8.1] | 16.1 [7.5, 37.1] | 16.1 | 6.1 | 67.7 | 4/4 (4) |
| arm E w10 | 76.4 | 32.9 | 3.5 | 2.5 [1.4, 8.5] | 1.7 [1.0, 6.6] | 13.7 [6.5, 36.8] | 13.7 | 4.3 | 65.6 | 4/4 (4) |
| CFES A (rev4) (mean of 3) | 73.2 | 48.2 | 16.4 | 10.6 [7.3, 21.8] | 5.5 [5.1, 14.1] | 12.1 [8.3, 35.2] | 12.1 | 4.1 | 62.6 | 4/4 (4) |
| control (det-only) (mean of 3) | 71.7 | 51.4 | 21.7 | 18.4 [13.2, 30.5] | 10.0 [7.7, 19.6] | 9.4 [7.3, 34.3] | 11.7 | 6.4 | 61.8 | 4/4 (4) |
| CFES-ResNet (mean of 3) | 58.7 | 29.9 | 19.5 | 17.7 [12.5, 31.6] | 12.2 [7.9, 24.0] | 13.6 [10.9, 31.6] | 16.2 | 10.8 | 50.0 | 4/4 (4) |
| CESel B0 | 76.5 | 35.5 | 7.2 | 12.4 [7.5, 19.3] | 6.1 [4.4, 10.0] | 20.2 [7.8, 51.0] | 20.2 | 1.7 | 61.8 | 4/4 (4) |
| CESel p1 full | 73.2 | 27.2 | 5.7 | 1.1 [2.1, 13.4] | 0.8 [1.4, 11.1] | 17.9 [6.7, 40.1] | 17.9 | 3.8 | 64.6 | 4/4 (4) |
| CESel p1 sens | 78.1 | 33.0 | 5.1 | 6.8 [4.3, 14.3] | 4.0 [2.7, 10.3] | 19.3 [7.9, 44.3] | 19.3 | 2.6 | 64.9 | 4/4 (4) |
| CESel p1 inv | 77.9 | 30.9 | 4.1 | 7.3 [4.0, 11.9] | 4.1 [2.9, 6.1] | 21.7 [7.2, 47.5] | 21.7 | 4.5 | 64.1 | 4/4 (4) |
| Ding repro (SFD+GDA) | 56.0 | 14.5 | 7.7 | 9.4 [4.2, 21.2] | 8.1 [3.0, 18.7] | 11.9 [3.8, 33.2] | 11.9 | 5.7 | 47.4 | 4/4 (4) |
| Ding Ori repro (SFD) (mean of 3) | 54.7 | 23.0 | 17.1 | 18.3 [14.3, 29.1] | 11.4 [8.6, 20.3] | 11.4 [9.8, 31.4] | 13.9 | 9.5 | 46.7 | 4/4 (4) |

### Celeb-DF

**gender (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 87.3 | 72.9 | 15.3 | 3.3 [1.7, 7.7] | 3.3 [1.7, 7.7] | 25.8 [16.3, 35.7] | 25.8 | 36.5 | 75.9 | 2/2 (2) |
| arm C (mean of 3) | 89.1 | 76.1 | 14.3 | 4.7 [1.5, 10.2] | 4.7 [1.5, 10.2] | 33.3 [23.4, 43.1] | 33.3 | 42.9 | 76.9 | 2/2 (2) |
| arm E w1 | 87.6 | 70.9 | 12.2 | 2.3 [0.1, 7.7] | 2.3 [0.1, 7.7] | 42.2 [32.2, 51.0] | 42.2 | 44.4 | 76.1 | 2/2 (2) |
| arm E w3 | 86.9 | 72.6 | 13.9 | 2.5 [0.2, 8.4] | 2.5 [0.2, 8.4] | 34.7 [25.2, 44.2] | 34.7 | 41.2 | 77.5 | 2/2 (2) |
| arm E w10 | 84.9 | 70.5 | 15.5 | 4.6 [0.3, 11.5] | 4.6 [0.3, 11.5] | 8.3 [0.8, 16.7] | 8.3 | 24.7 | 78.4 | 2/2 (2) |
| CFES A (rev4) (mean of 3) | 85.7 | 81.5 | 31.9 | 17.9 [11.0, 24.1] | 17.9 [11.0, 24.1] | 39.4 [30.3, 47.6] | 41.0 | 46.5 | 73.5 | 2/2 (2) |
| control (det-only) (mean of 3) | 81.4 | 75.2 | 28.9 | 17.6 [11.2, 23.8] | 17.6 [11.2, 23.8] | 45.8 [36.3, 53.9] | 45.8 | 47.8 | 67.9 | 2/2 (2) |
| CFES-ResNet (mean of 3) | 71.4 | 72.8 | 41.5 | 10.5 [5.5, 16.2] | 10.5 [5.5, 16.2] | 26.3 [19.9, 32.6] | 26.3 | 29.6 | 64.8 | 2/2 (2) |
| CESel B0 | 85.0 | 74.4 | 19.4 | 5.9 [0.5, 13.0] | 5.9 [0.5, 13.0] | 39.7 [28.6, 48.7] | 39.7 | 43.4 | 71.1 | 2/2 (2) |
| CESel p1 full | 82.2 | 86.8 | 36.0 | 7.1 [0.5, 17.1] | 7.1 [0.5, 17.1] | 25.8 [16.8, 34.6] | 25.8 | 36.0 | 75.7 | 2/2 (2) |
| CESel p1 sens | 87.1 | 85.5 | 28.1 | 11.9 [3.5, 20.2] | 11.9 [3.5, 20.2] | 42.8 [31.9, 52.4] | 42.8 | 48.6 | 74.5 | 2/2 (2) |
| CESel p1 inv | 85.7 | 76.1 | 20.0 | 3.7 [0.3, 11.8] | 3.7 [0.3, 11.8] | 33.0 [22.5, 42.1] | 33.0 | 39.9 | 73.8 | 2/2 (2) |
| Ding repro (SFD+GDA) | 71.2 | 58.2 | 25.9 | 11.0 [4.1, 18.4] | 11.0 [4.1, 18.4] | 48.1 [41.5, 53.8] | 48.1 | 39.9 | 59.8 | 2/2 (2) |
| Ding Ori repro (SFD) (mean of 3) | 67.1 | 68.7 | 40.4 | 11.4 [6.0, 17.9] | 11.4 [6.0, 17.9] | 34.1 [27.4, 40.6] | 34.1 | 32.2 | 60.6 | 2/2 (2) |

**race (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 87.3 | 72.9 | 15.3 | 13.8 [10.6, 35.1] | 12.7 [8.4, 33.9] | 38.5 [24.3, 59.7] | 38.5 | 37.4 | 64.4 | 3/3 (3) |
| arm C (mean of 3) | 89.1 | 76.1 | 14.3 | 25.1 [10.9, 42.4] | 24.9 [9.5, 41.9] | 33.6 [26.4, 60.3] | 33.6 | 35.0 | 62.8 | 3/3 (3) |
| arm E w1 | 87.6 | 70.9 | 12.2 | 17.7 [5.4, 36.0] | 17.6 [4.7, 35.6] | 42.2 [34.6, 57.5] | 42.2 | 39.4 | 62.1 | 3/3 (3) |
| arm E w3 | 86.9 | 72.6 | 13.9 | 17.1 [4.4, 38.7] | 16.9 [3.6, 38.3] | 40.3 [29.3, 57.0] | 40.3 | 35.6 | 60.2 | 3/3 (3) |
| arm E w10 | 84.9 | 70.5 | 15.5 | 26.3 [8.5, 46.6] | 22.1 [6.8, 40.0] | 34.1 [9.0, 61.1] | 34.1 | 24.2 | 56.3 | 3/3 (3) |
| CFES A (rev4) (mean of 3) | 85.7 | 81.5 | 31.9 | 24.8 [16.9, 44.1] | 21.4 [14.2, 41.3] | 39.4 [31.0, 51.0] | 41.1 | 41.0 | 70.6 | 3/3 (3) |
| control (det-only) (mean of 3) | 81.4 | 75.2 | 28.9 | 27.5 [19.5, 56.0] | 25.4 [16.3, 53.3] | 51.6 [39.8, 64.6] | 51.6 | 44.5 | 67.2 | 3/3 (3) |
| CFES-ResNet (mean of 3) | 71.4 | 72.8 | 41.5 | 21.8 [14.7, 37.5] | 18.6 [12.7, 33.4] | 36.3 [24.9, 48.4] | 36.3 | 27.2 | 61.4 | 3/3 (3) |
| CESel B0 | 85.0 | 74.4 | 19.4 | 18.4 [7.0, 32.1] | 10.8 [5.3, 20.8] | 39.7 [31.0, 51.7] | 39.7 | 48.4 | 74.3 | 3/3 (3) |
| CESel p1 full | 82.2 | 86.8 | 36.0 | 32.4 [19.8, 57.7] | 23.4 [16.3, 40.9] | 38.2 [22.4, 67.2] | 38.2 | 48.3 | 68.3 | 3/3 (3) |
| CESel p1 sens | 87.1 | 85.5 | 28.1 | 24.5 [12.5, 42.2] | 18.1 [9.5, 29.0] | 42.8 [32.4, 53.8] | 42.8 | 46.4 | 77.7 | 3/3 (3) |
| CESel p1 inv | 85.7 | 76.1 | 20.0 | 18.3 [12.3, 37.6] | 16.5 [11.0, 33.5] | 33.2 [26.5, 60.9] | 33.2 | 40.8 | 71.9 | 3/3 (3) |
| Ding repro (SFD+GDA) | 71.2 | 58.2 | 25.9 | 45.4 [26.8, 64.3] | 43.2 [23.8, 62.2] | 48.2 [43.5, 57.8] | 48.2 | 45.1 | 51.0 | 3/3 (3) |
| Ding Ori repro (SFD) (mean of 3) | 67.1 | 68.7 | 40.4 | 43.1 [27.1, 59.3] | 41.5 [25.0, 57.0] | 34.1 [28.8, 42.6] | 41.7 | 33.7 | 50.1 | 3/3 (3) |

**gender x race (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 87.3 | 72.9 | 15.3 | 27.0 [22.6, 69.8] | 18.9 [15.4, 55.9] | 38.5 [24.3, 59.7] | 38.5 | 34.0 | 63.4 | 5/3 (6) |
| arm C (mean of 3) | 89.1 | 76.1 | 14.3 | 43.6 [26.9, 62.6] | 36.4 [18.1, 51.5] | 33.6 [26.4, 60.3] | 40.4 | 36.9 | 60.0 | 5/3 (6) |
| arm E w1 | 87.6 | 70.9 | 12.2 | 26.6 [17.9, 60.7] | 25.3 [13.8, 52.1] | 42.2 [34.6, 57.5] | 42.2 | 41.2 | 60.6 | 5/3 (6) |
| arm E w3 | 86.9 | 72.6 | 13.9 | 25.7 [12.3, 69.2] | 24.0 [8.3, 58.3] | 40.3 [29.3, 57.0] | 40.3 | 36.6 | 58.9 | 5/3 (6) |
| arm E w10 | 84.9 | 70.5 | 15.5 | 43.3 [24.1, 77.8] | 30.2 [13.8, 55.3] | 34.1 [9.0, 61.1] | 34.1 | 20.2 | 55.4 | 5/3 (6) |
| CFES A (rev4) (mean of 3) | 85.7 | 81.5 | 31.9 | 57.3 [41.4, 83.1] | 30.0 [21.0, 54.5] | 39.4 [31.0, 51.0] | 44.5 | 42.2 | 67.8 | 5/3 (6) |
| control (det-only) (mean of 3) | 81.4 | 75.2 | 28.9 | 69.4 [47.6, 103.0] | 41.1 [24.1, 68.6] | 51.6 [39.8, 64.6] | 55.0 | 46.2 | 64.6 | 5/3 (6) |
| CFES-ResNet (mean of 3) | 71.4 | 72.8 | 41.5 | 51.4 [37.3, 77.2] | 35.4 [28.4, 55.8] | 36.3 [24.9, 48.4] | 41.7 | 30.1 | 59.1 | 5/3 (6) |
| CESel B0 | 85.0 | 74.4 | 19.4 | 25.2 [19.0, 54.9] | 14.2 [11.4, 33.3] | 39.7 [31.0, 51.7] | 39.7 | 51.2 | 73.6 | 5/3 (6) |
| CESel p1 full | 82.2 | 86.8 | 36.0 | 76.1 [44.5, 109.7] | 48.1 [25.6, 76.7] | 38.2 [22.4, 67.2] | 48.1 | 60.6 | 74.4 | 5/3 (6) |
| CESel p1 sens | 87.1 | 85.5 | 28.1 | 53.8 [35.5, 76.7] | 29.8 [21.3, 42.6] | 42.8 [32.4, 53.8] | 42.8 | 52.3 | 75.2 | 5/3 (6) |
| CESel p1 inv | 85.7 | 76.1 | 20.0 | 48.4 [27.5, 69.5] | 36.1 [19.7, 47.5] | 33.2 [26.5, 60.9] | 36.1 | 48.6 | 77.2 | 5/3 (6) |
| Ding repro (SFD+GDA) | 71.2 | 58.2 | 25.9 | 62.2 [40.2, 92.0] | 50.8 [28.7, 69.0] | 48.2 [43.5, 57.8] | 50.8 | 46.0 | 49.5 | 5/3 (6) |
| Ding Ori repro (SFD) (mean of 3) | 67.1 | 68.7 | 40.4 | 68.9 [47.8, 99.5] | 45.0 [34.4, 75.8] | 34.1 [28.8, 42.6] | 45.0 | 34.9 | 48.7 | 5/3 (6) |

**age (AI-Face 4-bin; FF2: InsightFace <30/30-44/45-59/60+)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 87.3 | 72.9 | 15.3 | 10.4 [6.0, 20.3] | 9.9 [5.0, 18.2] | 8.0 [5.6, 13.2] | 10.7 | 10.2 | 83.6 | 3/3 (3) |
| arm C (mean of 3) | 89.1 | 76.1 | 14.3 | 5.7 [3.9, 15.4] | 5.0 [3.5, 13.6] | 8.1 [4.1, 14.0] | 8.3 | 9.8 | 85.7 | 3/3 (3) |
| arm E w1 | 87.6 | 70.9 | 12.2 | 1.8 [0.9, 11.6] | 1.6 [0.7, 10.4] | 7.2 [2.6, 14.1] | 7.2 | 11.0 | 85.2 | 3/3 (3) |
| arm E w3 | 86.9 | 72.6 | 13.9 | 0.4 [0.8, 12.3] | 0.4 [0.8, 10.5] | 8.6 [3.8, 14.7] | 8.6 | 11.1 | 84.4 | 3/3 (3) |
| arm E w10 | 84.9 | 70.5 | 15.5 | 3.7 [1.3, 15.7] | 2.6 [1.1, 13.9] | 1.8 [0.6, 8.5] | 2.6 | 6.7 | 82.2 | 3/3 (3) |
| CFES A (rev4) (mean of 3) | 85.7 | 81.5 | 31.9 | 10.3 [7.9, 20.4] | 9.3 [6.9, 18.1] | 6.0 [3.7, 10.0] | 10.4 | 9.0 | 82.7 | 3/3 (3) |
| control (det-only) (mean of 3) | 81.4 | 75.2 | 28.9 | 8.3 [6.0, 21.3] | 6.8 [5.2, 18.7] | 6.4 [4.8, 11.1] | 7.4 | 7.2 | 78.5 | 3/3 (3) |
| CFES-ResNet (mean of 3) | 71.4 | 72.8 | 41.5 | 13.6 [8.6, 22.9] | 13.0 [7.7, 21.1] | 8.0 [4.6, 12.1] | 13.0 | 9.6 | 64.3 | 3/3 (3) |
| CESel B0 | 85.0 | 74.4 | 19.4 | 10.5 [3.1, 24.2] | 8.5 [2.6, 21.8] | 10.4 [5.8, 16.5] | 10.4 | 8.6 | 81.0 | 3/3 (3) |
| CESel p1 full | 82.2 | 86.8 | 36.0 | 4.0 [1.8, 20.7] | 2.8 [1.5, 18.3] | 3.1 [0.8, 7.2] | 3.1 | 8.4 | 81.0 | 3/3 (3) |
| CESel p1 sens | 87.1 | 85.5 | 28.1 | 3.5 [1.9, 20.3] | 3.2 [1.6, 18.4] | 10.5 [5.9, 15.4] | 10.5 | 10.8 | 82.5 | 3/3 (3) |
| CESel p1 inv | 85.7 | 76.1 | 20.0 | 6.2 [1.7, 21.4] | 5.5 [1.5, 19.2] | 10.3 [5.3, 16.9] | 10.3 | 8.4 | 81.2 | 3/3 (3) |
| Ding repro (SFD+GDA) | 71.2 | 58.2 | 25.9 | 10.5 [2.0, 23.9] | 9.5 [1.7, 22.2] | 11.9 [6.8, 17.6] | 11.9 | 12.8 | 65.0 | 3/3 (3) |
| Ding Ori repro (SFD) (mean of 3) | 67.1 | 68.7 | 40.4 | 7.5 [3.8, 18.5] | 6.6 [3.3, 16.6] | 9.0 [5.4, 13.0] | 9.0 | 9.8 | 61.6 | 3/3 (3) |

**skin tone (ITA, 4 bins)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 87.3 | 72.9 | 15.3 | 9.1 [7.2, 19.8] | 5.8 [5.0, 14.0] | 12.0 [8.4, 16.3] | 12.0 | 12.6 | 78.6 | 4/4 (4) |
| arm C (mean of 3) | 89.1 | 76.1 | 14.3 | 7.4 [6.4, 19.9] | 6.3 [4.9, 15.1] | 13.5 [10.4, 17.8] | 13.5 | 14.4 | 80.7 | 4/4 (4) |
| arm E w1 | 87.6 | 70.9 | 12.2 | 8.0 [3.4, 18.4] | 5.0 [2.5, 13.4] | 13.9 [9.8, 18.5] | 13.9 | 14.2 | 79.6 | 4/4 (4) |
| arm E w3 | 86.9 | 72.6 | 13.9 | 6.0 [2.7, 18.2] | 4.5 [2.0, 13.9] | 12.6 [9.0, 17.0] | 12.6 | 13.2 | 79.3 | 4/4 (4) |
| arm E w10 | 84.9 | 70.5 | 15.5 | 10.9 [5.6, 23.9] | 8.5 [4.1, 16.4] | 7.1 [3.6, 11.7] | 8.5 | 7.5 | 75.4 | 4/4 (4) |
| CFES A (rev4) (mean of 3) | 85.7 | 81.5 | 31.9 | 20.1 [12.4, 34.2] | 13.8 [9.2, 24.1] | 15.3 [12.7, 18.7] | 19.3 | 17.7 | 78.6 | 4/4 (4) |
| control (det-only) (mean of 3) | 81.4 | 75.2 | 28.9 | 20.1 [11.4, 35.8] | 14.4 [8.3, 27.5] | 20.4 [17.0, 24.3] | 20.6 | 21.1 | 72.0 | 4/4 (4) |
| CFES-ResNet (mean of 3) | 71.4 | 72.8 | 41.5 | 9.3 [7.8, 22.4] | 5.8 [5.7, 16.9] | 8.5 [6.8, 11.9] | 9.6 | 9.0 | 67.9 | 4/4 (4) |
| CESel B0 | 85.0 | 74.4 | 19.4 | 18.3 [7.5, 34.6] | 14.2 [5.6, 25.9] | 16.2 [12.4, 21.6] | 16.2 | 18.1 | 77.7 | 4/4 (4) |
| CESel p1 full | 82.2 | 86.8 | 36.0 | 11.4 [5.7, 32.9] | 8.8 [4.1, 24.8] | 9.5 [6.5, 12.6] | 9.5 | 11.5 | 75.1 | 4/4 (4) |
| CESel p1 sens | 87.1 | 85.5 | 28.1 | 19.6 [8.0, 38.0] | 12.6 [6.0, 27.8] | 11.3 [8.3, 14.8] | 12.6 | 13.7 | 81.7 | 4/4 (4) |
| CESel p1 inv | 85.7 | 76.1 | 20.0 | 15.6 [5.8, 33.0] | 9.9 [4.0, 23.5] | 15.1 [12.0, 20.1] | 15.1 | 16.9 | 80.7 | 4/4 (4) |
| Ding repro (SFD+GDA) | 71.2 | 58.2 | 25.9 | 17.8 [7.9, 32.8] | 10.8 [5.7, 21.9] | 15.1 [10.8, 20.0] | 15.1 | 15.8 | 65.9 | 4/4 (4) |
| Ding Ori repro (SFD) (mean of 3) | 67.1 | 68.7 | 40.4 | 19.8 [15.5, 31.9] | 14.0 [10.6, 22.2] | 11.7 [9.6, 15.2] | 17.4 | 13.0 | 60.8 | 4/4 (4) |

### DFD

**gender (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 89.9 | 75.6 | 9.9 | 8.4 [3.9, 13.2] | 8.4 [3.9, 13.2] | 3.7 [0.7, 12.0] | 8.4 | 1.8 | 85.1 | 2/2 (2) |
| arm C (mean of 3) | 90.7 | 73.4 | 7.5 | 7.5 [3.1, 12.0] | 7.5 [3.1, 12.0] | 1.2 [0.8, 10.3] | 7.5 | 0.9 | 87.1 | 2/2 (2) |
| arm E w1 | 89.7 | 55.3 | 3.5 | 3.2 [0.7, 5.7] | 3.2 [0.7, 5.7] | 5.0 [0.3, 15.5] | 5.0 | 2.8 | 84.9 | 2/2 (2) |
| arm E w3 | 89.7 | 59.2 | 3.8 | 3.1 [0.4, 5.7] | 3.1 [0.4, 5.7] | 0.4 [0.2, 11.6] | 3.1 | 0.4 | 85.2 | 2/2 (2) |
| arm E w10 | 90.0 | 68.5 | 4.8 | 3.7 [0.5, 6.8] | 3.7 [0.5, 6.8] | 4.2 [0.2, 13.7] | 4.2 | 3.6 | 86.3 | 2/2 (2) |
| CFES A (rev4) (mean of 3) | 87.0 | 74.0 | 16.6 | 5.0 [3.2, 11.3] | 5.0 [3.2, 11.3] | 2.3 [0.8, 9.3] | 5.1 | 2.1 | 84.8 | 2/2 (2) |
| control (det-only) (mean of 3) | 83.7 | 67.0 | 15.1 | 4.2 [1.4, 10.1] | 4.2 [1.4, 10.1] | 4.0 [1.5, 12.3] | 4.3 | 2.5 | 82.2 | 2/2 (2) |
| CFES-ResNet (mean of 3) | 71.4 | 69.1 | 36.0 | 7.7 [2.4, 16.8] | 7.7 [2.4, 16.8] | 2.6 [1.5, 10.4] | 7.7 | 3.2 | 69.4 | 2/2 (2) |
| CESel B0 | 88.0 | 67.6 | 5.2 | 0.0 [0.1, 4.8] | 0.0 [0.1, 4.8] | 3.5 [0.2, 12.9] | 3.5 | 4.3 | 86.3 | 2/2 (2) |
| CESel p1 full | 86.9 | 74.6 | 8.7 | 6.9 [2.3, 11.8] | 6.9 [2.3, 11.8] | 5.8 [0.3, 14.7] | 6.9 | 3.7 | 81.2 | 2/2 (2) |
| CESel p1 sens | 88.6 | 70.2 | 5.4 | 0.1 [0.0, 3.3] | 0.1 [0.0, 3.3] | 1.1 [0.1, 11.0] | 1.1 | 2.7 | 87.7 | 2/2 (2) |
| CESel p1 inv | 88.8 | 69.1 | 4.2 | 3.2 [0.5, 5.7] | 3.2 [0.5, 5.7] | 2.1 [0.2, 11.5] | 3.2 | 2.4 | 85.1 | 2/2 (2) |
| Ding repro (SFD+GDA) | 72.5 | 63.7 | 25.2 | 2.3 [0.2, 12.4] | 2.3 [0.2, 12.4] | 3.9 [0.1, 12.3] | 3.9 | 4.6 | 72.4 | 2/2 (2) |
| Ding Ori repro (SFD) (mean of 3) | 66.8 | 66.7 | 37.4 | 2.0 [1.2, 9.8] | 2.0 [1.2, 9.8] | 3.5 [1.1, 10.0] | 3.5 | 3.9 | 65.4 | 2/2 (2) |

**race (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 89.9 | 75.6 | 9.9 | 22.5 [12.2, 35.4] | 15.4 [8.9, 23.8] | 25.7 [14.0, 43.6] | 32.0 | 39.8 | 79.8 | 3/3 (3) |
| arm C (mean of 3) | 90.7 | 73.4 | 7.5 | 13.1 [4.9, 25.3] | 9.1 [3.5, 18.1] | 33.1 [16.5, 52.5] | 33.1 | 43.8 | 79.5 | 3/3 (3) |
| arm E w1 | 89.7 | 55.3 | 3.5 | 2.0 [0.7, 9.3] | 2.0 [0.6, 8.3] | 54.7 [41.4, 66.3] | 54.7 | 49.4 | 79.1 | 3/3 (3) |
| arm E w3 | 89.7 | 59.2 | 3.8 | 3.2 [0.7, 11.1] | 3.1 [0.6, 10.0] | 47.0 [27.1, 62.2] | 47.0 | 48.7 | 80.3 | 3/3 (3) |
| arm E w10 | 90.0 | 68.5 | 4.8 | 6.1 [1.3, 15.9] | 3.6 [1.1, 12.1] | 24.0 [10.9, 45.4] | 24.0 | 39.8 | 81.6 | 3/3 (3) |
| CFES A (rev4) (mean of 3) | 87.0 | 74.0 | 16.6 | 20.7 [7.2, 38.9] | 17.2 [6.2, 32.8] | 28.1 [15.2, 40.9] | 28.1 | 38.1 | 74.4 | 3/3 (3) |
| control (det-only) (mean of 3) | 83.7 | 67.0 | 15.1 | 23.0 [9.6, 41.1] | 19.9 [7.7, 34.9] | 27.9 [11.8, 44.5] | 27.9 | 37.2 | 71.5 | 3/3 (3) |
| CFES-ResNet (mean of 3) | 71.4 | 69.1 | 36.0 | 12.0 [6.5, 32.6] | 8.5 [5.2, 25.0] | 13.2 [4.8, 29.2] | 13.5 | 19.3 | 65.4 | 3/3 (3) |
| CESel B0 | 88.0 | 67.6 | 5.2 | 4.7 [1.3, 11.9] | 4.7 [1.0, 11.3] | 35.0 [17.9, 52.6] | 35.0 | 42.4 | 80.7 | 3/3 (3) |
| CESel p1 full | 86.9 | 74.6 | 8.7 | 16.6 [5.3, 29.7] | 10.4 [4.1, 19.7] | 41.8 [22.7, 60.0] | 41.8 | 47.5 | 70.1 | 3/3 (3) |
| CESel p1 sens | 88.6 | 70.2 | 5.4 | 11.0 [3.5, 23.4] | 10.5 [2.9, 22.2] | 43.0 [24.7, 61.4] | 43.0 | 48.7 | 75.4 | 3/3 (3) |
| CESel p1 inv | 88.8 | 69.1 | 4.2 | 11.2 [3.7, 20.8] | 7.5 [2.5, 15.1] | 42.3 [22.1, 59.7] | 42.3 | 46.9 | 76.8 | 3/3 (3) |
| Ding repro (SFD+GDA) | 72.5 | 63.7 | 25.2 | 16.0 [4.4, 33.9] | 13.5 [3.7, 29.3] | 7.5 [2.4, 27.1] | 13.5 | 7.5 | 64.6 | 3/3 (3) |
| Ding Ori repro (SFD) (mean of 3) | 66.8 | 66.7 | 37.4 | 27.5 [16.5, 42.0] | 21.0 [13.7, 32.3] | 10.0 [5.7, 21.2] | 21.0 | 10.5 | 56.9 | 3/3 (3) |

**gender x race (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 89.9 | 75.6 | 9.9 | 67.4 [50.9, 92.7] | 36.0 [27.7, 46.7] | 48.4 [26.2, 67.4] | 62.9 | 49.6 | 61.9 | 6/6 (6) |
| arm C (mean of 3) | 90.7 | 73.4 | 7.5 | 44.0 [29.4, 75.0] | 26.5 [17.7, 41.2] | 49.3 [29.3, 71.6] | 54.0 | 51.5 | 67.7 | 6/6 (6) |
| arm E w1 | 89.7 | 55.3 | 3.5 | 14.4 [8.1, 30.1] | 8.2 [4.3, 17.3] | 63.1 [52.5, 76.3] | 63.1 | 56.5 | 61.4 | 6/6 (6) |
| arm E w3 | 89.7 | 59.2 | 3.8 | 16.0 [8.2, 33.9] | 8.2 [4.5, 19.6] | 54.0 [38.6, 80.0] | 54.0 | 54.7 | 63.0 | 6/6 (6) |
| arm E w10 | 90.0 | 68.5 | 4.8 | 20.8 [11.0, 47.0] | 13.5 [6.2, 27.5] | 57.9 [25.1, 84.1] | 57.9 | 55.3 | 63.7 | 6/6 (6) |
| CFES A (rev4) (mean of 3) | 87.0 | 74.0 | 16.6 | 60.0 [38.9, 97.4] | 30.4 [21.6, 51.1] | 42.4 [31.6, 57.4] | 43.6 | 47.6 | 68.8 | 6/6 (6) |
| control (det-only) (mean of 3) | 83.7 | 67.0 | 15.1 | 61.0 [37.0, 102.4] | 33.0 [21.5, 51.8] | 49.1 [31.1, 67.1] | 49.1 | 48.0 | 68.3 | 6/6 (6) |
| CFES-ResNet (mean of 3) | 71.4 | 69.1 | 36.0 | 39.0 [34.6, 82.4] | 22.3 [19.6, 44.2] | 34.6 [19.2, 51.5] | 34.6 | 27.1 | 55.9 | 6/6 (6) |
| CESel B0 | 88.0 | 67.6 | 5.2 | 19.8 [10.8, 40.1] | 9.5 [6.7, 25.2] | 44.1 [29.4, 65.0] | 44.1 | 42.8 | 74.0 | 6/6 (6) |
| CESel p1 full | 86.9 | 74.6 | 8.7 | 46.1 [27.1, 74.0] | 30.2 [16.2, 45.0] | 62.0 [34.5, 89.0] | 62.0 | 49.5 | 52.9 | 6/6 (6) |
| CESel p1 sens | 88.6 | 70.2 | 5.4 | 28.5 [11.9, 59.8] | 15.5 [7.4, 33.6] | 48.4 [35.5, 76.4] | 48.4 | 52.1 | 65.9 | 6/6 (6) |
| CESel p1 inv | 88.8 | 69.1 | 4.2 | 32.7 [19.1, 56.1] | 16.9 [12.0, 32.7] | 49.8 [34.7, 76.4] | 49.8 | 50.9 | 62.7 | 6/6 (6) |
| Ding repro (SFD+GDA) | 72.5 | 63.7 | 25.2 | 55.1 [36.9, 90.3] | 33.4 [20.6, 52.5] | 45.3 [19.5, 76.6] | 45.3 | 40.2 | 56.1 | 6/6 (6) |
| Ding Ori repro (SFD) (mean of 3) | 66.8 | 66.7 | 37.4 | 57.0 [46.4, 88.2] | 32.8 [26.8, 48.0] | 40.1 [24.7, 60.8] | 49.1 | 31.9 | 46.4 | 6/6 (6) |

**age (AI-Face 4-bin; FF2: InsightFace <30/30-44/45-59/60+)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 89.9 | 75.6 | 9.9 | 3.2 [0.9, 10.3] | 3.2 [0.9, 10.3] | 2.4 [0.8, 12.4] | 3.2 | 9.5 | 87.5 | 2/2 (2) |
| arm C (mean of 3) | 90.7 | 73.4 | 7.5 | 1.0 [0.6, 6.4] | 1.0 [0.6, 6.4] | 4.5 [1.7, 15.1] | 4.5 | 11.8 | 89.0 | 2/2 (2) |
| arm E w1 | 89.7 | 55.3 | 3.5 | 0.8 [0.1, 3.7] | 0.8 [0.1, 3.7] | 17.2 [4.7, 28.4] | 17.2 | 17.4 | 87.0 | 2/2 (2) |
| arm E w3 | 89.7 | 59.2 | 3.8 | 1.1 [0.1, 4.1] | 1.1 [0.1, 4.1] | 9.8 [0.7, 21.8] | 9.8 | 13.3 | 87.3 | 2/2 (2) |
| arm E w10 | 90.0 | 68.5 | 4.8 | 0.6 [0.1, 4.4] | 0.6 [0.1, 4.4] | 1.4 [0.2, 12.9] | 1.4 | 9.7 | 88.3 | 2/2 (2) |
| CFES A (rev4) (mean of 3) | 87.0 | 74.0 | 16.6 | 1.1 [0.7, 8.5] | 1.1 [0.7, 8.5] | 7.8 [2.0, 16.8] | 7.8 | 12.6 | 83.7 | 2/2 (2) |
| control (det-only) (mean of 3) | 83.7 | 67.0 | 15.1 | 4.0 [2.4, 8.7] | 4.0 [2.4, 8.7] | 7.9 [1.9, 18.1] | 7.9 | 12.4 | 80.3 | 2/2 (2) |
| CFES-ResNet (mean of 3) | 71.4 | 69.1 | 36.0 | 5.7 [3.8, 15.6] | 5.7 [3.8, 15.6] | 6.6 [3.2, 14.5] | 10.7 | 7.6 | 67.0 | 2/2 (2) |
| CESel B0 | 88.0 | 67.6 | 5.2 | 0.7 [0.1, 4.6] | 0.7 [0.1, 4.6] | 10.7 [1.2, 21.8] | 10.7 | 15.5 | 84.6 | 2/2 (2) |
| CESel p1 full | 86.9 | 74.6 | 8.7 | 0.6 [0.1, 7.5] | 0.6 [0.1, 7.5] | 6.6 [0.3, 17.5] | 6.6 | 13.5 | 82.9 | 2/2 (2) |
| CESel p1 sens | 88.6 | 70.2 | 5.4 | 1.7 [0.1, 5.4] | 1.7 [0.1, 5.4] | 13.3 [1.8, 24.7] | 13.3 | 17.8 | 84.3 | 2/2 (2) |
| CESel p1 inv | 88.8 | 69.1 | 4.2 | 1.1 [0.1, 4.8] | 1.1 [0.1, 4.8] | 9.1 [0.6, 20.0] | 9.1 | 14.2 | 84.6 | 2/2 (2) |
| Ding repro (SFD+GDA) | 72.5 | 63.7 | 25.2 | 11.4 [2.0, 21.6] | 11.4 [2.0, 21.6] | 2.8 [0.2, 13.5] | 11.4 | 2.8 | 65.7 | 2/2 (2) |
| Ding Ori repro (SFD) (mean of 3) | 66.8 | 66.7 | 37.4 | 5.4 [2.0, 13.3] | 5.4 [2.0, 13.3] | 3.2 [1.5, 10.6] | 6.3 | 3.3 | 62.4 | 2/2 (2) |

**skin tone (ITA, 4 bins)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 89.9 | 75.6 | 9.9 | 6.4 [5.3, 23.0] | 4.8 [4.1, 18.2] | 18.4 [9.8, 30.2] | 18.4 | 8.0 | 81.4 | 4/4 (4) |
| arm C (mean of 3) | 90.7 | 73.4 | 7.5 | 5.6 [5.5, 21.5] | 4.7 [4.3, 17.6] | 20.7 [12.9, 32.5] | 20.7 | 9.7 | 83.0 | 4/4 (4) |
| arm E w1 | 89.7 | 55.3 | 3.5 | 3.0 [1.8, 11.0] | 2.0 [1.2, 8.1] | 22.5 [11.8, 38.5] | 22.5 | 10.4 | 82.0 | 4/4 (4) |
| arm E w3 | 89.7 | 59.2 | 3.8 | 2.9 [1.7, 11.1] | 2.0 [1.3, 7.6] | 21.0 [10.7, 37.4] | 21.0 | 9.9 | 81.8 | 4/4 (4) |
| arm E w10 | 90.0 | 68.5 | 4.8 | 3.6 [2.3, 15.5] | 3.2 [1.7, 12.5] | 23.6 [12.5, 36.1] | 23.6 | 10.1 | 81.5 | 4/4 (4) |
| CFES A (rev4) (mean of 3) | 87.0 | 74.0 | 16.6 | 15.0 [9.3, 29.3] | 11.3 [6.5, 23.1] | 17.1 [11.9, 27.9] | 17.1 | 6.3 | 76.7 | 4/4 (4) |
| control (det-only) (mean of 3) | 83.7 | 67.0 | 15.1 | 15.6 [10.7, 29.3] | 10.5 [7.8, 20.0] | 16.9 [9.9, 31.5] | 16.9 | 6.2 | 73.0 | 4/4 (4) |
| CFES-ResNet (mean of 3) | 71.4 | 69.1 | 36.0 | 26.8 [23.7, 44.2] | 20.0 [16.3, 31.4] | 11.4 [9.0, 24.1] | 20.0 | 13.4 | 62.7 | 4/4 (4) |
| CESel B0 | 88.0 | 67.6 | 5.2 | 10.6 [4.7, 18.8] | 7.2 [3.3, 12.8] | 17.9 [6.9, 35.0] | 17.9 | 9.1 | 77.4 | 4/4 (4) |
| CESel p1 full | 86.9 | 74.6 | 8.7 | 9.0 [4.0, 25.3] | 6.3 [2.9, 20.0] | 20.4 [9.8, 33.9] | 20.4 | 7.6 | 78.4 | 4/4 (4) |
| CESel p1 sens | 88.6 | 70.2 | 5.4 | 7.5 [3.3, 16.3] | 5.6 [2.5, 10.3] | 21.0 [8.0, 38.2] | 21.0 | 7.9 | 78.0 | 4/4 (4) |
| CESel p1 inv | 88.8 | 69.1 | 4.2 | 2.4 [1.4, 9.4] | 1.9 [1.0, 6.5] | 17.8 [8.5, 34.7] | 17.8 | 10.3 | 79.3 | 4/4 (4) |
| Ding repro (SFD+GDA) | 72.5 | 63.7 | 25.2 | 28.6 [16.9, 50.7] | 17.0 [11.9, 34.8] | 12.5 [6.2, 27.9] | 17.0 | 11.6 | 66.6 | 4/4 (4) |
| Ding Ori repro (SFD) (mean of 3) | 66.8 | 66.7 | 37.4 | 25.6 [17.2, 40.9] | 16.3 [12.6, 29.9] | 6.7 [6.8, 19.9] | 16.3 | 8.8 | 59.6 | 4/4 (4) |

### FF2-public TEST

**gender (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 71.0 | 69.1 | 45.9 | 1.3 [0.7, 2.6] | 1.3 [0.7, 2.6] | 5.0 [3.9, 6.1] | 5.2 | 4.6 | 67.8 | 2/2 (2) |
| arm C (mean of 3) | 69.9 | 53.2 | 19.0 | 1.2 [0.7, 2.9] | 1.2 [0.7, 2.9] | 7.2 [5.8, 8.6] | 7.2 | 6.6 | 67.6 | 2/2 (2) |
| arm E w1 | 76.9 | 39.3 | 5.9 | 0.1 [0.0, 1.6] | 0.1 [0.0, 1.6] | 5.4 [3.9, 6.9] | 5.4 | 5.0 | 74.1 | 2/2 (2) |
| arm E w3 | 78.5 | 41.0 | 5.8 | 0.7 [0.0, 2.0] | 0.7 [0.0, 2.0] | 8.1 [6.5, 9.6] | 8.1 | 7.5 | 75.2 | 2/2 (2) |
| arm E w10 | 79.2 | 49.0 | 9.2 | 0.7 [0.0, 2.4] | 0.7 [0.0, 2.4] | 7.9 [6.4, 9.3] | 7.9 | 7.2 | 75.4 | 2/2 (2) |
| CFES A (rev4) (mean of 3) | 66.6 | 54.5 | 28.6 | 1.9 [0.5, 3.9] | 1.9 [0.5, 3.9] | 7.4 [6.1, 8.7] | 7.4 | 6.9 | 64.2 | 2/2 (2) |
| control (det-only) (mean of 3) | 62.7 | 50.5 | 30.5 | 1.9 [0.9, 3.6] | 1.9 [0.9, 3.6] | 6.9 [5.6, 8.3] | 6.9 | 6.5 | 60.2 | 2/2 (2) |
| CFES-ResNet (mean of 3) | 38.5 | 23.5 | 37.9 | 3.2 [2.0, 5.0] | 3.2 [2.0, 5.0] | 2.7 [2.1, 3.7] | 3.2 | 2.8 | 38.1 | 2/2 (2) |
| Ding repro (SFD+GDA) | 36.4 | 20.7 | 38.3 | 0.4 [0.0, 3.2] | 0.4 [0.0, 3.2] | 0.4 [0.0, 1.6] | 0.4 | 0.4 | 36.1 | 2/2 (2) |
| Ding Ori repro (SFD) (mean of 3) | 37.9 | 25.1 | 43.4 | 1.3 [0.5, 3.5] | 1.3 [0.5, 3.5] | 2.1 [1.0, 3.3] | 2.1 | 2.1 | 37.7 | 2/2 (2) |

**race (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 71.0 | 69.1 | 45.9 | 20.1 [17.6, 23.2] | 13.5 [12.0, 15.8] | 26.7 [25.7, 27.8] | 26.7 | 25.1 | 58.5 | 4/4 (4) |
| arm C (mean of 3) | 69.9 | 53.2 | 19.0 | 21.2 [17.8, 24.9] | 13.4 [11.5, 15.7] | 34.6 [33.3, 36.1] | 34.6 | 32.2 | 57.2 | 4/4 (4) |
| arm E w1 | 76.9 | 39.3 | 5.9 | 6.7 [5.2, 9.3] | 5.9 [4.4, 7.4] | 39.1 [37.3, 40.8] | 39.1 | 35.9 | 65.5 | 4/4 (4) |
| arm E w3 | 78.5 | 41.0 | 5.8 | 6.9 [5.2, 9.6] | 6.0 [4.3, 7.5] | 37.1 [35.3, 38.9] | 37.1 | 33.9 | 67.1 | 4/4 (4) |
| arm E w10 | 79.2 | 49.0 | 9.2 | 9.6 [6.9, 12.5] | 6.6 [4.8, 8.5] | 33.1 [31.7, 34.8] | 33.1 | 30.0 | 67.9 | 4/4 (4) |
| CFES A (rev4) (mean of 3) | 66.6 | 54.5 | 28.6 | 23.5 [20.0, 27.0] | 14.8 [13.2, 17.4] | 30.5 [29.1, 31.9] | 30.5 | 28.6 | 56.8 | 4/4 (4) |
| control (det-only) (mean of 3) | 62.7 | 50.5 | 30.5 | 30.1 [26.2, 34.0] | 19.0 [17.2, 21.8] | 30.9 [29.6, 32.4] | 30.9 | 29.4 | 54.1 | 4/4 (4) |
| CFES-ResNet (mean of 3) | 38.5 | 23.5 | 37.9 | 30.5 [27.1, 34.7] | 22.4 [19.7, 25.4] | 24.2 [22.7, 25.8] | 24.2 | 24.3 | 35.9 | 4/4 (4) |
| Ding repro (SFD+GDA) | 36.4 | 20.7 | 38.3 | 30.5 [26.6, 36.0] | 21.6 [18.3, 25.5] | 21.0 [19.4, 22.6] | 21.6 | 21.5 | 34.1 | 4/4 (4) |
| Ding Ori repro (SFD) (mean of 3) | 37.9 | 25.1 | 43.4 | 33.2 [29.8, 36.8] | 23.0 [20.5, 25.7] | 23.9 [22.7, 25.5] | 23.9 | 24.2 | 35.2 | 4/4 (4) |

**gender x race (AI-Face clean, frame level; FF2: FF2 labels)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 71.0 | 69.1 | 45.9 | 41.0 [37.0, 48.2] | 16.3 [14.8, 19.4] | 30.1 [29.1, 31.6] | 30.1 | 28.2 | 49.5 | 8/8 (8) |
| arm C (mean of 3) | 69.9 | 53.2 | 19.0 | 42.5 [37.6, 50.1] | 17.5 [15.5, 21.0] | 39.1 [37.6, 40.7] | 39.1 | 36.5 | 48.2 | 8/8 (8) |
| arm E w1 | 76.9 | 39.3 | 5.9 | 16.3 [12.7, 22.1] | 7.2 [5.6, 10.0] | 41.8 [39.6, 43.8] | 41.8 | 38.3 | 56.6 | 8/8 (8) |
| arm E w3 | 78.5 | 41.0 | 5.8 | 17.4 [13.4, 22.5] | 7.7 [5.9, 10.3] | 40.6 [38.7, 42.9] | 40.6 | 36.8 | 58.5 | 8/8 (8) |
| arm E w10 | 79.2 | 49.0 | 9.2 | 20.0 [14.9, 26.5] | 8.8 [6.4, 11.8] | 36.9 [35.7, 39.6] | 36.9 | 33.2 | 59.1 | 8/8 (8) |
| CFES A (rev4) (mean of 3) | 66.6 | 54.5 | 28.6 | 47.6 [41.5, 54.6] | 16.8 [15.7, 20.7] | 35.1 [33.9, 36.7] | 35.1 | 33.0 | 49.2 | 8/8 (8) |
| control (det-only) (mean of 3) | 62.7 | 50.5 | 30.5 | 61.6 [54.5, 69.0] | 21.2 [20.0, 25.3] | 35.2 [34.0, 37.3] | 35.2 | 33.4 | 46.9 | 8/8 (8) |
| CFES-ResNet (mean of 3) | 38.5 | 23.5 | 37.9 | 63.0 [56.8, 71.9] | 24.7 [23.0, 29.0] | 26.3 [25.0, 28.2] | 26.3 | 26.5 | 33.2 | 8/8 (8) |
| Ding repro (SFD+GDA) | 36.4 | 20.7 | 38.3 | 66.7 [58.2, 77.9] | 27.9 [23.6, 33.5] | 25.9 [23.6, 28.2] | 27.9 | 26.4 | 31.6 | 8/8 (8) |
| Ding Ori repro (SFD) (mean of 3) | 37.9 | 25.1 | 43.4 | 68.3 [62.7, 76.5] | 27.2 [24.2, 31.5] | 26.5 [25.0, 28.6] | 27.3 | 26.9 | 32.7 | 8/8 (8) |

**age (AI-Face 4-bin; FF2: InsightFace <30/30-44/45-59/60+)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 71.0 | 69.1 | 45.9 | 10.2 [8.3, 14.1] | 8.9 [6.8, 11.5] | 12.4 [11.1, 13.7] | 13.2 | 12.3 | 64.5 | 4/4 (4) |
| arm C (mean of 3) | 69.9 | 53.2 | 19.0 | 10.3 [8.2, 13.9] | 8.4 [6.3, 10.7] | 15.5 [13.7, 17.2] | 15.5 | 15.2 | 63.5 | 4/4 (4) |
| arm E w1 | 76.9 | 39.3 | 5.9 | 5.2 [4.0, 8.4] | 4.3 [3.0, 6.2] | 14.2 [12.6, 15.9] | 14.2 | 13.5 | 72.1 | 4/4 (4) |
| arm E w3 | 78.5 | 41.0 | 5.8 | 8.0 [5.6, 10.8] | 5.5 [4.5, 7.9] | 16.3 [14.6, 18.0] | 16.3 | 15.8 | 74.5 | 4/4 (4) |
| arm E w10 | 79.2 | 49.0 | 9.2 | 9.5 [6.3, 13.3] | 6.0 [4.4, 9.4] | 16.0 [14.2, 17.8] | 16.0 | 15.6 | 75.4 | 4/4 (4) |
| CFES A (rev4) (mean of 3) | 66.6 | 54.5 | 28.6 | 10.0 [7.6, 14.3] | 7.3 [5.4, 10.4] | 13.9 [12.3, 15.4] | 13.9 | 13.5 | 60.8 | 4/4 (4) |
| control (det-only) (mean of 3) | 62.7 | 50.5 | 30.5 | 13.5 [10.1, 18.2] | 10.2 [7.9, 13.6] | 15.1 [13.4, 16.7] | 15.1 | 14.8 | 57.1 | 4/4 (4) |
| CFES-ResNet (mean of 3) | 38.5 | 23.5 | 37.9 | 15.1 [11.8, 19.9] | 11.8 [9.1, 15.7] | 11.2 [9.9, 12.5] | 11.8 | 11.0 | 36.9 | 4/4 (4) |
| Ding repro (SFD+GDA) | 36.4 | 20.7 | 38.3 | 4.1 [1.9, 10.7] | 3.3 [1.5, 8.3] | 7.3 [6.1, 8.5] | 7.3 | 7.1 | 34.1 | 4/4 (4) |
| Ding Ori repro (SFD) (mean of 3) | 37.9 | 25.1 | 43.4 | 8.8 [6.9, 13.5] | 6.9 [5.2, 10.2] | 8.2 [7.0, 9.4] | 8.5 | 8.0 | 36.2 | 4/4 (4) |

**skin tone (ITA, 4 bins)**

| model | AUC | TPR | FPR | F_FPR | FPR gap | TPR gap | EO gap | F_DP | es-AUC | elig. FPR/TPR (of) |
|---|---|---|---|---|---|---|---|---|---|---|
| CFES-D (mean of 3) | 71.0 | 69.1 | 45.9 | 5.1 [3.4, 8.7] | 3.8 [2.4, 6.5] | 8.7 [7.5, 9.9] | 9.1 | 8.0 | 65.4 | 4/4 (4) |
| arm C (mean of 3) | 69.9 | 53.2 | 19.0 | 6.2 [4.1, 9.9] | 4.7 [2.9, 7.4] | 12.4 [10.9, 14.1] | 12.4 | 11.5 | 64.3 | 4/4 (4) |
| arm E w1 | 76.9 | 39.3 | 5.9 | 3.5 [1.7, 6.0] | 2.4 [1.2, 4.5] | 6.6 [5.0, 8.5] | 6.6 | 5.8 | 72.1 | 4/4 (4) |
| arm E w3 | 78.5 | 41.0 | 5.8 | 3.6 [1.7, 6.4] | 2.5 [1.2, 4.7] | 10.9 [9.3, 12.6] | 10.9 | 9.9 | 72.5 | 4/4 (4) |
| arm E w10 | 79.2 | 49.0 | 9.2 | 5.1 [2.4, 8.3] | 3.5 [1.8, 6.2] | 15.3 [13.6, 17.0] | 15.3 | 14.0 | 72.7 | 4/4 (4) |
| CFES A (rev4) (mean of 3) | 66.6 | 54.5 | 28.6 | 3.5 [2.8, 7.6] | 2.5 [1.9, 5.5] | 7.3 [5.8, 8.9] | 7.3 | 6.6 | 63.4 | 4/4 (4) |
| control (det-only) (mean of 3) | 62.7 | 50.5 | 30.5 | 7.4 [5.3, 11.7] | 5.6 [3.9, 8.4] | 10.0 [8.4, 11.5] | 10.0 | 9.5 | 60.1 | 4/4 (4) |
| CFES-ResNet (mean of 3) | 38.5 | 23.5 | 37.9 | 19.7 [17.6, 23.0] | 14.4 [12.8, 16.3] | 9.9 [8.9, 11.3] | 14.4 | 10.1 | 37.2 | 4/4 (4) |
| Ding repro (SFD+GDA) | 36.4 | 20.7 | 38.3 | 12.7 [8.7, 17.9] | 9.3 [5.7, 13.3] | 5.8 [4.6, 7.5] | 9.3 | 6.0 | 35.1 | 4/4 (4) |
| Ding Ori repro (SFD) (mean of 3) | 37.9 | 25.1 | 43.4 | 22.0 [18.0, 26.6] | 15.7 [12.9, 18.6] | 12.2 [10.8, 13.7] | 15.7 | 12.4 | 36.2 | 4/4 (4) |

## 4b. Label-source and label-unit sensitivity (A1, seed means)

The AI-Face cleaner labels are per frame and change inside a video: they split 9 % (FF++ test reals) to 36 % (DFDC reals) of videos across gender x race groups (Xu labels: 0 %). The spec's min-support rule counts videos, which presumes one group per video, so the video-majority version (`*_vidmaj`, ties -> Xu label) is reported next to the frame-level primary. **The coordinator should fix the unit for H4 (recommendation: video majority) before the final model is scored.** Cells: F_FPR (eligible groups FPR/TPR).

**gender x race** (F_FPR)

| set | model | clean, frame | clean, video maj. | Xu (protocol) |
|---|---|---|---|---|
| FF++ test | CFES-D | 14.4 (8/8) | 18.2 (8/8) | 16.8 (8/8) |
| FF++ test | arm C | 17.6 (8/8) | 22.3 (8/8) | 18.5 (8/8) |
| FF++ test | arm E w3 | 7.3 (8/8) | 8.8 (8/8) | 7.0 (8/8) |
| FF++ test | control (det-only) | 39.0 (8/8) | 43.4 (8/8) | 38.8 (8/8) |
| FF++ test | CESel B0 | 36.7 (8/8) | 45.0 (8/8) | 36.2 (8/8) |
| FF++ test | CESel p1 full | 38.9 (8/8) | 41.3 (8/8) | 34.2 (8/8) |
| FF++ test | Ding repro (SFD+GDA) | 6.4 (8/8) | 6.5 (8/8) | 7.7 (8/8) |
| FF++ test | Ding Ori repro (SFD) | 24.5 (8/8) | 24.7 (8/8) | 21.8 (8/8) |
| DFDC | CFES-D | 14.6 (8/8) | 19.2 (8/7) | 14.1 (8/7) |
| DFDC | arm C | 20.3 (8/8) | 23.1 (8/7) | 23.4 (8/7) |
| DFDC | arm E w3 | 5.9 (8/8) | 10.8 (8/7) | 11.2 (8/7) |
| DFDC | control (det-only) | 55.2 (8/8) | 58.4 (8/7) | 45.1 (8/7) |
| DFDC | CESel B0 | 11.0 (8/8) | 15.4 (8/7) | 18.0 (8/7) |
| DFDC | CESel p1 full | 11.3 (8/8) | 11.8 (8/7) | 11.0 (8/7) |
| DFDC | Ding repro (SFD+GDA) | 18.7 (8/8) | 21.2 (8/7) | 18.3 (8/7) |
| DFDC | Ding Ori repro (SFD) | 37.7 (8/8) | 47.0 (8/7) | 36.0 (8/7) |
| Celeb-DF | CFES-D | 27.0 (5/3) | 22.4 (4/2) | 22.8 (4/2) |
| Celeb-DF | arm C | 43.6 (5/3) | 39.2 (4/2) | 36.4 (4/2) |
| Celeb-DF | arm E w3 | 25.7 (5/3) | 25.8 (4/2) | 15.9 (4/2) |
| Celeb-DF | control (det-only) | 69.4 (5/3) | 52.5 (4/2) | 50.7 (4/2) |
| Celeb-DF | CESel B0 | 25.2 (5/3) | 23.9 (4/2) | 23.0 (4/2) |
| Celeb-DF | CESel p1 full | 76.1 (5/3) | 54.1 (4/2) | 59.1 (4/2) |
| Celeb-DF | Ding repro (SFD+GDA) | 62.2 (5/3) | 55.7 (4/2) | 68.7 (4/2) |
| Celeb-DF | Ding Ori repro (SFD) | 68.9 (5/3) | 52.0 (4/2) | 64.6 (4/2) |
| DFD | CFES-D | 67.4 (6/6) | 34.8 (4/6) | 34.7 (4/6) |
| DFD | arm C | 44.0 (6/6) | 22.5 (4/6) | 23.6 (4/6) |
| DFD | arm E w3 | 16.0 (6/6) | 6.0 (4/6) | 4.8 (4/6) |
| DFD | control (det-only) | 61.0 (6/6) | 22.9 (4/6) | 20.8 (4/6) |
| DFD | CESel B0 | 19.8 (6/6) | 8.0 (4/6) | 4.3 (4/6) |
| DFD | CESel p1 full | 46.1 (6/6) | 25.4 (4/6) | 28.6 (4/6) |
| DFD | Ding repro (SFD+GDA) | 55.1 (6/6) | 37.5 (4/6) | 40.9 (4/6) |
| DFD | Ding Ori repro (SFD) | 57.0 (6/6) | 33.6 (4/6) | 34.5 (4/6) |

**age** (F_FPR)

| set | model | AI-Face 4-bin | InsightFace <30/30-44/45-59/60+ | AI-Face v2 young/middle/senior | FairFace 0-29/30+ (video majority) |
|---|---|---|---|---|---|
| FF++ test | CFES-D | 4.7 (4/3) | 5.8 (4/3) | 5.1 (3/3) | 1.7 (2/2) |
| FF++ test | arm C | 6.5 (4/3) | 5.8 (4/3) | 5.0 (3/3) | 1.4 (2/2) |
| FF++ test | arm E w3 | 3.1 (4/3) | 2.1 (4/3) | 3.1 (3/3) | 0.2 (2/2) |
| FF++ test | control (det-only) | 15.4 (4/3) | 15.5 (4/3) | 10.0 (3/3) | 2.1 (2/2) |
| FF++ test | CESel B0 | 79.2 (4/3) | 9.5 (4/3) | 8.0 (3/3) | 0.4 (2/2) |
| FF++ test | CESel p1 full | 6.7 (4/3) | 5.3 (4/3) | 5.2 (3/3) | 2.1 (2/2) |
| FF++ test | Ding repro (SFD+GDA) | 3.3 (4/3) | 2.5 (4/3) | 2.8 (3/3) | 0.4 (2/2) |
| FF++ test | Ding Ori repro (SFD) | 9.2 (4/3) | 10.2 (4/3) | 7.4 (3/3) | 1.6 (2/2) |
| DFDC | CFES-D | 6.8 (4/4) | 13.2 (4/4) | 4.6 (3/3) | 1.7 (2/2) |
| DFDC | arm C | 8.6 (4/4) | 5.9 (4/4) | 5.1 (3/3) | 0.6 (2/2) |
| DFDC | arm E w3 | 4.7 (4/4) | 7.2 (4/4) | 4.7 (3/3) | 1.1 (2/2) |
| DFDC | control (det-only) | 18.3 (4/4) | 19.3 (4/4) | 12.4 (3/3) | 2.0 (2/2) |
| DFDC | CESel B0 | 9.5 (4/4) | 6.8 (4/4) | 2.8 (3/3) | 1.4 (2/2) |
| DFDC | CESel p1 full | 4.3 (4/4) | 5.5 (4/4) | 4.2 (3/3) | 0.1 (2/2) |
| DFDC | Ding repro (SFD+GDA) | 4.1 (4/4) | 6.3 (4/4) | 2.9 (3/3) | 1.4 (2/2) |
| DFDC | Ding Ori repro (SFD) | 13.1 (4/4) | 14.1 (4/4) | 11.6 (3/3) | 4.3 (2/2) |
| Celeb-DF | CFES-D | 10.4 (3/3) | 9.3 (3/3) | 11.2 (3/2) | 4.8 (2/2) |
| Celeb-DF | arm C | 5.7 (3/3) | 11.4 (3/3) | 13.2 (3/2) | 6.2 (2/2) |
| Celeb-DF | arm E w3 | 0.4 (3/3) | 7.8 (3/3) | 16.4 (3/2) | 5.5 (2/2) |
| Celeb-DF | control (det-only) | 8.3 (3/3) | 16.5 (3/3) | 34.0 (3/2) | 10.3 (2/2) |
| Celeb-DF | CESel B0 | 10.5 (3/3) | 15.3 (3/3) | 10.5 (3/2) | 6.8 (2/2) |
| Celeb-DF | CESel p1 full | 4.0 (3/3) | 7.5 (3/3) | 14.6 (3/2) | 12.0 (2/2) |
| Celeb-DF | Ding repro (SFD+GDA) | 10.5 (3/3) | 9.9 (3/3) | 24.1 (3/2) | 13.4 (2/2) |
| Celeb-DF | Ding Ori repro (SFD) | 7.5 (3/3) | 13.6 (3/3) | 32.2 (3/2) | 12.3 (2/2) |
| DFD | CFES-D | 3.2 (2/2) | 4.7 (2/2) | 6.8 (2/2) | 3.7 (2/2) |
| DFD | arm C | 1.0 (2/2) | 4.3 (2/2) | 4.4 (2/2) | 3.7 (2/2) |
| DFD | arm E w3 | 1.1 (2/2) | 1.7 (2/2) | 1.8 (2/2) | 3.6 (2/2) |
| DFD | control (det-only) | 4.0 (2/2) | 5.1 (2/2) | 3.4 (2/2) | 6.3 (2/2) |
| DFD | CESel B0 | 0.7 (2/2) | 0.6 (2/2) | 2.7 (2/2) | 4.1 (2/2) |
| DFD | CESel p1 full | 0.6 (2/2) | 5.7 (2/2) | 7.4 (2/2) | 5.3 (2/2) |
| DFD | Ding repro (SFD+GDA) | 11.4 (2/2) | 7.4 (2/2) | 11.7 (2/2) | 1.7 (2/2) |
| DFD | Ding Ori repro (SFD) | 5.4 (2/2) | 6.2 (2/2) | 13.2 (2/2) | 3.7 (2/2) |

**skin tone (FPR gap)** (fpr_gap)

| set | model | ITA gw (primary) | ITA raw | ITA gw+exposure | ITA within-set quartiles | AI-Face MST 3-group |
|---|---|---|---|---|---|---|
| FF++ test | CFES-D | 3.1 (4/4) | 4.1 (4/4) | 4.1 (4/4) | 3.6 (4/4) | 4.1 (3/3) |
| FF++ test | arm C | 2.4 (4/4) | 4.2 (4/4) | 4.0 (4/4) | 3.1 (4/4) | 2.9 (3/3) |
| FF++ test | arm E w3 | 1.7 (4/4) | 2.3 (4/4) | 2.1 (4/4) | 1.6 (4/4) | 1.7 (3/3) |
| FF++ test | control (det-only) | 2.4 (4/4) | 8.2 (4/4) | 5.8 (4/4) | 3.4 (4/4) | 16.3 (3/3) |
| FF++ test | CESel B0 | 4.8 (4/4) | 7.0 (4/4) | 6.3 (4/4) | 4.5 (4/4) | 4.5 (3/3) |
| FF++ test | CESel p1 full | 5.0 (4/4) | 5.6 (4/4) | 6.5 (4/4) | 7.2 (4/4) | 8.1 (3/3) |
| FF++ test | Ding repro (SFD+GDA) | 1.2 (4/4) | 1.7 (4/4) | 2.3 (4/4) | 1.7 (4/4) | 0.8 (3/3) |
| FF++ test | Ding Ori repro (SFD) | 2.9 (4/4) | 3.7 (4/4) | 5.4 (4/4) | 3.8 (4/4) | 5.9 (3/3) |
| DFDC | CFES-D | 5.0 (4/4) | 5.1 (4/4) | 5.7 (4/4) | 5.2 (4/4) | 2.7 (2/2) |
| DFDC | arm C | 5.1 (4/4) | 4.6 (4/4) | 3.5 (4/4) | 4.4 (4/4) | 2.5 (2/2) |
| DFDC | arm E w3 | 0.9 (4/4) | 1.8 (4/4) | 2.3 (4/4) | 2.4 (4/4) | 0.5 (2/2) |
| DFDC | control (det-only) | 10.0 (4/4) | 10.9 (4/4) | 10.8 (4/4) | 11.3 (4/4) | 7.2 (2/2) |
| DFDC | CESel B0 | 6.1 (4/4) | 5.4 (4/4) | 4.3 (4/4) | 6.5 (4/4) | 4.3 (2/2) |
| DFDC | CESel p1 full | 0.8 (4/4) | 3.2 (4/4) | 2.6 (4/4) | 1.8 (4/4) | 0.3 (2/2) |
| DFDC | Ding repro (SFD+GDA) | 8.1 (4/4) | 4.6 (4/4) | 1.9 (4/4) | 1.4 (4/4) | 1.0 (2/2) |
| DFDC | Ding Ori repro (SFD) | 11.4 (4/4) | 8.4 (4/4) | 8.8 (4/4) | 8.0 (4/4) | 6.2 (2/2) |
| Celeb-DF | CFES-D | 5.8 (4/4) | 7.2 (4/4) | 8.6 (4/4) | 11.6 (4/4) | 2.6 (2/2) |
| Celeb-DF | arm C | 6.3 (4/4) | 5.1 (4/4) | 9.5 (4/4) | 12.2 (4/4) | 1.6 (2/2) |
| Celeb-DF | arm E w3 | 4.5 (4/4) | 5.9 (4/4) | 10.0 (4/4) | 9.8 (4/4) | 0.4 (2/2) |
| Celeb-DF | control (det-only) | 14.4 (4/4) | 11.8 (4/4) | 9.9 (4/4) | 17.8 (4/4) | 7.4 (2/2) |
| Celeb-DF | CESel B0 | 14.2 (4/4) | 9.8 (4/4) | 6.6 (4/4) | 14.0 (4/4) | 4.0 (2/2) |
| Celeb-DF | CESel p1 full | 8.8 (4/4) | 16.8 (4/4) | 15.8 (4/4) | 28.4 (4/4) | 3.3 (2/2) |
| Celeb-DF | Ding repro (SFD+GDA) | 10.8 (4/4) | 17.6 (4/4) | 14.3 (4/4) | 17.6 (4/4) | 11.3 (2/2) |
| Celeb-DF | Ding Ori repro (SFD) | 14.0 (4/4) | 17.3 (4/4) | 17.2 (4/4) | 16.4 (4/4) | 9.7 (2/2) |
| DFD | CFES-D | 4.8 (4/4) | 8.7 (4/4) | 16.4 (4/4) | 6.1 (4/4) | 4.0 (2/2) |
| DFD | arm C | 4.7 (4/4) | 9.0 (4/4) | 15.0 (4/4) | 3.6 (4/4) | 1.3 (2/2) |
| DFD | arm E w3 | 2.0 (4/4) | 2.3 (4/4) | 11.7 (4/4) | 1.9 (4/4) | 2.2 (2/2) |
| DFD | control (det-only) | 10.5 (4/4) | 13.6 (4/4) | 20.5 (4/4) | 8.5 (4/4) | 5.3 (2/2) |
| DFD | CESel B0 | 7.2 (4/4) | 7.3 (4/4) | 2.8 (4/4) | 6.9 (4/4) | 0.6 (2/2) |
| DFD | CESel p1 full | 6.3 (4/4) | 11.1 (4/4) | 12.3 (4/4) | 7.2 (4/4) | 3.0 (2/2) |
| DFD | Ding repro (SFD+GDA) | 17.0 (4/4) | 16.8 (4/4) | 17.7 (4/4) | 26.4 (4/4) | 16.5 (2/2) |
| DFD | Ding Ori repro (SFD) | 16.3 (4/4) | 16.8 (4/4) | 14.9 (4/4) | 15.4 (4/4) | 9.6 (2/2) |

## 5. Results at threshold 0.5 (compact: FPR gap per axis, with pooled TPR / FPR)

Full metric set at 0.5 in `results/seedmean_axes.csv` (threshold = t05).

**FF++ test**

| model | AUC | TPR | FPR | gender F_FPR / FPR gap | race F_FPR / FPR gap | gender_x_race F_FPR / FPR gap | age F_FPR / FPR gap | skin_ita4 F_FPR / FPR gap |
|---|---|---|---|---|---|---|---|---|
| CFES-D | 95.6 | 87.9 | 9.9 | 3.3 / 3.3 | 9.3 / 6.7 | 35.6 / 19.6 | 13.3 / 10.8 | 8.2 / 5.6 |
| arm C | 95.8 | 79.4 | 3.3 | 1.5 / 1.5 | 6.0 / 4.5 | 17.5 / 9.3 | 6.8 / 5.0 | 3.6 / 2.8 |
| arm E w1 | 97.0 | 85.3 | 2.8 | 1.2 / 1.2 | 5.6 / 3.6 | 14.2 / 7.0 | 6.2 / 4.7 | 2.1 / 1.8 |
| arm E w3 | 97.1 | 82.7 | 1.9 | 0.9 / 0.9 | 4.4 / 2.9 | 10.3 / 5.0 | 4.1 / 3.1 | 1.7 / 1.4 |
| arm E w10 | 96.7 | 65.6 | 0.3 | 0.1 / 0.1 | 0.5 / 0.5 | 1.7 / 1.0 | 0.8 / 0.6 | 0.7 / 0.5 |
| CFES A (rev4) | 93.0 | 87.6 | 18.0 | 6.5 / 6.5 | 19.1 / 11.2 | 58.1 / 29.1 | 12.6 / 10.2 | 7.2 / 6.1 |
| control (det-only) | 91.5 | 86.8 | 21.1 | 3.8 / 3.8 | 16.9 / 11.7 | 56.3 / 29.1 | 23.2 / 21.1 | 6.0 / 4.6 |
| CFES-ResNet | 92.5 | 88.3 | 19.6 | 4.5 / 4.5 | 23.5 / 18.4 | 52.0 / 27.4 | 20.2 / 16.4 | 7.0 / 5.4 |
| CESel B0 | 93.3 | 88.2 | 18.1 | 6.4 / 6.4 | 18.2 / 12.4 | 75.0 / 35.2 | 73.3 / 71.5 | 3.6 / 3.1 |
| CESel p1 full | 93.0 | 92.7 | 25.4 | 6.7 / 6.7 | 11.5 / 9.0 | 57.7 / 26.9 | 32.4 / 28.0 | 8.6 / 5.5 |
| CESel p1 sens | 93.4 | 89.3 | 18.8 | 4.3 / 4.3 | 20.5 / 17.6 | 62.8 / 31.1 | 51.1 / 50.0 | 4.7 / 4.2 |
| CESel p1 inv | 93.7 | 93.9 | 27.3 | 3.2 / 3.2 | 23.6 / 15.9 | 59.5 / 30.4 | 69.5 / 63.3 | 8.1 / 4.9 |
| Ding repro (SFD+GDA) | 97.4 | 94.8 | 12.5 | 1.3 / 1.3 | 11.1 / 8.7 | 30.2 / 12.0 | 21.1 / 15.7 | 11.9 / 8.3 |
| Ding Ori repro (SFD) | 94.1 | 93.8 | 26.3 | 4.6 / 4.6 | 21.4 / 15.7 | 47.8 / 27.0 | 20.1 / 17.4 | 14.7 / 9.7 |

**DFDC**

| model | AUC | TPR | FPR | gender F_FPR / FPR gap | race F_FPR / FPR gap | gender_x_race F_FPR / FPR gap | age F_FPR / FPR gap | skin_ita4 F_FPR / FPR gap |
|---|---|---|---|---|---|---|---|---|
| CFES-D | 76.9 | 61.2 | 21.8 | 6.9 / 6.9 | 16.7 / 12.0 | 50.2 / 24.6 | 13.2 / 9.5 | 20.6 / 11.0 |
| arm C | 75.1 | 37.3 | 6.1 | 3.4 / 3.4 | 9.1 / 6.7 | 22.8 / 10.1 | 8.9 / 5.0 | 8.0 / 5.4 |
| arm E w1 | 79.2 | 34.4 | 4.3 | 2.1 / 2.1 | 2.5 / 1.4 | 9.1 / 4.8 | 6.7 / 4.8 | 2.9 / 1.8 |
| arm E w3 | 77.9 | 31.1 | 3.7 | 1.3 / 1.3 | 1.7 / 1.0 | 6.2 / 3.4 | 5.0 / 3.7 | 1.6 / 0.9 |
| arm E w10 | 76.4 | 20.5 | 1.1 | 0.3 / 0.3 | 2.1 / 1.5 | 6.2 / 2.9 | 2.6 / 1.4 | 3.3 / 1.5 |
| CFES A (rev4) | 73.2 | 63.9 | 29.9 | 12.8 / 12.8 | 18.0 / 14.5 | 63.7 / 31.5 | 20.9 / 13.5 | 15.0 / 8.9 |
| control (det-only) | 71.7 | 68.5 | 38.1 | 12.1 / 12.1 | 18.8 / 13.8 | 79.5 / 39.8 | 19.9 / 12.9 | 23.2 / 13.2 |
| CFES-ResNet | 58.7 | 46.9 | 35.0 | 8.2 / 8.2 | 18.2 / 11.2 | 54.4 / 22.9 | 20.0 / 12.7 | 23.9 / 15.4 |
| CESel B0 | 76.5 | 58.4 | 21.9 | 3.9 / 3.9 | 3.7 / 3.1 | 33.4 / 14.4 | 14.8 / 11.6 | 26.6 / 10.8 |
| CESel p1 full | 73.2 | 63.3 | 28.4 | 1.0 / 1.0 | 22.4 / 17.7 | 40.3 / 22.0 | 20.6 / 15.7 | 15.0 / 9.6 |
| CESel p1 sens | 78.1 | 55.0 | 16.5 | 0.6 / 0.6 | 11.8 / 9.2 | 23.3 / 12.5 | 15.2 / 11.8 | 16.3 / 8.2 |
| CESel p1 inv | 77.9 | 62.9 | 22.7 | 1.1 / 1.1 | 12.6 / 9.2 | 27.6 / 14.3 | 18.4 / 12.9 | 20.4 / 10.3 |
| Ding repro (SFD+GDA) | 56.0 | 55.6 | 49.1 | 3.2 / 3.2 | 27.6 / 17.8 | 54.1 / 22.7 | 18.9 / 14.6 | 48.1 / 27.7 |
| Ding Ori repro (SFD) | 54.7 | 62.5 | 56.3 | 3.9 / 3.9 | 21.7 / 14.8 | 54.3 / 25.8 | 19.2 / 12.8 | 43.1 / 22.2 |

**Celeb-DF**

| model | AUC | TPR | FPR | gender F_FPR / FPR gap | race F_FPR / FPR gap | gender_x_race F_FPR / FPR gap | age F_FPR / FPR gap | skin_ita4 F_FPR / FPR gap |
|---|---|---|---|---|---|---|---|---|
| CFES-D | 87.3 | 90.5 | 42.0 | 8.0 / 8.0 | 24.7 / 23.7 | 45.0 / 27.3 | 11.4 / 10.8 | 10.8 / 7.7 |
| arm C | 89.1 | 75.3 | 14.5 | 4.2 / 4.2 | 24.0 / 23.7 | 42.5 / 34.8 | 5.5 / 5.0 | 6.8 / 5.9 |
| arm E w1 | 87.6 | 75.8 | 17.1 | 1.8 / 1.8 | 28.7 / 28.3 | 42.3 / 40.7 | 1.5 / 1.0 | 9.5 / 7.4 |
| arm E w3 | 86.9 | 75.7 | 16.1 | 2.7 / 2.7 | 19.8 / 19.7 | 28.0 / 27.7 | 1.3 / 1.1 | 6.2 / 4.3 |
| arm E w10 | 84.9 | 43.2 | 4.3 | 2.6 / 2.6 | 8.4 / 5.0 | 13.5 / 8.3 | 4.7 / 3.8 | 6.2 / 4.5 |
| CFES A (rev4) | 85.7 | 91.9 | 49.6 | 21.9 / 21.9 | 25.3 / 23.0 | 55.3 / 28.8 | 9.5 / 8.6 | 21.8 / 13.4 |
| control (det-only) | 81.4 | 89.0 | 49.9 | 24.1 / 24.1 | 30.2 / 28.7 | 73.8 / 38.5 | 9.8 / 8.3 | 24.7 / 15.1 |
| CFES-ResNet | 71.4 | 82.9 | 60.1 | 9.4 / 9.4 | 20.3 / 17.3 | 40.6 / 29.3 | 15.2 / 14.7 | 9.6 / 6.2 |
| CESel B0 | 85.0 | 92.4 | 50.2 | 22.4 / 22.4 | 41.7 / 32.1 | 65.7 / 35.0 | 3.9 / 3.6 | 25.3 / 14.1 |
| CESel p1 full | 82.2 | 97.9 | 70.0 | 18.5 / 18.5 | 35.0 / 31.5 | 54.3 / 42.7 | 5.2 / 4.4 | 11.6 / 8.9 |
| CESel p1 sens | 87.1 | 94.8 | 51.8 | 18.9 / 18.9 | 28.8 / 28.3 | 78.1 / 50.6 | 4.8 / 3.1 | 19.9 / 15.2 |
| CESel p1 inv | 85.7 | 96.4 | 63.7 | 18.8 / 18.8 | 33.8 / 33.2 | 68.3 / 49.3 | 4.6 / 4.6 | 17.9 / 13.6 |
| Ding repro (SFD+GDA) | 71.2 | 90.5 | 69.6 | 7.0 / 7.0 | 41.6 / 39.6 | 66.3 / 39.9 | 8.5 / 7.6 | 16.4 / 12.4 |
| Ding Ori repro (SFD) | 67.1 | 91.6 | 74.2 | 6.5 / 6.5 | 23.4 / 22.7 | 45.5 / 26.1 | 5.5 / 4.8 | 13.7 / 10.1 |

**DFD**

| model | AUC | TPR | FPR | gender F_FPR / FPR gap | race F_FPR / FPR gap | gender_x_race F_FPR / FPR gap | age F_FPR / FPR gap | skin_ita4 F_FPR / FPR gap |
|---|---|---|---|---|---|---|---|---|
| CFES-D | 89.9 | 85.9 | 24.6 | 14.9 / 14.9 | 40.9 / 28.4 | 122.2 / 62.2 | 2.9 / 2.9 | 8.3 / 7.1 |
| arm C | 90.7 | 73.0 | 7.3 | 6.9 / 6.9 | 12.7 / 8.9 | 41.6 / 24.1 | 1.0 / 1.0 | 5.3 / 4.6 |
| arm E w1 | 89.7 | 66.0 | 4.8 | 3.8 / 3.8 | 2.6 / 2.1 | 18.1 / 12.0 | 0.6 / 0.6 | 2.7 / 2.1 |
| arm E w3 | 89.7 | 62.7 | 4.5 | 3.2 / 3.2 | 3.0 / 2.9 | 16.6 / 8.9 | 0.7 / 0.7 | 3.5 / 2.4 |
| arm E w10 | 90.0 | 50.5 | 1.6 | 1.4 / 1.4 | 2.4 / 1.5 | 11.3 / 6.1 | 0.3 / 0.3 | 1.9 / 1.5 |
| CFES A (rev4) | 87.0 | 83.1 | 27.1 | 8.4 / 8.4 | 25.7 / 21.7 | 81.7 / 42.1 | 2.9 / 2.9 | 20.6 / 17.1 |
| control (det-only) | 83.7 | 78.3 | 28.2 | 6.9 / 6.9 | 23.1 / 21.5 | 81.6 / 43.3 | 7.2 / 7.2 | 22.0 / 15.9 |
| CFES-ResNet | 71.4 | 81.1 | 52.0 | 7.7 / 7.7 | 11.6 / 8.8 | 45.2 / 27.3 | 7.9 / 7.9 | 30.4 / 22.4 |
| CESel B0 | 88.0 | 81.9 | 22.5 | 11.4 / 11.4 | 22.5 / 21.3 | 64.5 / 40.1 | 6.6 / 6.6 | 22.2 / 19.9 |
| CESel p1 full | 86.9 | 86.6 | 38.8 | 6.1 / 6.1 | 26.1 / 23.8 | 92.2 / 47.8 | 1.6 / 1.6 | 16.1 / 15.2 |
| CESel p1 sens | 88.6 | 81.1 | 20.4 | 3.8 / 3.8 | 19.3 / 19.1 | 65.2 / 33.7 | 4.5 / 4.5 | 19.4 / 16.5 |
| CESel p1 inv | 88.8 | 86.7 | 33.9 | 7.1 / 7.1 | 26.2 / 22.0 | 118.4 / 55.2 | 7.0 / 7.0 | 12.6 / 11.0 |
| Ding repro (SFD+GDA) | 72.5 | 83.6 | 59.0 | 1.1 / 1.1 | 20.9 / 14.6 | 41.2 / 27.5 | 9.6 / 9.6 | 37.0 / 24.5 |
| Ding Ori repro (SFD) | 66.8 | 83.0 | 63.1 | 2.2 / 2.2 | 26.0 / 17.2 | 53.9 / 28.8 | 4.1 / 4.1 | 33.9 / 20.5 |

**FF2-public TEST**

| model | AUC | TPR | FPR | gender F_FPR / FPR gap | race F_FPR / FPR gap | gender_x_race F_FPR / FPR gap | age F_FPR / FPR gap | skin_ita4 F_FPR / FPR gap |
|---|---|---|---|---|---|---|---|---|
| CFES-D | 71.0 | 95.0 | 91.0 | 1.5 / 1.5 | 10.6 / 7.1 | 23.1 / 10.0 | 6.2 / 5.0 | 4.3 / 3.0 |
| arm C | 69.9 | 53.0 | 19.0 | 1.2 / 1.2 | 20.5 / 13.1 | 41.7 / 17.8 | 10.1 / 7.9 | 5.2 / 4.4 |
| arm E w1 | 76.9 | 51.9 | 10.4 | 0.1 / 0.1 | 9.9 / 8.1 | 21.6 / 10.9 | 8.5 / 7.0 | 4.1 / 2.9 |
| arm E w3 | 78.5 | 45.1 | 7.4 | 0.8 / 0.8 | 8.0 / 6.9 | 19.6 / 8.5 | 9.3 / 6.5 | 3.9 / 2.1 |
| arm E w10 | 79.2 | 30.4 | 2.3 | 0.3 / 0.3 | 3.5 / 2.7 | 7.7 / 2.9 | 3.8 / 2.6 | 1.7 / 0.9 |
| CFES A (rev4) | 66.6 | 65.4 | 43.3 | 3.6 / 3.6 | 30.5 / 19.6 | 60.6 / 22.3 | 11.9 / 9.1 | 4.2 / 3.1 |
| control (det-only) | 62.7 | 62.4 | 46.9 | 2.8 / 2.8 | 35.2 / 23.4 | 72.7 / 26.8 | 14.4 / 11.6 | 8.3 / 5.4 |
| CFES-ResNet | 38.5 | 36.3 | 53.8 | 4.4 / 4.4 | 36.5 / 26.2 | 74.8 / 28.7 | 18.2 / 15.1 | 28.4 / 19.5 |
| Ding repro (SFD+GDA) | 36.4 | 43.3 | 65.5 | 3.9 / 3.9 | 43.9 / 27.6 | 89.3 / 29.7 | 13.2 / 12.3 | 41.3 / 27.9 |
| Ding Ori repro (SFD) | 37.9 | 48.4 | 65.8 | 2.8 / 2.8 | 38.7 / 27.3 | 79.7 / 29.5 | 16.8 / 13.8 | 42.3 / 28.3 |
| FF2-trained phase 1 | 97.1 | 96.2 | 20.2 | 0.7 / 0.7 | 9.9 / 7.2 | 23.0 / 10.4 | 2.9 / 2.4 | 5.5 / 4.2 |
| FF2-trained control | 97.5 | 95.7 | 15.4 | 0.4 / 0.4 | 8.7 / 6.2 | 19.9 / 9.3 | 2.8 / 1.8 | 6.5 / 4.5 |
| FF2-trained CFES | 97.5 | 89.5 | 3.5 | 1.2 / 1.2 | 4.4 / 4.4 | 11.5 / 5.8 | 0.8 / 0.7 | 4.3 / 3.0 |

## 6. Per-group rates on the new axes (A1; CFES-D seed 7 / Ding repro / CESel B0)

FPR [video-cluster CI] and TPR per group; counts = real frames / real clusters / fake clusters (clusters = videos; FF2: persons). `*` = not FPR-eligible (< 3 real clusters). Seed 7 shown so that the CI is a single-model interval.

**FF++ test, age (AI-Face 4-bin; FF2: InsightFace <30/30-44/45-59/60+)**

| group | real fr / real cl. / fake cl. | CFES-D s7 FPR | TPR | Ding repro FPR | TPR | B0 FPR | TPR |
|---|---|---|---|---|---|---|---|
| 25-44 | 1873 / 108 / 320 | 2.3 [0.9, 4.5] | 81.3 | 2.0 [1.0, 3.2] | 83.2 | 5.4 [2.9, 8.3] | 80.1 |
| 45-64 | 1414 / 88 / 221 | 5.7 [2.3, 9.8] | 78.7 | 2.3 [1.1, 3.9] | 81.8 | 8.7 [3.5, 15.8] | 73.4 |
| 65+ | 34 / 5 / 2 | 0.0 [0.0, 0.0] | 98.3 | 0.0 [0.0, 0.0] | 98.3 | 82.4 [0.0, 91.8] | 100.0 |
| <25 | 1217 / 66 / 155 | 3.3 [1.0, 6.4] | 79.0 | 1.1 [0.3, 2.1] | 86.0 | 6.4 [2.4, 11.5] | 74.6 |

**FF++ test, skin tone (ITA, 4 bins)**

| group | real fr / real cl. / fake cl. | CFES-D s7 FPR | TPR | Ding repro FPR | TPR | B0 FPR | TPR |
|---|---|---|---|---|---|---|---|
| ITA-Q1 (darkest) | 1138 / 72 / 192 | 3.0 [1.1, 5.1] | 80.2 | 2.0 [0.8, 3.6] | 82.3 | 5.9 [2.5, 9.8] | 77.9 |
| ITA-Q2 | 1112 / 65 / 169 | 1.7 [0.1, 4.3] | 77.5 | 2.2 [0.7, 4.2] | 84.1 | 7.7 [2.2, 15.8] | 74.2 |
| ITA-Q3 | 1032 / 55 / 156 | 5.4 [1.0, 11.1] | 78.7 | 1.9 [0.6, 3.4] | 81.5 | 5.5 [2.1, 9.6] | 74.3 |
| ITA-Q4 (lightest) | 1165 / 68 / 171 | 4.6 [1.5, 8.7] | 82.3 | 1.0 [0.3, 2.1] | 85.7 | 10.3 [4.0, 18.5] | 79.5 |

**DFDC, age (AI-Face 4-bin; FF2: InsightFace <30/30-44/45-59/60+)**

| group | real fr / real cl. / fake cl. | CFES-D s7 FPR | TPR | Ding repro FPR | TPR | B0 FPR | TPR |
|---|---|---|---|---|---|---|---|
| 25-44 | 9474 / 303 / 177 | 5.1 [3.7, 6.5] | 32.2 | 7.9 [5.9, 10.1] | 12.5 | 8.7 [6.9, 10.7] | 32.9 |
| 45-64 | 2656 / 85 / 78 | 1.4 [0.7, 2.2] | 44.6 | 8.7 [4.9, 13.3] | 18.4 | 4.5 [2.1, 7.5] | 42.0 |
| 65+ | 896 / 32 / 16 | 4.2 [1.1, 8.1] | 30.0 | 6.1 [2.6, 10.5] | 14.1 | 5.4 [1.9, 9.9] | 23.9 |
| <25 | 1673 / 53 / 13 | 4.2 [2.1, 6.9] | 43.9 | 6.3 [3.3, 9.8] | 17.8 | 3.6 [1.7, 6.1] | 43.3 |

**DFDC, skin tone (ITA, 4 bins)**

| group | real fr / real cl. / fake cl. | CFES-D s7 FPR | TPR | Ding repro FPR | TPR | B0 FPR | TPR |
|---|---|---|---|---|---|---|---|
| ITA-Q1 (darkest) | 10645 / 341 / 194 | 4.8 [3.7, 6.1] | 34.6 | 8.0 [6.2, 10.0] | 14.6 | 8.5 [6.8, 10.3] | 35.5 |
| ITA-Q2 | 1043 / 35 / 16 | 3.1 [0.6, 6.7] | 48.0 | 5.1 [2.0, 8.7] | 24.6 | 2.4 [1.0, 4.1] | 54.0 |
| ITA-Q3 | 744 / 24 / 15 | 1.1 [0.1, 2.5] | 37.3 | 13.2 [6.3, 23.0] | 17.7 | 5.1 [1.1, 11.9] | 33.8 |
| ITA-Q4 (lightest) | 1553 / 50 / 40 | 2.4 [1.1, 4.2] | 43.1 | 6.6 [2.2, 12.5] | 12.7 | 2.9 [1.1, 5.1] | 35.1 |

**Celeb-DF, age (AI-Face 4-bin; FF2: InsightFace <30/30-44/45-59/60+)**

| group | real fr / real cl. / fake cl. | CFES-D s7 FPR | TPR | Ding repro FPR | TPR | B0 FPR | TPR |
|---|---|---|---|---|---|---|---|
| 25-44 | 3367 / 113 / 558 | 11.3 [8.0, 15.0] | 72.3 | 25.0 [20.5, 29.6] | 60.9 | 17.0 [13.5, 20.4] | 73.8 |
| 45-64 | 1512 / 51 / 133 | 13.8 [8.4, 19.9] | 67.9 | 23.9 [17.6, 30.8] | 49.0 | 21.4 [14.4, 28.9] | 72.2 |
| <25 | 834 / 28 / 73 | 17.9 [9.0, 27.6] | 77.1 | 33.5 [22.9, 43.8] | 53.8 | 25.5 [14.6, 38.1] | 82.6 |

**Celeb-DF, skin tone (ITA, 4 bins)**

| group | real fr / real cl. / fake cl. | CFES-D s7 FPR | TPR | Ding repro FPR | TPR | B0 FPR | TPR |
|---|---|---|---|---|---|---|---|
| ITA-Q1 (darkest) | 2652 / 89 / 283 | 14.1 [9.6, 18.8] | 67.3 | 21.9 [17.3, 26.7] | 50.3 | 17.3 [12.8, 21.7] | 65.7 |
| ITA-Q2 | 1212 / 41 / 178 | 12.3 [6.9, 18.9] | 76.3 | 32.8 [24.1, 40.9] | 65.5 | 21.5 [15.0, 28.8] | 81.2 |
| ITA-Q3 | 718 / 24 / 126 | 11.7 [5.8, 17.8] | 73.9 | 31.9 [24.0, 41.1] | 62.5 | 12.1 [6.6, 18.8] | 73.5 |
| ITA-Q4 (lightest) | 1101 / 37 / 174 | 11.8 [5.3, 19.9] | 74.0 | 24.9 [17.9, 32.6] | 61.0 | 26.3 [17.0, 36.6] | 81.9 |

**DFD, age (AI-Face 4-bin; FF2: InsightFace <30/30-44/45-59/60+)**

| group | real fr / real cl. / fake cl. | CFES-D s7 FPR | TPR | Ding repro FPR | TPR | B0 FPR | TPR |
|---|---|---|---|---|---|---|---|
| 25-44 | 1966 / 66 / 165 | 5.6 [3.0, 8.7] | 75.5 | 21.0 [16.4, 26.0] | 64.3 | 5.4 [2.9, 8.2] | 70.1 |
| <25 | 1126 / 38 / 50 | 9.9 [4.9, 15.4] | 72.4 | 32.4 [24.5, 41.1] | 61.5 | 4.7 [2.3, 7.6] | 59.4 |

**DFD, skin tone (ITA, 4 bins)**

| group | real fr / real cl. / fake cl. | CFES-D s7 FPR | TPR | Ding repro FPR | TPR | B0 FPR | TPR |
|---|---|---|---|---|---|---|---|
| ITA-Q1 (darkest) | 1486 / 50 / 88 | 5.5 [2.8, 8.1] | 76.6 | 31.8 [25.3, 38.7] | 69.9 | 4.9 [2.6, 8.1] | 65.6 |
| ITA-Q2 | 359 / 12 / 29 | 10.9 [1.5, 22.0] | 76.4 | 29.8 [14.3, 46.8] | 65.0 | 1.4 [0.0, 3.0] | 61.4 |
| ITA-Q3 | 386 / 13 / 22 | 9.6 [0.2, 21.8] | 89.6 | 18.1 [8.2, 29.4] | 59.5 | 2.1 [0.0, 5.9] | 79.3 |
| ITA-Q4 (lightest) | 861 / 29 / 76 | 7.3 [2.3, 14.0] | 67.8 | 14.9 [9.4, 21.6] | 57.4 | 8.6 [3.9, 13.7] | 68.8 |

**FF2-public TEST, age (AI-Face 4-bin; FF2: InsightFace <30/30-44/45-59/60+)**

| group | real fr / real cl. / fake cl. | CFES-D s7 FPR | TPR | Ding repro FPR | TPR | B0 FPR | TPR |
|---|---|---|---|---|---|---|---|
| 30-44 | 2527 / 2238 / 10610 | 43.9 [41.9, 45.8] | 72.6 | 38.0 [36.2, 40.0] | 21.7 | - | - |
| 45-59 | 1550 / 1381 / 6765 | 41.2 [38.8, 43.8] | 71.7 | 39.7 [37.3, 42.1] | 22.4 | - | - |
| 60+ | 869 / 762 / 5011 | 34.2 [31.0, 37.5] | 57.6 | 36.4 [33.1, 39.6] | 15.1 | - | - |
| <30 | 495 / 463 / 5225 | 41.8 [37.2, 46.4] | 68.2 | 38.8 [34.2, 43.3] | 20.4 | - | - |

**FF2-public TEST, skin tone (ITA, 4 bins)**

| group | real fr / real cl. / fake cl. | CFES-D s7 FPR | TPR | Ding repro FPR | TPR | B0 FPR | TPR |
|---|---|---|---|---|---|---|---|
| ITA-Q1 (darkest) | 1300 / 1180 / 4804 | 41.3 [38.2, 44.2] | 74.8 | 41.3 [38.5, 44.2] | 23.9 | - | - |
| ITA-Q2 | 1247 / 1158 / 6533 | 44.0 [41.3, 46.8] | 71.4 | 44.2 [41.3, 47.2] | 23.6 | - | - |
| ITA-Q3 | 1289 / 1158 / 7303 | 42.4 [39.7, 45.2] | 68.0 | 38.6 [36.0, 41.4] | 19.7 | - | - |
| ITA-Q4 (lightest) | 1288 / 1191 / 7735 | 38.8 [36.1, 41.6] | 65.1 | 34.9 [32.2, 37.9] | 18.1 | - | - |

## 7. H1-H4 quantities for existing models (DESCRIPTIVE ONLY; not the confirmatory test)

H1 = FF2-public TEST gender x race F_FPR (FF2 labels); H2 = FF2 age max-min FPR gap (InsightFace age, 4 bins); H3 = FF2 ITA 4-bin max-min FPR gap; H4 = mean over DFDC/Celeb-DF/DFD of gender x race F_FPR (AI-Face cleaner labels, >= 3 real videos per group). A1 threshold, seed means. NI columns = pooled TPR and AUC (FF2 for H1-H3; mean of the three sets for H4). The final-model comparator (best matched-CLIP baseline) does not exist yet.

| model | H1 i-F_FPR | H2 age FPR gap | H3 ITA FPR gap | FF2 TPR / AUC | H4 i-F_FPR (mean 3 sets; frame labels) | H4 with video-majority labels | cross-domain TPR / AUC |
|---|---|---|---|---|---|---|---|
| CFES-D | 41.0 | 8.9 | 3.8 | 69.1 / 71.0 | 36.4 | 25.4 | 62.4 / 84.7 |
| arm C | 42.5 | 8.4 | 4.7 | 53.2 / 69.9 | 36.0 | 28.3 | 62.2 / 84.9 |
| arm E w1 | 16.3 | 4.3 | 2.4 | 39.3 / 76.9 | 15.5 | 14.2 | 51.4 / 85.5 |
| arm E w3 | 17.4 | 5.5 | 2.5 | 41.0 / 78.5 | 15.9 | 14.2 | 53.3 / 84.8 |
| arm E w10 | 20.0 | 6.0 | 3.5 | 49.0 / 79.2 | 24.8 | 20.4 | 57.3 / 83.8 |
| CFES A (rev4) | 47.6 | 7.3 | 2.5 | 54.5 / 66.6 | 53.7 | 39.4 | 67.9 / 81.9 |
| control (det-only) | 61.6 | 10.2 | 5.6 | 50.5 / 62.7 | 61.9 | 44.6 | 64.5 / 78.9 |
| CFES-ResNet | 63.0 | 11.8 | 14.4 | 23.5 / 38.5 | 43.1 | 29.7 | 57.3 / 67.1 |
| CESel B0 | - | - | - | - / - | 18.7 | 15.8 | 59.1 / 83.2 |
| CESel p1 full | - | - | - | - / - | 44.5 | 30.4 | 62.8 / 80.8 |
| CESel p1 sens | - | - | - | - / - | 31.6 | 20.4 | 62.9 / 84.6 |
| CESel p1 inv | - | - | - | - / - | 30.2 | 17.8 | 58.7 / 84.1 |
| Ding repro (SFD+GDA) | 66.7 | 3.3 | 9.3 | 20.7 / 36.4 | 45.3 | 38.1 | 45.4 / 66.6 |
| Ding Ori repro (SFD) | 68.3 | 6.9 | 15.7 | 25.1 / 37.9 | 54.5 | 44.2 | 52.8 / 62.9 |

Paired descriptive differences (model - Ding repro (SFD+GDA), seed-mean, same cluster resample; percentile 95% interval; Ding has one seed; no multiplicity correction; NOT a test of H1-H4):

| model | H1 diff | H2 diff | H3 diff | H4 diff | H4 diff (video-majority labels) | dTPR FF2 / cross-domain |
|---|---|---|---|---|---|---|
| CFES-D | -25.7 [-36.6, -14.7] | +5.6 [+0.2, +8.3] | -5.5 [-9.1, -1.1] | -9.0 [-23.9, +11.0] | -12.7 [-24.0, +6.9] | 48.4 / 16.9 |
| arm C | -24.2 [-35.6, -13.3] | +5.1 [-0.0, +7.7] | -4.6 [-8.3, -0.5] | -9.3 [-26.9, +6.5] | -9.9 [-22.0, +3.2] | 32.5 / 16.8 |
| control (det-only) | -5.1 [-17.9, +5.7] | +6.8 [+1.6, +10.3] | -3.7 [-7.8, +1.1] | +16.6 [-1.6, +35.7] | +6.5 [-7.1, +24.0] | 29.8 / 19.1 |
| CESel B0 | - | - | - | -26.7 [-38.9, -9.1] | -22.3 [-34.7, -9.6] | - / 13.7 |
| Ding Ori repro (SFD) | +1.6 [-5.6, +9.1] | +3.5 [-0.1, +6.2] | +6.4 [+3.3, +9.2] | +9.2 [+0.7, +21.4] | +6.1 [-1.7, +12.8] | 4.4 / 7.4 |

## 8. Mechanism: B0 per-group cue deficits vs per-group errors

`code/mechanism.py`. Per (test set, group) point: cue deficit (set class mean - group mean of the removal utility u_k; positive = the group gets less evidence from cue k) vs the group's error deviation (reals: FPR_g - FPR_set; fakes: FNR_g - FNR_set). Spearman over all eligible points pooled across FF++ test, DFDC, Celeb-DF, DFD (>= 3 videos per group and side), video-cluster bootstrap 95% CI (B = 1000). `margin` row = the same correlation using the group's mean training-margin deficit (the arithmetic reference: summed utilities are margin differences). Threshold A1 (B0 a1.json).

| grouping | side | n points | tex | freq | bound | sem | total | margin (reference) |
|---|---|---|---|---|---|---|---|---|
| gender_x_race (Xu) | real | 24 | +0.15 [-0.23, +0.40] | -0.34 [-0.51, +0.11] | +0.31 [-0.02, +0.65] | +0.29 [-0.07, +0.45] | +0.48 [+0.13, +0.66] | +0.85 [+0.56, +0.93] |
| gender_x_race (Xu) | fake | 23 | -0.12 [-0.31, +0.19] | +0.62 [+0.28, +0.71] | +0.02 [-0.28, +0.30] | +0.38 [+0.03, +0.57] | +0.17 [-0.06, +0.46] | +0.97 [+0.92, +0.99] |
| gender_x_race (AI-Face clean) | real | 27 | +0.04 [-0.25, +0.39] | -0.20 [-0.46, +0.20] | +0.21 [-0.18, +0.52] | +0.45 [+0.10, +0.62] | +0.46 [+0.11, +0.66] | +0.77 [+0.49, +0.89] |
| gender_x_race (AI-Face clean) | fake | 25 | -0.15 [-0.44, +0.14] | +0.43 [+0.01, +0.58] | -0.14 [-0.40, +0.17] | +0.35 [+0.08, +0.66] | +0.21 [-0.15, +0.49] | +0.92 [+0.85, +0.98] |
| gender_x_race (AI-Face clean, video majority) | real | 24 | +0.05 [-0.26, +0.38] | -0.14 [-0.48, +0.17] | +0.14 [-0.18, +0.56] | +0.43 [+0.05, +0.61] | +0.45 [+0.14, +0.70] | +0.79 [+0.62, +0.93] |
| gender_x_race (AI-Face clean, video majority) | fake | 23 | -0.09 [-0.40, +0.22] | +0.44 [+0.09, +0.68] | -0.17 [-0.41, +0.23] | +0.66 [+0.24, +0.78] | +0.42 [+0.03, +0.70] | +0.95 [+0.89, +0.99] |
| age (AI-Face) | real | 13 | +0.49 [-0.15, +0.84] | +0.35 [-0.35, +0.72] | +0.66 [-0.08, +0.86] | +0.31 [+0.08, +0.84] | +0.79 [+0.26, +0.94] | +0.64 [+0.34, +0.93] |
| age (AI-Face) | fake | 12 | +0.28 [-0.54, +0.61] | -0.11 [-0.48, +0.68] | -0.24 [-0.60, +0.34] | +0.01 [-0.52, +0.57] | +0.10 [-0.55, +0.52] | +0.97 [+0.71, +0.99] |
| skin (ITA4) | real | 16 | +0.63 [+0.14, +0.78] | +0.21 [-0.22, +0.52] | +0.45 [+0.06, +0.70] | +0.44 [+0.04, +0.80] | +0.82 [+0.42, +0.89] | +0.79 [+0.46, +0.91] |
| skin (ITA4) | fake | 16 | -0.09 [-0.53, +0.46] | -0.45 [-0.71, +0.13] | -0.10 [-0.61, +0.24] | -0.08 [-0.49, +0.51] | -0.02 [-0.57, +0.39] | +0.92 [+0.77, +0.99] |

Partial Spearman of (cue deficit, error deviation) given the margin deficit (does the cue carry information beyond the group's overall margin?), same bootstrap:

| grouping | side | tex | freq | bound | sem | total |
|---|---|---|---|---|---|---|
| gender_x_race (Xu) | real | +0.10 [-0.23, +0.51] | +0.20 [-0.21, +0.47] | +0.23 [+0.05, +0.60] | +0.43 [-0.06, +0.57] | +0.57 [+0.12, +0.74] |
| gender_x_race (Xu) | fake | -0.32 [-0.64, +0.00] | +0.40 [-0.19, +0.57] | +0.11 [-0.28, +0.42] | -0.00 [-0.48, +0.17] | -0.01 [-0.53, +0.15] |
| gender_x_race (AI-Face clean) | real | +0.07 [-0.24, +0.46] | +0.05 [-0.34, +0.44] | +0.04 [-0.28, +0.43] | +0.50 [+0.03, +0.59] | +0.41 [+0.07, +0.61] |
| gender_x_race (AI-Face clean) | fake | -0.25 [-0.53, +0.24] | +0.26 [-0.31, +0.44] | +0.12 [-0.41, +0.27] | -0.30 [-0.59, +0.11] | -0.32 [-0.64, +0.01] |
| gender_x_race (AI-Face clean, video majority) | real | +0.10 [-0.20, +0.51] | +0.40 [-0.10, +0.63] | -0.02 [-0.18, +0.50] | +0.34 [-0.09, +0.54] | +0.30 [+0.06, +0.67] |
| gender_x_race (AI-Face clean, video majority) | fake | -0.37 [-0.57, +0.12] | +0.27 [-0.15, +0.58] | +0.17 [-0.34, +0.40] | -0.24 [-0.47, +0.35] | -0.35 [-0.55, +0.20] |
| age (AI-Face) | real | +0.48 [-0.23, +0.80] | +0.46 [-0.17, +0.75] | +0.71 [+0.02, +0.85] | +0.12 [-0.39, +0.70] | +0.73 [+0.03, +0.91] |
| age (AI-Face) | fake | -0.15 [-0.68, +0.56] | +0.49 [-0.63, +0.64] | +0.18 [-0.63, +0.65] | -0.35 [-0.74, +0.36] | -0.22 [-0.71, +0.35] |
| skin (ITA4) | real | +0.58 [+0.03, +0.80] | +0.32 [-0.23, +0.59] | +0.50 [-0.10, +0.71] | -0.05 [-0.41, +0.58] | +0.64 [+0.10, +0.83] |
| skin (ITA4) | fake | -0.19 [-0.61, +0.48] | -0.29 [-0.69, +0.33] | +0.07 [-0.67, +0.32] | -0.56 [-0.70, +0.24] | -0.39 [-0.74, +0.08] |

## Appendix A. Group support per test set and axis

Counts are model-independent: real frames / real videos / real persons | fake frames / fake videos / fake persons (FF2: video = person; de-novo fakes without a person are their own cluster); F = FPR-eligible (>= 3 real clusters), T = TPR-eligible (>= 3 fake clusters). With frame-level cleaner labels one video can contribute frames to several groups (Section 4b), so per-group video counts can add up to more than the set's videos.

**FF++ test**

| axis | group | real fr / vid / persons | fake fr / vid / persons | elig. |
|---|---|---|---|---|
| gender | male | 2041 / 120 / 120 | 9750 / 331 / 154 | FT |
| gender | nonmale | 2497 / 152 / 152 | 11113 / 385 / 195 | FT |
| race | asian | 720 / 51 / 51 | 2728 / 136 / 85 | FT |
| race | black | 208 / 11 / 11 | 857 / 35 / 25 | FT |
| race | others | 201 / 20 / 20 | 898 / 77 / 54 | FT |
| race | white | 3409 / 209 / 209 | 16380 / 594 / 295 | FT |
| gender_x_race | male,asian | 276 / 18 / 18 | 951 / 49 / 35 | FT |
| gender_x_race | male,black | 91 / 6 / 6 | 426 / 18 / 13 | FT |
| gender_x_race | male,others | 98 / 13 / 13 | 562 / 48 / 34 | FT |
| gender_x_race | male,white | 1576 / 94 / 94 | 7811 / 284 / 131 | FT |
| gender_x_race | nonmale,asian | 444 / 35 / 35 | 1777 / 95 / 57 | FT |
| gender_x_race | nonmale,black | 117 / 5 / 5 | 431 / 17 / 12 | FT |
| gender_x_race | nonmale,others | 103 / 7 / 7 | 336 / 31 / 21 | FT |
| gender_x_race | nonmale,white | 1833 / 117 / 117 | 8569 / 318 / 171 | FT |
| age | 25-44 | 1873 / 108 / 108 | 9553 / 320 / 199 | FT |
| age | 45-64 | 1414 / 88 / 88 | 6626 / 221 / 127 | FT |
| age | 65+ | 34 / 5 / 5 | 60 / 2 / 2 | F |
| age | <25 | 1217 / 66 / 66 | 4624 / 155 / 113 | FT |
| skin_ita4 | ITA-Q1 (darkest) | 1138 / 72 / 72 | 5758 / 192 / 111 | FT |
| skin_ita4 | ITA-Q2 | 1112 / 65 / 65 | 5066 / 169 / 108 | FT |
| skin_ita4 | ITA-Q3 | 1032 / 55 / 55 | 4648 / 156 / 96 | FT |
| skin_ita4 | ITA-Q4 (lightest) | 1165 / 68 / 68 | 5108 / 171 / 95 | FT |

**DFDC**

| axis | group | real fr / vid / persons | fake fr / vid / persons | elig. |
|---|---|---|---|---|
| gender | male | 6609 / 272 / 272 | 2905 / 135 / 135 | FT |
| gender | nonmale | 8090 / 308 / 308 | 5253 / 207 / 207 | FT |
| race | asian | 1569 / 106 / 106 | 282 / 45 / 45 | FT |
| race | black | 4363 / 190 / 190 | 1789 / 82 / 82 | FT |
| race | others | 1199 / 103 / 103 | 409 / 50 / 50 | FT |
| race | white | 7568 / 300 / 300 | 5678 / 219 / 219 | FT |
| gender_x_race | male,asian | 463 / 36 / 36 | 127 / 14 / 14 | FT |
| gender_x_race | male,black | 1564 / 109 / 109 | 230 / 33 / 33 | FT |
| gender_x_race | male,others | 722 / 71 / 71 | 222 / 30 / 30 | FT |
| gender_x_race | male,white | 3860 / 186 / 186 | 2326 / 112 / 112 | FT |
| gender_x_race | nonmale,asian | 1106 / 87 / 87 | 155 / 36 / 36 | FT |
| gender_x_race | nonmale,black | 2799 / 142 / 142 | 1559 / 74 / 74 | FT |
| gender_x_race | nonmale,others | 477 / 48 / 48 | 187 / 30 / 30 | FT |
| gender_x_race | nonmale,white | 3708 / 180 / 180 | 3352 / 150 / 150 | FT |
| age | 25-44 | 9474 / 303 / 303 | 5102 / 177 / 177 | FT |
| age | 45-64 | 2656 / 85 / 85 | 2270 / 78 / 78 | FT |
| age | 65+ | 896 / 32 / 32 | 426 / 16 / 16 | FT |
| age | <25 | 1673 / 53 / 53 | 360 / 13 / 13 | FT |
| skin_ita4 | ITA-Q1 (darkest) | 10645 / 341 / 341 | 5565 / 194 / 194 | FT |
| skin_ita4 | ITA-Q2 | 1043 / 35 / 35 | 452 / 16 / 16 | FT |
| skin_ita4 | ITA-Q3 | 744 / 24 / 24 | 480 / 15 / 15 | FT |
| skin_ita4 | ITA-Q4 (lightest) | 1553 / 50 / 50 | 1168 / 40 / 40 | FT |

**Celeb-DF**

| axis | group | real fr / vid / persons | fake fr / vid / persons | elig. |
|---|---|---|---|---|
| gender | male | 1667 / 57 / 36 | 1212 / 41 / 3 | FT |
| gender | nonmale | 4046 / 136 / 31 | 21533 / 723 / 12 | FT |
| race | black | 711 / 25 / 4 | 1212 / 41 / 3 | FT |
| race | others | 185 / 8 / 8 | 38 / 8 / 3 | FT |
| race | white | 4817 / 163 / 58 | 21495 / 723 / 12 | FT |
| gender_x_race | male,black | 710 / 24 / 3 | 1212 / 41 / 3 | FT |
| gender_x_race | male,others | 63 / 3 / 3 | 0 / 0 / 0 | F |
| gender_x_race | male,white | 894 / 30 / 30 | 0 / 0 / 0 | F |
| gender_x_race | nonmale,black | 1 / 1 / 1 | 0 / 0 / 0 | - |
| gender_x_race | nonmale,others | 122 / 6 / 6 | 38 / 8 / 3 | FT |
| gender_x_race | nonmale,white | 3923 / 133 / 28 | 21495 / 723 / 12 | FT |
| age | 25-44 | 3367 / 113 / 46 | 16612 / 558 / 14 | FT |
| age | 45-64 | 1512 / 51 / 18 | 3963 / 133 / 12 | FT |
| age | <25 | 834 / 28 / 11 | 2170 / 73 / 7 | FT |
| skin_ita4 | ITA-Q1 (darkest) | 2652 / 89 / 39 | 8454 / 283 / 15 | FT |
| skin_ita4 | ITA-Q2 | 1212 / 41 / 20 | 5229 / 178 / 11 | FT |
| skin_ita4 | ITA-Q3 | 718 / 24 / 18 | 3762 / 126 / 10 | FT |
| skin_ita4 | ITA-Q4 (lightest) | 1101 / 37 / 19 | 5210 / 174 / 11 | FT |

**DFD**

| axis | group | real fr / vid / persons | fake fr / vid / persons | elig. |
|---|---|---|---|---|
| gender | male | 1173 / 51 / 9 | 2596 / 124 / 8 | FT |
| gender | nonmale | 1919 / 67 / 6 | 3697 / 145 / 9 | FT |
| race | black | 630 / 28 / 5 | 549 / 28 / 5 | FT |
| race | others | 167 / 24 / 6 | 933 / 59 / 8 | FT |
| race | white | 2295 / 79 / 7 | 4811 / 189 / 10 | FT |
| gender_x_race | male,black | 383 / 18 / 4 | 438 / 19 / 4 | FT |
| gender_x_race | male,others | 89 / 14 / 4 | 499 / 34 / 7 | FT |
| gender_x_race | male,white | 701 / 32 / 7 | 1659 / 89 / 8 | FT |
| gender_x_race | nonmale,black | 247 / 12 / 2 | 111 / 12 / 3 | FT |
| gender_x_race | nonmale,others | 78 / 14 / 5 | 434 / 42 / 5 | FT |
| gender_x_race | nonmale,white | 1594 / 56 / 5 | 3152 / 141 / 9 | FT |
| age | 25-44 | 1966 / 66 / 8 | 4810 / 165 / 10 | FT |
| age | <25 | 1126 / 38 / 8 | 1483 / 50 / 8 | FT |
| skin_ita4 | ITA-Q1 (darkest) | 1486 / 50 / 9 | 2518 / 88 / 10 | FT |
| skin_ita4 | ITA-Q2 | 359 / 12 / 7 | 868 / 29 / 7 | FT |
| skin_ita4 | ITA-Q3 | 386 / 13 / 6 | 657 / 22 / 7 | FT |
| skin_ita4 | ITA-Q4 (lightest) | 861 / 29 / 8 | 2250 / 76 / 8 | FT |

**FF2-public TEST**

| axis | group | real fr / vid / persons | fake fr / vid / persons | elig. |
|---|---|---|---|---|
| gender | male | 3259 / 2621 / 2621 | 47711 / 10418 / 10418 | FT |
| gender | nonmale | 2182 / 1904 / 1904 | 34412 / 9434 / 9434 | FT |
| race | asian | 1171 / 1025 / 1025 | 20284 / 6471 / 6471 | FT |
| race | black | 1060 / 887 / 887 | 18653 / 5808 / 5808 | FT |
| race | others | 2031 / 1598 / 1598 | 23184 / 1597 / 1597 | FT |
| race | white | 1179 / 1018 / 1018 | 20002 / 5979 / 5979 | FT |
| gender_x_race | male,asian | 722 / 620 / 620 | 11573 / 3324 / 3324 | FT |
| gender_x_race | male,black | 585 / 471 / 471 | 10261 / 2931 / 2931 | FT |
| gender_x_race | male,others | 1340 / 1020 / 1020 | 15223 / 1019 / 1019 | FT |
| gender_x_race | male,white | 612 / 512 / 512 | 10654 / 3146 / 3146 | FT |
| gender_x_race | nonmale,asian | 449 / 405 / 405 | 8711 / 3147 / 3147 | FT |
| gender_x_race | nonmale,black | 475 / 416 / 416 | 8392 / 2877 / 2877 | FT |
| gender_x_race | nonmale,others | 691 / 578 / 578 | 7961 / 578 / 578 | FT |
| gender_x_race | nonmale,white | 567 / 506 / 506 | 9348 / 2833 / 2833 | FT |
| age | 30-44 | 2527 / 2238 / 2238 | 38377 / 10610 / 10610 | FT |
| age | 45-59 | 1550 / 1381 / 1381 | 19756 / 6765 / 6765 | FT |
| age | 60+ | 869 / 762 / 762 | 11672 / 5011 / 5011 | FT |
| age | <30 | 495 / 463 / 463 | 12318 / 5225 / 5225 | FT |
| skin_ita4 | ITA-Q1 (darkest) | 1300 / 1180 / 1180 | 18119 / 4804 / 4804 | FT |
| skin_ita4 | ITA-Q2 | 1247 / 1158 / 1158 | 19287 / 6533 / 6533 | FT |
| skin_ita4 | ITA-Q3 | 1289 / 1158 / 1158 | 19740 / 7303 / 7303 | FT |
| skin_ita4 | ITA-Q4 (lightest) | 1288 / 1191 / 1191 | 20309 / 7735 / 7735 | FT |

