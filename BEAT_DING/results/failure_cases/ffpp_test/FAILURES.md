# BD0_s7 failure cases on FF++ test (in-domain), 26 Oct 2026

**Model.** BD0 seed 7: CLIP ViT-L/14 with rank-1 adapters, CFES-D, Arm E base and ESA.

**Inputs.**
- Scores: `runs/BD0_s7/eval/ffpp_test_scores.npz`.
- Thresholds: from `bd_eval.json`.
  - matched (primary): logit ≥ −73.76. This matches the val TPR of the Ding reproduction (94.18).
  - native: −65.97.
  - A1: −19.13.
- Ding reproduction: `DINGREPRO_LIN/.../full_s7_ep49_ff++.npz`, using its own 0.5 rule (logit ≥ 0).
- Labels: `DEMO_AXES/labels/labels_ffpp_test.csv`, plus FairFace race7.

**Test set.** 4,538 real frames (267 videos) and 20,863 fake frames (698 videos).

**Sanity check.** The script reproduces bd_eval at the matched threshold: TPR 93.13, FPR 13.51, i-F_FPR 46.56.

**Code and outputs.**
- Code: `failure_analysis.py`, which writes every number below.
- Raw tables: `tables/*.csv`.
- Machine-readable results: `results.json`.
- Contact sheets: `fp_sheet.png`, `fn_sheet.png`. Each sheet shows the 24 videos with the most confident errors, one frame per video. Captions are green when Ding is right and red when Ding is also wrong.

## Headline numbers (matched)

| | BD0 matched | Ding repro @0.5 |
|---|---|---|
| FP (real flagged fake) | 613 / 4538 = 13.51 % (97/267 videos) | 569 = 12.54 % |
| FN (fake missed) | 1434 / 20863 = 6.87 % (178/698 videos) | 1075 = 5.15 % |
| i-F_FPR | 46.56 | 24.24 |
| FP also wrong for Ding | 36.1 % (221 frames); 392 FP are ours-only | 348 Ding-only FP |
| FN also wrong for Ding | 33.2 % (476 frames); 958 FN are ours-only | 599 Ding-only FN |
| Spearman(our logit, Ding logit) | reals 0.356, fakes 0.365 | |

## Failure modes (plain words, every number is from `failure_analysis.py`)

1. **Our errors are mostly different from Ding's.**
   - Only 36.1 % of our 613 FPs and 33.2 % of our 1,434 FNs are also errors for Ding. The Jaccard overlap of the two error sets is 0.23 for both FP and FN.
   - Our scores and Ding's rank the same images only weakly alike: Spearman 0.36 on reals and 0.37 on fakes.
   - So each model has its own failure set, not a shared "hard core" that both fail on. On the contact sheets Ding is also wrong for only 12 of 24 FP tiles and 13 of 24 FN tiles.

2. **Half of the intersectional F_FPR comes from under-flagging two small male groups.**
   - The pooled FPR is 13.51 %. M-Black has an FPR of 3.26 % (3 FP frames, 2 of 7 videos) and M-Others 2.76 % (5 frames, 3 of 12 videos).
   - Their absolute deviations from the pooled FPR are 10.25 and 10.75. Together that is 21.0 of the 46.56 points (45 %).
   - F-Asian over-flagging adds 12.10 points (26 %): an FPR of 25.6 % on 414 frames, with FPs in 12 of 23 videos.
   - These three cells make up 71 % of i-F_FPR.
   - Ding's largest per-group deviations are 6.1 (F-Black) and 5.9 (M-Asian). Ding's F-Asian FPR is 10.87 %, against our 25.6 %.

3. **Over-flagging of F-Asian reals is ours alone.**
   - 80 of the 106 F-Asian FP frames (75 %) are frames Ding calls real.
   - The three worst F-Asian videos hold 51 % of that group's FPs:
     - youtube_062: our FP rate 69 %, Ding 0 %.
     - youtube_616: ours 60 %, Ding 60 %.
     - youtube_930: ours 57 %, Ding 18 %.
   - 11 of the top-12 most confident FPs are Xu-White presenters (the other is youtube_062). The highest is youtube_018 at logit 47.0, and Ding also flags it.

4. **FPs are fairly concentrated by video, but the disparity is not just "a few bad videos".**
   - At matched, 97 of 267 real videos have at least one FP. 18 videos are flagged on at least 50 % of their frames. 17 videos hold half of all FP frames, and the top 15 hold 46.3 %.
   - Dropping the top-k FP videos without changing the threshold lowers i-F_FPR from 46.56 to 38.51 (k=5) and to 24.90 (k=15). Most of that fall comes from the lower pooled FPR (8.03 % at k=15).
   - If we re-threshold to the same 13.51 % FPR after dropping, i-F_FPR reaches its lowest value, 36.05, at k=5. It rises again to 40.85 at k=15 and 47.13 at k=20, because other videos cross the threshold.
   - Random-drop control (k=5, re-thresholded): median 46.36, 90 % range 43.75–49.76.
   - So the 5 worst videos (305, 845, 307, 383, 438, all labelled White by Xu) explain about 10 points. The rest is spread out and dominated by the group-size effect in mode 2.
   - bd_eval's own permutation null at this FPR has a median of 42.61, with P(null ≥ obs) = 0.366. Our 46.56 is therefore indistinguishable from "no group effect plus video clustering".

5. **The FP pattern does not follow darker skin, and the FN pattern is male-skewed.**
   - **FPR by ITA quartile (lightest to darkest):**
     - Q4 lightest: 17.5 %.
     - Q3: 15.1 %.
     - Q2: 10.0 %.
     - Q1 darkest: 12.3 %.
   - **FPR by age:** under 30: 14.7 %; 30–44: 13.4 %; 45–59: 8.5 %; 60+: 0 of 34 frames.
   - **FPR by gender:** female 14.3 % vs male 12.6 % (Xu labels); 14.9 % vs 11.8 % (cleaner labels).
   - **FNR by gender:** male 10.2 % vs female 3.8 % (Xu) or 3.95 % (cleaner). 71 % of all FNs are fakes of male targets.
   - **FNR by group:**
     - M-Asian: 17.4 %.
     - M-Others: 13.0 %.
     - M-White: 9.5 %.
     - Every female group: at most 4.0 %.

6. **Most misses are NeuralTextures (reenactment).**
   - FNR by method:
     - NeuralTextures: 17.2 % (777 of 1,434 FNs, 54 %).
     - Face2Face: 5.3 %.
     - FaceShifter: 5.05 %.
     - FaceSwap: 3.9 %.
     - Deepfakes: 1.5 %.
   - The reenactment family misses 11.4 % of its frames and accounts for 70 % of FNs.
   - Of the 34 fake videos missed on at least 50 % of their frames, 22 are NeuralTextures.
   - Of the 958 ours-only FNs, 483 are NeuralTextures. Example: NT_306_278 is missed on 93 % of frames, while Ding catches it on all 30.

7. **Misses cluster by target identity, and the target's real video predicts them.**
   - 15 of 192 target identities hold half of all FNs, and the top 10 hold 38.9 %.
   - Five targets are missed under at least 2 methods. Examples: 705 (3 methods, 49 % FN, Ding 0 %), 259, 016, 619, 492.
   - Across the 152 targets whose real video is in test, there is an appearance prior:
     - A lower mean logit on the target's real video goes with a higher FN rate on its fakes: Spearman −0.446 for us, −0.369 for Ding.
     - The real-video and fake-video mean logits correlate with Spearman 0.453.
     - The real-video FP rate and the fake FN rate correlate with Spearman −0.391.
   - So the model partly scores how fake the person or broadcast looks, not the manipulation itself.
   - Two examples: target 800 has a real-video mean logit of −116.8 and has 54 FN frames, spread over NeuralTextures (28), FaceSwap (11), Face2Face (10) and FaceShifter (5). Target 619 has a real-video mean logit of −114.3.

8. **The operating point trades one failure for the other. Errors at A1 are the confident ones.**
   - **native:** FPR 8.09 %, FNR 9.54 %, i-F_FPR 36.11.
   - **A1:** FPR 1.87 %, FNR 19.63 %, i-F_FPR 8.43.
   - At A1 the 85 FPs come from 31 videos, and 6 videos hold half of them. youtube_845 alone has 12 FP frames.
   - At A1, 74 % of Ding's FNs are also ours, and 131 fake videos (75 of them NeuralTextures) are missed on at least 50 % of frames.
   - Ding's FP set barely overlaps ours at A1: only 8.4 % of Ding's FPs are ours.

## Label caveat seen in the top FP videos
youtube_305, our #1 FP video at matched (80 % of frames flagged), is Xu "M-White". The cleaner AI-Face labels say female/Asian, and FairFace says Middle Eastern. Group-level FP numbers for small cells shift with such single-video label disagreements.

---

# Full tables (generated by failure_analysis.py; also in FAILURES_tables.md)

## Operating points (FF++ test, n_real=4538 frames / 267 videos, n_fake=20863 / 698)

| op | logit t | #FP | FPR % | #FN | FNR % | i-F_FPR | FP also Ding % | FN also Ding % | FP ours-only | FP Ding-only | FN ours-only | FN Ding-only |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matched | -73.76 | 613 | 13.51 | 1434 | 6.87 | 46.56 | 36.1 | 33.2 | 392 | 348 | 958 | 599 |
| native | -65.97 | 367 | 8.09 | 1991 | 9.54 | 36.11 | 43.6 | 29.7 | 207 | 409 | 1399 | 483 |
| A1 | -19.13 | 85 | 1.87 | 4095 | 19.63 | 8.43 | 56.5 | 19.5 | 37 | 521 | 3297 | 277 |
| Ding repro @0.5 | 0.0 | 569 | 12.54 | 1075 | 5.15 | 24.24 |  |  |  |  |  |  |


## matched: false positives by attribute


**gname**

| gname | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| F-Asian | 414 | 106 | 23 | 12 | 25.6 | 17.29 |
| F-Black | 118 | 14 | 6 | 2 | 11.86 | 2.28 |
| F-Others | 151 | 12 | 9 | 3 | 7.95 | 1.96 |
| F-White | 1755 | 216 | 109 | 35 | 12.31 | 35.24 |
| M-Asian | 238 | 23 | 13 | 6 | 9.66 | 3.75 |
| M-Black | 92 | 3 | 7 | 2 | 3.26 | 0.49 |
| M-Others | 181 | 5 | 12 | 3 | 2.76 | 0.82 |
| M-White | 1589 | 234 | 88 | 34 | 14.73 | 38.17 |


**xu_gender**

| xu_gender | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| female | 2438 | 348 | 147 | 52 | 14.27 | 56.77 |
| male | 2100 | 265 | 120 | 45 | 12.62 | 43.23 |


**cl_gender**

| cl_gender | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| female | 2497 | 373 | 152 | 57 | 14.94 | 60.85 |
| male | 2041 | 240 | 120 | 44 | 11.76 | 39.15 |


**xu_race**

| xu_race | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| asian | 652 | 129 | 36 | 18 | 19.78 | 21.04 |
| black | 210 | 17 | 13 | 4 | 8.1 | 2.77 |
| others | 332 | 17 | 21 | 6 | 5.12 | 2.77 |
| white | 3344 | 450 | 197 | 69 | 13.46 | 73.41 |


**cl_race**

| cl_race | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| asian | 720 | 139 | 51 | 22 | 19.31 | 22.68 |
| black | 208 | 17 | 11 | 4 | 8.17 | 2.77 |
| others | 201 | 17 | 20 | 8 | 8.46 | 2.77 |
| white | 3409 | 440 | 209 | 75 | 12.91 | 71.78 |


**age4**

| age4 | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| 30-44 | 1451 | 195 | 92 | 30 | 13.44 | 31.81 |
| 45-59 | 509 | 43 | 29 | 11 | 8.45 | 7.02 |
| 60+ | 34 | 0 | 6 | 0 | 0.0 | 0.0 |
| <30 | 2544 | 375 | 140 | 56 | 14.74 | 61.18 |


**ita4**

| ita4 | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| Q1 darkest | 1138 | 140 | 72 | 25 | 12.3 | 22.84 |
| Q2 | 1112 | 111 | 65 | 23 | 9.98 | 18.11 |
| Q3 | 1032 | 156 | 55 | 19 | 15.12 | 25.45 |
| Q4 lightest | 1165 | 204 | 68 | 28 | 17.51 | 33.28 |
| missing | 91 | 2 | 7 | 2 | 2.2 | 0.33 |


## matched: false negatives by attribute


**method**

| method | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| Deepfakes | 3880 | 59 | 130 | 6 | 1.52 | 4.11 |
| Face2Face | 4373 | 233 | 146 | 28 | 5.33 | 16.25 |
| FaceShifter | 4280 | 216 | 143 | 41 | 5.05 | 15.06 |
| FaceSwap | 3822 | 149 | 128 | 28 | 3.9 | 10.39 |
| NeuralTextures | 4508 | 777 | 151 | 75 | 17.24 | 54.18 |


**family**

| family | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| identity-swap | 11982 | 424 | 401 | 75 | 3.54 | 29.57 |
| reenactment | 8881 | 1010 | 297 | 103 | 11.37 | 70.43 |


**gname**

| gname | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| F-Asian | 1768 | 59 | 60 | 7 | 3.34 | 4.11 |
| F-Black | 480 | 11 | 16 | 3 | 2.29 | 0.77 |
| F-Others | 690 | 25 | 23 | 2 | 3.62 | 1.74 |
| F-White | 7968 | 320 | 267 | 55 | 4.02 | 22.32 |
| M-Asian | 929 | 162 | 31 | 16 | 17.44 | 11.3 |
| M-Black | 510 | 12 | 17 | 3 | 2.35 | 0.84 |
| M-Others | 990 | 129 | 33 | 10 | 13.03 | 9.0 |
| M-White | 7528 | 716 | 251 | 82 | 9.51 | 49.93 |


**cl_gender**

| cl_gender | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| female | 11113 | 439 | 385 | 75 | 3.95 | 30.61 |
| male | 9750 | 995 | 331 | 106 | 10.2 | 69.39 |


**cl_race**

| cl_race | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| asian | 2728 | 297 | 136 | 30 | 10.89 | 20.71 |
| black | 857 | 22 | 35 | 5 | 2.57 | 1.53 |
| others | 898 | 58 | 77 | 10 | 6.46 | 4.04 |
| white | 16380 | 1057 | 594 | 145 | 6.45 | 73.71 |


**age4**

| age4 | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| 30-44 | 6563 | 541 | 219 | 63 | 8.24 | 37.73 |
| 45-59 | 2549 | 198 | 85 | 26 | 7.77 | 13.81 |
| 60+ | 30 | 18 | 1 | 1 | 60.0 | 1.25 |
| <30 | 11721 | 677 | 393 | 88 | 5.78 | 47.21 |


**ita4**

| ita4 | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| Q1 darkest | 5758 | 373 | 192 | 51 | 6.48 | 26.01 |
| Q2 | 5066 | 405 | 169 | 48 | 7.99 | 28.24 |
| Q3 | 4648 | 394 | 156 | 45 | 8.48 | 27.48 |
| Q4 lightest | 5108 | 262 | 171 | 34 | 5.13 | 18.27 |
| missing | 283 | 0 | 10 | 0 | 0.0 | 0.0 |


## matched: FP video concentration

- real videos with >=1 FP frame: 97 / 267; with >=50% FP frames: 18
- top-5 videos hold 17.9% of FP frames, top-15 hold 46.3%; 17 videos hold half


| top-k FP videos dropped | i-F_FPR (same t) | FPR % (same t) | i-F_FPR (re-thresholded to same FPR) | last dropped (<=5) |
|---|---|---|---|---|
| 0 | 46.56 | 13.51 | 46.56 |  |
| 1 | 43.53 | 13.07 | 44.74 | 305 |
| 2 | 41.4 | 12.64 | 42.19 | 305, 845 |
| 3 | 40.32 | 12.23 | 39.97 | 305, 845, 307 |
| 5 | 38.51 | 11.46 | 36.05 | 305, 845, 307, 383, 438 |
| 10 | 32.09 | 9.62 | 37.99 | 237, 062, 445, 018, 446 |
| 15 | 24.9 | 8.03 | 40.85 | 616, 204, 930, 472, 547 |
| 20 | 21.21 | 6.78 | 47.13 | 923, 202, 458, 985, 372 |


random-drop control, re-thresholded to same FPR (200 draws): k=5: median 46.36 [p5 43.75, p95 49.76]; k=10: median 45.77 [p5 42.62, p95 52.18]


| group | group_FPR_% | |FPR_g-FPR_all| | video-weighted FPR % | videos with FP | top-3 video share of group FPs % | Ding FPR_g % | Ding |dev| |
|---|---|---|---|---|---|---|---|
| F-Asian | 25.6 | 12.1 | 15.95 | 12/23 | 50.94 | 10.87 | 1.67 |
| F-Black | 11.86 | 1.64 | 8.05 | 2/6 | 100.0 | 18.64 | 6.11 |
| F-Others | 7.95 | 5.56 | 4.5 | 3/9 | 100.0 | 14.57 | 2.03 |
| F-White | 12.31 | 1.2 | 7.73 | 35/109 | 25.93 | 11.4 | 1.14 |
| M-Asian | 9.66 | 3.84 | 13.53 | 6/13 | 78.26 | 18.49 | 5.95 |
| M-Black | 3.26 | 10.25 | 1.48 | 2/7 | 100.0 | 8.7 | 3.84 |
| M-Others | 2.76 | 10.75 | 9.45 | 3/12 | 100.0 | 16.02 | 3.48 |
| M-White | 14.73 | 1.22 | 11.19 | 34/88 | 29.49 | 12.52 | 0.01 |


**top-15 FP videos (matched)**

| video | group | cl_gender | cl_race | ff_race7 | age_vid | ita4 | n | n_fp | fp_rate_% | ding_fp_rate_% | max_logit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0_youtube_305 | M-White | female | asian | Middle Eastern | 28.0 | Q4 lightest | 30 | 24 | 80.0 | 46.67 | 29.74 |
| 0_youtube_845 | M-White | male | white | White | 42.0 | Q1 darkest | 29 | 23 | 79.31 | 72.41 | 31.92 |
| 0_youtube_307 | M-White | male | white | Middle Eastern | 55.5 | Q4 lightest | 30 | 22 | 73.33 | 90.0 | 46.22 |
| 0_youtube_383 | M-White | male | white | Middle Eastern | 34.0 | Q3 | 30 | 21 | 70.0 | 6.67 | 12.44 |
| 0_youtube_438 | F-White | female | white | White | 29.0 | Q4 lightest | 29 | 20 | 68.97 | 20.69 | 33.59 |
| 0_youtube_237 | M-White | male | white | White | 24.0 | Q3 | 30 | 20 | 66.67 | 3.33 | 28.78 |
| 0_youtube_062 | F-Asian | female | asian | East Asian | 31.0 | Q3 | 29 | 20 | 68.97 | 0.0 | 27.5 |
| 0_youtube_445 | M-White | male | white | White | 42.0 | Q2 | 30 | 19 | 63.33 | 0.0 | -50.82 |
| 0_youtube_018 | F-White | female | white | White | 31.0 | Q2 | 29 | 18 | 62.07 | 72.41 | 47.02 |
| 0_youtube_446 | F-White | female | white | White | 20.0 | Q4 lightest | 30 | 18 | 60.0 | 13.33 | 38.15 |
| 0_youtube_616 | F-Asian | female | asian | East Asian | 37.0 | Q2 | 30 | 18 | 60.0 | 60.0 | -41.47 |
| 0_youtube_204 | F-White | female | white | White | 29.0 | Q4 lightest | 30 | 18 | 60.0 | 6.67 | -55.17 |
| 0_youtube_930 | F-Asian | female | asian | East Asian | 18.0 | Q4 lightest | 28 | 16 | 57.14 | 17.86 | 24.07 |
| 0_youtube_472 | F-White | female | white | Latino_Hispanic | 23.0 | Q3 | 29 | 14 | 48.28 | 20.69 | -49.62 |
| 0_youtube_547 | M-White | male | white | Middle Eastern | 23.5 | Q3 | 30 | 13 | 43.33 | 43.33 | -2.17 |


**top-15 FN videos (matched)**

| video | method | group | cl_gender | cl_race | age_vid | ita4 | n | n_fn | fn_rate_% | ding_fn_rate_% | min_logit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0_NeuralTextures_840_800 | NeuralTextures | M-White | male | white | 33.0 | Q1 darkest | 30 | 30 | 100.0 | 96.67 | -103.29 |
| 0_NeuralTextures_794_779 | NeuralTextures | M-Others | male | white | 28.0 | Q2 | 30 | 30 | 100.0 | 100.0 | -96.22 |
| 0_Deepfakes_492_325 | Deepfakes | M-White | male | white | 42.0 | Q3 | 30 | 29 | 96.67 | 96.67 | -105.82 |
| 0_NeuralTextures_235_194 | NeuralTextures | M-Asian | male | asian | 24.0 | Q3 | 30 | 29 | 96.67 | 96.67 | -99.81 |
| 0_NeuralTextures_800_840 | NeuralTextures | M-White | male | white | 56.0 | Q1 darkest | 30 | 28 | 93.33 | 10.0 | -110.75 |
| 0_NeuralTextures_016_209 | NeuralTextures | M-Others | male | asian | 3.0 | Q2 | 30 | 28 | 93.33 | 20.0 | -106.07 |
| 0_NeuralTextures_306_278 | NeuralTextures | M-White | male | white | 32.5 | Q2 | 30 | 28 | 93.33 | 0.0 | -100.85 |
| 0_NeuralTextures_578_636 | NeuralTextures | F-White | female | white | 32.5 | Q4 lightest | 30 | 27 | 90.0 | 3.33 | -113.82 |
| 0_NeuralTextures_867_765 | NeuralTextures | F-White | female | white | 25.0 | Q4 lightest | 30 | 27 | 90.0 | 43.33 | -110.34 |
| 0_NeuralTextures_619_620 | NeuralTextures | M-White | male | white | 54.0 | Q3 | 30 | 26 | 86.67 | 96.67 | -111.46 |
| 0_Face2Face_016_209 | Face2Face | M-Others | male | white | 3.0 | Q2 | 30 | 26 | 86.67 | 3.33 | -109.54 |
| 0_NeuralTextures_294_292 | NeuralTextures | F-Asian | female | asian | 23.0 | Q3 | 30 | 25 | 83.33 | 93.33 | -114.75 |
| 0_NeuralTextures_548_632 | NeuralTextures | M-White | male | white | 35.5 | Q3 | 30 | 25 | 83.33 | 90.0 | -101.77 |
| 0_NeuralTextures_494_445 | NeuralTextures | M-Others | male | white | 38.0 | Q1 darkest | 30 | 25 | 83.33 | 3.33 | -89.79 |
| 0_NeuralTextures_010_005 | NeuralTextures | F-Others | female | others | 23.0 | Q2 | 30 | 24 | 80.0 | 30.0 | -103.47 |


**FP ours-only (we flag, Ding says real), top videos (matched)** — by group: {'F-Asian': 80, 'F-Black': 4, 'F-Others': 9, 'F-White': 148, 'M-Asian': 13, 'M-Black': 3, 'M-Others': 5, 'M-White': 130}

| video | group | method | n | n_err | rate_% | ours_mean_logit | ding_mean_logit |
|---|---|---|---|---|---|---|---|
| 0_youtube_062 | F-Asian | real | 29 | 20 | 68.97 | -57.19 | -5.09 |
| 0_youtube_445 | M-White | real | 30 | 19 | 63.33 | -71.32 | -4.9 |
| 0_youtube_237 | M-White | real | 30 | 19 | 63.33 | -53.75 | -4.63 |
| 0_youtube_383 | M-White | real | 30 | 19 | 63.33 | -48.41 | -3.2 |
| 0_youtube_204 | F-White | real | 30 | 17 | 56.67 | -73.19 | -3.66 |
| 0_youtube_446 | F-White | real | 30 | 14 | 46.67 | -57.43 | -3.54 |
| 0_youtube_438 | F-White | real | 29 | 14 | 48.28 | -65.91 | -2.62 |
| 0_youtube_372 | F-Asian | real | 28 | 12 | 42.86 | -78.41 | -4.81 |


**FP Ding-only, top videos (matched)** — by group: {'F-Asian': 19, 'F-Black': 12, 'F-Others': 19, 'F-White': 132, 'M-Asian': 34, 'M-Black': 8, 'M-Others': 29, 'M-White': 95}

| video | group | method | n | n_err | rate_% | ours_mean_logit | ding_mean_logit |
|---|---|---|---|---|---|---|---|
| 0_youtube_494 | M-Others | real | 29 | 24 | 82.76 | -92.55 | 2.31 |
| 0_youtube_763 | F-Others | real | 29 | 13 | 44.83 | -91.24 | -0.07 |
| 0_youtube_602 | F-White | real | 28 | 13 | 46.43 | -96.73 | 0.07 |
| 0_youtube_923 | M-White | real | 29 | 12 | 41.38 | -75.79 | -0.26 |
| 0_youtube_623 | F-White | real | 29 | 12 | 41.38 | -102.71 | -0.7 |
| 0_youtube_606 | F-White | real | 29 | 12 | 41.38 | -103.71 | -0.67 |
| 0_youtube_513 | M-Asian | real | 29 | 10 | 34.48 | -79.31 | -0.81 |
| 0_youtube_019 | F-White | real | 30 | 10 | 33.33 | -85.64 | -0.55 |


**FN ours-only (we miss, Ding catches), top videos (matched)** — by group: {'F-Asian': 28, 'F-Black': 8, 'F-Others': 18, 'F-White': 256, 'M-Asian': 78, 'M-Black': 10, 'M-Others': 85, 'M-White': 475}; by method: {'Deepfakes': 25, 'Face2Face': 172, 'FaceShifter': 168, 'FaceSwap': 110, 'NeuralTextures': 483}

| video | group | method | n | n_err | rate_% | ours_mean_logit | ding_mean_logit |
|---|---|---|---|---|---|---|---|
| 0_NeuralTextures_306_278 | M-White | NeuralTextures | 30 | 28 | 93.33 | -82.18 | 4.92 |
| 0_NeuralTextures_578_636 | F-White | NeuralTextures | 30 | 26 | 86.67 | -89.85 | 4.21 |
| 0_NeuralTextures_800_840 | M-White | NeuralTextures | 30 | 25 | 83.33 | -94.6 | 3.77 |
| 0_NeuralTextures_494_445 | M-Others | NeuralTextures | 30 | 25 | 83.33 | -79.75 | 2.85 |
| 0_Face2Face_016_209 | M-Others | Face2Face | 30 | 25 | 83.33 | -85.0 | 3.71 |
| 0_NeuralTextures_872_873 | M-White | NeuralTextures | 30 | 23 | 76.67 | -80.73 | 4.03 |
| 0_NeuralTextures_016_209 | M-Others | NeuralTextures | 30 | 22 | 73.33 | -88.57 | 2.49 |
| 0_NeuralTextures_194_235 | M-Asian | NeuralTextures | 30 | 22 | 73.33 | -76.46 | 4.0 |


**FN Ding-only, top videos (matched)** — by group: {'F-Asian': 54, 'F-Black': 9, 'F-Others': 9, 'F-White': 216, 'M-Asian': 49, 'M-Black': 5, 'M-Others': 35, 'M-White': 222}; by method: {'Deepfakes': 88, 'Face2Face': 87, 'FaceShifter': 68, 'FaceSwap': 60, 'NeuralTextures': 296}

| video | group | method | n | n_err | rate_% | ours_mean_logit | ding_mean_logit |
|---|---|---|---|---|---|---|---|
| 0_NeuralTextures_381_376 | F-White | NeuralTextures | 30 | 25 | 83.33 | -15.32 | -3.75 |
| 0_NeuralTextures_662_641 | F-Asian | NeuralTextures | 28 | 24 | 85.71 | -34.64 | -3.08 |
| 0_NeuralTextures_534_490 | F-White | NeuralTextures | 29 | 21 | 72.41 | 11.05 | -1.1 |
| 0_Face2Face_193_030 | F-White | Face2Face | 30 | 20 | 66.67 | 20.4 | -1.87 |
| 0_NeuralTextures_620_619 | M-White | NeuralTextures | 30 | 20 | 66.67 | -55.93 | -1.2 |
| 0_NeuralTextures_482_465 | F-White | NeuralTextures | 30 | 17 | 56.67 | -52.82 | -0.77 |
| 0_Deepfakes_948_965 | F-White | Deepfakes | 30 | 16 | 53.33 | 42.52 | -0.51 |
| 0_NeuralTextures_857_909 | M-White | NeuralTextures | 30 | 15 | 50.0 | -58.44 | -0.52 |


**top-12 most confident FP (matched)**

| frame | gname | logit | ding_logit | cl_race | age4 | ita4 | img_path |
|---|---|---|---|---|---|---|---|
| 0_youtube_018_291.png | F-White | 47.02 | 5.2 | white | 30-44 | Q2 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_youtube_018_291.png |
| 0_youtube_307_131.png | M-White | 46.22 | 2.95 | white | 45-59 | Q4 lightest | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_youtube_307_131.png |
| 0_youtube_095_121.png | F-White | 38.66 | -1.07 | white | <30 | Q3 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_youtube_095_121.png |
| 0_youtube_446_141.png | F-White | 38.15 | -5.46 | white | <30 | Q4 lightest | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_youtube_446_141.png |
| 0_youtube_438_131.png | F-White | 33.59 | -4.8 | white | <30 | Q4 lightest | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_youtube_438_131.png |
| 0_youtube_845_51.png | M-White | 31.92 | 1.23 | white | 30-44 | Q1 darkest | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_youtube_845_51.png |
| 0_youtube_305_21.png | M-White | 29.74 | 0.03 | asian | <30 | Q4 lightest | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_youtube_305_21.png |
| 0_youtube_288_221.png | F-White | 28.8 | 2.2 | white | <30 | Q1 darkest | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_youtube_288_221.png |
| 0_youtube_237_81.png | M-White | 28.78 | -2.65 | white | <30 | Q3 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_youtube_237_81.png |
| 0_youtube_018_1.png | F-White | 28.41 | 4.86 | white | 30-44 | Q2 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_youtube_018_1.png |
| 0_youtube_062_21.png | F-Asian | 27.5 | -5.17 | asian | 30-44 | Q3 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_youtube_062_21.png |
| 0_youtube_305_1.png | M-White | 27.3 | 4.14 | white | <30 | Q4 lightest | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_youtube_305_1.png |


**top-12 most confident FN (matched)**

| frame | gname | method | logit | ding_logit | cl_race | age4 | ita4 | img_path |
|---|---|---|---|---|---|---|---|---|
| 0_NeuralTextures_294_292_1.png | F-Asian | NeuralTextures | -114.75 | -5.44 | asian | <30 | Q3 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_NeuralTextures_294_292_1.png |
| 0_NeuralTextures_578_636_101.png | F-White | NeuralTextures | -113.82 | 0.66 | white | 30-44 | Q4 lightest | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_NeuralTextures_578_636_101.png |
| 0_FaceSwap_800_840_101.png | M-White | FaceSwap | -113.25 | 4.58 | white | 45-59 | Q1 darkest | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_FaceSwap_800_840_101.png |
| 0_FaceShifter_951_947_101.png | F-White | FaceShifter | -111.9 | -5.65 | white | <30 | Q2 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_FaceShifter_951_947_101.png |
| 0_NeuralTextures_619_620_231.png | M-White | NeuralTextures | -111.46 | -4.18 | white | 45-59 | Q3 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_NeuralTextures_619_620_231.png |
| 0_FaceSwap_800_840_111.png | M-White | FaceSwap | -111.27 | 5.55 | white | 45-59 | Q1 darkest | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_FaceSwap_800_840_111.png |
| 0_NeuralTextures_800_840_31.png | M-White | NeuralTextures | -110.75 | 4.8 | white | 45-59 | Q1 darkest | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_NeuralTextures_800_840_31.png |
| 0_FaceShifter_800_840_41.png | M-White | FaceShifter | -110.75 | 5.18 | white | 45-59 | Q1 darkest | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_FaceShifter_800_840_41.png |
| 0_Face2Face_619_620_1.png | M-White | Face2Face | -110.65 | -0.09 | white | 45-59 | Q3 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_Face2Face_619_620_1.png |
| 0_NeuralTextures_619_620_11.png | M-White | NeuralTextures | -110.36 | -5.79 | white | 45-59 | Q3 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_NeuralTextures_619_620_11.png |
| 0_NeuralTextures_867_765_1.png | F-White | NeuralTextures | -110.34 | -5.77 | white | <30 | Q4 lightest | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_NeuralTextures_867_765_1.png |
| 0_FaceShifter_867_765_231.png | F-Asian | FaceShifter | -109.92 | 5.26 | asian | <30 | Q4 lightest | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/ffpp/0_FaceShifter_867_765_231.png |


## native: false positives by attribute


**gname**

| gname | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| F-Asian | 414 | 63 | 23 | 11 | 15.22 | 17.17 |
| F-Black | 118 | 12 | 6 | 2 | 10.17 | 3.27 |
| F-Others | 151 | 4 | 9 | 3 | 2.65 | 1.09 |
| F-White | 1755 | 128 | 109 | 26 | 7.29 | 34.88 |
| M-Asian | 238 | 9 | 13 | 5 | 3.78 | 2.45 |
| M-Black | 92 | 0 | 7 | 0 | 0.0 | 0.0 |
| M-Others | 181 | 2 | 12 | 2 | 1.1 | 0.55 |
| M-White | 1589 | 149 | 88 | 28 | 9.38 | 40.6 |


**xu_gender**

| xu_gender | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| female | 2438 | 207 | 147 | 42 | 8.49 | 56.4 |
| male | 2100 | 160 | 120 | 35 | 7.62 | 43.6 |


**cl_gender**

| cl_gender | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| female | 2497 | 220 | 152 | 46 | 8.81 | 59.95 |
| male | 2041 | 147 | 120 | 34 | 7.2 | 40.05 |


**xu_race**

| xu_race | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| asian | 652 | 72 | 36 | 16 | 11.04 | 19.62 |
| black | 210 | 12 | 13 | 2 | 5.71 | 3.27 |
| others | 332 | 6 | 21 | 5 | 1.81 | 1.64 |
| white | 3344 | 277 | 197 | 54 | 8.28 | 75.48 |


**cl_race**

| cl_race | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| asian | 720 | 79 | 51 | 18 | 10.97 | 21.53 |
| black | 208 | 12 | 11 | 2 | 5.77 | 3.27 |
| others | 201 | 9 | 20 | 6 | 4.48 | 2.45 |
| white | 3409 | 267 | 209 | 58 | 7.83 | 72.75 |


**age4**

| age4 | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| 30-44 | 1451 | 122 | 92 | 23 | 8.41 | 33.24 |
| 45-59 | 509 | 29 | 29 | 7 | 5.7 | 7.9 |
| 60+ | 34 | 0 | 6 | 0 | 0.0 | 0.0 |
| <30 | 2544 | 216 | 140 | 47 | 8.49 | 58.86 |


**ita4**

| ita4 | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| Q1 darkest | 1138 | 90 | 72 | 21 | 7.91 | 24.52 |
| Q2 | 1112 | 66 | 65 | 17 | 5.94 | 17.98 |
| Q3 | 1032 | 96 | 55 | 18 | 9.3 | 26.16 |
| Q4 lightest | 1165 | 114 | 68 | 20 | 9.78 | 31.06 |
| missing | 91 | 1 | 7 | 1 | 1.1 | 0.27 |


## native: false negatives by attribute


**method**

| method | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| Deepfakes | 3880 | 85 | 130 | 10 | 2.19 | 4.27 |
| Face2Face | 4373 | 324 | 146 | 36 | 7.41 | 16.27 |
| FaceShifter | 4280 | 314 | 143 | 47 | 7.34 | 15.77 |
| FaceSwap | 3822 | 209 | 128 | 38 | 5.47 | 10.5 |
| NeuralTextures | 4508 | 1059 | 151 | 93 | 23.49 | 53.19 |


**family**

| family | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| identity-swap | 11982 | 608 | 401 | 95 | 5.07 | 30.54 |
| reenactment | 8881 | 1383 | 297 | 129 | 15.57 | 69.46 |


**gname**

| gname | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| F-Asian | 1768 | 74 | 60 | 9 | 4.19 | 3.72 |
| F-Black | 480 | 18 | 16 | 3 | 3.75 | 0.9 |
| F-Others | 690 | 29 | 23 | 4 | 4.2 | 1.46 |
| F-White | 7968 | 469 | 267 | 73 | 5.89 | 23.56 |
| M-Asian | 929 | 218 | 31 | 18 | 23.47 | 10.95 |
| M-Black | 510 | 25 | 17 | 4 | 4.9 | 1.26 |
| M-Others | 990 | 171 | 33 | 11 | 17.27 | 8.59 |
| M-White | 7528 | 987 | 251 | 102 | 13.11 | 49.57 |


**cl_gender**

| cl_gender | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| female | 11113 | 645 | 385 | 101 | 5.8 | 32.4 |
| male | 9750 | 1346 | 331 | 132 | 13.8 | 67.6 |


**cl_race**

| cl_race | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| asian | 2728 | 386 | 136 | 38 | 14.15 | 19.39 |
| black | 857 | 42 | 35 | 7 | 4.9 | 2.11 |
| others | 898 | 78 | 77 | 14 | 8.69 | 3.92 |
| white | 16380 | 1485 | 594 | 185 | 9.07 | 74.59 |


**age4**

| age4 | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| 30-44 | 6563 | 742 | 219 | 80 | 11.31 | 37.27 |
| 45-59 | 2549 | 275 | 85 | 33 | 10.79 | 13.81 |
| 60+ | 30 | 21 | 1 | 1 | 70.0 | 1.06 |
| <30 | 11721 | 953 | 393 | 110 | 8.13 | 47.86 |


**ita4**

| ita4 | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| Q1 darkest | 5758 | 578 | 192 | 64 | 10.04 | 29.03 |
| Q2 | 5066 | 536 | 169 | 64 | 10.58 | 26.92 |
| Q3 | 4648 | 505 | 156 | 54 | 10.86 | 25.36 |
| Q4 lightest | 5108 | 372 | 171 | 42 | 7.28 | 18.68 |
| missing | 283 | 0 | 10 | 0 | 0.0 | 0.0 |


## native: FP video concentration

- real videos with >=1 FP frame: 77 / 267; with >=50% FP frames: 10
- top-5 videos hold 25.6% of FP frames, top-15 hold 56.9%; 12 videos hold half


| top-k FP videos dropped | i-F_FPR (same t) | FPR % (same t) | i-F_FPR (re-thresholded to same FPR) | last dropped (<=5) |
|---|---|---|---|---|
| 0 | 36.11 | 8.09 | 36.11 |  |
| 1 | 34.11 | 7.67 | 35.56 | 845 |
| 2 | 32.64 | 7.3 | 34.3 | 845, 307 |
| 3 | 32.99 | 6.92 | 36.24 | 845, 307, 305 |
| 5 | 31.95 | 6.22 | 34.87 | 845, 307, 305, 383, 018 |
| 10 | 23.34 | 4.83 | 27.57 | 237, 930, 062, 446, 202 |
| 15 | 14.02 | 3.85 | 26.66 | 458, 616, 438, 348, 385 |
| 20 | 11.5 | 3.21 | 27.76 | 472, 288, 722, 985, 547 |


random-drop control, re-thresholded to same FPR (200 draws): k=5: median 36.65 [p5 34.88, p95 38.81]; k=10: median 36.79 [p5 33.98, p95 40.52]


| group | group_FPR_% | |FPR_g-FPR_all| | video-weighted FPR % | videos with FP | top-3 video share of group FPs % | Ding FPR_g % | Ding |dev| |
|---|---|---|---|---|---|---|---|
| F-Asian | 15.22 | 7.13 | 9.48 | 11/23 | 63.49 | 10.87 | 1.67 |
| F-Black | 10.17 | 2.08 | 6.9 | 2/6 | 100.0 | 18.64 | 6.11 |
| F-Others | 2.65 | 5.44 | 1.52 | 3/9 | 100.0 | 14.57 | 2.03 |
| F-White | 7.29 | 0.79 | 4.06 | 26/109 | 30.47 | 11.4 | 1.14 |
| M-Asian | 3.78 | 4.31 | 9.81 | 5/13 | 77.78 | 18.49 | 5.95 |
| M-Black | 0.0 | 8.09 | 0.0 | 0/7 | 0.0 | 8.7 | 3.84 |
| M-Others | 1.1 | 6.98 | 8.62 | 2/12 | 100.0 | 16.02 | 3.48 |
| M-White | 9.38 | 1.29 | 6.82 | 28/88 | 39.6 | 12.52 | 0.01 |


**top-15 FP videos (native)**

| video | group | cl_gender | cl_race | ff_race7 | age_vid | ita4 | n | n_fp | fp_rate_% | ding_fp_rate_% | max_logit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0_youtube_845 | M-White | male | white | White | 42.0 | Q1 darkest | 29 | 21 | 72.41 | 72.41 | 31.92 |
| 0_youtube_307 | M-White | male | white | Middle Eastern | 55.5 | Q4 lightest | 30 | 19 | 63.33 | 90.0 | 46.22 |
| 0_youtube_305 | M-White | female | asian | Middle Eastern | 28.0 | Q4 lightest | 30 | 19 | 63.33 | 46.67 | 29.74 |
| 0_youtube_383 | M-White | male | white | Middle Eastern | 34.0 | Q3 | 30 | 19 | 63.33 | 6.67 | 12.44 |
| 0_youtube_018 | F-White | female | white | White | 31.0 | Q2 | 29 | 16 | 55.17 | 72.41 | 47.02 |
| 0_youtube_237 | M-White | male | white | White | 24.0 | Q3 | 30 | 16 | 53.33 | 3.33 | 28.78 |
| 0_youtube_930 | F-Asian | female | asian | East Asian | 18.0 | Q4 lightest | 28 | 16 | 57.14 | 17.86 | 24.07 |
| 0_youtube_062 | F-Asian | female | asian | East Asian | 31.0 | Q3 | 29 | 13 | 44.83 | 0.0 | 27.5 |
| 0_youtube_446 | F-White | female | white | White | 20.0 | Q4 lightest | 30 | 12 | 40.0 | 13.33 | 38.15 |
| 0_youtube_202 | F-White | female | white | Middle Eastern | 27.0 | Q1 darkest | 29 | 11 | 37.93 | 6.9 | 14.5 |
| 0_youtube_458 | F-Black | female | black | Black | 14.0 | Q1 darkest | 29 | 11 | 37.93 | 48.28 | 1.36 |
| 0_youtube_616 | F-Asian | female | asian | East Asian | 37.0 | Q2 | 30 | 11 | 36.67 | 60.0 | -41.47 |
| 0_youtube_438 | F-White | female | white | White | 29.0 | Q4 lightest | 29 | 10 | 34.48 | 20.69 | 33.59 |
| 0_youtube_348 | F-White | female | white | White | 25.0 | Q2 | 27 | 8 | 29.63 | 11.11 | 24.68 |
| 0_youtube_385 | F-White | female | asian | Latino_Hispanic | 27.0 | Q4 lightest | 30 | 7 | 23.33 | 10.0 | -24.86 |


**top-15 FN videos (native)**

| video | method | group | cl_gender | cl_race | age_vid | ita4 | n | n_fn | fn_rate_% | ding_fn_rate_% | min_logit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0_Deepfakes_492_325 | Deepfakes | M-White | male | white | 42.0 | Q3 | 30 | 30 | 100.0 | 96.67 | -105.82 |
| 0_NeuralTextures_840_800 | NeuralTextures | M-White | male | white | 33.0 | Q1 darkest | 30 | 30 | 100.0 | 96.67 | -103.29 |
| 0_NeuralTextures_235_194 | NeuralTextures | M-Asian | male | asian | 24.0 | Q3 | 30 | 30 | 100.0 | 96.67 | -99.81 |
| 0_NeuralTextures_794_779 | NeuralTextures | M-Others | male | white | 28.0 | Q2 | 30 | 30 | 100.0 | 100.0 | -96.22 |
| 0_NeuralTextures_578_636 | NeuralTextures | F-White | female | white | 32.5 | Q4 lightest | 30 | 29 | 96.67 | 3.33 | -113.82 |
| 0_NeuralTextures_619_620 | NeuralTextures | M-White | male | white | 54.0 | Q3 | 30 | 29 | 96.67 | 96.67 | -111.46 |
| 0_NeuralTextures_800_840 | NeuralTextures | M-White | male | white | 56.0 | Q1 darkest | 30 | 29 | 96.67 | 10.0 | -110.75 |
| 0_NeuralTextures_306_278 | NeuralTextures | M-White | male | white | 32.5 | Q2 | 30 | 29 | 96.67 | 0.0 | -100.85 |
| 0_NeuralTextures_494_445 | NeuralTextures | M-Others | male | white | 38.0 | Q1 darkest | 30 | 29 | 96.67 | 3.33 | -89.79 |
| 0_NeuralTextures_867_765 | NeuralTextures | F-White | female | white | 25.0 | Q4 lightest | 30 | 28 | 93.33 | 43.33 | -110.34 |
| 0_NeuralTextures_016_209 | NeuralTextures | M-Others | male | asian | 3.0 | Q2 | 30 | 28 | 93.33 | 20.0 | -106.07 |
| 0_NeuralTextures_548_632 | NeuralTextures | M-White | male | white | 35.5 | Q3 | 30 | 28 | 93.33 | 90.0 | -101.77 |
| 0_Face2Face_619_620 | Face2Face | M-White | male | white | 53.0 | Q3 | 30 | 27 | 90.0 | 33.33 | -110.65 |
| 0_NeuralTextures_259_345 | NeuralTextures | M-White | male | white | 10.0 | Q3 | 30 | 27 | 90.0 | 60.0 | -99.11 |
| 0_NeuralTextures_594_530 | NeuralTextures | M-White | male | white | 41.0 | Q1 darkest | 30 | 27 | 90.0 | 90.0 | -97.84 |


**FP ours-only (we flag, Ding says real), top videos (native)** — by group: {'F-Asian': 44, 'F-Black': 4, 'F-Others': 3, 'F-White': 78, 'M-Asian': 4, 'M-Black': 0, 'M-Others': 2, 'M-White': 72}

| video | group | method | n | n_err | rate_% | ours_mean_logit | ding_mean_logit |
|---|---|---|---|---|---|---|---|
| 0_youtube_383 | M-White | real | 30 | 17 | 56.67 | -48.41 | -3.2 |
| 0_youtube_237 | M-White | real | 30 | 15 | 50.0 | -53.75 | -4.63 |
| 0_youtube_062 | F-Asian | real | 29 | 13 | 44.83 | -57.19 | -5.09 |
| 0_youtube_930 | F-Asian | real | 28 | 11 | 39.29 | -56.62 | -2.91 |
| 0_youtube_202 | F-White | real | 29 | 10 | 34.48 | -68.4 | -3.92 |
| 0_youtube_446 | F-White | real | 30 | 9 | 30.0 | -57.43 | -3.54 |
| 0_youtube_305 | M-White | real | 30 | 6 | 20.0 | -48.34 | -1.0 |
| 0_youtube_438 | F-White | real | 29 | 6 | 20.69 | -65.91 | -2.62 |


**FP Ding-only, top videos (native)** — by group: {'F-Asian': 26, 'F-Black': 14, 'F-Others': 21, 'F-White': 150, 'M-Asian': 39, 'M-Black': 8, 'M-Others': 29, 'M-White': 122}

| video | group | method | n | n_err | rate_% | ours_mean_logit | ding_mean_logit |
|---|---|---|---|---|---|---|---|
| 0_youtube_494 | M-Others | real | 29 | 24 | 82.76 | -92.55 | 2.31 |
| 0_youtube_923 | M-White | real | 29 | 19 | 65.52 | -75.79 | -0.26 |
| 0_youtube_763 | F-Others | real | 29 | 13 | 44.83 | -91.24 | -0.07 |
| 0_youtube_602 | F-White | real | 28 | 13 | 46.43 | -96.73 | 0.07 |
| 0_youtube_095 | F-White | real | 30 | 13 | 43.33 | -79.3 | 0.71 |
| 0_youtube_623 | F-White | real | 29 | 12 | 41.38 | -102.71 | -0.7 |
| 0_youtube_606 | F-White | real | 29 | 12 | 41.38 | -103.71 | -0.67 |
| 0_youtube_513 | M-Asian | real | 29 | 11 | 37.93 | -79.31 | -0.81 |


**FN ours-only (we miss, Ding catches), top videos (native)** — by group: {'F-Asian': 37, 'F-Black': 14, 'F-Others': 22, 'F-White': 387, 'M-Asian': 113, 'M-Black': 22, 'M-Others': 114, 'M-White': 690}; by method: {'Deepfakes': 44, 'Face2Face': 240, 'FaceShifter': 253, 'FaceSwap': 166, 'NeuralTextures': 696}

| video | group | method | n | n_err | rate_% | ours_mean_logit | ding_mean_logit |
|---|---|---|---|---|---|---|---|
| 0_NeuralTextures_306_278 | M-White | NeuralTextures | 30 | 29 | 96.67 | -82.18 | 4.92 |
| 0_NeuralTextures_494_445 | M-Others | NeuralTextures | 30 | 28 | 93.33 | -79.75 | 2.85 |
| 0_NeuralTextures_578_636 | F-White | NeuralTextures | 30 | 28 | 93.33 | -89.85 | 4.21 |
| 0_NeuralTextures_800_840 | M-White | NeuralTextures | 30 | 26 | 86.67 | -94.6 | 3.77 |
| 0_NeuralTextures_872_873 | M-White | NeuralTextures | 30 | 26 | 86.67 | -80.73 | 4.03 |
| 0_Face2Face_016_209 | M-Others | Face2Face | 30 | 25 | 83.33 | -85.0 | 3.71 |
| 0_NeuralTextures_194_235 | M-Asian | NeuralTextures | 30 | 25 | 83.33 | -76.46 | 4.0 |
| 0_Deepfakes_923_023 | M-White | Deepfakes | 30 | 24 | 80.0 | -54.81 | 1.42 |


**FN Ding-only, top videos (native)** — by group: {'F-Asian': 48, 'F-Black': 8, 'F-Others': 9, 'F-White': 198, 'M-Asian': 28, 'M-Black': 4, 'M-Others': 22, 'M-White': 166}; by method: {'Deepfakes': 81, 'Face2Face': 64, 'FaceShifter': 55, 'FaceSwap': 56, 'NeuralTextures': 227}

| video | group | method | n | n_err | rate_% | ours_mean_logit | ding_mean_logit |
|---|---|---|---|---|---|---|---|
| 0_NeuralTextures_381_376 | F-White | NeuralTextures | 30 | 25 | 83.33 | -15.32 | -3.75 |
| 0_NeuralTextures_662_641 | F-Asian | NeuralTextures | 28 | 22 | 78.57 | -34.64 | -3.08 |
| 0_NeuralTextures_534_490 | F-White | NeuralTextures | 29 | 21 | 72.41 | 11.05 | -1.1 |
| 0_Face2Face_193_030 | F-White | Face2Face | 30 | 20 | 66.67 | 20.4 | -1.87 |
| 0_Deepfakes_948_965 | F-White | Deepfakes | 30 | 16 | 53.33 | 42.52 | -0.51 |
| 0_NeuralTextures_482_465 | F-White | NeuralTextures | 30 | 15 | 50.0 | -52.82 | -0.77 |
| 0_NeuralTextures_620_619 | M-White | NeuralTextures | 30 | 15 | 50.0 | -55.93 | -1.2 |
| 0_Deepfakes_300_304 | M-Others | Deepfakes | 30 | 14 | 46.67 | 33.97 | 0.23 |


## A1: false positives by attribute


**gname**

| gname | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| F-Asian | 414 | 8 | 23 | 4 | 1.93 | 9.41 |
| F-Black | 118 | 1 | 6 | 1 | 0.85 | 1.18 |
| F-Others | 151 | 0 | 9 | 0 | 0.0 | 0.0 |
| F-White | 1755 | 36 | 109 | 14 | 2.05 | 42.35 |
| M-Asian | 238 | 2 | 13 | 2 | 0.84 | 2.35 |
| M-Black | 92 | 0 | 7 | 0 | 0.0 | 0.0 |
| M-Others | 181 | 0 | 12 | 0 | 0.0 | 0.0 |
| M-White | 1589 | 38 | 88 | 10 | 2.39 | 44.71 |


**xu_gender**

| xu_gender | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| female | 2438 | 45 | 147 | 19 | 1.85 | 52.94 |
| male | 2100 | 40 | 120 | 12 | 1.9 | 47.06 |


**cl_gender**

| cl_gender | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| female | 2497 | 47 | 152 | 21 | 1.88 | 55.29 |
| male | 2041 | 38 | 120 | 11 | 1.86 | 44.71 |


**xu_race**

| xu_race | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| asian | 652 | 10 | 36 | 6 | 1.53 | 11.76 |
| black | 210 | 1 | 13 | 1 | 0.48 | 1.18 |
| others | 332 | 0 | 21 | 0 | 0.0 | 0.0 |
| white | 3344 | 74 | 197 | 24 | 2.21 | 87.06 |


**cl_race**

| cl_race | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| asian | 720 | 11 | 51 | 7 | 1.53 | 12.94 |
| black | 208 | 1 | 11 | 1 | 0.48 | 1.18 |
| others | 201 | 2 | 20 | 2 | 1.0 | 2.35 |
| white | 3409 | 71 | 209 | 24 | 2.08 | 83.53 |


**age4**

| age4 | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| 30-44 | 1451 | 37 | 92 | 9 | 2.55 | 43.53 |
| 45-59 | 509 | 7 | 29 | 2 | 1.38 | 8.24 |
| 60+ | 34 | 0 | 6 | 0 | 0.0 | 0.0 |
| <30 | 2544 | 41 | 140 | 20 | 1.61 | 48.24 |


**ita4**

| ita4 | n | n_err | n_videos | n_videos_with_err | rate_% | share_of_errors_% |
|---|---|---|---|---|---|---|
| Q1 darkest | 1138 | 25 | 72 | 10 | 2.2 | 29.41 |
| Q2 | 1112 | 16 | 65 | 5 | 1.44 | 18.82 |
| Q3 | 1032 | 21 | 55 | 7 | 2.04 | 24.71 |
| Q4 lightest | 1165 | 23 | 68 | 9 | 1.97 | 27.06 |
| missing | 91 | 0 | 7 | 0 | 0.0 | 0.0 |


## A1: FP video concentration

- real videos with >=1 FP frame: 31 / 267; with >=50% FP frames: 0
- top-5 videos hold 44.7% of FP frames, top-15 hold 81.2%; 6 videos hold half


| top-k FP videos dropped | i-F_FPR (same t) | FPR % (same t) | i-F_FPR (re-thresholded to same FPR) | last dropped (<=5) |
|---|---|---|---|---|
| 0 | 8.43 | 1.87 | 8.43 |  |
| 1 | 7.2 | 1.62 | 8.84 | 845 |
| 2 | 6.54 | 1.47 | 8.08 | 845, 018 |
| 3 | 5.99 | 1.33 | 8.89 | 845, 018, 383 |
| 5 | 5.01 | 1.07 | 10.7 | 845, 018, 383, 307, 348 |
| 10 | 3.16 | 0.64 | 13.75 | 446, 237, 062, 722, 305 |
| 15 | 2.42 | 0.39 | 12.73 | 019, 930, 202, 547, 323 |
| 20 | 1.95 | 0.28 | 13.32 | 095, 438, 288, 345, 736 |


random-drop control, re-thresholded to same FPR (200 draws): k=5: median 8.50 [p5 8.32, p95 8.91]; k=10: median 8.56 [p5 8.22, p95 9.33]


| group | group_FPR_% | |FPR_g-FPR_all| | video-weighted FPR % | videos with FP | top-3 video share of group FPs % | Ding FPR_g % | Ding |dev| |
|---|---|---|---|---|---|---|---|
| F-Asian | 1.93 | 0.06 | 1.21 | 4/23 | 87.5 | 10.87 | 1.67 |
| F-Black | 0.85 | 1.03 | 0.57 | 1/6 | 100.0 | 18.64 | 6.11 |
| F-Others | 0.0 | 1.87 | 0.0 | 0/9 | 0.0 | 14.57 | 2.03 |
| F-White | 2.05 | 0.18 | 1.14 | 14/109 | 50.0 | 11.4 | 1.14 |
| M-Asian | 0.84 | 1.03 | 0.53 | 2/13 | 100.0 | 18.49 | 5.95 |
| M-Black | 0.0 | 1.87 | 0.0 | 0/7 | 0.0 | 8.7 | 3.84 |
| M-Others | 0.0 | 1.87 | 0.0 | 0/12 | 0.0 | 16.02 | 3.48 |
| M-White | 2.39 | 0.52 | 1.46 | 10/88 | 65.79 | 12.52 | 0.01 |


**top-15 FP videos (A1)**

| video | group | cl_gender | cl_race | ff_race7 | age_vid | ita4 | n | n_fp | fp_rate_% | ding_fp_rate_% | max_logit |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0_youtube_845 | M-White | male | white | White | 42.0 | Q1 darkest | 29 | 12 | 41.38 | 72.41 | 31.92 |
| 0_youtube_018 | F-White | female | white | White | 31.0 | Q2 | 29 | 7 | 24.14 | 72.41 | 47.02 |
| 0_youtube_383 | M-White | male | white | Middle Eastern | 34.0 | Q3 | 30 | 7 | 23.33 | 6.67 | 12.44 |
| 0_youtube_307 | M-White | male | white | Middle Eastern | 55.5 | Q4 lightest | 30 | 6 | 20.0 | 90.0 | 46.22 |
| 0_youtube_348 | F-White | female | white | White | 25.0 | Q2 | 27 | 6 | 22.22 | 11.11 | 24.68 |
| 0_youtube_446 | F-White | female | white | White | 20.0 | Q4 lightest | 30 | 5 | 16.67 | 13.33 | 38.15 |
| 0_youtube_237 | M-White | male | white | White | 24.0 | Q3 | 30 | 4 | 13.33 | 3.33 | 28.78 |
| 0_youtube_062 | F-Asian | female | asian | East Asian | 31.0 | Q3 | 29 | 4 | 13.79 | 0.0 | 27.5 |
| 0_youtube_722 | F-White | female | white | White | 25.0 | Q1 darkest | 29 | 4 | 13.79 | 51.72 | 22.96 |
| 0_youtube_305 | M-White | female | asian | Middle Eastern | 28.0 | Q4 lightest | 30 | 3 | 10.0 | 46.67 | 29.74 |
| 0_youtube_019 | F-White | female | white | White | 31.0 | Q4 lightest | 30 | 3 | 10.0 | 46.67 | 20.55 |
| 0_youtube_930 | F-Asian | female | asian | East Asian | 18.0 | Q4 lightest | 28 | 2 | 7.14 | 17.86 | 24.07 |
| 0_youtube_202 | F-White | female | white | Middle Eastern | 27.0 | Q1 darkest | 29 | 2 | 6.9 | 6.9 | 14.5 |
| 0_youtube_547 | M-White | male | white | Middle Eastern | 23.5 | Q3 | 30 | 2 | 6.67 | 43.33 | -2.17 |
| 0_youtube_323 | F-White | female | white | White | 25.0 | Q3 | 29 | 2 | 6.9 | 0.0 | -5.97 |


**FP ours-only (we flag, Ding says real), top videos (A1)** — by group: {'F-Asian': 5, 'F-Black': 1, 'F-Others': 0, 'F-White': 15, 'M-Asian': 0, 'M-Black': 0, 'M-Others': 0, 'M-White': 16}

| video | group | method | n | n_err | rate_% | ours_mean_logit | ding_mean_logit |
|---|---|---|---|---|---|---|---|
| 0_youtube_383 | M-White | real | 30 | 6 | 20.0 | -48.41 | -3.2 |
| 0_youtube_062 | F-Asian | real | 29 | 4 | 13.79 | -57.19 | -5.09 |
| 0_youtube_237 | M-White | real | 30 | 4 | 13.33 | -53.75 | -4.63 |
| 0_youtube_348 | F-White | real | 27 | 3 | 11.11 | -67.31 | -3.54 |
| 0_youtube_446 | F-White | real | 30 | 3 | 10.0 | -57.43 | -3.54 |
| 0_youtube_845 | M-White | real | 29 | 2 | 6.9 | -34.05 | 1.1 |
| 0_youtube_323 | F-White | real | 29 | 2 | 6.9 | -90.28 | -5.46 |
| 0_youtube_202 | F-White | real | 29 | 1 | 3.45 | -68.4 | -3.92 |


**FP Ding-only, top videos (A1)** — by group: {'F-Asian': 42, 'F-Black': 22, 'F-Others': 22, 'F-White': 179, 'M-Asian': 42, 'M-Black': 8, 'M-Others': 29, 'M-White': 177}

| video | group | method | n | n_err | rate_% | ours_mean_logit | ding_mean_logit |
|---|---|---|---|---|---|---|---|
| 0_youtube_923 | M-White | real | 29 | 24 | 82.76 | -75.79 | -0.26 |
| 0_youtube_494 | M-Others | real | 29 | 24 | 82.76 | -92.55 | 2.31 |
| 0_youtube_307 | M-White | real | 30 | 21 | 70.0 | -46.59 | 2.13 |
| 0_youtube_616 | F-Asian | real | 30 | 18 | 60.0 | -70.26 | 0.95 |
| 0_youtube_095 | F-White | real | 30 | 16 | 53.33 | -79.3 | 0.71 |
| 0_youtube_909 | M-White | real | 29 | 14 | 48.28 | -75.07 | -0.19 |
| 0_youtube_458 | F-Black | real | 29 | 14 | 48.28 | -69.26 | 0.0 |
| 0_youtube_763 | F-Others | real | 29 | 14 | 48.28 | -91.24 | -0.07 |


## FN by target identity (matched; fake video Method_TTT_SSS alters target TTT)

- 101/192 targets have >=1 FN; 15 targets hold half of all FNs; top-10 targets hold 38.9%; 5 targets are missed (>=50% frames) under >=2 methods
- over 152 targets whose real video is in test: Spearman(real-video mean logit, FN rate of its fakes) = -0.446 (ours), -0.369 (Ding); Spearman(real mean logit, fake mean logit) = 0.453

| target | group | n_fake | n_fn | fn_rate_% | methods_ge50pct_fn | fake_mean_logit | real_mean_logit | real_fp_rate | ding_fn_rate |
|---|---|---|---|---|---|---|---|---|---|
| 705 | M-White | 150 | 74 | 49.33 | 3 | -20.74 | -90.48 | 0.03 | 0.0 |
| 259 | M-White | 150 | 70 | 46.67 | 2 | -55.94 | -101.08 | 0.0 | 38.67 |
| 923 | M-White | 150 | 61 | 40.67 | 1 | -56.85 | -75.79 | 0.45 | 5.33 |
| 016 | M-Others | 150 | 56 | 37.33 | 2 | -13.36 | -108.84 | 0.0 | 4.67 |
| 800 | M-White | 150 | 54 | 36.0 | 1 | -27.36 | -116.83 | 0.0 | 2.67 |
| 567 | F-White | 150 | 54 | 36.0 | 1 | -51.69 | nan | nan | 16.0 |
| 306 | M-White | 120 | 51 | 42.5 | 1 | -37.97 | -104.51 | 0.0 | 0.0 |
| 619 | M-White | 149 | 49 | 32.89 | 2 | -15.95 | -114.32 | 0.0 | 28.19 |
| 492 | M-White | 150 | 48 | 32.0 | 2 | -33.05 | -87.99 | 0.14 | 32.67 |
| 867 | F-White | 60 | 41 | 68.33 | 1 | -65.8 | nan | nan | 21.67 |
