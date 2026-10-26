# BD0 seed 7 on Celeb-DF: failure cases

The model is BD0 s7 (CLIP ViT-L/14 with rank-1 adapters, CFES-D, Arm E base and ESA). The scores come from `runs/BD0_s7/eval/celebdf_scores.npz`. Ding is our reproduction of Ding's model (official code, Xception, s7, `full_s7_ep49_celebdf.npz`), which decides at z >= 0.

The script is `analyze_celebdf_failures.py`. Every number below comes from that script; the raw values are in `failures.json` and the per-axis CSVs are in `tables/`. The script recomputes F_FPR, FPR and TPR at all three thresholds and asserts that they match `bd_eval.json` (test.celebdf) to 1e-3.

**Test set.** Celeb-DF has 5,713 real frames in 192 videos (141 Celeb-real, 51 YouTube-real) and 22,745 fake frames in 764 videos. All fakes come from one method, Celeb-synthesis face swap. That swap only happens within the 15 test identities id47 to id61.

**Group labels.** In the Xu labels, the M-Black group is three people (id59, id60 and id61, 24 real videos). F-Others is three YouTube videos. M-White and M-Others have no fakes, so the metric skips them for intersectional F_FPR, but their reals still count in the overall FPR.

**Operating points:**

| threshold | t (logit) | FP | FPR % | FN | FNR % | iF_FPR |
|---|---|---|---|---|---|---|
| matched (primary) | -73.76 | 3349 | 58.62 | 1015 | 4.46 | 32.48 |
| native | -65.97 | 2325 | 40.70 | 2089 | 9.18 | 31.83 |
| A1 | -19.13 | 582 | 10.19 | 8007 | 35.20 | 22.99 |
| Ding repro (z >= 0) | 0 | 3976 | 69.60 | 2164 | 9.51 | 47.59 |

Contact sheets (matched threshold, one frame per video, the most confident error in each video):
- `fp_sheet.png`: 24 real videos. 18 are F-White, 5 M-White and 1 M-Black, and Ding also flags 23 of the 24.
- `fn_sheet.png`: 24 fake videos. 16 are M-Black and 8 F-White, and Ding catches 7 of the 24.

## Failure modes, in plain words

1. **At the matched threshold most of the failure is a threshold that does not transfer, not a few bad videos.**
   - The matched threshold (-73.76) sits just below the median Celeb-DF real logit (-70.15). It therefore flags 58.6% of real frames.
   - Those false positives are spread out. 183 of 192 real videos have at least one FP and 126 are flagged on 50% of frames or more. The top 15 videos hold only 13.3% of all FP frames, and it takes 62 videos to reach half of them.
   - Celeb-real (58.6%) and YouTube-real (58.8%) are hit equally.
   - Ding is worse on the same reals (69.6% FPR).
   - The native threshold gives 40.7% FPR and A1 gives 10.2%. A1 pays for that with a 35.2% miss rate.

2. **Intersectional F_FPR at matched (32.48) comes from the model flagging Black men less, not from any group being over-flagged.**
   - M-Black real FPR is 36.3%, against 58.6% overall and 63.0% for F-White. That one group term is 22.28 of the 32.48.
   - The drop comes mostly from two of the three people. id60 is flagged on 12.6% of real frames and id61 on 27.3%; id59 is at 59.5%.
   - Removing the single worst FP video changes the metric by less than 0.3 (to 32.70 or 32.41).
   - Halving it takes 8 video removals. Seven of them are M-Black videos with only 0 to 3 FP frames each; the eighth is YouTube-real_00262.
   - A null that shuffles group labels across real videos gives a mean of 19.94 (5 to 95% range 5.17 to 39.10). The observed 32.48 is not unusual under that null (p = 0.139).

3. **At A1, F_FPR (22.99) rests on 3 videos.**
   - The F-Others group is three YouTube videos of Asian women (by the cleaner AI-Face labels; 90 frames). Its FPR is 30.0% against 10.2% overall, which makes its term 19.81 of the 22.99.
   - Removing two of those videos (YouTube-real_00262 and 00255) drops iF_FPR to 6.03.
   - The shuffle null gives p = 0.036 here.
   - This is the same kind of failure as the FF++ non-male-Asian false positives found earlier.

4. **Missed fakes pile up on the Black male identities, and the scores for those faces are pushed toward "real" as a whole.**
   - At matched, M-Black fakes are missed on 24.4% of frames, against 3.3% for F-White.
     - That group is 5.3% of the fakes but 29.2% of the misses.
     - Ding misses them at a similar rate (25.9% against 8.6%).
   - The worst target identity is id60, missed on 40.0% of its frames.
   - The worst swap pair is id60->id61 at 50.0% (Ding 51.2%).
   - The whole score distribution shifts for this group:
     - The median real logit is -79.5 for M-Black against -68.9 for F-White.
     - The median fake logit is -28.6 against 19.9.
     - AUC within the group is 76.5 against 88.4.
   - At A1 this gets worse: M-Black misses rise to 53.6% (id60 to 71.0%), while Ding stays at 25.9%.

5. **The first frame of each video is read as real by both models (a keyframe effect).**
   - Frame index 1 is missed on 35.5% of fakes (271 of 764) at matched, against 3.4% for later frames. Those 271 frames are 26.7% of all misses.
   - The same frames lower the real FPR: 22.9% against 59.9%.
   - Ding shows the same pattern, even more strongly: 63.9% against 7.6% on fakes and 25.5% against 71.1% on reals.
   - At A1, 89.9% of first frames of fakes are missed and 0 first frames of reals are flagged.
   - Both detectors appear to read per-frame compression or quality as evidence of forgery.

6. **Missed fakes are often shared with Ding; false positives overlap only a little above chance.**
   - At matched, Ding also misses 41.7% of our missed fakes. If the overlap were chance, that would be 9.5% (Ding's own miss rate), so this is about 4.4 times chance. Hard fakes are hard for both.
   - Ding also flags 76.8% of our false positives, against a chance level of 69.6%.
   - Errors only we make: 778 FP and 592 FN frames. Errors only Ding makes: 1,405 FP and 1,741 FN frames.
   - At video level, 22 real videos are flagged on at least half their frames by us but on less than half by Ding. 16 of them are F-White Celeb-real, for example Celeb-real_id57_0001 (90% against 7%) and Celeb-real_id55_0006 (100% against 15%).
   - 5 fake videos are missed on at least half their frames by us and caught by Ding. 4 of the 5 have id60 as the target.
   - At A1 our misses are mostly our own: 6,421 FN frames Ding catches, and 229 fake videos we miss on at least half their frames while Ding does not.

7. **The most confident errors are a few specific videos.**
   - 9 of the 12 highest-scoring false positives are frames of YouTube-real_00247, a man filmed in broadcast/news footage. They score logits of 59 to 72 and Ding flags them too.
   - 10 of the 12 lowest-scoring missed fakes are M-Black swaps (logits -92 to -104).
   - Looking at `fp_sheet.png`, the false positives are mostly soft-focus or low-resolution, heavily made-up, broadcast-style frames. This is a visual impression, not a measurement.

8. **Group labels are noisy and the groups are tiny, so per-group claims are fragile.**
   - The Xu labels disagree with the cleaner AI-Face labels on 6 of the 192 real videos. One example is YouTube-real_00263: Xu calls it M-White, the cleaner labels say non-male/others, and the frame on the sheet shows an older woman. It is flagged on 100% of frames.
   - Gender gaps hold under both label sets. FPR is 48.2% for male against 63.0% for non-male with Xu labels, and 47.3% against 63.3% with the cleaner labels.
   - Skin tone is flat with the FF++-train ITA edges (56.2% to 65.2%). With Celeb-DF's own quartiles, the darkest quartile is lowest at 43.7%, which matches mode 2.

---
# Full tables (auto-generated by the script)

## A. Operating points

| threshold | t | FP | FPR | FN | FNR | FP_videos | FN_videos | gF_FPR | rF_FPR | iF_FPR |
|---|---|---|---|---|---|---|---|---|---|---|
| matched | -73.76 | 3349 | 58.62 | 1015 | 4.46 | 183 | 349 | 14.86 | 37.86 | 32.48 |
| native | -65.97 | 2325 | 40.70 | 2089 | 9.18 | 179 | 518 | 7.08 | 34.84 | 31.83 |
| A1 | -19.13 | 582 | 10.19 | 8007 | 35.20 | 98 | 715 | 2.81 | 14.47 | 22.99 |
| Ding repro (z>=0) | 0.00 | 3976 | 69.60 | 2164 | 9.51 |  |  | 6.96 | 45.47 | 47.59 |

Real frames 5713 in 192 videos ({'Celeb-real': 141, 'YouTube-real': 51}); fake frames 22745 in 764 videos, one method ({'Celeb-synthesis': 22745}).


## B-matched. False positives by axis (matched, t = -73.76)


**Xu intersectional group (metric)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| M-Others | 60 | 2 | 49 | 2 | 81.7 | 90.0 |
| F-Others | 90 | 3 | 58 | 3 | 64.4 | 96.7 |
| F-White | 3929 | 132 | 2475 | 128 | 63.0 | 71.1 |
| M-White | 924 | 31 | 509 | 30 | 55.1 | 73.9 |
| M-Black | 710 | 24 | 258 | 20 | 36.3 | 50.6 |

**Xu gender (metric)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| nonmale | 4019 | 135 | 2533 | 131 | 63.0 | 71.7 |
| male | 1694 | 57 | 816 | 52 | 48.2 | 64.7 |

**Xu race (metric)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| others | 150 | 5 | 107 | 5 | 71.3 | 94.0 |
| white | 4853 | 163 | 2984 | 158 | 61.5 | 71.6 |
| black | 710 | 24 | 258 | 20 | 36.3 | 50.6 |

**AI-Face cleaner gender (frame)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| nonmale | 4046 | 136 | 2560 | 132 | 63.3 | 71.7 |
| male | 1667 | 57 | 789 | 52 | 47.3 | 64.6 |

**AI-Face cleaner race (frame)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| others | 24 | 4 | 24 | 4 | 100.0 | 70.8 |
| asian | 161 | 6 | 118 | 6 | 73.3 | 93.2 |
| white | 4817 | 163 | 2948 | 158 | 61.2 | 71.6 |
| black | 711 | 25 | 259 | 21 | 36.4 | 50.6 |

**AI-Face cleaner race (video majority)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| others | 30 | 1 | 30 | 1 | 100.0 | 73.3 |
| asian | 150 | 5 | 107 | 5 | 71.3 | 94.0 |
| white | 4823 | 162 | 2954 | 157 | 61.2 | 71.6 |
| black | 710 | 24 | 258 | 20 | 36.3 | 50.6 |

**InsightFace age, video median**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| <30 | 3031 | 102 | 1845 | 99 | 60.9 | 72.0 |
| 30-44 | 2267 | 76 | 1312 | 72 | 57.9 | 68.3 |
| 60+ | 56 | 2 | 31 | 2 | 55.4 | 58.9 |
| 45-59 | 359 | 12 | 161 | 10 | 44.8 | 58.8 |

**ITA skin tone, FF++-train quartile edges**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| no ITA | 30 | 1 | 20 | 1 | 66.7 | 36.7 |
| Q2 | 1212 | 41 | 790 | 40 | 65.2 | 74.8 |
| Q4 lightest | 1101 | 37 | 637 | 35 | 57.9 | 62.3 |
| Q3 | 718 | 24 | 412 | 23 | 57.4 | 73.5 |
| Q1 darkest | 2652 | 89 | 1490 | 84 | 56.2 | 69.6 |

**ITA skin tone, Celeb-DF real-video quartiles**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Q2 | 1405 | 47 | 983 | 47 | 70.0 | 72.7 |
| no ITA | 30 | 1 | 20 | 1 | 66.7 | 36.7 |
| Q3 | 1420 | 48 | 882 | 46 | 62.1 | 74.0 |
| Q4 lightest | 1431 | 48 | 841 | 46 | 58.8 | 65.8 |
| Q1 darkest | 1427 | 48 | 623 | 43 | 43.7 | 66.7 |

**FairFace race7 (frame)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Indian | 1 | 1 | 1 | 1 | 100.0 | 100.0 |
| East Asian | 144 | 13 | 105 | 11 | 72.9 | 86.8 |
| Latino_Hispanic | 77 | 15 | 55 | 13 | 71.4 | 79.2 |
| Middle Eastern | 340 | 51 | 228 | 43 | 67.1 | 72.1 |
| White | 4449 | 165 | 2713 | 156 | 61.0 | 71.6 |
| Southeast Asian | 27 | 6 | 16 | 6 | 59.3 | 77.8 |
| Black | 675 | 25 | 231 | 20 | 34.2 | 49.8 |

**FairFace sex (frame)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Female | 4027 | 140 | 2552 | 133 | 63.4 | 71.7 |
| Male | 1686 | 64 | 797 | 58 | 47.3 | 64.6 |

**real source**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| YouTube-real | 1517 | 51 | 892 | 48 | 58.8 | 74.1 |
| Celeb-real | 4196 | 141 | 2457 | 135 | 58.6 | 68.0 |

**frame position in video**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| later frames | 5521 | 192 | 3305 | 183 | 59.9 | 71.1 |
| first frame (idx 1) | 192 | 192 | 44 | 44 | 22.9 | 25.5 |

## B-native. False positives by axis (native, t = -65.97)


**Xu intersectional group (metric)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| M-Others | 60 | 2 | 39 | 2 | 65.0 | 90.0 |
| F-Others | 90 | 3 | 51 | 3 | 56.7 | 96.7 |
| F-White | 3929 | 132 | 1669 | 125 | 42.5 | 71.1 |
| M-White | 924 | 31 | 377 | 29 | 40.8 | 73.9 |
| M-Black | 710 | 24 | 189 | 20 | 26.6 | 50.6 |

**Xu gender (metric)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| nonmale | 4019 | 135 | 1720 | 128 | 42.8 | 71.7 |
| male | 1694 | 57 | 605 | 51 | 35.7 | 64.7 |

**Xu race (metric)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| others | 150 | 5 | 90 | 5 | 60.0 | 94.0 |
| white | 4853 | 163 | 2046 | 154 | 42.2 | 71.6 |
| black | 710 | 24 | 189 | 20 | 26.6 | 50.6 |

**AI-Face cleaner gender (frame)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| nonmale | 4046 | 136 | 1749 | 129 | 43.2 | 71.7 |
| male | 1667 | 57 | 576 | 50 | 34.6 | 64.6 |

**AI-Face cleaner race (frame)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| others | 24 | 4 | 18 | 3 | 75.0 | 70.8 |
| asian | 161 | 6 | 101 | 6 | 62.7 | 93.2 |
| white | 4817 | 163 | 2017 | 154 | 41.9 | 71.6 |
| black | 711 | 25 | 189 | 20 | 26.6 | 50.6 |

**AI-Face cleaner race (video majority)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| others | 30 | 1 | 29 | 1 | 96.7 | 73.3 |
| asian | 150 | 5 | 90 | 5 | 60.0 | 94.0 |
| white | 4823 | 162 | 2017 | 153 | 41.8 | 71.6 |
| black | 710 | 24 | 189 | 20 | 26.6 | 50.6 |

**InsightFace age, video median**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| 60+ | 56 | 2 | 25 | 2 | 44.6 | 58.9 |
| <30 | 3031 | 102 | 1319 | 95 | 43.5 | 72.0 |
| 30-44 | 2267 | 76 | 883 | 72 | 39.0 | 68.3 |
| 45-59 | 359 | 12 | 98 | 10 | 27.3 | 58.8 |

**ITA skin tone, FF++-train quartile edges**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| no ITA | 30 | 1 | 15 | 1 | 50.0 | 36.7 |
| Q2 | 1212 | 41 | 527 | 40 | 43.5 | 74.8 |
| Q4 lightest | 1101 | 37 | 448 | 33 | 40.7 | 62.3 |
| Q1 darkest | 2652 | 89 | 1055 | 83 | 39.8 | 69.6 |
| Q3 | 718 | 24 | 280 | 22 | 39.0 | 73.5 |

**ITA skin tone, Celeb-DF real-video quartiles**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Q2 | 1405 | 47 | 708 | 47 | 50.4 | 72.7 |
| no ITA | 30 | 1 | 15 | 1 | 50.0 | 36.7 |
| Q4 lightest | 1431 | 48 | 594 | 43 | 41.5 | 65.8 |
| Q3 | 1420 | 48 | 585 | 46 | 41.2 | 74.0 |
| Q1 darkest | 1427 | 48 | 423 | 42 | 29.6 | 66.7 |

**FairFace race7 (frame)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Indian | 1 | 1 | 1 | 1 | 100.0 | 100.0 |
| East Asian | 144 | 13 | 84 | 10 | 58.3 | 86.8 |
| Latino_Hispanic | 77 | 15 | 43 | 11 | 55.8 | 79.2 |
| Middle Eastern | 340 | 51 | 146 | 34 | 42.9 | 72.1 |
| White | 4449 | 165 | 1875 | 151 | 42.1 | 71.6 |
| Southeast Asian | 27 | 6 | 9 | 6 | 33.3 | 77.8 |
| Black | 675 | 25 | 167 | 20 | 24.7 | 49.8 |

**FairFace sex (frame)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Female | 4027 | 140 | 1741 | 130 | 43.2 | 71.7 |
| Male | 1686 | 64 | 584 | 55 | 34.6 | 64.6 |

**real source**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| YouTube-real | 1517 | 51 | 689 | 47 | 45.4 | 74.1 |
| Celeb-real | 4196 | 141 | 1636 | 132 | 39.0 | 68.0 |

**frame position in video**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| later frames | 5521 | 192 | 2305 | 179 | 41.7 | 71.1 |
| first frame (idx 1) | 192 | 192 | 20 | 20 | 10.4 | 25.5 |

## B-A1. False positives by axis (A1, t = -19.13)


**Xu intersectional group (metric)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| F-Others | 90 | 3 | 27 | 3 | 30.0 | 96.7 |
| M-White | 924 | 31 | 140 | 18 | 15.2 | 73.9 |
| M-Others | 60 | 2 | 7 | 2 | 11.7 | 90.0 |
| F-White | 3929 | 132 | 349 | 64 | 8.9 | 71.1 |
| M-Black | 710 | 24 | 59 | 11 | 8.3 | 50.6 |

**Xu gender (metric)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| male | 1694 | 57 | 206 | 31 | 12.2 | 64.7 |
| nonmale | 4019 | 135 | 376 | 67 | 9.4 | 71.7 |

**Xu race (metric)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| others | 150 | 5 | 34 | 5 | 22.7 | 94.0 |
| white | 4853 | 163 | 489 | 82 | 10.1 | 71.6 |
| black | 710 | 24 | 59 | 11 | 8.3 | 50.6 |

**AI-Face cleaner gender (frame)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| male | 1667 | 57 | 191 | 30 | 11.5 | 64.6 |
| nonmale | 4046 | 136 | 391 | 68 | 9.7 | 71.7 |

**AI-Face cleaner race (frame)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| others | 24 | 4 | 12 | 2 | 50.0 | 70.8 |
| asian | 161 | 6 | 37 | 6 | 23.0 | 93.2 |
| white | 4817 | 163 | 474 | 81 | 9.8 | 71.6 |
| black | 711 | 25 | 59 | 11 | 8.3 | 50.6 |

**AI-Face cleaner race (video majority)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| others | 30 | 1 | 15 | 1 | 50.0 | 73.3 |
| asian | 150 | 5 | 34 | 5 | 22.7 | 94.0 |
| white | 4823 | 162 | 474 | 81 | 9.8 | 71.6 |
| black | 710 | 24 | 59 | 11 | 8.3 | 50.6 |

**InsightFace age, video median**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| <30 | 3031 | 102 | 358 | 54 | 11.8 | 72.0 |
| 30-44 | 2267 | 76 | 197 | 34 | 8.7 | 68.3 |
| 60+ | 56 | 2 | 4 | 2 | 7.1 | 58.9 |
| 45-59 | 359 | 12 | 23 | 8 | 6.4 | 58.8 |

**ITA skin tone, FF++-train quartile edges**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Q1 darkest | 2652 | 89 | 317 | 48 | 12.0 | 69.6 |
| Q2 | 1212 | 41 | 135 | 26 | 11.1 | 74.8 |
| Q3 | 718 | 24 | 66 | 14 | 9.2 | 73.5 |
| Q4 lightest | 1101 | 37 | 64 | 10 | 5.8 | 62.3 |
| no ITA | 30 | 1 | 0 | 0 | 0.0 | 36.7 |

**ITA skin tone, Celeb-DF real-video quartiles**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Q2 | 1405 | 47 | 227 | 29 | 16.2 | 72.7 |
| Q3 | 1420 | 48 | 159 | 30 | 11.2 | 74.0 |
| Q1 darkest | 1427 | 48 | 103 | 22 | 7.2 | 66.7 |
| Q4 lightest | 1431 | 48 | 93 | 17 | 6.5 | 65.8 |
| no ITA | 30 | 1 | 0 | 0 | 0.0 | 36.7 |

**FairFace race7 (frame)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Indian | 1 | 1 | 1 | 1 | 100.0 | 100.0 |
| Latino_Hispanic | 77 | 15 | 17 | 5 | 22.1 | 79.2 |
| East Asian | 144 | 13 | 20 | 5 | 13.9 | 86.8 |
| Middle Eastern | 340 | 51 | 45 | 11 | 13.2 | 72.1 |
| White | 4449 | 165 | 447 | 81 | 10.0 | 71.6 |
| Black | 675 | 25 | 50 | 10 | 7.4 | 49.8 |
| Southeast Asian | 27 | 6 | 2 | 2 | 7.4 | 77.8 |

**FairFace sex (frame)**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Male | 1686 | 64 | 193 | 32 | 11.4 | 64.6 |
| Female | 4027 | 140 | 389 | 68 | 9.7 | 71.7 |

**real source**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| YouTube-real | 1517 | 51 | 241 | 34 | 15.9 | 74.1 |
| Celeb-real | 4196 | 141 | 341 | 64 | 8.1 | 68.0 |

**frame position in video**

| value | n_real | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| later frames | 5521 | 192 | 582 | 98 | 10.5 | 71.1 |
| first frame (idx 1) | 192 | 192 | 0 | 0 | 0.0 | 25.5 |

## C-matched. FP concentration by video (matched)

183/192 real videos have >= 1 FP frame; 126 have FP rate >= 50%; 15 are flagged on every frame. Top-k videos' share of all FP frames: top-1 0.9%, top-3 2.7%, top-5 4.5%, top-10 9.0%, top-15 13.3%, top-20 17.6%. Videos needed for 50% / 80% of FPs: 62 / 112.


Top-15 real videos by FP frames

| vid_short | gname | cl_gender_vid | cl_race_vid | age_vid_if | ita_vid | n | fp | fp_rate% | ding_fp_rate% | max_logit |
|---|---|---|---|---|---|---|---|---|---|---|
| YouTube-real_00247 | M-White | male | white | 29.0 | -3.0 | 30 | 30 | 100.0 | 100.0 | 72.2 |
| Celeb-real_id48_0007 | F-White | nonmale | white | 26.5 | 7.4 | 30 | 30 | 100.0 | 86.7 | 63.2 |
| YouTube-real_00263 | M-White | nonmale | others | 27.0 | -18.1 | 30 | 30 | 100.0 | 73.3 | 46.4 |
| Celeb-real_id57_0000 | F-White | nonmale | white | 32.0 | 68.8 | 30 | 30 | 100.0 | 73.3 | 44.7 |
| YouTube-real_00268 | M-White | male | white | 28.0 | 14.9 | 30 | 30 | 100.0 | 93.3 | 43.9 |
| YouTube-real_00291 | F-White | nonmale | white | 23.0 | 74.3 | 30 | 30 | 100.0 | 83.3 | 35.3 |
| Celeb-real_id54_0009 | F-White | nonmale | white | 26.0 | -28.3 | 30 | 30 | 100.0 | 66.7 | 18.6 |
| YouTube-real_00281 | F-White | nonmale | white | 24.0 | 9.2 | 30 | 30 | 100.0 | 93.3 | 1.4 |
| Celeb-real_id56_0009 | F-White | nonmale | white | 23.0 | 57.0 | 30 | 30 | 100.0 | 96.7 | -26.5 |
| Celeb-real_id52_0003 | F-White | nonmale | white | 33.0 | 20.6 | 30 | 30 | 100.0 | 100.0 | -41.0 |
| Celeb-real_id51_0007 | F-White | nonmale | white | 33.0 | -8.8 | 30 | 30 | 100.0 | 93.3 | -46.8 |
| Celeb-real_id50_0007 | F-White | nonmale | white | 25.0 | 13.7 | 30 | 29 | 96.7 | 96.7 | 67.8 |
| Celeb-real_id47_0006 | F-White | nonmale | white | 34.0 | 26.6 | 30 | 29 | 96.7 | 96.7 | 52.9 |
| Celeb-real_id54_0001 | F-White | nonmale | white | 37.0 | 56.1 | 29 | 29 | 100.0 | 93.1 | 43.5 |
| Celeb-real_id49_0002 | F-White | nonmale | white | 23.0 | 30.3 | 30 | 29 | 96.7 | 90.0 | 40.7 |

Intersectional F_FPR = 32.48. Greedy removal of the real video that most reduces it:

| k | removed | group | fp | n | iF_FPR_after |
|---|---|---|---|---|---|
| 1 | YouTube-real_00262 | F-Others | 21 | 30 | 29.76 |
| 2 | Celeb-real_id61_0008 | M-Black | 0 | 30 | 27.85 |
| 3 | Celeb-real_id60_0007 | M-Black | 0 | 30 | 25.78 |
| 4 | Celeb-real_id61_0004 | M-Black | 0 | 30 | 23.54 |
| 5 | Celeb-real_id60_0002 | M-Black | 0 | 30 | 21.11 |
| 6 | Celeb-real_id59_0004 | M-Black | 1 | 30 | 18.64 |
| 7 | Celeb-real_id60_0000 | M-Black | 3 | 30 | 16.33 |
| 8 | Celeb-real_id59_0002 | M-Black | 3 | 30 | 13.78 |
| 9 | Celeb-real_id60_0001 | M-Black | 3 | 30 | 10.93 |
| 10 | Celeb-real_id60_0006 | M-Black | 2 | 27 | 7.90 |

Videos to remove to halve iF_FPR: 8.


Permutation null (Xu group labels shuffled across the 192 real videos, fakes fixed, 2000 perms): mean 19.94, 5-95% [5.17, 39.10]; observed 32.48; P(null >= observed) = 0.139.


Per-group FP video structure (video FP rates, %, descending)

| group | real_videos | fp | videos_with_fp | top1_share | top3_share | video_fp_rates |
|---|---|---|---|---|---|---|
| F-Others | 3 | 58 | 3 | 36.206896551724135 | 100.0 | 70, 63, 60 |
| F-White | 132 | 2475 | 128 | 1.2121212121212122 | 3.6363636363636362 | 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100 ... |
| M-Black | 24 | 258 | 20 | 11.24031007751938 | 30.62015503875969 | 97, 87, 80, 80, 67, 63, 62, 57, 53, 41, 37, 30 ... |
| M-Others | 2 | 49 | 2 | 57.142857142857146 | 100.0 | 93, 70 |
| M-White | 31 | 509 | 30 | 5.893909626719057 | 17.68172888015717 | 100, 100, 100, 93, 93, 87, 80, 80, 80, 80, 80, 73 ... |

Group terms of intersectional F_FPR (|FPR_g - FPR_all|, %)

| group (both classes) | FPR | n_real | term |
|---|---|---|---|
| male,black | 36.34 | 710.00 | 22.28 |
| nonmale,white | 62.99 | 3929.00 | 4.37 |
| nonmale,others | 64.44 | 90.00 | 5.82 |

## C-native. FP concentration by video (native)

179/192 real videos have >= 1 FP frame; 81 have FP rate >= 50%; 3 are flagged on every frame. Top-k videos' share of all FP frames: top-1 1.3%, top-3 3.9%, top-5 6.4%, top-10 12.5%, top-15 18.3%, top-20 23.6%. Videos needed for 50% / 80% of FPs: 49 / 93.


Top-15 real videos by FP frames

| vid_short | gname | cl_gender_vid | cl_race_vid | age_vid_if | ita_vid | n | fp | fp_rate% | ding_fp_rate% | max_logit |
|---|---|---|---|---|---|---|---|---|---|---|
| Celeb-real_id48_0007 | F-White | nonmale | white | 26.5 | 7.4 | 30 | 30 | 100.0 | 86.7 | 63.2 |
| YouTube-real_00291 | F-White | nonmale | white | 23.0 | 74.3 | 30 | 30 | 100.0 | 83.3 | 35.3 |
| Celeb-real_id54_0009 | F-White | nonmale | white | 26.0 | -28.3 | 30 | 30 | 100.0 | 66.7 | 18.6 |
| YouTube-real_00247 | M-White | male | white | 29.0 | -3.0 | 30 | 29 | 96.7 | 100.0 | 72.2 |
| YouTube-real_00263 | M-White | nonmale | others | 27.0 | -18.1 | 30 | 29 | 96.7 | 73.3 | 46.4 |
| Celeb-real_id57_0000 | F-White | nonmale | white | 32.0 | 68.8 | 30 | 29 | 96.7 | 73.3 | 44.7 |
| YouTube-real_00268 | M-White | male | white | 28.0 | 14.9 | 30 | 29 | 96.7 | 93.3 | 43.9 |
| Celeb-real_id54_0001 | F-White | nonmale | white | 37.0 | 56.1 | 29 | 28 | 96.5 | 93.1 | 43.5 |
| Celeb-real_id48_0006 | F-White | nonmale | white | 28.5 | -1.7 | 30 | 28 | 93.3 | 80.0 | 18.4 |
| YouTube-real_00281 | F-White | nonmale | white | 24.0 | 9.2 | 30 | 28 | 93.3 | 93.3 | 1.4 |
| Celeb-real_id54_0003 | F-White | nonmale | white | 29.0 | 54.8 | 30 | 28 | 93.3 | 96.7 | -19.5 |
| Celeb-real_id56_0009 | F-White | nonmale | white | 23.0 | 57.0 | 30 | 28 | 93.3 | 96.7 | -26.5 |
| Celeb-real_id59_0005 | M-Black | male | black | 42.5 | -24.4 | 30 | 27 | 90.0 | 93.3 | 33.0 |
| Celeb-real_id50_0007 | F-White | nonmale | white | 25.0 | 13.7 | 30 | 26 | 86.7 | 96.7 | 67.8 |
| Celeb-real_id54_0005 | F-White | nonmale | white | 28.0 | 24.9 | 27 | 26 | 96.3 | 81.5 | 57.5 |

Intersectional F_FPR = 31.83. Greedy removal of the real video that most reduces it:

| k | removed | group | fp | n | iF_FPR_after |
|---|---|---|---|---|---|
| 1 | YouTube-real_00262 | F-Others | 20 | 30 | 26.97 |
| 2 | YouTube-real_00255 | F-Others | 16 | 30 | 25.37 |
| 3 | YouTube-real_00276 | F-Others | 15 | 30 | 15.86 |
| 4 | Celeb-real_id61_0008 | M-Black | 0 | 30 | 14.68 |
| 5 | Celeb-real_id60_0007 | M-Black | 0 | 30 | 13.40 |
| 6 | Celeb-real_id61_0004 | M-Black | 0 | 30 | 12.00 |
| 7 | Celeb-real_id60_0002 | M-Black | 0 | 30 | 10.45 |
| 8 | Celeb-real_id59_0004 | M-Black | 1 | 30 | 8.91 |
| 9 | Celeb-real_id59_0002 | M-Black | 1 | 30 | 7.20 |
| 10 | Celeb-real_id60_0001 | M-Black | 1 | 30 | 5.28 |

Videos to remove to halve iF_FPR: 3.


Permutation null (Xu group labels shuffled across the 192 real videos, fakes fixed, 2000 perms): mean 19.52, 5-95% [5.51, 38.01]; observed 31.83; P(null >= observed) = 0.138.


Per-group FP video structure (video FP rates, %, descending)

| group | real_videos | fp | videos_with_fp | top1_share | top3_share | video_fp_rates |
|---|---|---|---|---|---|---|
| F-Others | 3 | 51 | 3 | 39.21568627450981 | 100.0 | 67, 53, 50 |
| F-White | 132 | 1669 | 125 | 1.7974835230677053 | 5.392450569203116 | 100, 100, 100, 97, 97, 96, 93, 93, 93, 93, 92, 87 ... |
| M-Black | 24 | 189 | 20 | 14.285714285714286 | 38.095238095238095 | 90, 77, 73, 67, 58, 53, 50, 30, 27, 23, 21, 17 ... |
| M-Others | 2 | 39 | 2 | 58.97435897435897 | 100.0 | 77, 53 |
| M-White | 31 | 377 | 29 | 7.6923076923076925 | 23.076923076923077 | 97, 97, 97, 80, 73, 67, 63, 60, 57, 57, 57, 57 ... |

Group terms of intersectional F_FPR (|FPR_g - FPR_all|, %)

| group (both classes) | FPR | n_real | term |
|---|---|---|---|
| male,black | 26.62 | 710.00 | 14.08 |
| nonmale,white | 42.48 | 3929.00 | 1.78 |
| nonmale,others | 56.67 | 90.00 | 15.97 |

## C-A1. FP concentration by video (A1)

98/192 real videos have >= 1 FP frame; 10 have FP rate >= 50%; 0 are flagged on every frame. Top-k videos' share of all FP frames: top-1 4.3%, top-3 11.0%, top-5 17.0%, top-10 30.1%, top-15 40.7%, top-20 49.1%. Videos needed for 50% / 80% of FPs: 21 / 45.


Top-15 real videos by FP frames

| vid_short | gname | cl_gender_vid | cl_race_vid | age_vid_if | ita_vid | n | fp | fp_rate% | ding_fp_rate% | max_logit |
|---|---|---|---|---|---|---|---|---|---|---|
| YouTube-real_00247 | M-White | male | white | 29.0 | -3.0 | 30 | 25 | 83.3 | 100.0 | 72.2 |
| YouTube-real_00291 | F-White | nonmale | white | 23.0 | 74.3 | 30 | 21 | 70.0 | 83.3 | 35.3 |
| Celeb-real_id54_0005 | F-White | nonmale | white | 28.0 | 24.9 | 27 | 18 | 66.7 | 81.5 | 57.5 |
| YouTube-real_00278 | M-White | male | white | 27.5 | -1.1 | 30 | 18 | 60.0 | 96.7 | 19.7 |
| YouTube-real_00268 | M-White | male | white | 28.0 | 14.9 | 30 | 17 | 56.7 | 93.3 | 43.9 |
| Celeb-real_id48_0006 | F-White | nonmale | white | 28.5 | -1.7 | 30 | 16 | 53.3 | 80.0 | 18.4 |
| Celeb-real_id47_0009 | F-White | nonmale | white | 30.0 | 16.9 | 30 | 15 | 50.0 | 96.7 | 57.4 |
| Celeb-real_id47_0006 | F-White | nonmale | white | 34.0 | 26.6 | 30 | 15 | 50.0 | 96.7 | 52.9 |
| YouTube-real_00263 | M-White | nonmale | others | 27.0 | -18.1 | 30 | 15 | 50.0 | 73.3 | 46.4 |
| Celeb-real_id59_0005 | M-Black | male | black | 42.5 | -24.4 | 30 | 15 | 50.0 | 93.3 | 33.0 |
| Celeb-real_id50_0007 | F-White | nonmale | white | 25.0 | 13.7 | 30 | 14 | 46.7 | 96.7 | 67.8 |
| YouTube-real_00262 | F-Others | nonmale | asian | 23.0 | 38.5 | 30 | 14 | 46.7 | 96.7 | 19.3 |
| YouTube-real_00279 | M-White | male | white | 37.0 | -6.8 | 30 | 13 | 43.3 | 96.7 | 42.8 |
| Celeb-real_id48_0005 | F-White | nonmale | white | 25.5 | 15.0 | 30 | 11 | 36.7 | 66.7 | 20.6 |
| Celeb-real_id48_0007 | F-White | nonmale | white | 26.5 | 7.4 | 30 | 10 | 33.3 | 86.7 | 63.2 |

Intersectional F_FPR = 22.99. Greedy removal of the real video that most reduces it:

| k | removed | group | fp | n | iF_FPR_after |
|---|---|---|---|---|---|
| 1 | YouTube-real_00262 | F-Others | 14 | 30 | 14.47 |
| 2 | YouTube-real_00255 | F-Others | 9 | 30 | 6.03 |
| 3 | YouTube-real_00276 | F-Others | 4 | 30 | 2.55 |
| 4 | YouTube-real_00247 | M-White | 25 | 30 | 1.76 |
| 5 | YouTube-real_00278 | M-White | 18 | 30 | 1.21 |
| 6 | YouTube-real_00268 | M-White | 17 | 30 | 0.70 |
| 7 | Celeb-real_id60_0003 | M-Black | 0 | 30 | 0.43 |
| 8 | Celeb-real_id60_0006 | M-Black | 0 | 27 | 0.16 |
| 9 | YouTube-real_00249 | M-White | 9 | 30 | 0.15 |
| 10 | Celeb-real_id54_0003 | F-White | 0 | 30 | 0.08 |

Videos to remove to halve iF_FPR: 2.


Permutation null (Xu group labels shuffled across the 192 real videos, fakes fixed, 2000 perms): mean 10.37, 5-95% [2.54, 21.05]; observed 22.99; P(null >= observed) = 0.036.


Per-group FP video structure (video FP rates, %, descending)

| group | real_videos | fp | videos_with_fp | top1_share | top3_share | video_fp_rates |
|---|---|---|---|---|---|---|
| F-Others | 3 | 27 | 3 | 51.851851851851855 | 100.0 | 47, 30, 13 |
| F-White | 132 | 349 | 64 | 6.017191977077364 | 15.759312320916905 | 70, 67, 53, 50, 50, 47, 37, 33, 33, 33, 33, 33 ... |
| M-Black | 24 | 59 | 11 | 25.423728813559322 | 52.54237288135593 | 50, 30, 29, 23, 23, 17, 10, 7, 7, 3, 3, 0 ... |
| M-Others | 2 | 7 | 2 | 57.142857142857146 | 100.0 | 13, 10 |
| M-White | 31 | 140 | 18 | 17.857142857142858 | 42.857142857142854 | 83, 60, 57, 50, 43, 30, 30, 27, 23, 20, 13, 7 ... |

Group terms of intersectional F_FPR (|FPR_g - FPR_all|, %)

| group (both classes) | FPR | n_real | term |
|---|---|---|---|
| male,black | 8.31 | 710.00 | 1.88 |
| nonmale,white | 8.88 | 3929.00 | 1.30 |
| nonmale,others | 30.00 | 90.00 | 19.81 |

## D-matched. False negatives (matched)


**Xu intersectional group (metric)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| M-Black | 1212 | 41 | 296 | 38 | 24.4 | 25.9 |
| F-White | 21503 | 722 | 719 | 311 | 3.3 | 8.6 |
| F-Others | 30 | 1 | 0 | 0 | 0.0 | 0.0 |

**Xu gender**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| male | 1212 | 41 | 296 | 38 | 24.4 | 25.9 |
| nonmale | 21533 | 723 | 719 | 311 | 3.3 | 8.6 |

**Xu race**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| black | 1212 | 41 | 296 | 38 | 24.4 | 25.9 |
| white | 21503 | 722 | 719 | 311 | 3.3 | 8.6 |
| others | 30 | 1 | 0 | 0 | 0.0 | 0.0 |

**AI-Face cleaner gender (frame)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| male | 1212 | 41 | 296 | 38 | 24.4 | 25.9 |
| nonmale | 21533 | 723 | 719 | 311 | 3.3 | 8.6 |

**AI-Face cleaner race (frame)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| black | 1212 | 41 | 296 | 38 | 24.4 | 25.9 |
| asian | 38 | 8 | 3 | 2 | 7.9 | 2.6 |
| white | 21495 | 723 | 716 | 311 | 3.3 | 8.6 |

**InsightFace age, video median**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| 45-59 | 504 | 17 | 113 | 16 | 22.4 | 31.5 |
| 30-44 | 9052 | 303 | 498 | 148 | 5.5 | 9.7 |
| <30 | 13189 | 444 | 404 | 185 | 3.1 | 8.6 |

**ITA skin tone (FF++-train edges)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Q1 darkest | 8454 | 283 | 588 | 165 | 7.0 | 10.8 |
| Q3 | 3762 | 126 | 144 | 67 | 3.8 | 8.1 |
| Q4 lightest | 5210 | 174 | 158 | 56 | 3.0 | 10.2 |
| Q2 | 5229 | 178 | 125 | 61 | 2.4 | 7.6 |
| no ITA | 90 | 3 | 0 | 0 | 0.0 | 25.6 |

**ITA skin tone (Celeb-DF quartiles)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Q1 darkest | 4236 | 142 | 442 | 98 | 10.4 | 13.3 |
| Q4 lightest | 6380 | 213 | 213 | 82 | 3.3 | 9.6 |
| Q2 | 5070 | 170 | 159 | 75 | 3.1 | 8.2 |
| Q3 | 6969 | 236 | 201 | 94 | 2.9 | 7.9 |
| no ITA | 90 | 3 | 0 | 0 | 0.0 | 25.6 |

**FairFace race7 (frame)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Black | 1137 | 41 | 278 | 37 | 24.5 | 24.2 |
| Southeast Asian | 45 | 18 | 11 | 4 | 24.4 | 44.4 |
| Latino_Hispanic | 130 | 42 | 13 | 10 | 10.0 | 13.8 |
| Middle Eastern | 428 | 128 | 23 | 21 | 5.4 | 13.3 |
| East Asian | 148 | 38 | 7 | 5 | 4.7 | 12.8 |
| White | 20842 | 727 | 683 | 306 | 3.3 | 8.5 |
| Indian | 15 | 4 | 0 | 0 | 0.0 | 46.7 |

**target identity (video whose face is replaced)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| id60 | 480 | 16 | 192 | 15 | 40.0 | 30.6 |
| id61 | 294 | 10 | 44 | 9 | 15.0 | 15.3 |
| id59 | 438 | 15 | 60 | 14 | 13.7 | 27.9 |
| id52 | 2364 | 79 | 138 | 40 | 5.8 | 6.5 |
| id50 | 2092 | 70 | 101 | 39 | 4.8 | 7.3 |
| id51 | 2009 | 67 | 91 | 46 | 4.5 | 7.6 |
| id57 | 1710 | 57 | 58 | 22 | 3.4 | 8.4 |
| id56 | 2247 | 77 | 75 | 34 | 3.3 | 5.4 |
| id48 | 300 | 10 | 10 | 4 | 3.3 | 15.0 |
| id53 | 2489 | 83 | 75 | 27 | 3.0 | 12.1 |
| id55 | 1892 | 64 | 56 | 35 | 3.0 | 13.2 |
| id58 | 1470 | 49 | 36 | 23 | 2.4 | 9.5 |
| id49 | 2288 | 77 | 56 | 26 | 2.4 | 9.9 |
| id54 | 2372 | 80 | 23 | 15 | 1.0 | 6.2 |
| id47 | 300 | 10 | 0 | 0 | 0.0 | 6.7 |

**source identity (face swapped in)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| id61 | 474 | 16 | 157 | 15 | 33.1 | 47.3 |
| id59 | 387 | 13 | 89 | 11 | 23.0 | 11.6 |
| id60 | 351 | 12 | 50 | 12 | 14.2 | 12.8 |
| id50 | 1754 | 59 | 82 | 27 | 4.7 | 6.3 |
| id55 | 2413 | 81 | 101 | 40 | 4.2 | 14.7 |
| id58 | 1966 | 66 | 77 | 29 | 3.9 | 7.4 |
| id51 | 2141 | 72 | 83 | 36 | 3.9 | 5.6 |
| id57 | 2264 | 76 | 87 | 34 | 3.8 | 12.1 |
| id56 | 2334 | 78 | 84 | 38 | 3.6 | 9.1 |
| id47 | 300 | 10 | 10 | 4 | 3.3 | 15.0 |
| id53 | 2026 | 68 | 62 | 27 | 3.1 | 6.0 |
| id49 | 1903 | 64 | 52 | 28 | 2.7 | 2.9 |
| id52 | 1901 | 64 | 38 | 22 | 2.0 | 11.6 |
| id54 | 2231 | 75 | 43 | 26 | 1.9 | 7.8 |
| id48 | 300 | 10 | 0 | 0 | 0.0 | 6.7 |

**frame position in video**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| first frame (idx 1) | 764 | 764 | 271 | 271 | 35.5 | 63.9 |
| later frames | 21981 | 764 | 744 | 232 | 3.4 | 7.6 |

349/764 fake videos have >= 1 FN frame; 7 have FN rate >= 50%. top-1 3.0%, top-5 11.4%, top-15 23.3%, top-50 46.3% of FN frames.


Top-15 fake videos by FN frames

| vid_short | gname | tgt_id | src_id | cl_race_vid | n | fn | fn_rate% | ding_fn_rate% | min_logit |
|---|---|---|---|---|---|---|---|---|---|
| Celeb-synthesis_id60_id61_0004 | M-Black | id60 | id61 | black | 30 | 30 | 100.0 | 13.3 | -103.7 |
| Celeb-synthesis_id60_id59_0004 | M-Black | id60 | id59 | black | 30 | 25 | 83.3 | 0.0 | -92.6 |
| Celeb-synthesis_id60_id61_0002 | M-Black | id60 | id61 | black | 30 | 23 | 76.7 | 33.3 | -98.3 |
| Celeb-synthesis_id60_id61_0007 | M-Black | id60 | id61 | black | 30 | 21 | 70.0 | 86.7 | -95.1 |
| Celeb-synthesis_id52_id58_0002 | F-White | id52 | id58 | white | 30 | 17 | 56.7 | 3.3 | -90.0 |
| Celeb-synthesis_id59_id61_0004 | M-Black | id59 | id61 | black | 30 | 15 | 50.0 | 60.0 | -101.4 |
| Celeb-synthesis_id60_id59_0001 | M-Black | id60 | id59 | black | 30 | 15 | 50.0 | 23.3 | -93.9 |
| Celeb-synthesis_id61_id59_0005 | M-Black | id61 | id59 | black | 30 | 13 | 43.3 | 3.3 | -86.3 |
| Celeb-synthesis_id59_id61_0007 | M-Black | id59 | id61 | black | 30 | 12 | 40.0 | 33.3 | -92.3 |
| Celeb-synthesis_id60_id61_0000 | M-Black | id60 | id61 | black | 30 | 12 | 40.0 | 56.7 | -89.8 |
| Celeb-synthesis_id60_id59_0002 | M-Black | id60 | id59 | black | 30 | 11 | 36.7 | 6.7 | -101.8 |
| Celeb-synthesis_id60_id59_0006 | M-Black | id60 | id59 | black | 30 | 11 | 36.7 | 20.0 | -92.1 |
| Celeb-synthesis_id60_id61_0003 | M-Black | id60 | id61 | black | 30 | 11 | 36.7 | 56.7 | -85.9 |
| Celeb-synthesis_id59_id60_0004 | M-Black | id59 | id60 | black | 30 | 10 | 33.3 | 3.3 | -93.5 |
| Celeb-synthesis_id53_id56_0007 | F-White | id53 | id56 | white | 30 | 10 | 33.3 | 16.7 | -85.9 |

Swap pairs with FN rate >= 50%: 1/94. Top-10 pairs:

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| id60->id61 | 240 | 8 | 120 | 8 | 50.0 | 51.2 |
| id60->id59 | 240 | 8 | 72 | 7 | 30.0 | 10.0 |
| id61->id60 | 147 | 5 | 27 | 5 | 18.4 | 16.3 |
| id59->id61 | 234 | 8 | 37 | 7 | 15.8 | 43.2 |
| id61->id59 | 147 | 5 | 17 | 4 | 11.6 | 14.3 |
| id59->id60 | 204 | 7 | 23 | 7 | 11.3 | 10.3 |
| id51->id50 | 240 | 8 | 20 | 7 | 8.3 | 4.6 |
| id52->id55 | 298 | 10 | 23 | 7 | 7.7 | 14.1 |
| id52->id58 | 269 | 9 | 20 | 3 | 7.4 | 4.1 |
| id52->id50 | 300 | 10 | 21 | 5 | 7.0 | 6.7 |

## D-native. False negatives (native)


**Xu intersectional group (metric)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| M-Black | 1212 | 41 | 391 | 39 | 32.3 | 25.9 |
| F-White | 21503 | 722 | 1698 | 479 | 7.9 | 8.6 |
| F-Others | 30 | 1 | 0 | 0 | 0.0 | 0.0 |

**Xu gender**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| male | 1212 | 41 | 391 | 39 | 32.3 | 25.9 |
| nonmale | 21533 | 723 | 1698 | 479 | 7.9 | 8.6 |

**Xu race**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| black | 1212 | 41 | 391 | 39 | 32.3 | 25.9 |
| white | 21503 | 722 | 1698 | 479 | 7.9 | 8.6 |
| others | 30 | 1 | 0 | 0 | 0.0 | 0.0 |

**AI-Face cleaner gender (frame)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| male | 1212 | 41 | 391 | 39 | 32.3 | 25.9 |
| nonmale | 21533 | 723 | 1698 | 479 | 7.9 | 8.6 |

**AI-Face cleaner race (frame)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| black | 1212 | 41 | 391 | 39 | 32.3 | 25.9 |
| asian | 38 | 8 | 9 | 3 | 23.7 | 2.6 |
| white | 21495 | 723 | 1689 | 479 | 7.9 | 8.6 |

**InsightFace age, video median**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| 45-59 | 504 | 17 | 157 | 16 | 31.2 | 31.5 |
| 30-44 | 9052 | 303 | 952 | 213 | 10.5 | 9.7 |
| <30 | 13189 | 444 | 980 | 289 | 7.4 | 8.6 |

**ITA skin tone (FF++-train edges)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Q1 darkest | 8454 | 283 | 1079 | 214 | 12.8 | 10.8 |
| Q3 | 3762 | 126 | 305 | 95 | 8.1 | 8.1 |
| Q4 lightest | 5210 | 174 | 385 | 105 | 7.4 | 10.2 |
| Q2 | 5229 | 178 | 320 | 104 | 6.1 | 7.6 |
| no ITA | 90 | 3 | 0 | 0 | 0.0 | 25.6 |

**ITA skin tone (Celeb-DF quartiles)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Q1 darkest | 4236 | 142 | 739 | 116 | 17.4 | 13.3 |
| Q4 lightest | 6380 | 213 | 489 | 137 | 7.7 | 9.6 |
| Q2 | 5070 | 170 | 385 | 116 | 7.6 | 8.2 |
| Q3 | 6969 | 236 | 476 | 149 | 6.8 | 7.9 |
| no ITA | 90 | 3 | 0 | 0 | 0.0 | 25.6 |

**FairFace race7 (frame)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Black | 1137 | 41 | 366 | 38 | 32.2 | 24.2 |
| Southeast Asian | 45 | 18 | 14 | 6 | 31.1 | 44.4 |
| Latino_Hispanic | 130 | 42 | 17 | 13 | 13.1 | 13.8 |
| Middle Eastern | 428 | 128 | 50 | 36 | 11.7 | 13.3 |
| East Asian | 148 | 38 | 13 | 8 | 8.8 | 12.8 |
| White | 20842 | 727 | 1628 | 472 | 7.8 | 8.5 |
| Indian | 15 | 4 | 1 | 1 | 6.7 | 46.7 |

**target identity (video whose face is replaced)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| id60 | 480 | 16 | 231 | 15 | 48.1 | 30.6 |
| id61 | 294 | 10 | 65 | 10 | 22.1 | 15.3 |
| id59 | 438 | 15 | 95 | 14 | 21.7 | 27.9 |
| id52 | 2364 | 79 | 287 | 60 | 12.1 | 6.5 |
| id50 | 2092 | 70 | 203 | 50 | 9.7 | 7.3 |
| id56 | 2247 | 77 | 213 | 53 | 9.5 | 5.4 |
| id57 | 1710 | 57 | 151 | 28 | 8.8 | 8.4 |
| id55 | 1892 | 64 | 166 | 52 | 8.8 | 13.2 |
| id51 | 2009 | 67 | 153 | 60 | 7.6 | 7.6 |
| id53 | 2489 | 83 | 185 | 54 | 7.4 | 12.1 |
| id49 | 2288 | 77 | 168 | 45 | 7.3 | 9.9 |
| id48 | 300 | 10 | 20 | 7 | 6.7 | 15.0 |
| id58 | 1470 | 49 | 83 | 35 | 5.6 | 9.5 |
| id54 | 2372 | 80 | 66 | 33 | 2.8 | 6.2 |
| id47 | 300 | 10 | 3 | 2 | 1.0 | 6.7 |

**source identity (face swapped in)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| id61 | 474 | 16 | 200 | 15 | 42.2 | 47.3 |
| id59 | 387 | 13 | 117 | 12 | 30.2 | 11.6 |
| id60 | 351 | 12 | 74 | 12 | 21.1 | 12.8 |
| id50 | 1754 | 59 | 213 | 47 | 12.1 | 6.3 |
| id51 | 2141 | 72 | 208 | 47 | 9.7 | 5.6 |
| id55 | 2413 | 81 | 225 | 52 | 9.3 | 14.7 |
| id57 | 2264 | 76 | 192 | 57 | 8.5 | 12.1 |
| id53 | 2026 | 68 | 170 | 47 | 8.4 | 6.0 |
| id58 | 1966 | 66 | 163 | 43 | 8.3 | 7.4 |
| id56 | 2334 | 78 | 172 | 51 | 7.4 | 9.1 |
| id47 | 300 | 10 | 20 | 7 | 6.7 | 15.0 |
| id52 | 1901 | 64 | 126 | 39 | 6.6 | 11.6 |
| id49 | 1903 | 64 | 112 | 46 | 5.9 | 2.9 |
| id54 | 2231 | 75 | 94 | 41 | 4.2 | 7.8 |
| id48 | 300 | 10 | 3 | 2 | 1.0 | 6.7 |

**frame position in video**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| first frame (idx 1) | 764 | 764 | 424 | 424 | 55.5 | 63.9 |
| later frames | 21981 | 764 | 1665 | 395 | 7.6 | 7.6 |

518/764 fake videos have >= 1 FN frame; 19 have FN rate >= 50%. top-1 1.4%, top-5 6.0%, top-15 14.7%, top-50 34.6% of FN frames.


Top-15 fake videos by FN frames

| vid_short | gname | tgt_id | src_id | cl_race_vid | n | fn | fn_rate% | ding_fn_rate% | min_logit |
|---|---|---|---|---|---|---|---|---|---|
| Celeb-synthesis_id60_id61_0004 | M-Black | id60 | id61 | black | 30 | 30 | 100.0 | 13.3 | -103.7 |
| Celeb-synthesis_id60_id59_0004 | M-Black | id60 | id59 | black | 30 | 26 | 86.7 | 0.0 | -92.6 |
| Celeb-synthesis_id60_id61_0002 | M-Black | id60 | id61 | black | 30 | 24 | 80.0 | 33.3 | -98.3 |
| Celeb-synthesis_id60_id61_0007 | M-Black | id60 | id61 | black | 30 | 23 | 76.7 | 86.7 | -95.1 |
| Celeb-synthesis_id52_id58_0002 | F-White | id52 | id58 | white | 30 | 23 | 76.7 | 3.3 | -90.0 |
| Celeb-synthesis_id49_id50_0007 | F-White | id49 | id50 | white | 30 | 21 | 70.0 | 0.0 | -87.4 |
| Celeb-synthesis_id60_id61_0000 | M-Black | id60 | id61 | black | 30 | 20 | 66.7 | 56.7 | -89.8 |
| Celeb-synthesis_id61_id59_0005 | M-Black | id61 | id59 | black | 30 | 20 | 66.7 | 3.3 | -86.3 |
| Celeb-synthesis_id53_id51_0007 | F-White | id53 | id51 | white | 30 | 20 | 66.7 | 3.3 | -83.9 |
| Celeb-synthesis_id60_id61_0003 | M-Black | id60 | id61 | black | 30 | 18 | 60.0 | 56.7 | -85.9 |
| Celeb-synthesis_id60_id59_0001 | M-Black | id60 | id59 | black | 30 | 17 | 56.7 | 23.3 | -93.9 |
| Celeb-synthesis_id59_id61_0007 | M-Black | id59 | id61 | black | 30 | 17 | 56.7 | 33.3 | -92.3 |
| Celeb-synthesis_id53_id56_0007 | F-White | id53 | id56 | white | 30 | 17 | 56.7 | 16.7 | -85.9 |
| Celeb-synthesis_id60_id59_0002 | M-Black | id60 | id59 | black | 30 | 16 | 53.3 | 6.7 | -101.8 |
| Celeb-synthesis_id59_id61_0004 | M-Black | id59 | id61 | black | 30 | 16 | 53.3 | 60.0 | -101.4 |

Swap pairs with FN rate >= 50%: 1/94. Top-10 pairs:

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| id60->id61 | 240 | 8 | 143 | 8 | 59.6 | 51.2 |
| id60->id59 | 240 | 8 | 88 | 7 | 36.7 | 10.0 |
| id61->id60 | 147 | 5 | 36 | 5 | 24.5 | 16.3 |
| id59->id61 | 234 | 8 | 57 | 7 | 24.4 | 43.2 |
| id49->id50 | 115 | 4 | 24 | 2 | 20.9 | 2.6 |
| id61->id59 | 147 | 5 | 29 | 5 | 19.7 | 14.3 |
| id59->id60 | 204 | 7 | 38 | 7 | 18.6 | 10.3 |
| id57->id50 | 120 | 4 | 22 | 2 | 18.3 | 5.8 |
| id52->id50 | 300 | 10 | 51 | 9 | 17.0 | 6.7 |
| id56->id50 | 207 | 7 | 33 | 7 | 15.9 | 2.9 |

## D-A1. False negatives (A1)


**Xu intersectional group (metric)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| M-Black | 1212 | 41 | 650 | 41 | 53.6 | 25.9 |
| F-Others | 30 | 1 | 11 | 1 | 36.7 | 0.0 |
| F-White | 21503 | 722 | 7346 | 673 | 34.2 | 8.6 |

**Xu gender**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| male | 1212 | 41 | 650 | 41 | 53.6 | 25.9 |
| nonmale | 21533 | 723 | 7357 | 674 | 34.2 | 8.6 |

**Xu race**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| black | 1212 | 41 | 650 | 41 | 53.6 | 25.9 |
| others | 30 | 1 | 11 | 1 | 36.7 | 0.0 |
| white | 21503 | 722 | 7346 | 673 | 34.2 | 8.6 |

**AI-Face cleaner gender (frame)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| male | 1212 | 41 | 650 | 41 | 53.6 | 25.9 |
| nonmale | 21533 | 723 | 7357 | 674 | 34.2 | 8.6 |

**AI-Face cleaner race (frame)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| asian | 38 | 8 | 29 | 7 | 76.3 | 2.6 |
| black | 1212 | 41 | 650 | 41 | 53.6 | 25.9 |
| white | 21495 | 723 | 7328 | 674 | 34.1 | 8.6 |

**InsightFace age, video median**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| 45-59 | 504 | 17 | 271 | 17 | 53.8 | 31.5 |
| 30-44 | 9052 | 303 | 3345 | 289 | 37.0 | 9.7 |
| <30 | 13189 | 444 | 4391 | 409 | 33.3 | 8.6 |

**ITA skin tone (FF++-train edges)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| no ITA | 90 | 3 | 46 | 3 | 51.1 | 25.6 |
| Q1 darkest | 8454 | 283 | 3263 | 263 | 38.6 | 10.8 |
| Q4 lightest | 5210 | 174 | 1893 | 163 | 36.3 | 10.2 |
| Q3 | 3762 | 126 | 1219 | 123 | 32.4 | 8.1 |
| Q2 | 5229 | 178 | 1586 | 163 | 30.3 | 7.6 |

**ITA skin tone (Celeb-DF quartiles)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| no ITA | 90 | 3 | 46 | 3 | 51.1 | 25.6 |
| Q1 darkest | 4236 | 142 | 1784 | 140 | 42.1 | 13.3 |
| Q4 lightest | 6380 | 213 | 2316 | 202 | 36.3 | 9.6 |
| Q2 | 5070 | 170 | 1766 | 150 | 34.8 | 8.2 |
| Q3 | 6969 | 236 | 2095 | 220 | 30.1 | 7.9 |

**FairFace race7 (frame)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| Black | 1137 | 41 | 612 | 40 | 53.8 | 24.2 |
| Southeast Asian | 45 | 18 | 24 | 11 | 53.3 | 44.4 |
| East Asian | 148 | 38 | 71 | 26 | 48.0 | 12.8 |
| Middle Eastern | 428 | 128 | 168 | 76 | 39.3 | 13.3 |
| White | 20842 | 727 | 7084 | 675 | 34.0 | 8.5 |
| Latino_Hispanic | 130 | 42 | 44 | 24 | 33.8 | 13.8 |
| Indian | 15 | 4 | 4 | 2 | 26.7 | 46.7 |

**target identity (video whose face is replaced)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| id60 | 480 | 16 | 341 | 16 | 71.0 | 30.6 |
| id53 | 2489 | 83 | 1176 | 80 | 47.2 | 12.1 |
| id61 | 294 | 10 | 132 | 10 | 44.9 | 15.3 |
| id56 | 2247 | 77 | 957 | 75 | 42.6 | 5.4 |
| id59 | 438 | 15 | 177 | 15 | 40.4 | 27.9 |
| id52 | 2364 | 79 | 932 | 78 | 39.4 | 6.5 |
| id55 | 1892 | 64 | 698 | 61 | 36.9 | 13.2 |
| id58 | 1470 | 49 | 516 | 47 | 35.1 | 9.5 |
| id49 | 2288 | 77 | 798 | 69 | 34.9 | 9.9 |
| id50 | 2092 | 70 | 704 | 62 | 33.7 | 7.3 |
| id57 | 1710 | 57 | 504 | 47 | 29.5 | 8.4 |
| id48 | 300 | 10 | 77 | 10 | 25.7 | 15.0 |
| id51 | 2009 | 67 | 469 | 66 | 23.3 | 7.6 |
| id54 | 2372 | 80 | 508 | 72 | 21.4 | 6.2 |
| id47 | 300 | 10 | 18 | 7 | 6.0 | 6.7 |

**source identity (face swapped in)**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| id61 | 474 | 16 | 309 | 16 | 65.2 | 47.3 |
| id59 | 387 | 13 | 196 | 13 | 50.6 | 11.6 |
| id51 | 2141 | 72 | 1029 | 70 | 48.1 | 5.6 |
| id50 | 1754 | 59 | 806 | 55 | 46.0 | 6.3 |
| id52 | 1901 | 64 | 804 | 61 | 42.3 | 11.6 |
| id60 | 351 | 12 | 145 | 12 | 41.3 | 12.8 |
| id55 | 2413 | 81 | 846 | 73 | 35.1 | 14.7 |
| id53 | 2026 | 68 | 667 | 65 | 32.9 | 6.0 |
| id56 | 2334 | 78 | 730 | 71 | 31.3 | 9.1 |
| id57 | 2264 | 76 | 696 | 71 | 30.7 | 12.1 |
| id49 | 1903 | 64 | 561 | 60 | 29.5 | 2.9 |
| id54 | 2231 | 75 | 598 | 71 | 26.8 | 7.8 |
| id58 | 1966 | 66 | 525 | 60 | 26.7 | 7.4 |
| id47 | 300 | 10 | 77 | 10 | 25.7 | 15.0 |
| id48 | 300 | 10 | 18 | 7 | 6.0 | 6.7 |

**frame position in video**

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| first frame (idx 1) | 764 | 764 | 687 | 687 | 89.9 | 63.9 |
| later frames | 21981 | 764 | 7320 | 666 | 33.3 | 7.6 |

715/764 fake videos have >= 1 FN frame; 239 have FN rate >= 50%. top-1 0.4%, top-5 1.9%, top-15 5.5%, top-50 17.0% of FN frames.


Top-15 fake videos by FN frames

| vid_short | gname | tgt_id | src_id | cl_race_vid | n | fn | fn_rate% | ding_fn_rate% | min_logit |
|---|---|---|---|---|---|---|---|---|---|
| Celeb-synthesis_id60_id61_0004 | M-Black | id60 | id61 | black | 30 | 30 | 100.0 | 13.3 | -103.7 |
| Celeb-synthesis_id60_id61_0002 | M-Black | id60 | id61 | black | 30 | 30 | 100.0 | 33.3 | -98.3 |
| Celeb-synthesis_id49_id50_0007 | F-White | id49 | id50 | white | 30 | 30 | 100.0 | 0.0 | -87.4 |
| Celeb-synthesis_id61_id59_0005 | M-Black | id61 | id59 | black | 30 | 30 | 100.0 | 3.3 | -86.3 |
| Celeb-synthesis_id52_id50_0002 | F-White | id52 | id50 | white | 30 | 30 | 100.0 | 6.7 | -85.7 |
| Celeb-synthesis_id53_id51_0007 | F-White | id53 | id51 | white | 30 | 30 | 100.0 | 3.3 | -83.9 |
| Celeb-synthesis_id55_id52_0002 | F-White | id55 | id52 | white | 30 | 30 | 100.0 | 73.3 | -82.6 |
| Celeb-synthesis_id57_id50_0007 | F-White | id57 | id50 | white | 30 | 30 | 100.0 | 20.0 | -80.2 |
| Celeb-synthesis_id56_id52_0009 | F-White | id56 | id52 | white | 30 | 30 | 100.0 | 0.0 | -63.6 |
| Celeb-synthesis_id60_id61_0007 | M-Black | id60 | id61 | black | 30 | 29 | 96.7 | 86.7 | -95.1 |
| Celeb-synthesis_id57_id51_0003 | F-White | id57 | id51 | white | 30 | 29 | 96.7 | 3.3 | -76.7 |
| Celeb-synthesis_id53_id50_0003 | F-White | id53 | id50 | white | 30 | 29 | 96.7 | 0.0 | -74.8 |
| Celeb-synthesis_id60_id59_0004 | M-Black | id60 | id59 | black | 30 | 28 | 93.3 | 0.0 | -92.6 |
| Celeb-synthesis_id52_id58_0002 | F-White | id52 | id58 | white | 30 | 28 | 93.3 | 3.3 | -90.0 |
| Celeb-synthesis_id53_id56_0007 | F-White | id53 | id56 | white | 30 | 28 | 93.3 | 16.7 | -85.9 |

Swap pairs with FN rate >= 50%: 15/94. Top-10 pairs:

| value | n_fake | videos | errors | err_videos | rate% | ding_rate% |
|---|---|---|---|---|---|---|
| id60->id61 | 240 | 8 | 206 | 8 | 85.8 | 51.2 |
| id53->id51 | 300 | 10 | 199 | 10 | 66.3 | 6.3 |
| id53->id56 | 300 | 10 | 178 | 10 | 59.3 | 11.3 |
| id53->id58 | 300 | 10 | 174 | 10 | 58.0 | 15.3 |
| id58->id51 | 240 | 8 | 138 | 7 | 57.5 | 8.3 |
| id56->id50 | 207 | 7 | 119 | 7 | 57.5 | 2.9 |
| id53->id50 | 270 | 9 | 154 | 9 | 57.0 | 5.9 |
| id56->id51 | 261 | 9 | 148 | 9 | 56.7 | 2.7 |
| id49->id50 | 115 | 4 | 65 | 3 | 56.5 | 2.6 |
| id52->id50 | 300 | 10 | 169 | 10 | 56.3 | 6.7 |

## G. Score shift by group (all thresholds; logit medians)

| group | n_real | n_fake | real_median_logit | fake_median_logit | ding_real_median | ding_fake_median | auc | ding_auc |
|---|---|---|---|---|---|---|---|---|
| F-Others | 90.0 | 30.0 | -58.3 | 6.8 | 4.9 | 4.6 | 82.2 | 42.9 |
| F-White | 3929.0 | 21503.0 | -68.9 | 19.9 | 2.7 | 4.6 | 88.4 | 71.1 |
| M-Black | 710.0 | 1212.0 | -79.5 | -28.6 | 0.0 | 2.0 | 76.5 | 66.2 |
| M-Others | 60.0 | 0.0 | -60.0 |  | 3.5 |  |  |  |
| M-White | 924.0 | 0.0 | -71.2 |  | 2.7 |  |  |  |

## H. Matched FP by real identity (Celeb-real ids; YouTube pooled)

| ident | n_real | videos | fp | ding_fp | median_logit | gname | fp_rate% | ding_fp_rate% |
|---|---|---|---|---|---|---|---|---|
| id54 | 294 | 10 | 246 | 209 | -55.6 | F-White | 83.7 | 71.1 |
| id56 | 291 | 10 | 202 | 256 | -67.5 | F-White | 69.4 | 88.0 |
| id47 | 300 | 10 | 208 | 196 | -66.2 | F-White | 69.3 | 65.3 |
| id48 | 300 | 10 | 201 | 237 | -62.3 | F-White | 67.0 | 79.0 |
| id52 | 298 | 10 | 197 | 210 | -68.5 | F-White | 66.1 | 70.5 |
| id51 | 300 | 10 | 196 | 253 | -67.0 | F-White | 65.3 | 84.3 |
| id53 | 300 | 10 | 187 | 210 | -70.7 | F-White | 62.3 | 70.0 |
| id55 | 296 | 10 | 179 | 135 | -71.4 | F-White | 60.5 | 45.6 |
| id57 | 300 | 10 | 179 | 171 | -71.0 | F-White | 59.7 | 57.0 |
| id59 | 294 | 10 | 175 | 186 | -67.8 | M-Black | 59.5 | 63.3 |
| YouTube (51 people) | 1517 | 51 | 892 | 1124 | -69.0 | mixed | 58.8 | 74.1 |
| id58 | 300 | 10 | 158 | 178 | -72.4 | F-White | 52.7 | 59.3 |
| id50 | 269 | 9 | 137 | 239 | -73.2 | F-White | 50.9 | 88.8 |
| id49 | 238 | 8 | 109 | 199 | -75.3 | F-White | 45.8 | 83.6 |
| id61 | 209 | 7 | 57 | 66 | -82.9 | M-Black | 27.3 | 31.6 |
| id60 | 207 | 7 | 26 | 107 | -86.6 | M-Black | 12.6 | 51.7 |

M-Black fakes at matched: {'fn': 296, 'share_of_fn': 29.16256157635468, 'share_of_fakes': 5.3286436579468015, 'also_ding': 42.9054054054054, 'fw_also_ding': 41.1682892906815}


## E. Shared vs unique with the Ding reproduction (Ding rule z >= 0)

| threshold | our_FP | FP_also_Ding | Ding_FPR_base | FP_ours_only | FP_Ding_only | our_FN | FN_also_Ding | Ding_FNR_base | FN_ours_only | FN_Ding_only | FPvid50_ours_both_ding | FNvid50_ours_both_ding |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matched | 3349 | 76.8 | 69.6 | 778 | 1405 | 1015 | 41.7 | 9.5 | 592 | 1741 | 126/104/150 | 7/2/13 |
| native | 2325 | 82.0 | 69.6 | 419 | 2070 | 2089 | 34.3 | 9.5 | 1372 | 1447 | 81/70/150 | 19/4/13 |
| A1 | 582 | 90.5 | 69.6 | 55 | 3449 | 8007 | 19.8 | 9.5 | 6421 | 578 | 10/10/150 | 239/10/13 |

(*_also_Ding: % of our errors on which Ding is also wrong; *_base: Ding's error rate on all reals/fakes, the chance overlap; vid50 = videos with >= 50% erroneous frames, ours / both / Ding.)


**matched: real videos we flag on >= 50% of frames while Ding flags < 50%** (22): Celeb-real_id54_0007 F-White 93% vs 20%; Celeb-real_id53_0006 F-White 90% vs 43%; Celeb-real_id57_0001 F-White 90% vs 7%; Celeb-real_id47_0001 F-White 87% vs 20%; Celeb-real_id55_0006 F-White 100% vs 15%; Celeb-real_id52_0005 F-White 87% vs 43%; Celeb-real_id54_0004 F-White 80% vs 43%; Celeb-real_id59_0008 M-Black 80% vs 43%; YouTube-real_00298 M-White 80% vs 27%; YouTube-real_00283 M-White 73% vs 40%; YouTube-real_00249 M-White 70% vs 47%; Celeb-real_id47_0007 F-White 70% vs 23%; Celeb-real_id55_0004 F-White 70% vs 40%; Celeb-real_id58_0008 F-White 67% vs 37%; Celeb-real_id55_0000 F-White 67% vs 43%


**matched: fake videos we miss on >= 50% of frames while Ding misses < 50%** (5): Celeb-synthesis_id60_id61_0004 100% vs 13%; Celeb-synthesis_id60_id59_0004 83% vs 0%; Celeb-synthesis_id60_id61_0002 77% vs 33%; Celeb-synthesis_id52_id58_0002 57% vs 3%; Celeb-synthesis_id60_id59_0001 50% vs 23%


**matched: fake videos Ding misses on >= 50% while we miss < 50%** (11): Celeb-synthesis_id60_id61_0000 57% vs ours 40%; Celeb-synthesis_id60_id61_0003 57% vs ours 37%; Celeb-synthesis_id60_id61_0001 73% vs ours 30%; Celeb-synthesis_id53_id58_0006 63% vs ours 20%; Celeb-synthesis_id59_id61_0005 73% vs ours 10%; Celeb-synthesis_id55_id52_0002 73% vs ours 10%; Celeb-synthesis_id59_id61_0008 80% vs ours 3%; Celeb-synthesis_id55_id52_0009 70% vs ours 3%; Celeb-synthesis_id58_id57_0008 60% vs ours 0%; Celeb-synthesis_id53_id57_0008 50% vs ours 0%; Celeb-synthesis_id53_id57_0004 50% vs ours 0%


**matched: FP where we fail and Ding does not (highest BD logit)** (n = 778)

| vid_short | gname | cl_gender | cl_race | if_age | ita_gw | logit | ding_logit | frame |
|---|---|---|---|---|---|---|---|---|
| Celeb-real_id59_0009 | M-Black | male | black | 41 | -36.6 | 37.1 | -1.5 | 0_Celeb-real_id59_0009_221.png |
| Celeb-real_id47_0001 | F-White | nonmale | white | 27 | 34.7 | 30.1 | -0.8 | 0_Celeb-real_id47_0001_201.png |
| Celeb-real_id54_0004 | F-White | nonmale | white | 26 | 80.2 | 27.8 | -1.9 | 0_Celeb-real_id54_0004_261.png |
| YouTube-real_00267 | F-White | nonmale | white | 20 | 80.0 | 24.6 | -3.5 | 0_YouTube-real_00267_11.png |
| Celeb-real_id47_0001 | F-White | nonmale | white | 28 | 21.6 | 24.2 | -0.7 | 0_Celeb-real_id47_0001_71.png |
| YouTube-real_00263 | M-White | nonmale | asian | 26 | -17.6 | 24.1 | -1.0 | 0_YouTube-real_00263_281.png |
| Celeb-real_id47_0001 | F-White | nonmale | white | 29 | 30.2 | 20.9 | -3.9 | 0_Celeb-real_id47_0001_261.png |
| Celeb-real_id48_0006 | F-White | nonmale | white | 29 | 9.7 | 18.3 | -0.2 | 0_Celeb-real_id48_0006_141.png |
| YouTube-real_00249 | M-White | male | white | 28 | 17.1 | 15.9 | -1.2 | 0_YouTube-real_00249_281.png |
| YouTube-real_00249 | M-White | male | white | 25 | 36.3 | 15.7 | -3.2 | 0_YouTube-real_00249_261.png |
| YouTube-real_00287 | M-White | male | white | 32 | -24.7 | 15.5 | -1.6 | 0_YouTube-real_00287_271.png |
| Celeb-real_id48_0006 | F-White | nonmale | white | 27 | -9.5 | 15.0 | -0.5 | 0_Celeb-real_id48_0006_61.png |

**matched: FP where Ding fails and we do not (highest Ding logit)** (n = 1405)

| vid_short | gname | cl_gender | cl_race | if_age | ita_gw | logit | ding_logit | frame |
|---|---|---|---|---|---|---|---|---|
| YouTube-real_00266 | M-White | male | white | 38 | -8.4 | -91.0 | 5.7 | 0_YouTube-real_00266_31.png |
| YouTube-real_00248 | M-White | male | white | 22 | 47.6 | -84.4 | 5.5 | 0_YouTube-real_00248_31.png |
| YouTube-real_00266 | M-White | male | white | 30 | -20.6 | -101.5 | 5.5 | 0_YouTube-real_00266_271.png |
| YouTube-real_00266 | M-White | male | white | 27 | -21.3 | -90.4 | 5.5 | 0_YouTube-real_00266_261.png |
| Celeb-real_id57_0008 | F-White | nonmale | white | 27 | 63.3 | -90.4 | 5.5 | 0_Celeb-real_id57_0008_271.png |
| Celeb-real_id50_0005 | F-White | nonmale | white | 23 | 63.4 | -99.6 | 5.4 | 0_Celeb-real_id50_0005_61.png |
| YouTube-real_00245 | F-White | nonmale | white | 24 | 60.0 | -74.4 | 5.4 | 0_YouTube-real_00245_131.png |
| YouTube-real_00277 | M-White | male | white | 24 | 70.4 | -95.7 | 5.4 | 0_YouTube-real_00277_291.png |
| Celeb-real_id57_0004 | F-White | nonmale | white | 27 | 17.0 | -84.8 | 5.4 | 0_Celeb-real_id57_0004_171.png |
| Celeb-real_id57_0004 | F-White | nonmale | white | 37 | 25.7 | -77.2 | 5.4 | 0_Celeb-real_id57_0004_91.png |
| Celeb-real_id55_0007 | F-White | nonmale | white | 26 | -10.5 | -78.6 | 5.4 | 0_Celeb-real_id55_0007_281.png |
| YouTube-real_00266 | M-White | male | white | 47 | -22.7 | -89.8 | 5.4 | 0_YouTube-real_00266_71.png |

**matched: FN where we fail and Ding does not (lowest BD logit)** (n = 592)

| vid_short | gname | cl_gender | cl_race | if_age | ita_gw | logit | ding_logit | tgt_id | src_id | frame |
|---|---|---|---|---|---|---|---|---|---|---|
| Celeb-synthesis_id59_id60_0001 | M-Black | male | black | 37 | -63.3 | -104.4 | 0.1 | id59 | id60 | 0_Celeb-synthesis_id59_id60_0001_1.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 48 | -67.5 | -103.7 | 3.5 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_71.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 45 | -68.5 | -98.5 | 1.8 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_111.png |
| Celeb-synthesis_id60_id61_0002 | M-Black | male | black | 46 | -73.7 | -98.3 | 0.2 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0002_21.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 45 | -62.8 | -97.9 | 1.7 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_251.png |
| Celeb-synthesis_id60_id61_0002 | M-Black | male | black | 18 | -55.2 | -97.8 | 0.6 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0002_151.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 37 | -65.4 | -97.7 | 1.4 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_231.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 39 | -60.9 | -97.5 | 0.3 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_181.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 45 | -62.6 | -97.1 | 2.2 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_101.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 47 | -68.5 | -96.5 | 3.2 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_81.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 44 | -68.4 | -96.1 | 1.9 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_211.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 49 | -64.9 | -95.8 | 1.4 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_261.png |

**matched: FN where Ding fails and we do not (lowest Ding logit)** (n = 1741)

| vid_short | gname | cl_gender | cl_race | if_age | ita_gw | logit | ding_logit | tgt_id | src_id | frame |
|---|---|---|---|---|---|---|---|---|---|---|
| Celeb-synthesis_id50_id55_0006 | F-White | nonmale | white | 23 | -81.4 | -64.7 | -5.7 | id50 | id55 | 0_Celeb-synthesis_id50_id55_0006_1.png |
| Celeb-synthesis_id49_id51_0008 | F-White | nonmale | white | 25 | 83.6 | -56.6 | -5.6 | id49 | id51 | 0_Celeb-synthesis_id49_id51_0008_1.png |
| Celeb-synthesis_id49_id50_0000 | F-White | nonmale | white | 21 | 45.7 | -71.4 | -5.6 | id49 | id50 | 0_Celeb-synthesis_id49_id50_0000_1.png |
| Celeb-synthesis_id60_id61_0001 | M-Black | male | black | 42 | -53.6 | -47.3 | -5.6 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0001_121.png |
| Celeb-synthesis_id58_id55_0002 | F-White | nonmale | white | 25 | 35.4 | -71.4 | -5.6 | id58 | id55 | 0_Celeb-synthesis_id58_id55_0002_1.png |
| Celeb-synthesis_id53_id58_0006 | F-White | nonmale | white | 39 | -10.0 | -29.7 | -5.6 | id53 | id58 | 0_Celeb-synthesis_id53_id58_0006_121.png |
| Celeb-synthesis_id55_id50_0009 | F-White | nonmale | white | 21 | 59.2 | -65.8 | -5.6 | id55 | id50 | 0_Celeb-synthesis_id55_id50_0009_51.png |
| Celeb-synthesis_id57_id55_0001 | F-White | nonmale | white | 27 | 81.3 | -65.0 | -5.5 | id57 | id55 | 0_Celeb-synthesis_id57_id55_0001_1.png |
| Celeb-synthesis_id50_id57_0009 | F-White | nonmale | white | 35 | 16.2 | -66.5 | -5.5 | id50 | id57 | 0_Celeb-synthesis_id50_id57_0009_1.png |
| Celeb-synthesis_id55_id50_0006 | F-White | nonmale | white | 23 | 26.3 | -67.7 | -5.5 | id55 | id50 | 0_Celeb-synthesis_id55_id50_0006_1.png |
| Celeb-synthesis_id49_id54_0006 | F-White | nonmale | white | 28 | 68.3 | -53.3 | -5.5 | id49 | id54 | 0_Celeb-synthesis_id49_id54_0006_201.png |
| Celeb-synthesis_id57_id53_0000 | F-White | nonmale | white | 39 | 69.2 | -61.6 | -5.5 | id57 | id53 | 0_Celeb-synthesis_id57_id53_0000_1.png |

**A1: real videos we flag on >= 50% of frames while Ding flags < 50%** (0): 


**A1: fake videos we miss on >= 50% of frames while Ding misses < 50%** (229): Celeb-synthesis_id60_id61_0004 100% vs 13%; Celeb-synthesis_id60_id61_0002 100% vs 33%; Celeb-synthesis_id49_id50_0007 100% vs 0%; Celeb-synthesis_id61_id59_0005 100% vs 3%; Celeb-synthesis_id52_id50_0002 100% vs 7%; Celeb-synthesis_id53_id51_0007 100% vs 3%; Celeb-synthesis_id57_id50_0007 100% vs 20%; Celeb-synthesis_id56_id52_0009 100% vs 0%; Celeb-synthesis_id57_id51_0003 97% vs 3%; Celeb-synthesis_id53_id50_0003 97% vs 0%; Celeb-synthesis_id60_id59_0004 93% vs 0%; Celeb-synthesis_id52_id58_0002 93% vs 3%; Celeb-synthesis_id53_id56_0007 93% vs 17%; Celeb-synthesis_id56_id49_0005 93% vs 13%; Celeb-synthesis_id49_id53_0007 93% vs 0%


**A1: fake videos Ding misses on >= 50% while we miss < 50%** (3): Celeb-synthesis_id59_id61_0005 73% vs ours 30%; Celeb-synthesis_id58_id57_0008 60% vs ours 27%; Celeb-synthesis_id59_id61_0008 80% vs ours 7%


**A1: FP where we fail and Ding does not (highest BD logit)** (n = 55)

| vid_short | gname | cl_gender | cl_race | if_age | ita_gw | logit | ding_logit | frame |
|---|---|---|---|---|---|---|---|---|
| Celeb-real_id59_0009 | M-Black | male | black | 41 | -36.6 | 37.1 | -1.5 | 0_Celeb-real_id59_0009_221.png |
| Celeb-real_id47_0001 | F-White | nonmale | white | 27 | 34.7 | 30.1 | -0.8 | 0_Celeb-real_id47_0001_201.png |
| Celeb-real_id54_0004 | F-White | nonmale | white | 26 | 80.2 | 27.8 | -1.9 | 0_Celeb-real_id54_0004_261.png |
| YouTube-real_00267 | F-White | nonmale | white | 20 | 80.0 | 24.6 | -3.5 | 0_YouTube-real_00267_11.png |
| Celeb-real_id47_0001 | F-White | nonmale | white | 28 | 21.6 | 24.2 | -0.7 | 0_Celeb-real_id47_0001_71.png |
| YouTube-real_00263 | M-White | nonmale | asian | 26 | -17.6 | 24.1 | -1.0 | 0_YouTube-real_00263_281.png |
| Celeb-real_id47_0001 | F-White | nonmale | white | 29 | 30.2 | 20.9 | -3.9 | 0_Celeb-real_id47_0001_261.png |
| Celeb-real_id48_0006 | F-White | nonmale | white | 29 | 9.7 | 18.3 | -0.2 | 0_Celeb-real_id48_0006_141.png |
| YouTube-real_00249 | M-White | male | white | 28 | 17.1 | 15.9 | -1.2 | 0_YouTube-real_00249_281.png |
| YouTube-real_00249 | M-White | male | white | 25 | 36.3 | 15.7 | -3.2 | 0_YouTube-real_00249_261.png |
| YouTube-real_00287 | M-White | male | white | 32 | -24.7 | 15.5 | -1.6 | 0_YouTube-real_00287_271.png |
| Celeb-real_id48_0006 | F-White | nonmale | white | 27 | -9.5 | 15.0 | -0.5 | 0_Celeb-real_id48_0006_61.png |

**A1: FP where Ding fails and we do not (highest Ding logit)** (n = 3449)

| vid_short | gname | cl_gender | cl_race | if_age | ita_gw | logit | ding_logit | frame |
|---|---|---|---|---|---|---|---|---|
| YouTube-real_00266 | M-White | male | white | 38 | -8.4 | -91.0 | 5.7 | 0_YouTube-real_00266_31.png |
| YouTube-real_00248 | M-White | male | white | 22 | 54.2 | -59.5 | 5.6 | 0_YouTube-real_00248_101.png |
| YouTube-real_00248 | M-White | male | white | 22 | 47.6 | -84.4 | 5.5 | 0_YouTube-real_00248_31.png |
| YouTube-real_00266 | M-White | male | white | 30 | -20.6 | -101.5 | 5.5 | 0_YouTube-real_00266_271.png |
| YouTube-real_00248 | M-White | male | white | 29 | 58.1 | -65.9 | 5.5 | 0_YouTube-real_00248_161.png |
| YouTube-real_00271 | M-White | male | white | 47 | 1.2 | -38.6 | 5.5 | 0_YouTube-real_00271_91.png |
| YouTube-real_00281 | F-White | nonmale | white | 24 | 11.0 | -25.9 | 5.5 | 0_YouTube-real_00281_71.png |
| YouTube-real_00248 | M-White | male | white | 28 | 41.9 | -73.2 | 5.5 | 0_YouTube-real_00248_11.png |
| YouTube-real_00266 | M-White | male | white | 27 | -21.3 | -90.4 | 5.5 | 0_YouTube-real_00266_261.png |
| YouTube-real_00281 | F-White | nonmale | white | 24 | 22.0 | -52.2 | 5.5 | 0_YouTube-real_00281_81.png |
| YouTube-real_00248 | M-White | male | white | 24 | 55.3 | -70.8 | 5.5 | 0_YouTube-real_00248_241.png |
| YouTube-real_00295 | M-White | male | white | 54 | 21.0 | -64.4 | 5.5 | 0_YouTube-real_00295_231.png |

**A1: FN where we fail and Ding does not (lowest BD logit)** (n = 6421)

| vid_short | gname | cl_gender | cl_race | if_age | ita_gw | logit | ding_logit | tgt_id | src_id | frame |
|---|---|---|---|---|---|---|---|---|---|---|
| Celeb-synthesis_id59_id60_0001 | M-Black | male | black | 37 | -63.3 | -104.4 | 0.1 | id59 | id60 | 0_Celeb-synthesis_id59_id60_0001_1.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 48 | -67.5 | -103.7 | 3.5 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_71.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 45 | -68.5 | -98.5 | 1.8 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_111.png |
| Celeb-synthesis_id60_id61_0002 | M-Black | male | black | 46 | -73.7 | -98.3 | 0.2 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0002_21.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 45 | -62.8 | -97.9 | 1.7 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_251.png |
| Celeb-synthesis_id60_id61_0002 | M-Black | male | black | 18 | -55.2 | -97.8 | 0.6 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0002_151.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 37 | -65.4 | -97.7 | 1.4 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_231.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 39 | -60.9 | -97.5 | 0.3 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_181.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 45 | -62.6 | -97.1 | 2.2 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_101.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 47 | -68.5 | -96.5 | 3.2 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_81.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 44 | -68.4 | -96.1 | 1.9 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_211.png |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 49 | -64.9 | -95.8 | 1.4 | id60 | id61 | 0_Celeb-synthesis_id60_id61_0004_261.png |

**A1: FN where Ding fails and we do not (lowest Ding logit)** (n = 578)

| vid_short | gname | cl_gender | cl_race | if_age | ita_gw | logit | ding_logit | tgt_id | src_id | frame |
|---|---|---|---|---|---|---|---|---|---|---|
| Celeb-synthesis_id55_id52_0009 | F-White | nonmale | white | 25 | 59.4 | -9.4 | -5.5 | id55 | id52 | 0_Celeb-synthesis_id55_id52_0009_251.png |
| Celeb-synthesis_id48_id47_0003 | F-White | nonmale | white | 22 | 20.2 | 10.4 | -5.4 | id48 | id47 | 0_Celeb-synthesis_id48_id47_0003_181.png |
| Celeb-synthesis_id59_id61_0008 | M-Black | male | black | 44 | -30.5 | 15.2 | -5.4 | id59 | id61 | 0_Celeb-synthesis_id59_id61_0008_181.png |
| Celeb-synthesis_id59_id61_0008 | M-Black | male | black | 42 | -34.3 | 17.7 | -5.2 | id59 | id61 | 0_Celeb-synthesis_id59_id61_0008_221.png |
| Celeb-synthesis_id55_id57_0009 | F-White | nonmale | white | 22 | 65.1 | 15.3 | -5.2 | id55 | id57 | 0_Celeb-synthesis_id55_id57_0009_11.png |
| Celeb-synthesis_id53_id57_0006 | F-White | nonmale | white | 40 | -24.5 | 31.3 | -5.2 | id53 | id57 | 0_Celeb-synthesis_id53_id57_0006_121.png |
| Celeb-synthesis_id59_id61_0008 | M-Black | male | black | 46 | -29.9 | 19.9 | -5.2 | id59 | id61 | 0_Celeb-synthesis_id59_id61_0008_121.png |
| Celeb-synthesis_id49_id54_0004 | F-White | nonmale | white | 32 | 47.1 | -14.4 | -5.1 | id49 | id54 | 0_Celeb-synthesis_id49_id54_0004_61.png |
| Celeb-synthesis_id59_id61_0008 | M-Black | male | black | 47 | -28.8 | -14.9 | -5.1 | id59 | id61 | 0_Celeb-synthesis_id59_id61_0008_241.png |
| Celeb-synthesis_id55_id52_0009 | F-White | nonmale | white | 23 | 57.7 | 8.2 | -5.0 | id55 | id52 | 0_Celeb-synthesis_id55_id52_0009_271.png |
| Celeb-synthesis_id59_id61_0008 | M-Black | male | black | 45 | -32.6 | 38.5 | -5.0 | id59 | id61 | 0_Celeb-synthesis_id59_id61_0008_231.png |
| Celeb-synthesis_id55_id57_0009 | F-White | nonmale | white | 23 | 62.5 | 24.5 | -4.9 | id55 | id57 | 0_Celeb-synthesis_id55_id57_0009_181.png |

## F. Most confident errors (matched)


**Top-12 FP (highest BD logit)** (full paths in failures.json and tables/)

| vid_short | gname | cl_gender | cl_race | if_age | ita_gw | logit | ding_logit | img_path |
|---|---|---|---|---|---|---|---|---|
| YouTube-real_00247 | M-White | male | white | 34 | -3.2 | 72.2 | 4.0 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_YouTube-real_00247_71.png |
| YouTube-real_00247 | M-White | male | white | 34 | -2.4 | 72.1 | 4.7 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_YouTube-real_00247_91.png |
| Celeb-real_id50_0007 | F-White | nonmale | white | 26 | 17.7 | 67.8 | 5.1 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-real_id50_0007_251.png |
| YouTube-real_00247 | M-White | male | white | 32 | -10.4 | 67.7 | 5.3 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_YouTube-real_00247_191.png |
| YouTube-real_00247 | M-White | male | white | 25 | -5.8 | 64.7 | 5.0 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_YouTube-real_00247_281.png |
| YouTube-real_00247 | M-White | male | white | 29 | -0.1 | 64.7 | 5.2 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_YouTube-real_00247_101.png |
| YouTube-real_00247 | M-White | male | white | 29 | -9.3 | 63.5 | 5.2 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_YouTube-real_00247_131.png |
| Celeb-real_id48_0007 | F-White | nonmale | white | 32 | 12.3 | 63.2 | 3.3 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-real_id48_0007_31.png |
| YouTube-real_00247 | M-White | male | white | 27 | -6.7 | 61.2 | 5.3 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_YouTube-real_00247_141.png |
| YouTube-real_00247 | M-White | male | white | 26 | -1.2 | 59.7 | 4.9 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_YouTube-real_00247_161.png |
| YouTube-real_00247 | M-White | male | white | 24 | -1.2 | 59.3 | 5.2 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_YouTube-real_00247_251.png |
| Celeb-real_id50_0007 | F-White | nonmale | white | 24 | 10.8 | 58.7 | 5.2 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-real_id50_0007_221.png |

**Top-12 FN (lowest BD logit)** (full paths in failures.json and tables/)

| vid_short | gname | cl_gender | cl_race | if_age | ita_gw | logit | ding_logit | img_path | tgt_id | src_id |
|---|---|---|---|---|---|---|---|---|---|---|
| Celeb-synthesis_id59_id60_0001 | M-Black | male | black | 37 | -63.3 | -104.4 | 0.1 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-synthesis_id59_id60_0001_1.png | id59 | id60 |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 48 | -67.5 | -103.7 | 3.5 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-synthesis_id60_id61_0004_71.png | id60 | id61 |
| Celeb-synthesis_id60_id59_0002 | M-Black | male | black | 42 | -71.8 | -101.8 | -0.1 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-synthesis_id60_id59_0002_21.png | id60 | id59 |
| Celeb-synthesis_id59_id61_0004 | M-Black | male | black | 43 | -61.9 | -101.4 | -2.0 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-synthesis_id59_id61_0004_101.png | id59 | id61 |
| Celeb-synthesis_id51_id50_0008 | F-White | nonmale | white | 28 | 35.9 | -99.8 | -3.6 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-synthesis_id51_id50_0008_241.png | id51 | id50 |
| Celeb-synthesis_id50_id51_0008 | F-White | nonmale | white | 29 | 28.2 | -99.5 | -0.9 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-synthesis_id50_id51_0008_1.png | id50 | id51 |
| Celeb-synthesis_id59_id61_0001 | M-Black | male | black | 46 | -62.1 | -99.0 | -2.0 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-synthesis_id59_id61_0001_1.png | id59 | id61 |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 45 | -68.5 | -98.5 | 1.8 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-synthesis_id60_id61_0004_111.png | id60 | id61 |
| Celeb-synthesis_id60_id61_0002 | M-Black | male | black | 46 | -73.7 | -98.3 | 0.2 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-synthesis_id60_id61_0002_21.png | id60 | id61 |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 45 | -62.8 | -97.9 | 1.7 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-synthesis_id60_id61_0004_251.png | id60 | id61 |
| Celeb-synthesis_id60_id61_0002 | M-Black | male | black | 18 | -55.2 | -97.8 | 0.6 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-synthesis_id60_id61_0002_151.png | id60 | id61 |
| Celeb-synthesis_id60_id61_0004 | M-Black | male | black | 37 | -65.4 | -97.7 | 1.4 | /home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan/work/extracted/celebdf/0_Celeb-synthesis_id60_id61_0004_231.png | id60 | id61 |
