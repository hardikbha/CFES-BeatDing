# BD0 seed 7: failure cases on FF++, DFDC, Celeb-DF and DFD

26 Oct 2026. Model: BD0 seed 7, the provisional "Beat-Ding" model (CLIP ViT-L/14 with rank-1 adapters, CFES-D, Arm E base, and ESA score alignment, i.e. component B). Main operating point: **matched**, logit >= -73.76, set on FF++ val so that TPR equals the Ding reproduction's 94.18%. The same threshold is used on every test set. Native (-65.97) and A1 (-19.13) are shown where they change the picture. Comparison model: Ding's official code reproduced (Xception, seed 7), deciding at logit >= 0 (his 0.5 rule), on the same rows.

Sources: the four per-set analyses in `ffpp_test/`, `dfdc/`, `celebdf/` and `dfd/` (each has its own FAILURES.md, scripts and tables), with three verifier corrections applied (marked *[corrected]*). The headline table and one label spot check come from `report_build/report_tables.py` (output in `report_tables.log`). Claims that are visual impressions, not measurements, are marked *[visual]*.

## 1. Summary

At the matched threshold BD0 has the higher AUC on all three cross-domain sets (DFDC 77.41, Celeb-DF 87.32, DFD 90.21, against Ding's published 61.86 / 76.75 / 81.46). In-domain it is behind (FF++ AUC 96.78 vs 97.71), and its FF++ intersectional FPR gap is about twice Ding's (i-F_FPR 46.56 vs reproduction 24.24, published 20.18). The failures fall into four kinds. (i) **The FF++ threshold does not carry over.** Real frames score higher off-domain, so 33.2% (DFDC), 58.6% (Celeb-DF) and 32.4% (DFD) of real frames are flagged. These false positives are spread over most real videos (380/473, 183/192, 83/104), not a few. Ding's reproduction at 0.5 is worse on every one (49.1 / 69.6 / 59.0%). (ii) **Specific real people or videos are over-flagged.** In FF++ it is F-Asian news presenters (youtube_062/616/930), and 75% of those FPs are ours alone. In Celeb-DF it is one broadcast video (YouTube-real_00247). In DFD it is one actor (actor 19). (iii) **Systematic misses.** NeuralTextures accounts for 54% of FF++ misses. Celeb-DF misses fall on swaps among the Black male identities id59-61 (29% of misses) and on the first frame of each video (27% of misses). DFDC misses fall on clean clips of white women (50% of misses). In DFD there are probable crops of the unswapped person (FNR 70.6%). (iv) **Group disparities are mostly small-cell effects.** On the Xu labels the metric uses, BD0's i-F_FPR is not above a label-permutation null on FF++ (p=0.366), DFDC (p=0.57) or Celeb-DF matched (p=0.139). The real, label-robust patterns are these: women's real videos are flagged more in all four sets; cleaner labels show a significant DFDC gap that falls on women of colour, and Ding has it too; and Celeb-DF's Black male identities are scored lower (more real) on both reals and fakes. Our errors are largely **not** Ding's in-domain: only 36% of FF++ FPs and 33% of FNs are shared. Cross-domain, the shared share is higher (FP 55-77%, FN 42-54%).

| Test set (matched) | AUC | TPR | FPR | i-F_FPR | Ding published AUC / i-F_FPR | Ding repro @0.5 TPR / FPR | Columns won vs Ding published |
|---|---|---|---|---|---|---|---|
| FF++ test | 96.78 | 93.13 | 13.51 | 46.56 | 97.71 / 20.18 | 94.85 / 12.54 | 0/10 |
| DFDC | 77.41 | 72.47 | 33.23 | 32.23 | 61.86 / 37.37 | 55.60 / 49.13 | 7/10 |
| Celeb-DF | 87.32 | 95.54 | 58.62 | 32.48 | 76.75 / 23.32 | 90.49 / 69.60 | 4/10 |
| DFD | 90.21 | 89.16 | 32.44 | 29.05 | 81.46 / 27.32 | 83.57 / 59.02 | 7/10 |

Total columns won vs Ding published: matched 18/40, native 19/40, A1 27/40 (from bd_eval.json).

## 2. Cross-set failure taxonomy

### 2.1 The FF++ threshold floods cross-domain sets with false positives

| | FF++ | DFDC | Celeb-DF | DFD |
|---|---|---|---|---|
| FP frames / reals | 613 / 4,538 (13.51%) | 4,885 / 14,699 (33.23%) | 3,349 / 5,713 (58.62%) | 1,003 / 3,092 (32.44%) |
| Real videos with >= 1 FP | 97 / 267 | 380 / 473 | 183 / 192 | 83 / 104 |
| Videos flagged on >= 50% of frames | 18 | 142 | 126 | 30 |
| Videos needed for half of all FP frames | 17 | 79 | 62 | 21 |
| Median real logit (t = -73.76) | n/a | -80.86 | -70.15 | -80.2 |

- On DFDC, real scores sit about 11 logits higher than on FF++ val (median -80.86 vs -92.22). At the same threshold, FF++ val FPR is 15.81%. 40.86% of DFDC reals lie within ±10 logits of t. On Celeb-DF the threshold is *above* the median real logit.
- Raising the threshold trades FPs for misses. A1 brings FPR to 1.90 (DFDC), 10.19 (Celeb-DF) and 3.33% (DFD), but TPR falls to 27.16, 64.80 and 68.28%. On FF++, A1 gives FPR 1.87% at FNR 19.63%.
- Ding's reproduction floods more (FPR 49.13 / 69.60 / 59.02% at TPR 55.60 / 90.49 / 83.57%). This is an operating-point failure common to both models, not something BD introduced.

### 2.2 Over-flagging of specific real people or videos

- **FF++ F-Asian presenters (ours only).** F-Asian FPR is 25.6% (106/414 frames, 12 of 23 videos), against Ding's 10.87%. 80 of the 106 frames (75%) are frames Ding calls real. youtube_062, 616 and 930 hold 50.9% of them (FP rate ours vs Ding: 69% vs 0%, 60% vs 60%, 57% vs 18%). The earlier diagnosis described these as TV news anchors with heavy makeup and broadcast compression *[visual, from the 26 Oct plan]*.
- **FF++ worst videos overall.** youtube_305, 845, 307, 383 and 438 lead. Dropping the top 5 and re-thresholding to the same FPR lowers i-F_FPR from 46.56 to only 36.05 (random-drop median 46.36, p5-p95 43.75-49.76). The disparity is not "a few bad videos".
- **Celeb-DF YouTube-real_00247.** It holds 9 of the top-12 FP frames (logits 59.34-72.25), and Ding flags all of them. The frame shows a man in a suit against a studio/broadcast background *[visual, verifier]*.
- **DFD actor 19.** All 3 "F-Black" real videos (88 frames) are actor 19, flagged at 50.0% vs 32.4% pooled (video rates 96.7 / 33.3 / 17.9%). This one person drives i-F_FPR: see 2.5.
- **DFDC.** The 12 most confident FPs come from only 4 videos (0_7434 F-Black has 6 frames, logits 61.53 to 47.22). Some "real" clips show visible compositing or overlay artefacts (0_7406 a pasted face patch, 0_4516 magenta glitch bands, 0_4543 blue line overlays) *[visual; needs a manual label check]*.

### 2.3 Systematic misses

| Set | Concentration of misses |
|---|---|
| FF++ | NeuralTextures: FNR 17.24%, 777 FN = 54.2% of all FNs. Reenactment (F2F+NT): FNR 11.37%, 70.4% of FNs. Other methods: Face2Face 5.33, FaceShifter 5.05, FaceSwap 3.90, Deepfakes 1.52%. 22 of 34 videos missed on >= 50% of frames are NT. |
| FF++ | By target identity: 15 of 192 targets hold half of FNs, and 5 targets are missed under >= 2 methods (705, 259, 016, 619, 492). Male targets: FNR 10.23 vs 3.81% female (Xu), 71% of FNs. |
| DFDC | F-White holds 1,119 / 2,246 FN (49.82%) from 37.7% of fake rows; FNR 36.39% vs M-White 16.92%. Fakes in the brightness Q3 bin are missed more (34.44%) than in Q1 (16.67%). No method labels exist on DFDC. |
| Celeb-DF | Black male swaps (ids 59-61 swapped among themselves): FNR 24.4% (296/1,212), 5.3% of fakes but 29.2% of FNs; pair id60->id61 50.0%. **Frame 1** of each video: FNR 35.5% vs 3.4% for later frames, 26.7% of all FNs (Ding: 63.9 vs 7.6%). |
| DFD | Pairs 01<-27: FNR 39.91%; **17<-08: 35.27% (79/224)** *[corrected; 33.5% was actor 17 over all its pairs]*. Scenes walking_and_outside_surprised 44.44% and walking_down_indoor_hall_disgust 36.67%, vs kitchen_still 0.00%. Frames whose AI-Face gender matches neither named actor (a proxy for crops of the unswapped person): 136 frames / 10 videos, FNR 70.59% (Ding 26.47%) vs 10.46% elsewhere. |

### 2.4 Appearance and capture-quality cues read as forgery evidence (or as realness)

- **DFDC.** Dark, low-contrast and blurry real frames are flagged. Brightness Q1 vs Q4: FPR 42.91 vs 26.63%. Contrast Q1 vs Q4: 43.28 vs 23.90%. The brightness effect holds **within white, Black and Asian** (cleaner race; Q1 vs Q4: 43.1 vs 26.8, 46.5 vs 28.5, 54.3 vs 20.3) but **not within "others"** (Q1 12.4%, n=370, vs Q4 32.8%, n=122) *[corrected]*. Only the blurriest quartile stands out on sharpness (Q1 47.35%). The effect is not monotone: Q2 28.60, Q3 23.25, Q4 35.15% *[corrected]*. Black real videos are darker (median brightness 71.5 vs 100.8 for white), so lighting partly explains the higher FPR on darker skin.
- **DFD.** Outdoor walking scenes are flagged (walking_down_street_outside_angry 69.0%, walking_outside_cafe_disgusted 65.8%), static indoor scenes rarely (kitchen_still 14.8%, talking_against_wall 10.8%).
- **Celeb-DF.** The first extracted frame of every video reads as real for both models. Real FPR on frame 1 is 22.9% vs 59.9% on later frames. At A1, 89.9% of fake first frames are missed. This points to a keyframe or compression cue (inferred, not tested).
- **FF++ appearance prior.** How fake the model finds a target's real video predicts how often that target's fakes are missed. Spearman(real-video mean logit, FN rate of its fakes) = -0.446 for ours (-0.369 for Ding), over 152 targets. Target 800 (real mean logit -116.8) has 54 FN frames spread over 4 methods.

### 2.5 Group and attribute concentration

| Set | Largest i-F_FPR contributors (matched) | Is it beyond chance? |
|---|---|---|
| FF++ | i-F_FPR 46.56. M-Black (FPR 3.26%, 7 videos) and M-Others (2.76%, 12 videos) are *under*-flagged: 45.1% of the metric. F-Asian is over-flagged (25.6%): 26%. Together 71.1%. | Permutation null median 42.61, P(null >= obs) = 0.366: no. |
| DFDC | Xu: 32.23 (F-Black +12.24 pp, M-Asian -10.74 pp). Cleaner labels: 68.79 (F-Black 45.47, F-Asian 43.85 vs M-Asian 14.86, M-Others 17.03). | Xu: null median 34.18, p = 0.57, no. Cleaner: null median 36.25, p = 0.0025, yes. Ding repro on cleaner labels: 62.52. |
| Celeb-DF | 32.48. M-Black under-flagged (36.3% vs 58.6%): term 22.28. Real FPR for id60 12.6%, id61 27.3%. At A1 (22.99): 3 YouTube videos of Asian women in Xu F-Others (term 19.81). | Matched: null mean 19.94, p = 0.139, no. A1: p = 0.036, weakly. |
| DFD | 29.05, driven by actor 19 (F-Black). Removing one video gives 18.06, two give 13.38. | Video bootstrap 95% CI [9.27, 81.53]: undetermined. |

Patterns across attributes:

- **Women's real videos are flagged more in all four sets.**
  - FF++: 14.27 vs 12.62% (Xu); 14.94 vs 11.76% (cleaner).
  - DFDC: 37.56 vs 27.93% (cleaner).
  - Celeb-DF: 63.0 vs 48.2% (Xu); 63.3 vs 47.3% (cleaner).
  - DFD: 34.4 vs 30.2% (Xu); 38.7 vs 22.2% (cleaner); 38.0 vs 23.1% (FairFace).
  - The Xu labels understate the gap on DFD, because actor 19 is labelled M-Black in 8 videos and F-Black in 3.
- **Misses by gender go in opposite directions.** On FF++, men are missed more (FNR 10.23 vs 3.81%). On DFDC, women are missed more (31.90 vs 21.00%, Xu).
- **Skin tone has no consistent direction.**
  - FF++: the lightest ITA quartile has the highest FPR (17.51%; darkest 12.30%).
  - DFDC: darker is higher on every measure (ITA Q1 36.35 vs Q4 23.82%; MST 7-10 36.78 vs 4-6 24.64%), partly through lighting (2.4).
  - Celeb-DF: flat with FF++-train ITA edges (56.2-65.2%). With Celeb-DF's own edges, the darkest quartile is *lowest* (43.7%).
  - DFD: flat for us (31.6-33.3%), while Ding's FPR runs from 82.7% (darkest) to 43.5% (lightest).
- **Celeb-DF Black men are scored more real overall.** Median logit M-Black vs F-White: reals -79.5 vs -68.9; fakes -28.6 vs 19.9. Within-group AUC is 76.5 vs 88.4. This one score offset drives both the iF_FPR term and 29% of misses.

### 2.6 Shared with Ding vs unique to BD0 (matched vs Ding @0.5)

| Set | Our FPs also Ding FP (Ding base FPR) | Our FNs also Ding FN (Ding base FNR) | Ours-only FP / FN frames | Contact sheets: Ding also wrong |
|---|---|---|---|---|
| FF++ | 36.1% (12.54%) | 33.2% (5.15%) | 392 / 958 | FP 12/24, FN 13/24 |
| DFDC | 54.78% (49.13%) | 54.32% (44.40%) | 2,209 / 1,026 | FP 22/24, FN 23/24 |
| Celeb-DF | 76.8% (69.6%) | 41.7% (9.5%, 4.38x) | 778 / 592 | FP 23/24, FN 17/24 |
| DFD | 71.7% (59.0%) | 43.1% (16.4%, 2.6x) | not reported / 388 | FP 23/24, FN 10/24 |

- Ranks agree only weakly. Spearman of the two models' logits is 0.356 / 0.365 on FF++ reals / fakes and 0.246 / 0.287 on DFDC.
- Failures unique to BD0:
  - FF++: F-Asian presenter FPs, and 483 ours-only NeuralTextures FNs (e.g. NT_306_278, which we miss on 28/30 frames and Ding catches on all 30).
  - Celeb-DF: 22 real videos flagged on >= 50% of frames by us and < 50% by Ding (16 of them F-White, e.g. Celeb-real_id57_0001, 90% vs 7%). 5 fake videos we miss and Ding catches, 4 of them with target id60. At A1, 229 fake videos are missed on >= 50% of frames that Ding catches.
  - DFD: 0_17_08__podium_speech_happy__ISS1V0B1, which we miss on 100% of frames and Ding on 0%. The unswapped-person crops (FNR 70.6% vs Ding 26.5%).
- Shared with Ding: the cross-domain FP floods (2.1), the Celeb-DF frame-1 misses and the id60->id61 pair (50.0% vs Ding 51.2%). DFDC's most confident errors are also shared (2.6 table).

## 3. Per-set sections

### 3.1 FF++ test (in-domain)
4,538 real frames (267 videos) and 20,863 fake frames (698 videos). Matched: 613 FP (FPR 13.51%) and 1,434 FN (FNR 6.87%), against Ding's 569 FP (12.54%) and 1,075 FN (5.15%). BD0 wins 0/10 columns against Ding published.

- **i-F_FPR 46.56 vs Ding 24.24.** 71.1% of it comes from three cells: M-Black and M-Others under-flagged (deviations 10.25 and 10.75), and F-Asian over-flagged (12.10). Ding's largest deviations are F-Black 6.11 and M-Asian 5.95.
- **Video concentration.** 17 videos hold half of all FP frames. At a fixed threshold, dropping 15 videos gives i-F_FPR 24.90, but only because pooled FPR falls to 8.03%. Dropped and re-thresholded to the same FPR, 15 videos give 40.85 and 20 give 47.13.
- **Misses.** NeuralTextures holds 54.2% of FNs. Misses cluster by target identity and on male targets. M-Asian FNR is 17.44% and M-Others 13.03%, while every female group is at most 4.02%.
- **Operating points.**
  - Native: FPR 8.09%, FNR 9.54%, i-F_FPR 36.11.
  - A1: FPR 1.87%, FNR 19.63%, i-F_FPR 8.43. The 85 FPs come from 31 videos, half of them from 6 (youtube_845 has 12). 131 fake videos (75 NT) are missed on >= 50% of frames.
  - At A1, 74.2% of Ding's FNs are also ours, while only 8.4% of Ding's FPs are.
- **Label caveat.** youtube_305, our top FP video, is Xu M-White, but the cleaner video-level labels say female / Asian (spot check in `report_build/report_tables.log`).

![FF++ test: 24 most confident false-positive videos (one frame each; caption colour shows whether Ding is also wrong)](ffpp_test/fp_sheet.png)

![FF++ test: 24 most confident false-negative videos](ffpp_test/fn_sheet.png)

### 3.2 DFDC (cross-domain)
14,699 real and 8,158 fake rows (22,857 protocol rows; 1,476 paths appear twice and are kept, as the metric keeps them). Matched: FPR 33.23%, FNR 27.53%, AUC 77.41. Wins 7/10 against Ding published.

- **Threshold shift** (2.1). FPs are spread over 380 of 473 real videos; median per-video FPR is 20.0%, and the top 15 videos hold only 14.70% of FP frames.
- **Lighting and quality** (2.4, corrected wording): dark, low-contrast and the blurriest frames are flagged.
- **Groups.** On Xu labels, i-F_FPR is at chance (p = 0.57) and fragile: greedily removing 9 videos halves it, while random labels need a median of 7 (range 4-14). 76/473 real videos change group between Xu and cleaner labels, including 6 of the top-15 FP videos. On cleaner labels the gap is real (68.79, p = 0.0025) and falls on F-Black and F-Asian. Ding shows it too (62.52).
- **Misses.** Clean clips of white women (2.3). 72/284 fake videos are missed on >= 50% of frames and 8 on every frame (Ding: 128 at >= 50%). The 12 most confident FNs come from 6 videos; 0_4680 (F-White) has 6 frames, logits -119.21 to -111.50.
- **Asian fakes.** DFDC has only 4 Asian fake videos (120 rows; FNR 57.50%), so these numbers are unreliable.

![DFDC: 24 most confident false-positive videos](dfdc/fp_sheet.png)

![DFDC: 24 most confident false-negative videos](dfdc/fn_sheet.png)

### 3.3 Celeb-DF (cross-domain)
Matched: FP 3,349 / 5,713 (58.62%). Celeb-real (58.56%) and YouTube-real (58.8%) are flagged at the same rate. TPR 95.54%, AUC 87.32. Wins 4/10 against Ding published.

- **Flood** (2.1). Native FPR is 40.70%. A1 gives FPR 10.19% at FNR 35.20%.
- **M-Black.** Reals are under-flagged and fakes missed, for the same reason: their scores are lower overall. At A1, M-Black FNR is 53.6% (id60 71.0%) vs Ding's 25.9%. Greedy removal needs 8 videos to halve i-F_FPR (7 of them M-Black videos with 0-3 FP frames each).
- **Frame-1 effect**, shared with Ding (2.4).
- **Most confident errors** *[corrected]*. 9 of the top-12 FP frames are YouTube-real_00247 (logits 59.34-72.25). 10 of the top-12 FN frames are M-Black swaps, with BD logits **-97.67 to -104.38**. The FP sheet is 18 F-White, 5 M-White and 1 M-Black, and Ding flags 23/24. The FN sheet is 16 M-Black and 8 F-White, and Ding detects 7/24.
- **Labels.** Xu and cleaner labels disagree on 6/192 real videos. Example: YouTube-real_00263 is Xu M-White, cleaner non-male, flagged on 100% of frames. Its FP tile visibly shows an older woman *[visual, verifier]*. The FP frames look mostly soft-focus, low-resolution, made-up broadcast frames *[visual]*.

![Celeb-DF: 24 most confident false-positive videos](celebdf/fp_sheet.png)

![Celeb-DF: 24 most confident false-negative videos](celebdf/fn_sheet.png)

### 3.4 DFD (cross-domain)
Matched: FP 1,003 / 3,092 (32.44%), FN 682 / 6,293 (10.84%), AUC 90.21. Ding's reproduction: 59.02% FPR at 83.57% TPR. Wins 7/10 against Ding published.

- **Flood** (2.1). Median real logit per actor ranges from -94.7 to -67.1.
- **Actor 19** drives i-F_FPR (2.2, 2.5). Greedy removal of 10 videos takes it to 0.59.
- **Scene effects.** Outdoor walking scenes cause FPs (2.4) and moving or two-person scenes cause misses (2.3).
- **Confident errors.** The highest FP logit is +66.5 (0_01__walking_down_street_outside_angry). 68 FPs have logit >= 0, and 103 survive A1. The lowest FN logit is -110.5. On the contact sheets Ding is right on only 1/24 FP tiles but on 14/24 FN tiles.
- **No fixed per-identity offset.** Spearman(actor FPR, actor FNR) = -0.25 (p = 0.52, 9 actors). Mean within-actor AUC is 89.1 vs 89.8 pooled.
- **Data issues.**
  - 120 real frames in 4 videos are pixel-identical across actors 22 and 23.
  - The npz `method` column on DFD fakes is the first actor id (100% of rows), not a manipulation method.

![DFD: 24 most confident false-positive videos](dfd/fp_sheet.png)

![DFD: 24 most confident false-negative videos](dfd/fn_sheet.png)

## 4. What the pending BD components are expected to address

BD0 already contains **B (ESA, weight 1, 4 stratified images per cell)**. Every failure above is therefore what remains *with* B. The other arms: BD1 = A without B, BD2 = A+B, BD3 = A+B+C, BD4 = A+B+C+D. What follows is the design intent, matched to the failures; none of it has been measured yet.

| Component | Failures it targets (expected) | Why it may not be enough |
|---|---|---|
| **A: processing augmentation** (JPEG q30-90, bilateral smoothing, blur, down-up resize, gamma/brightness/contrast/saturation, unsharp) | DFDC dark / low-contrast / blurry FPs (2.4). FF++ F-Asian presenter FPs, if they come from makeup, smoothing or broadcast compression as the plan hypothesises. Celeb-DF frame-1 and DFD outdoor-scene effects, if those are compression or blur cues. Possibly part of the cross-domain real-score shift. | The cues are inferred, not shown. A only teaches invariance to processing seen in training. It cannot set a threshold for a new domain. |
| **B: ESA** (already in BD0) | Group-level score offsets on FF++-train groups: the upper tail of real scores per group (FPR gaps), the lower tail of fake scores (FNR gaps). | BD0 still has FF++ i-F_FPR 46.56 and the F-Asian over-flagging. ESA aligns training groups. It cannot fix identity-level offsets in unseen sets (Celeb-DF id60/id61, DFD actor 19) or test-set label noise. Its effect is measured only by BD1 vs BD2. |
| **C: multi-layer evidence** (blocks 8/12/16/20, gate starts at 0) | NeuralTextures and reenactment misses (54.2% / 70.4% of FF++ FNs) and the FF++ appearance prior (the model scoring how fake a person looks rather than the manipulation), through mid-level local evidence. | The gate starts at 0 and the heads get no direct map loss, so C may stay unused. It does not address FPs. |
| **D: group-AUC floor** (BD4 only) | Within-group ranking gaps: FF++ male-target and M-Asian misses (FNR 17.44%), Celeb-DF M-Black within-group AUC 76.5 vs 88.4. | Applied on FF++-train groups. The worst AUC gaps are cross-domain and on 1-3 identities. |

**What nothing in BD addresses:**

1. Transferring the operating point across domains. The matched threshold is set on FF++ val. The 33-59% cross-domain FPRs are mainly this, and only a per-domain threshold or calibration step would change it.
2. Statistical fragility of i-F_FPR on tiny cells: FF++ M-Black has 7 real videos, DFD F-Black is one actor, Celeb-DF M-Black is 3 identities. Most Xu-label gaps are inside the permutation null.
3. Label noise in the Xu groups. 76/473 DFDC real videos change group under cleaner labels; DFD actor 19 is split across genders; FF++ youtube_305 is mislabelled.
4. Dataset defects:
   - DFD: probable crops of the unswapped person (FNR 70.6%) and 120 duplicate real frames.
   - DFDC: "real" clips with compositing or overlay artefacts *[visual]*, and no method labels.
5. Celeb-DF frame-1 misses, unless A happens to cover the cue; the cause is untested.

## 5. Caveats

- **Single seed.** These are BD0 seed 7 only. Seed-to-seed variation in which videos fail is unmeasured, so the per-video findings may not hold for other seeds.
- **Tiny groups.**
  - FF++ M-Black: 7 real videos, M-Others: 12.
  - DFDC Asian fakes: 4 videos.
  - Celeb-DF M-Black: 3 identities; F-Others: 3 real videos and 1 fake video.
  - DFD F-Black: 1 actor, 3 videos.
  - Frame-level counts overstate the effective sample size; videos are the real unit.
- **Label noise.** The Xu labels (which the metric uses) disagree with the cleaner AI-Face labels on 76/473 DFDC and 6/192 Celeb-DF real videos, and on key FP videos in FF++ and DFD. Group conclusions that flip between label sets are flagged above.
- **Matched threshold.** It is defined on FF++ val TPR, so cross-domain error rates measure threshold transfer as much as detection.
- **Ding comparison** uses the reproduction (seed 7, logit >= 0), not Ding's own scores. The published column counts come from bd_eval.json.
- **Visual observations** (news-anchor appearance, compositing artefacts, YouTube-real_00263) are not quantified.
- **Other labels.** DFD "methods" are actor ids, and DFDC has no method labels.
- **Inferred causes.** The frame-1, lighting and appearance-prior explanations are associations, not tested causes.
