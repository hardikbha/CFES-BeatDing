I could not save the requested file at `.../agents/RESNET_STRAT/2026-11-06-codex-resnet-strategy.md`: the read-only sandbox rejected both the direct write and the patch, and a check confirmed the file does not exist. I did not launch jobs, GPU processes, or tests. The memo is below.

## Recommendation and feasibility

Keep Ding’s natural real/fake training stream and add a **low-weight, class-conditional feature alignment and real-score tail loss** on a separate, balanced FF++ training batch. Apply the same SWA procedure to the fairness arm and its R0 control. This avoids R1’s main confound: its CE sampler changes the training distribution by assigning equal mass to every intersection cell.

A mean of 25/40 is not a credible forecast from the available evidence. R0 scored 6/40, R1 scored 7/40, and the earlier ResNet CES-FL result scored 11/40. R1’s four AUCs all fell versus R0. A realistic near-term range is **10–16/40**, with **18/40 an optimistic ceiling**. The required AUC at least equal to Ding on all four sets is also a difficult, unmet condition.

## Diagnosis

Ding’s scorer defines F_FPR as the sum of each eligible group’s absolute FPR deviation from the overall FPR. Each group contributes equally, so small groups can dominate. F_DP measures the range of group prediction rates and is label-blind. es-AUC depends on both overall AUC and subgroup AUC deviations, so it reflects ranking quality as well as group disparity.

At probability 0.5, the current R0 and R1 results compare as follows:

| Test set | Ding AUC | R0 AUC | R1 AUC | R0 wins | R1 wins |
|---|---:|---:|---:|---:|---:|
| FF++ | 95.53 | 95.11 | 89.31 | 3/10 | 1/10 |
| DFDC | 62.54 | 59.37 | 55.48 | 0/10 | 0/10 |
| Celeb-DF | 77.35 | 76.56 | 68.61 | 1/10 | 4/10 |
| DFD | 78.63 | 73.87 | 67.03 | 2/10 | 2/10 |
| **Total** | — | — | — | **6/40** | **7/40** |

These are single-seed screens, not seed-mean estimates. R1’s four Celeb-DF wins are threshold fairness columns; its AUC and all es-AUC columns still lose.

The R0 group metrics explain the main F_FPR losses:

- **FF++:** Overall FPR is 34.22%. Gender FPRs are nearly equal: 34.19% male and 34.25% nonmale, so gender F_FPR is only 0.06. Race FPRs are Asian 44.17%, White 31.13%, Black 45.71%, and Others 38.55%; their deviations sum to 28.87 points, versus Ding’s 18.67. Intersection F_FPR is 58.69, driven especially by nonmale-Black FPR of 55.93%. R0 wins three columns but misses AUC (95.11 vs 95.53) and intersection es-AUC (80.84 vs 82.87).

- **DFDC:** Overall FPR is 55.85%. Gender FPRs of 50.33% and 62.27% yield F_FPR 11.94, versus Ding’s 3.97. Race FPRs—Asian 63.63%, White 52.01%, Black 63.85%, Others 49.56%—yield 25.90, versus 11.94. Intersection F_FPR is 62.83, versus 32.14: male-Asian is 68.12%, male-White 43.39%, and nonmale-Black 64.95%. Overall AUC is also low (59.37 vs 62.54), as is intersection es-AUC (32.24 vs 37.55).

- **Celeb-DF:** Overall FPR is 78.23%; R0 predicts fake for most images at the 0.5 threshold. Gender FPRs stay near that rate, producing a gender F_FPR win (2.78 vs Ding 4.53). Race FPRs are White 78.03%, Black 75.77%, and Others 96.00%, yielding F_FPR 20.42 vs Ding 6.76. The Others group has 180 total images and contributes about 17.78 points. The evaluated intersection group nonmale-Others has 120 images, including 90 real, and 94.44% FPR; it contributes about 16.22 points to intersection F_FPR 19.14, versus Ding’s 5.56. Overall AUC is close (76.56 vs 77.35), but race and intersection es-AUC are 56.29 and 54.73, versus 71.03 and 68.25.

- **DFD:** Overall FPR is 62.29%. Gender F_FPR is a win (1.78 vs 3.12), but race FPRs of 57.62% White and 76.36% Black yield 18.74, versus 6.51. Intersection FPRs are 49.55% male-White, 78.74% male-Black, 61.65% nonmale-White, and 57.95% nonmale-Black; their deviations sum to 34.16, versus 20.99. Overall AUC is 73.87 vs 78.63 and intersection es-AUC is 61.82 vs 66.19.

**Threshold placement and ranking are separate problems.** R0’s FF++-validation FPR10 threshold is logit 3.8372. Applying it to test data changes F_FPR inconsistently: Celeb-DF race rises from 20.42 to 42.80 and intersection from 19.14 to 43.89, while DFD race improves from 18.74 to 12.41. Celeb-DF gender F_DP rises from 12.14 to 31.60. AUC and es-AUC are unchanged by threshold selection, so calibration cannot repair the cross-domain ranking deficits. Ding’s published table lacks overall FPR, so it is not possible to quantify how much of each R0/Ding fairness gap is due to global threshold shift.

R1 is a poor base to extend. It samples 8 real and 8 fake images from each of eight cells, despite FF++ train cell sizes ranging from 182 to 5,190 reals and 929 to 26,091 fakes. This reweights the CE stream substantially. The R2 smoke trace shows the current worst-two real CE term’s gradient norm at 0.5–4.83× the CE gradient; the margin term is 0.90–3.77× CE at its current coefficient. Those are strong losses, not small regularizers.

## Ranked strategies

### 1. Natural-prior CE with weak feature and real-tail constraints

**Recipe:** Keep ImageNet ResNet-50, Ding’s FF++ split and transforms, SGD (lr 1e-3, momentum 0.9, weight decay 5e-3), batches of 64 real + 64 fake, no augmentation, and 50 epochs. Every fourth update, draw a separate fair batch of 8 real and 8 fake training images from each gender×race cell. Use an independent deterministic sampler.

Let s = z_fake − z_real and f be the L2-normalized pooled 2048-D penultimate feature. On the fair batch:

- L_C is the mean, over class and group, of the squared difference between group and class-pooled feature means plus the squared Frobenius difference between their population covariances.
- L_R is the mean of the two largest per-cell real-image means of softplus(s + 1).
- On fair steps, optimize L_CE(R0 batch) + 4 × [0.5 L_C + 0.02 L_R]. On the other three updates, optimize only L_CE. The factor 4 compensates for applying the auxiliary loss once per four updates.

The feature term targets cross-domain AUC and es-AUC; the real-tail term targets the large race/intersection F_FPR gaps. The auxiliary batch adds about 25% compute. Its small cells make covariance estimates noisy, and FF++ group fairness may not transfer.

This is an incremental method. Deep CORAL aligns second-order features, PG-FDD studies demographic/domain-agnostic forgery features, DAW-FDD uses group-aware fairness losses, and Ding’s CVPR 2026 work combines channel decoupling and global distribution alignment. The proposed distinction is a controlled, single-stage ResNet study that preserves natural-prior CE and adds class-conditional group alignment plus a real-score tail anchor; it is not a new general fairness principle. [Deep CORAL](https://arxiv.org/abs/1607.01719), [PG-FDD](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_Preserving_Fairness_Generalization_in_Deepfake_Detection_CVPR_2024_paper.html), [DAW-FDD](https://openaccess.thecvf.com/content/WACV2024/html/Ju_Improving_Fairness_in_Deepfake_Detection_WACV_2024_paper.html), [Ding et al.](https://github.com/ywh1093/Fairness-Optimization).

**Compute:** Six runs for three paired seeds: roughly 18–24 GPU-h for base training, 2–3 for auxiliary forwards, and 2–4 for scoring and BatchNorm passes; about **22–31 GPU-h total**.

### 2. SWA as a shared recipe

Average weights from epochs 40–49, then recompute BatchNorm statistics over FF++ train. Apply it identically to R0 and the fairness arm. It may improve AUC and es-AUC, especially on DFD and DFDC, but is not expected to repair F_DP by itself. It has no extra training FLOPs, but a matched three-seed experiment still costs around **20–28 GPU-h including evaluation**. SWA is established optimization, not method novelty. [Izmailov et al.](https://proceedings.mlr.press/r16/izmailov18a.html).

### 3. SBI pseudo-fakes, only with a matched control

Pre-generate SBIs from FF++ train reals only. Keep batch size fixed and replace exactly 32 of the 64 fake slots with SBIs in both R0 and the fairness arm. Apply identical pseudo-fakes, quotas, transforms, and seeds to both arms. This may improve cross-dataset AUC and es-AUC, but pseudo-fakes can become their own shortcut and do not directly optimize real-score tails. SBI is prior work, so any fairness claim must come from a paired method comparison. Budget **18–24 GPU-h for six training runs**, plus an uncertain **3–8 GPU-h** pre-generation allowance. [Shiohara and Yamasaki, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Shiohara_Detecting_Deepfakes_With_Self-Blended_Images_CVPR_2022_paper.pdf).

### 4. One global FF++-validation threshold as a secondary analysis

For each seed, select the 90th percentile logit among FF++ validation reals and apply that same threshold to all four test sets. Do not fit per-group thresholds or report this as primary. It may change F_FPR and F_DP, but cannot change AUC or es-AUC. Existing results show inconsistent transfer and large Celeb-DF regressions. It requires no training, only validation inference.

## Best plan to execute

**Arms:** R0-SWA is Ding’s unchanged CE path and natural sampler with SWA/BatchNorm handling. R3-SWA is identical except for the auxiliary loss above. Do not include R1’s balanced CE sampler, W1 alignment, or full-strength worst-two CE loss.

**Seeds:** 37, 47, 57 as fresh confirmatory seeds; earlier 7/17/27 results remain screening evidence. Fix 50 epochs, SWA epochs 40–49, and all lambdas before the runs.

**Eight-GPU schedule:**

1. On 6 Nov, implement R3 and complete CPU/unit checks. Freeze code hash, config, seeds, success rule, and kill rule.
2. On 7 Nov, run a short finite-gradient smoke for R0 and R3. Use it only to verify correctness, not tune settings.
3. On 8–10 Nov, use six GPU slots for the six jobs: paired R0/R3 runs for seeds 37, 47, and 57. Keep each pair on the same host if possible. Reserve two slots for scoring/BatchNorm passes and shared-cluster work. Budget about 4–5 hours wall time for training at full utilization, plus queue contention.
4. By 11 Nov, produce the fixed SWA checkpoints and score each once on all four test sets at probability 0.5. Compute the secondary threshold from FF++ validation and apply it unchanged to every test set.
5. By 12–13 Nov, report each seed, mean, standard deviation, paired R0 comparison, and both operating points. Do not tune against the test results.

**Predeclared criteria:** Target success requires at least 25/40 mean wins and mean AUC at least Ding’s published AUC on each of the four datasets. For evidence that the fairness method helps, require at least 3/40 more mean wins than paired R0-SWA and no more than 0.5 AUC-point regression versus paired R0-SWA on any dataset. Report per-seed differences; three seeds do not support a precise population estimate.

**Kill rule:** Stop an R3 seed without restoring an earlier checkpoint if its FF++ validation AUC is more than 2 points below paired R0 at both epochs 10 and 15, or if the combined auxiliary-gradient norm exceeds the CE norm on more than 20% of audited fair steps. Record it as killed; do not substitute a best-validation epoch. Nonfinite losses or gradients are an immediate run failure.

## Implementation notes

Modify `agents/SAMEBB/code_r2/Ding_r2/train_r2.py` to add an R3 branch using the existing R0 `CombinedDataset` CE stream and a separate fair-batch sampler built from `manifest_gids`. Keep its RNG independent of the R0 sampler. Reuse `r2_losses.py`’s CORAL and margin functions with margin 1 and coefficient 0.02; use the ResNet-50 `adjust_channel` feature hook for the pooled feature. Add SWA accumulation for epochs 40–49 and a BatchNorm pass over FF++ train only. Make the scorer accept the averaged checkpoint directly; keep `op05` primary and `valFPR10` secondary.

Required checks include preserving R0’s bit-identical update path, exact balanced-batch quotas and train-only sampling, finite/nonzero gradients for the losses, the fair-step frequency and factor-of-four scaling, ten SWA snapshots, train-only BatchNorm recalibration, and assertions that the test scorer performs no test threshold search or calibration. **These checks were not run.**
2,16,775
I could not save the requested file at `.../agents/RESNET_STRAT/2026-11-06-codex-resnet-strategy.md`: the read-only sandbox rejected both the direct write and the patch, and a check confirmed the file does not exist. I did not launch jobs, GPU processes, or tests. The memo is below.

## Recommendation and feasibility

Keep Ding’s natural real/fake training stream and add a **low-weight, class-conditional feature alignment and real-score tail loss** on a separate, balanced FF++ training batch. Apply the same SWA procedure to the fairness arm and its R0 control. This avoids R1’s main confound: its CE sampler changes the training distribution by assigning equal mass to every intersection cell.

A mean of 25/40 is not a credible forecast from the available evidence. R0 scored 6/40, R1 scored 7/40, and the earlier ResNet CES-FL result scored 11/40. R1’s four AUCs all fell versus R0. A realistic near-term range is **10–16/40**, with **18/40 an optimistic ceiling**. The required AUC at least equal to Ding on all four sets is also a difficult, unmet condition.

## Diagnosis

Ding’s scorer defines F_FPR as the sum of each eligible group’s absolute FPR deviation from the overall FPR. Each group contributes equally, so small groups can dominate. F_DP measures the range of group prediction rates and is label-blind. es-AUC depends on both overall AUC and subgroup AUC deviations, so it reflects ranking quality as well as group disparity.

At probability 0.5, the current R0 and R1 results compare as follows:

| Test set | Ding AUC | R0 AUC | R1 AUC | R0 wins | R1 wins |
|---|---:|---:|---:|---:|---:|
| FF++ | 95.53 | 95.11 | 89.31 | 3/10 | 1/10 |
| DFDC | 62.54 | 59.37 | 55.48 | 0/10 | 0/10 |
| Celeb-DF | 77.35 | 76.56 | 68.61 | 1/10 | 4/10 |
| DFD | 78.63 | 73.87 | 67.03 | 2/10 | 2/10 |
| **Total** | — | — | — | **6/40** | **7/40** |

These are single-seed screens, not seed-mean estimates. R1’s four Celeb-DF wins are threshold fairness columns; its AUC and all es-AUC columns still lose.

The R0 group metrics explain the main F_FPR losses:

- **FF++:** Overall FPR is 34.22%. Gender FPRs are nearly equal: 34.19% male and 34.25% nonmale, so gender F_FPR is only 0.06. Race FPRs are Asian 44.17%, White 31.13%, Black 45.71%, and Others 38.55%; their deviations sum to 28.87 points, versus Ding’s 18.67. Intersection F_FPR is 58.69, driven especially by nonmale-Black FPR of 55.93%. R0 wins three columns but misses AUC (95.11 vs 95.53) and intersection es-AUC (80.84 vs 82.87).

- **DFDC:** Overall FPR is 55.85%. Gender FPRs of 50.33% and 62.27% yield F_FPR 11.94, versus Ding’s 3.97. Race FPRs—Asian 63.63%, White 52.01%, Black 63.85%, Others 49.56%—yield 25.90, versus 11.94. Intersection F_FPR is 62.83, versus 32.14: male-Asian is 68.12%, male-White 43.39%, and nonmale-Black 64.95%. Overall AUC is also low (59.37 vs 62.54), as is intersection es-AUC (32.24 vs 37.55).

- **Celeb-DF:** Overall FPR is 78.23%; R0 predicts fake for most images at the 0.5 threshold. Gender FPRs stay near that rate, producing a gender F_FPR win (2.78 vs Ding 4.53). Race FPRs are White 78.03%, Black 75.77%, and Others 96.00%, yielding F_FPR 20.42 vs Ding 6.76. The Others group has 180 total images and contributes about 17.78 points. The evaluated intersection group nonmale-Others has 120 images, including 90 real, and 94.44% FPR; it contributes about 16.22 points to intersection F_FPR 19.14, versus Ding’s 5.56. Overall AUC is close (76.56 vs 77.35), but race and intersection es-AUC are 56.29 and 54.73, versus 71.03 and 68.25.

- **DFD:** Overall FPR is 62.29%. Gender F_FPR is a win (1.78 vs 3.12), but race FPRs of 57.62% White and 76.36% Black yield 18.74, versus 6.51. Intersection FPRs are 49.55% male-White, 78.74% male-Black, 61.65% nonmale-White, and 57.95% nonmale-Black; their deviations sum to 34.16, versus 20.99. Overall AUC is 73.87 vs 78.63 and intersection es-AUC is 61.82 vs 66.19.

**Threshold placement and ranking are separate problems.** R0’s FF++-validation FPR10 threshold is logit 3.8372. Applying it to test data changes F_FPR inconsistently: Celeb-DF race rises from 20.42 to 42.80 and intersection from 19.14 to 43.89, while DFD race improves from 18.74 to 12.41. Celeb-DF gender F_DP rises from 12.14 to 31.60. AUC and es-AUC are unchanged by threshold selection, so calibration cannot repair the cross-domain ranking deficits. Ding’s published table lacks overall FPR, so it is not possible to quantify how much of each R0/Ding fairness gap is due to global threshold shift.

R1 is a poor base to extend. It samples 8 real and 8 fake images from each of eight cells, despite FF++ train cell sizes ranging from 182 to 5,190 reals and 929 to 26,091 fakes. This reweights the CE stream substantially. The R2 smoke trace shows the current worst-two real CE term’s gradient norm at 0.5–4.83× the CE gradient; the margin term is 0.90–3.77× CE at its current coefficient. Those are strong losses, not small regularizers.

## Ranked strategies

### 1. Natural-prior CE with weak feature and real-tail constraints

**Recipe:** Keep ImageNet ResNet-50, Ding’s FF++ split and transforms, SGD (lr 1e-3, momentum 0.9, weight decay 5e-3), batches of 64 real + 64 fake, no augmentation, and 50 epochs. Every fourth update, draw a separate fair batch of 8 real and 8 fake training images from each gender×race cell. Use an independent deterministic sampler.

Let s = z_fake − z_real and f be the L2-normalized pooled 2048-D penultimate feature. On the fair batch:

- L_C is the mean, over class and group, of the squared difference between group and class-pooled feature means plus the squared Frobenius difference between their population covariances.
- L_R is the mean of the two largest per-cell real-image means of softplus(s + 1).
- On fair steps, optimize L_CE(R0 batch) + 4 × [0.5 L_C + 0.02 L_R]. On the other three updates, optimize only L_CE. The factor 4 compensates for applying the auxiliary loss once per four updates.

The feature term targets cross-domain AUC and es-AUC; the real-tail term targets the large race/intersection F_FPR gaps. The auxiliary batch adds about 25% compute. Its small cells make covariance estimates noisy, and FF++ group fairness may not transfer.

This is an incremental method. Deep CORAL aligns second-order features, PG-FDD studies demographic/domain-agnostic forgery features, DAW-FDD uses group-aware fairness losses, and Ding’s CVPR 2026 work combines channel decoupling and global distribution alignment. The proposed distinction is a controlled, single-stage ResNet study that preserves natural-prior CE and adds class-conditional group alignment plus a real-score tail anchor; it is not a new general fairness principle. [Deep CORAL](https://arxiv.org/abs/1607.01719), [PG-FDD](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_Preserving_Fairness_Generalization_in_Deepfake_Detection_CVPR_2024_paper.html), [DAW-FDD](https://openaccess.thecvf.com/content/WACV2024/html/Ju_Improving_Fairness_in_Deepfake_Detection_WACV_2024_paper.html), [Ding et al.](https://github.com/ywh1093/Fairness-Optimization).

**Compute:** Six runs for three paired seeds: roughly 18–24 GPU-h for base training, 2–3 for auxiliary forwards, and 2–4 for scoring and BatchNorm passes; about **22–31 GPU-h total**.

### 2. SWA as a shared recipe

Average weights from epochs 40–49, then recompute BatchNorm statistics over FF++ train. Apply it identically to R0 and the fairness arm. It may improve AUC and es-AUC, especially on DFD and DFDC, but is not expected to repair F_DP by itself. It has no extra training FLOPs, but a matched three-seed experiment still costs around **20–28 GPU-h including evaluation**. SWA is established optimization, not method novelty. [Izmailov et al.](https://proceedings.mlr.press/r16/izmailov18a.html).

### 3. SBI pseudo-fakes, only with a matched control

Pre-generate SBIs from FF++ train reals only. Keep batch size fixed and replace exactly 32 of the 64 fake slots with SBIs in both R0 and the fairness arm. Apply identical pseudo-fakes, quotas, transforms, and seeds to both arms. This may improve cross-dataset AUC and es-AUC, but pseudo-fakes can become their own shortcut and do not directly optimize real-score tails. SBI is prior work, so any fairness claim must come from a paired method comparison. Budget **18–24 GPU-h for six training runs**, plus an uncertain **3–8 GPU-h** pre-generation allowance. [Shiohara and Yamasaki, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Shiohara_Detecting_Deepfakes_With_Self-Blended_Images_CVPR_2022_paper.pdf).

### 4. One global FF++-validation threshold as a secondary analysis

For each seed, select the 90th percentile logit among FF++ validation reals and apply that same threshold to all four test sets. Do not fit per-group thresholds or report this as primary. It may change F_FPR and F_DP, but cannot change AUC or es-AUC. Existing results show inconsistent transfer and large Celeb-DF regressions. It requires no training, only validation inference.

## Best plan to execute

**Arms:** R0-SWA is Ding’s unchanged CE path and natural sampler with SWA/BatchNorm handling. R3-SWA is identical except for the auxiliary loss above. Do not include R1’s balanced CE sampler, W1 alignment, or full-strength worst-two CE loss.

**Seeds:** 37, 47, 57 as fresh confirmatory seeds; earlier 7/17/27 results remain screening evidence. Fix 50 epochs, SWA epochs 40–49, and all lambdas before the runs.

**Eight-GPU schedule:**

1. On 6 Nov, implement R3 and complete CPU/unit checks. Freeze code hash, config, seeds, success rule, and kill rule.
2. On 7 Nov, run a short finite-gradient smoke for R0 and R3. Use it only to verify correctness, not tune settings.
3. On 8–10 Nov, use six GPU slots for the six jobs: paired R0/R3 runs for seeds 37, 47, and 57. Keep each pair on the same host if possible. Reserve two slots for scoring/BatchNorm passes and shared-cluster work. Budget about 4–5 hours wall time for training at full utilization, plus queue contention.
4. By 11 Nov, produce the fixed SWA checkpoints and score each once on all four test sets at probability 0.5. Compute the secondary threshold from FF++ validation and apply it unchanged to every test set.
5. By 12–13 Nov, report each seed, mean, standard deviation, paired R0 comparison, and both operating points. Do not tune against the test results.

**Predeclared criteria:** Target success requires at least 25/40 mean wins and mean AUC at least Ding’s published AUC on each of the four datasets. For evidence that the fairness method helps, require at least 3/40 more mean wins than paired R0-SWA and no more than 0.5 AUC-point regression versus paired R0-SWA on any dataset. Report per-seed differences; three seeds do not support a precise population estimate.

**Kill rule:** Stop an R3 seed without restoring an earlier checkpoint if its FF++ validation AUC is more than 2 points below paired R0 at both epochs 10 and 15, or if the combined auxiliary-gradient norm exceeds the CE norm on more than 20% of audited fair steps. Record it as killed; do not substitute a best-validation epoch. Nonfinite losses or gradients are an immediate run failure.

## Implementation notes

Modify `agents/SAMEBB/code_r2/Ding_r2/train_r2.py` to add an R3 branch using the existing R0 `CombinedDataset` CE stream and a separate fair-batch sampler built from `manifest_gids`. Keep its RNG independent of the R0 sampler. Reuse `r2_losses.py`’s CORAL and margin functions with margin 1 and coefficient 0.02; use the ResNet-50 `adjust_channel` feature hook for the pooled feature. Add SWA accumulation for epochs 40–49 and a BatchNorm pass over FF++ train only. Make the scorer accept the averaged checkpoint directly; keep `op05` primary and `valFPR10` secondary.

Required checks include preserving R0’s bit-identical update path, exact balanced-batch quotas and train-only sampling, finite/nonzero gradients for the losses, the fair-step frequency and factor-of-four scaling, ten SWA snapshots, train-only BatchNorm recalibration, and assertions that the test scorer performs no test threshold search or calibration. **These checks were not run.**
