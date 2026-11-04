# FairForge-2 seed-17 model results and findings

**Observation:** 2026-11-05, approximately 01:57 IST (management host). This is a validation-only snapshot from actual run receipts and histories. No TEST scores are included. The report distinguishes verified runs from epoch histories that lack completion verification.

## Executive summary

- **Six training runs are completed and verified:** three matched 50-epoch Xception baselines (BCE, DAG-FDD, DAW-FDD) and three CES-FL experiments (P, A, B).
- **F3Net and CORE each have a 50-epoch history but are not verified complete:** neither output has `DONE` or `RUN_VERIFIED.json`. Their epoch-50 values below are provisional history records.
- **The FairForge-2 Ding-style run is still in corrected GDA training:** latest recorded event is epoch 40/50, batch 3,800; last completed validation event is epoch 39. It has no final selected result.
- **Age and skin-tone proxy metrics are available** in addition to gender, race and gender×race. Age labels are automatic estimates; skin bins are automatic ITA proxies, not human-verified ground truth.
- CES-FL's full sensitivity/response-alignment objective has **not** been evaluated on FairForge-2. The verified CES-FL rows are P (original-only BCE), A (weighted-BCE control), and B (A plus soft error-disparity loss).

## Dataset and evaluation protocol

All six verified runs use seed 17 and the same public FairForge-2 TRAIN/VAL manifests:

| Split | Images | Real | Fake | Manifest SHA256 |
|---|---:|---:|---:|---|
| TRAIN | 218,266 | 16,777 | 201,489 | `4f83e7dc8042533329d7047fefd0f30cbd96f45d42bde2666dd0edc7d58f29cc` |
| VAL | 31,107 | 2,438 | 28,669 | `aeda051259d9e26b07c2810152b601ae115ce7c9e10523f99ea07bbc9f92afa8` |

The global threshold is calibrated separately per model from VAL real images for a 10% target FPR; achieved VAL FPR is 243/2,438 = **9.9672%**. Checkpoints use the configured validation-AUC selection rule. These are in-domain validation results, not TEST results or direct comparisons to Ding's published TEST numbers. No TEST inference, test-based model selection, or test-calibrated threshold was used. Where the preparation workflow inspected test metadata for split-integrity checks, that was not test inference.

## Verified detection results

Percentages are shown for AUC/FPR/FNR. `selected epoch` is the validation-selected checkpoint; all rows have a completion/verification receipt.

| Run | Training | Selected epoch | VAL AUC % | VAL FPR % | VAL FNR % |
|---|---:|---:|---:|---:|---:|
| Xception BCE | 50/50 | 40 | 99.8046 | 9.9672 | 0.3000 |
| DAG-FDD | 50/50 | 50 | 99.8136 | 9.9672 | 0.2197 |
| DAW-FDD | 50/50 | 41 | 99.8360 | 9.9672 | 0.2442 |
| CES-FL P — original-only BCE | 10/10 | 9 | 99.8105 | 9.9672 | 0.3802 |
| CES-FL A — weighted-BCE control | 10/10 | 7 | 99.8085 | 9.9672 | 0.3314 |
| CES-FL B — A + soft error disparity | 10/10 | 1 | 99.8075 | 9.9672 | 0.3697 |

## Verified fairness results

The following are the run-record `F_FPR` values in native fraction units (lower is better). They are aggregate group-versus-pooled FPR disparity measures, not the overall FPR and not a percentage. `F_DP` and `es-AUC` are also in the run records; this table emphasizes the primary fairness family across all available demographic axes.

| Run | Gender | Race | Gender×race | Age bins | Skin bins |
|---|---:|---:|---:|---:|---:|
| Xception BCE | 0.0343 | 0.1954 | 0.4052 | 0.0874 | 0.1457 |
| DAG-FDD | 0.0428 | 0.1832 | 0.3789 | 0.1045 | 0.1278 |
| DAW-FDD | 0.0343 | 0.1287 | 0.3070 | 0.1045 | 0.0680 |
| CES-FL P | 0.0360 | 0.1216 | 0.2839 | 0.0923 | 0.1168 |
| CES-FL A | 0.0326 | 0.1280 | 0.2910 | 0.0902 | 0.1062 |
| CES-FL B | 0.0343 | 0.1317 | 0.2955 | 0.0742 | 0.1156 |

### Validation group support (real/fake counts)

- Gender: Female 1,001/11,613; Male 1,437/17,056.
- Race: Asian 524/6,326; Black 432/5,344; White 513/6,144; Others 969/10,855.
- Gender×race: Female-Asian 202/2,493; Female-Black 194/2,272; Female-White 257/2,966; Female-Others 348/3,882; Male-Asian 322/3,833; Male-Black 238/3,072; Male-White 256/3,178; Male-Others 621/6,973.
- Age bins: `<30` 260/4,586; `30–44` 1,142/14,025; `45–59` 662/6,686; `60+` 374/3,372.
- Skin bins: `0` 630/7,713; `1` 584/7,238; `2` 551/6,199; `3` 515/5,746; unknown 158/1,773. Unknown is retained in support reporting and excluded from the skin-bin fairness aggregate.

All listed groups exceed the configured minimum support of 20 examples per class. Age and skin categories remain proxy-label analyses; support does not establish label validity.

## Additional histories and active run

These values are **not counted among the six verified runs**. They are the latest epoch-50 validation records, not certified selected-checkpoint replays.

| Model | Recorded epochs | Epoch-50 AUC % | FPR % | FNR % | Gender×race F_FPR | Age F_FPR | Skin F_FPR | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| F3Net AI-Face FAD-only adaptation | 50/50 | 99.3585 | 9.9672 | 2.4207 | 0.5542 | 0.1908 | 0.1753 | No `DONE` / `RUN_VERIFIED.json` |
| CORE paired original/flip adaptation | 50/50 | 99.7781 | 9.9672 | 0.4046 | 0.4084 | 0.1537 | 0.1348 | No `DONE` / `RUN_VERIFIED.json` |

CORE is an external paired-view consistency baseline, not our method. F3Net is explicitly the AI-Face FAD-only adaptation, not the complete dual-branch paper method. Both need selected-checkpoint replay and provenance verification before being tabled as completed results.

The separate Ding-style FairForge-2 job is running its **corrected GDA** stage, not a final published-method result. The latest recorded training event is epoch 40/50, batch 3,800; the latest completed GDA validation event is epoch 39 with AUC 99.0579%. Do not compare this partial validation score with a completed run or Ding's paper TEST results.

## What the results say and how to improve the evidence

1. **Detection is strong on this validation split, but this is not a TEST claim.** The six verified AUCs are 99.8046–99.8360%. The repeated 9.9672% FPR reflects the common validation-real calibration target; it does not show equal test FPR.
2. **Fairness is axis-dependent.** Among the three 50-epoch Xception fairness baselines, DAW-FDD has the lowest race, gender×race and skin-bin `F_FPR`, while its age-bin `F_FPR` is higher than BCE's. No method is uniformly best across all axes.
3. **CES-FL B is mixed versus its A control.** B lowers age-bin `F_FPR` from 0.0902 to 0.0742, but increases gender, race, gender×race and skin-bin `F_FPR`; its AUC is lower by about 0.0011 percentage points. This does not establish an overall fairness improvement.
4. **CES-FL P/A/B do not validate the full counterfactual claim.** P is original-only BCE, A is weighted BCE, and B adds the soft error-disparity term. Sensitivity/control-stability and conditional-response-alignment arms have no verified FairForge-2 run. Their launch remains gated by the unresolved intervention review; do not relax that gate or infer validity from visual appearance.
5. **The cross-method table is not yet paper-ready.** Baseline and CES-FL runs use different training budgets/recipes, F3Net and CORE lack completion verification, Ding's corrected GDA is incomplete, and there is one seed. The existing baseline uncertainty analysis has only 490 usable bootstrap draws for the primary gender×race `F_FPR`, below its 900-draw acceptance minimum. No TEST scores are included.

Recommended sequence: verify F3Net/CORE checkpoints and selected-VAL replays; finish and verify Ding's current stage without calling it an exact paper reproduction; preserve a common protocol and add the full CES-FL arms only after valid intervention evidence exists; then run a preregistered multi-seed confirmation with contributor-cluster uncertainty. Freeze model selection and thresholds before any one-time TEST evaluation. Human-review or improve the age/skin labels before making strong claims on those axes.

## Provenance

All checkpoint binaries, private images, full row-level predictions and logs remain on the cluster. The report contains aggregates and hashes only.

| Run family | Source commit | Configuration / checkpoint evidence |
|---|---|---|
| BCE / DAG-FDD / DAW-FDD | `815b4d4bc679c55d4ee6323cc7284335ce210568` | Config digests: BCE `904ac0b50d745b4ef2f53fa3f84b688cc01c405eb12b0d192ec4f317570827bd`; DAG `36dbef68848db48761470018002aeac0fc44c2333a3d0f5709fdecc2c1083163`; DAW `2425e539b1afcf5f5f667fd6e424d8468b62601b4df4fccb966e12e05e8be6cd`. Selected checkpoint SHA256s: BCE `a62af6bc1ff02f4280fcb38e9b9a4b035ccee3427472d160880b4b9e0c31d835`; DAG `7cd8e8511910ea8e94d6adb8f44c5890cc1f08d7d1fe8b63f41043f73576309f`; DAW `63b53852dfd61d8c44a25bd8fcb1baf90a67719fcccb94225892db49c6ea4036`. |
| CES-FL P | `875bce025084871438ef799ca9086985c4375672` | Config SHA256 `db0c886527635295f0394dddbca9da1a8a3c40807609ab78f324b75b382fbee8`; selected checkpoint `43152de1b98dce48801114efb2454cb0bc6aa7400445dd5f139759022229895a`. |
| CES-FL A / B | `c332fd34fb603b8919b15fee08e8d022fb9f57c8` | Source file SHA256 `89530025a00b9b0405fba7026d3eed5f2749d86620174ad3ffde93a44f8b770b`; config digest `212645fe0e196651e0f6728708edfac5f2b87fa7cde508e19f7b799887a1f522`; selected checkpoints A `3b504d4e038c1ddc266f3e5179cdf78d052176f936705fe94b4c2caf2e227dd6`, B `542111034536fc0b35a3e203192d1b377f9477b3c60d593eb28c8d4cd6531bd3`. |
| F3Net / CORE | `e7648fb25aeb16d49bbef33ad998e1b1cf5a5016` | Config SHA256s: F3Net `410a36caa9fafb22384421211b7fbdd8c652858c7b36022c27ad6d1f17b5030e`; CORE `e8a318c2d30a9c240b5f8fa5f6e9150e34d88727899732a6ba51992523fc6e4a`. Histories only; no verified selected checkpoint receipt. |
| Ding corrected-GDA run | `8487d98303fd5d7766fef2fa3dd32c37f38e5f4b` | Active/incomplete at observation; no final checkpoint result in this report. |

The report branch is based on results snapshot `a84672b35437e107247b7b9de23f72d87ed1aa74`. The private machine-readable group support and all run receipts remain at their existing cluster paths; no prior artifact was overwritten.
