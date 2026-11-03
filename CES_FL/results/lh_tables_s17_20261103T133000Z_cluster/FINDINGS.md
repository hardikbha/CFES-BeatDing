# Locked L_A/L_H benchmark findings

This report covers the one authorized, frozen TEST evaluation of the two completed seed-17 Xception models. L_A is the plain-head BCE control; L_H is the plain-head hierarchical-fairness model. Both completed ten downstream epochs. Their original validation-AUC rule selected epochs 3 and 1, respectively. Neither checkpoint nor its threshold was changed after TEST evaluation.

| Dataset | L_A AUC (%) | L_H AUC (%) | H minus A (percentage points) |
|---|---:|---:|---:|
| FF++ | 98.470943 | 98.488888 | +0.017945 |
| DFDC | 61.768966 | 60.455631 | -1.313335 |
| Celeb-DF | 70.747452 | 71.403911 | +0.656459 |
| DFD | 81.156119 | 80.878773 | -0.277347 |

The [paired intervals](UNCERTAINTY.md) and [all interval cells](BOOTSTRAP_INTERVALS.csv) are necessary to interpret these differences. They measure recorded-contributor sampling uncertainty conditional on these two trained models, not training-seed variability.

At fixed probability 0.5, L_H reduces the FF++ gender×race F_FPR sum from 0.227934 to 0.202166, but its FF++ fake-image FNR increases from 3.878% to 7.070% while overall FPR decreases from 9.365% to 4.319%. A lower false-positive disparity does not by itself establish a utility-preserving fairness improvement. On Celeb-DF and DFD, the fixed-0.5 intersection F_FPR point estimates worsen. DFDC AUC also declines.

The source-validation operating point tells a different story: with each checkpoint's original FF++ validation-calibrated 10% target-FPR threshold applied unchanged, FF++ intersection F_FPR rises from 0.248587 to 0.314022. The achieved FF++ TEST FPRs are 11.305% for L_A and 10.688% for L_H; the target is not a guaranteed TEST rate. Target-dataset labels and scores never determine the threshold. Both views are reported without choosing the more favorable view.

The retained uncertainty requirement is at least 900 usable draws out of 1,000. Intersection F_FPR has 990 usable draws on FF++, 608 on DFDC, 587 on Celeb-DF and 957 on DFD. DFDC and Celeb-DF therefore do not meet that requirement. Their conditional intervals are exported with explicit warnings, not treated as confirmatory evidence. Eligibility and the 900-draw rule were not relaxed.

[Tables 1 and 2](TABLE_1_FIXED05.md) use Ding's layout and separately identify published paper references, historical local Ding runs and current CES-FL rows. The historical GDA fairness-gradient limitation remains disclosed. Different stage budgets, preprocessing, selection and support policies prevent a strict paper-protocol win claim from this table alone. The new connected-gradient Ding ten-epoch adaptation has not been implemented or run. [Table 3](TABLE_3_FIXED05.md) remains explicitly PENDING because a current ResNet-50 L_H model has not been implemented and trained.

FF++ retains the contributor-overlap declarations from its original audit, and DFDC retains its 1,476 repeated native manifest rows. Byte overlap checks and recorded-component resampling do not establish biological identity independence. Existing age and skin-tone labels retain their proxy-label limitations. All per-group denominators and achieved FPR/FNR values are in [the group export](GROUP_COUNTS_AND_METRICS.csv).

The software/evaluation workflow can pass while the method's overall fairness claim remains unsupported. This evidence does not establish that CES-FL beats Ding, or that L_H reliably improves all demographic axes. No new model, loss, threshold, checkpoint or schedule is selected from these TEST outcomes. The prospective training work orders were frozen before this evaluation and remain separate implementation, verification and budget tasks.
