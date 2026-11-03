# Locked benchmark uncertainty

2026-11-03T13:58:36.118739+00:00 management UTC.

All eight native analyses and four paired H-minus-A analyses completed. Each view uses 1,000 connected-recorded-contributor draws, seed 17, fixed original group eligibility and unchanged stored thresholds. The interval is a percentile interval conditional on these fixed trained models. It measures dataset-sampling uncertainty, not training-seed variability.

Numbers below are 100 times fraction values: AUC/FPR/FNR differences are percentage points; F_FPR is a sum of absolute disparities and is not a probability. The n value is usable draws out of 1,000. Cells with fewer than 900 usable draws do not meet the retained uncertainty requirement, even when an interval is displayed. An interval crossing zero does not prove equivalence. These tables introduce no new checkpoint, threshold or training-selection rule.

## fixed_probability_0.5

### Detection differences: H minus A

| Dataset | overall/auc | overall/fpr | overall/fnr |
|---|---|---|---|
| ffpp | +0.018 [-0.077, +0.113]; n=1000 | -5.046 [-6.474, -3.606]; n=1000 | +3.192 [+2.308, +4.186]; n=1000 |
| dfdc | -1.313 [-2.462, -0.165]; n=1000 | -12.457 [-13.908, -11.175]; n=1000 | +15.420 [+13.638, +17.415]; n=1000 |
| celebdf | +0.656 [+0.227, +1.085]; n=1000 | -19.359 [-20.720, -18.035]; n=1000 | +11.695 [+11.057, +12.250]; n=1000 |
| dfd | -0.277 [-0.640, +0.094]; n=1000 | -14.521 [-16.726, -12.339]; n=1000 | +8.041 [+6.742, +9.302]; n=1000 |

### Primary demographic differences: H minus A

| Dataset | gender/F_FPR | race/F_FPR | gender_race/F_FPR |
|---|---|---|---|
| ffpp | -0.233 [-3.586, +1.070]; n=1000 | -0.197 [-11.899, +2.317]; n=1000 | -2.577 [-30.783, -1.769]; n=990 |
| dfdc | +0.303 [-3.099, +1.700]; n=1000 | -6.945 [-13.356, +0.693]; n=982 | -12.818 [-26.357, -0.943]; n=608 |
| celebdf | -1.422 [-4.053, +1.952]; n=1000 | +3.505 [-6.126, +13.504]; n=609 | +7.324 [-4.839, +18.110]; n=587 |
| dfd | -1.854 [-6.317, +3.648]; n=1000 | +2.389 [-3.795, +6.598]; n=1000 | +5.800 [-13.125, +12.786]; n=957 |

### Auxiliary demographic differences: H minus A

| Dataset | age_bin/F_FPR | skin_bin/F_FPR |
|---|---|---|
| ffpp | -6.981 [-13.226, -3.951]; n=675 | -2.733 [-9.196, +0.081]; n=1000 |
| dfdc | undefined | undefined |
| celebdf | undefined | undefined |
| dfd | undefined | undefined |

### Resampling support

| Dataset | Components | Largest component (rows) |
|---|---:|---:|
| ffpp | 213 | 360 |
| dfdc | 757 | 60 |
| celebdf | 956 | 30 |
| dfd | 319 | 30 |

## native_selected_VAL_FPR10

### Detection differences: H minus A

| Dataset | overall/auc | overall/fpr | overall/fnr |
|---|---|---|---|
| ffpp | +0.018 [-0.077, +0.113]; n=1000 | -0.617 [-1.444, +0.187]; n=1000 | +0.398 [+0.098, +0.812]; n=1000 |
| dfdc | -1.313 [-2.462, -0.165]; n=1000 | +2.987 [+1.855, +4.081]; n=1000 | -2.268 [-3.731, -0.891]; n=1000 |
| celebdf | +0.656 [+0.227, +1.085]; n=1000 | -0.333 [-1.365, +0.636]; n=1000 | -0.418 [-0.712, -0.121]; n=1000 |
| dfd | -0.277 [-0.640, +0.094]; n=1000 | -0.841 [-2.373, +0.691]; n=1000 | +0.636 [+0.080, +1.160]; n=1000 |

### Primary demographic differences: H minus A

| Dataset | gender/F_FPR | race/F_FPR | gender_race/F_FPR |
|---|---|---|---|
| ffpp | +0.724 [-2.407, +2.075]; n=1000 | +2.986 [-5.364, +9.320]; n=1000 | +6.543 [-9.722, +15.898]; n=990 |
| dfdc | +0.464 [-1.991, +2.290]; n=1000 | -1.145 [-5.297, +3.790]; n=982 | -2.819 [-12.478, +9.130]; n=608 |
| celebdf | +1.961 [-0.867, +4.054]; n=1000 | +0.425 [-4.374, +4.975]; n=609 | +1.573 [-5.459, +7.091]; n=587 |
| dfd | +0.878 [-3.282, +3.489]; n=1000 | +0.428 [-2.882, +3.157]; n=1000 | -0.575 [-4.505, +7.413]; n=957 |

### Auxiliary demographic differences: H minus A

| Dataset | age_bin/F_FPR | skin_bin/F_FPR |
|---|---|---|
| ffpp | +1.116 [-4.902, +8.021]; n=675 | +1.199 [-3.055, +4.214]; n=1000 |
| dfdc | undefined | undefined |
| celebdf | undefined | undefined |
| dfd | undefined | undefined |

### Resampling support

| Dataset | Components | Largest component (rows) |
|---|---:|---:|
| ffpp | 213 | 360 |
| dfdc | 757 | 60 |
| celebdf | 956 | 30 |
| dfd | 319 | 30 |

198 defined model/view/metric cells have fewer than 900 usable draws. See BOOTSTRAP_INTERVALS.csv for every cell, including undefined values and all supported subgroup intervals.

Recorded contributors are not verified biological identities. FF++ retains the explicitly declared historical fit:TEST contributor overlap; DFDC retains 1,476 repeated native manifest rows. Connected resampling accounts for recorded image/source/identity links but cannot repair incomplete provenance. Age and skin-tone labels retain their original proxy-label limitations.

A small FF++ AUC difference or a lower FPR at fixed 0.5 is not sufficient to claim a win over Ding. Compare missed fakes and both operating-point views; published and historical Ding rows have different protocols. No new training decision is made from these TEST outcomes.
