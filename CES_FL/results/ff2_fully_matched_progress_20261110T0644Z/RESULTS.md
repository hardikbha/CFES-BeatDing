# Matched FF2 TEST results

Observed management UTC: 2026-11-10T06:35:39.515417+00:00

Final table cells are filled only after all three selected-checkpoint TEST evaluations for that method pass.
Mean ± sample SD is over downstream seeds7/17/27 conditional on one shared pretrained encoder; it is not a confidence interval.

| Method | Verified TEST seeds | AUC % ↑ | FNR % ↓ | Joint 100F_FPR ↓ |
|---|---:|---:|---:|---:|
| BCE | 1/3 | pending | pending | pending |
| DAG-FDD | 1/3 | pending | pending | pending |
| DAW-FDD | 1/3 | pending | pending | pending |
| Ding SFD+GDA (common representation) | 2/3 | pending | pending | pending |
| L_H | 1/3 | pending | pending | pending |
| L_H ramp | 1/3 | pending | pending | pending |
| L_H low response | 1/3 | pending | pending | pending |

All preprocessing, initialization, trainability, optimizer, exposure, selection and evaluator choices are shared.
Only the declared method objective and Ding TRAIN-derived feature mask differ. These are controlled adaptations, not native-paper reproductions.
Per-seed/per-axis/family CSV files retain the full available verified measurements. No best-cell oracle or validation-to-TEST substitution is used.
