# Matched FF2 TEST results

Observed management UTC: 2026-11-10T11:45:04.584320+00:00

Final table cells are filled only after all three selected-checkpoint TEST evaluations for that method pass.
Mean ± sample SD is over downstream seeds7/17/27 conditional on one shared pretrained encoder; it is not a confidence interval.

| Method | Verified TEST seeds | AUC % ↑ | FNR % ↓ | Joint 100F_FPR ↓ |
|---|---:|---:|---:|---:|
| BCE | 3/3 | 99.0255 ± 0.0191 | 1.2599 ± 0.1547 | 68.5463 ± 2.8925 |
| DAG-FDD | 3/3 | 99.0063 ± 0.0064 | 1.3735 ± 0.0455 | 65.7146 ± 0.7141 |
| DAW-FDD | 3/3 | 99.0095 ± 0.0097 | 1.4227 ± 0.0412 | 64.4452 ± 0.3355 |
| Ding SFD+GDA (common representation) | 3/3 | 99.0111 ± 0.0100 | 1.3248 ± 0.0402 | 65.9988 ± 0.7489 |
| L_H | 3/3 | 98.9358 ± 0.0946 | 1.4348 ± 0.9809 | 79.4685 ± 11.8340 |
| L_H ramp | 3/3 | 99.0141 ± 0.0066 | 1.3841 ± 0.1784 | 65.2960 ± 3.4445 |
| L_H low response | 3/3 | 99.0141 ± 0.0066 | 1.3841 ± 0.1784 | 65.2960 ± 3.4445 |

All preprocessing, initialization, trainability, optimizer, exposure, selection and evaluator choices are shared.
Only the declared method objective and Ding TRAIN-derived feature mask differ. These are controlled adaptations, not native-paper reproductions.
Per-seed/per-axis/family CSV files retain the full available verified measurements. No best-cell oracle or validation-to-TEST substitution is used.
