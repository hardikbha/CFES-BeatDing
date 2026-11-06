# R3 vs R0 (ResNet-50, Ding recipe, SWA 40-49), seeds 37/47/57 vs Ding published Table 3

Amendment 4 (pre-registered). Primary: SWA model, op05 (prob >= 0.5). Secondary: valFPR10 (one global FF++-val-real
90th-percentile threshold) and raw epoch 49 (labelled). 36 fairness + 4 AUC = 40 columns. No test threshold search.

## Run status

| run | status | FF++ val AUC ep10 | ep15 | aux>CE audited |
|---|---|---|---|---|
| R0_s37 | DONE | 0.944050406674476 | 0.944978236541512 | 0/0 |
| R0_s47 | running (epoch 50/50) | 0.9438000653881549 | 0.9537041512847966 | 0/0 |
| R0_s57 | DONE | 0.9429914240708285 | 0.9374411545627427 | 0/0 |
| R3_s37 | KILLED | - | - | 248/988 |
| R3_s47 | KILLED | - | - | 246/988 |
| R3_s57 | KILLED | - | - | 247/988 |

## Kill rule (R3 seeds)

- R3_s37: val-AUC rule pending; aux rule TRIGGERED (248/988); status KILLED
- R3_s47: val-AUC rule pending; aux rule TRIGGERED (246/988); status KILLED
- R3_s57: val-AUC rule pending; aux rule TRIGGERED (247/988); status KILLED

## SWA (primary), op05 (primary)

| run | wins/40 | fair/36 | AUC FF++ | DFDC | Celeb-DF | DFD |
|---|---|---|---|---|---|---|
| R0_s37 | 15 | 15 | 95.01 | 60.64 | 76.45 | 73.84 |
| R0_s47 | 13 | 13 | 95.08 | 60.86 | 75.61 | 72.63 |
| R0_s57 | 14 | 14 | 95.44 | 60.71 | 75.09 | 71.93 |
| Ding T3 published | - | - | 95.53 | 62.54 | 77.35 | 78.63 |
| **R0 mean (n=3)** | 14.00 ± 1.00 | 14.00 | 95.18 | 60.74 | 75.72 | 72.80 |  wins of the mean-metric row: 15/40

## SWA (primary), valFPR10

| run | wins/40 | fair/36 | AUC FF++ | DFDC | Celeb-DF | DFD |
|---|---|---|---|---|---|---|
| R0_s37 | 15 | 15 | 95.01 | 60.64 | 76.45 | 73.84 |
| R0_s47 | 12 | 12 | 95.08 | 60.86 | 75.61 | 72.63 |
| R0_s57 | 14 | 14 | 95.44 | 60.71 | 75.09 | 71.93 |
| Ding T3 published | - | - | 95.53 | 62.54 | 77.35 | 78.63 |
| **R0 mean (n=3)** | 13.67 ± 1.53 | 13.67 | 95.18 | 60.74 | 75.72 | 72.80 |  wins of the mean-metric row: 15/40

## raw epoch 49 (secondary), op05 (primary)

| run | wins/40 | fair/36 | AUC FF++ | DFDC | Celeb-DF | DFD |
|---|---|---|---|---|---|---|
| R0_s37 | 8 | 8 | 94.44 | 59.20 | 74.35 | 70.39 |
| R0_s47 | 9 | 9 | 94.72 | 58.69 | 70.38 | 69.76 |
| R0_s57 | 12 | 12 | 95.39 | 59.00 | 71.85 | 68.78 |
| Ding T3 published | - | - | 95.53 | 62.54 | 77.35 | 78.63 |
| **R0 mean (n=3)** | 9.67 ± 2.08 | 9.67 | 94.85 | 58.96 | 72.20 | 69.64 |  wins of the mean-metric row: 11/40

## raw epoch 49 (secondary), valFPR10

| run | wins/40 | fair/36 | AUC FF++ | DFDC | Celeb-DF | DFD |
|---|---|---|---|---|---|---|
| R0_s37 | 8 | 8 | 94.44 | 59.20 | 74.35 | 70.39 |
| R0_s47 | 10 | 10 | 94.72 | 58.69 | 70.38 | 69.76 |
| R0_s57 | 12 | 12 | 95.39 | 59.00 | 71.85 | 68.78 |
| Ding T3 published | - | - | 95.53 | 62.54 | 77.35 | 78.63 |
| **R0 mean (n=3)** | 10.00 ± 2.00 | 10.00 | 94.85 | 58.96 | 72.20 | 69.64 |  wins of the mean-metric row: 8/40
