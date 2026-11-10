# Fully matched FF2 training: timestamped snapshot

Observed management UTC: 2026-11-10T05:03:28.945766+00:00

Seven controlled objective adaptations, three downstream seeds, ten epochs each.
One shared seed17 P50 encoder; this is conditional downstream replication.

| Method | Seed | Recorded state | Stage | Host / GPU | Epochs |
|---|---:|---|---|---|---:|
| BCE | 17 | RUNNING | TRAINING | gpu01 / 4 | 7/10 |
| DAG-FDD | 17 | RUNNING | TRAINING | gpu02 / 5 | 8/10 |
| DAW-FDD | 17 | RUNNING | TRAINING | gpu01 / 1 | 7/10 |
| Ding SFD+GDA (common representation) | 17 | RUNNING | TRAINING | gpu02 / 7 | 8/10 |
| L_H | 17 | RUNNING | TRAINING | gpu01 / 5 | 7/10 |
| L_H ramp | 17 | RUNNING | TRAINING | gpu02 / 4 | 8/10 |
| L_H low response | 17 | RUNNING | TRAINING | gpu01 / 6 | 7/10 |
| BCE | 7 | PREPARED | -- | None / None | 0/10 |
| DAG-FDD | 7 | PREPARED | -- | None / None | 0/10 |
| DAW-FDD | 7 | PREPARED | -- | None / None | 0/10 |
| Ding SFD+GDA (common representation) | 7 | RUNNING | TRAINING | gpu02 / 6 | 6/10 |
| L_H | 7 | PREPARED | -- | None / None | 0/10 |
| L_H ramp | 7 | PREPARED | -- | None / None | 0/10 |
| L_H low response | 7 | PREPARED | -- | None / None | 0/10 |
| BCE | 27 | PREPARED | -- | None / None | 0/10 |
| DAG-FDD | 27 | PREPARED | -- | None / None | 0/10 |
| DAW-FDD | 27 | PREPARED | -- | None / None | 0/10 |
| Ding SFD+GDA (common representation) | 27 | PREPARED | -- | None / None | 0/10 |
| L_H | 27 | PREPARED | -- | None / None | 0/10 |
| L_H ramp | 27 | PREPARED | -- | None / None | 0/10 |
| L_H low response | 27 | PREPARED | -- | None / None | 0/10 |

States are worker/artifact records, not independent continuous process-liveness checks.
