# Fully matched FF2 training: timestamped snapshot

Observed management UTC: 2026-11-10T06:35:39.515417+00:00

Seven controlled objective adaptations, three downstream seeds, ten epochs each.
One shared seed17 P50 encoder; this is conditional downstream replication.

| Method | Seed | Recorded state | Stage | Host / GPU | Epochs |
|---|---:|---|---|---|---:|
| BCE | 17 | COMPLETED_VERIFIED | DONE | gpu01 / 4 | 10/10 |
| DAG-FDD | 17 | COMPLETED_VERIFIED | DONE | gpu02 / 5 | 10/10 |
| DAW-FDD | 17 | COMPLETED_VERIFIED | DONE | gpu01 / 1 | 10/10 |
| Ding SFD+GDA (common representation) | 17 | COMPLETED_VERIFIED | DONE | gpu02 / 7 | 10/10 |
| L_H | 17 | COMPLETED_VERIFIED | DONE | gpu01 / 5 | 10/10 |
| L_H ramp | 17 | COMPLETED_VERIFIED | DONE | gpu02 / 4 | 10/10 |
| L_H low response | 17 | COMPLETED_VERIFIED | DONE | gpu01 / 6 | 10/10 |
| BCE | 7 | RUNNING | TRAINING | gpu02 / 4 | 4/10 |
| DAG-FDD | 7 | RUNNING | TRAINING | gpu02 / 5 | 4/10 |
| DAW-FDD | 7 | RUNNING | TRAINING | gpu02 / 7 | 4/10 |
| Ding SFD+GDA (common representation) | 7 | COMPLETED_VERIFIED | DONE | gpu02 / 6 | 10/10 |
| L_H | 7 | RUNNING | TRAINING | gpu01 / 4 | 2/10 |
| L_H ramp | 7 | RUNNING | TRAINING | gpu01 / 6 | 2/10 |
| L_H low response | 7 | RUNNING | TRAINING | gpu01 / 5 | 2/10 |
| BCE | 27 | RUNNING | TRAINING | gpu01 / 3 | 2/10 |
| DAG-FDD | 27 | RUNNING | TRAINING | gpu01 / 1 | 1/10 |
| DAW-FDD | 27 | RUNNING | TRAINING | gpu02 / 6 | 1/10 |
| Ding SFD+GDA (common representation) | 27 | RUNNING | TRAINING | gpu01 / 2 | 5/10 |
| L_H | 27 | PREPARED | -- | None / None | 0/10 |
| L_H ramp | 27 | PREPARED | -- | None / None | 0/10 |
| L_H low response | 27 | PREPARED | -- | None / None | 0/10 |

States are worker/artifact records, not independent continuous process-liveness checks.
