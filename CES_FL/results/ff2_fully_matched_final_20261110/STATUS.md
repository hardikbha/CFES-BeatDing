# Fully matched FF2 training: timestamped snapshot

Observed management UTC: 2026-11-10T11:45:04.584320+00:00

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
| BCE | 7 | COMPLETED_VERIFIED | DONE | gpu02 / 4 | 10/10 |
| DAG-FDD | 7 | COMPLETED_VERIFIED | DONE | gpu02 / 5 | 10/10 |
| DAW-FDD | 7 | COMPLETED_VERIFIED | DONE | gpu02 / 7 | 10/10 |
| Ding SFD+GDA (common representation) | 7 | COMPLETED_VERIFIED | DONE | gpu02 / 6 | 10/10 |
| L_H | 7 | COMPLETED_VERIFIED | DONE | gpu01 / 4 | 10/10 |
| L_H ramp | 7 | COMPLETED_VERIFIED | DONE | gpu01 / 6 | 10/10 |
| L_H low response | 7 | COMPLETED_VERIFIED | DONE | gpu01 / 5 | 10/10 |
| BCE | 27 | COMPLETED_VERIFIED | DONE | gpu01 / 3 | 10/10 |
| DAG-FDD | 27 | COMPLETED_VERIFIED | DONE | gpu01 / 1 | 10/10 |
| DAW-FDD | 27 | COMPLETED_VERIFIED | DONE | gpu02 / 6 | 10/10 |
| Ding SFD+GDA (common representation) | 27 | COMPLETED_VERIFIED | DONE | gpu01 / 2 | 10/10 |
| L_H | 27 | COMPLETED_VERIFIED | DONE | gpu01 / 2 | 10/10 |
| L_H ramp | 27 | COMPLETED_VERIFIED | DONE | gpu01 / 4 | 10/10 |
| L_H low response | 27 | COMPLETED_VERIFIED | DONE | gpu01 / 6 | 10/10 |

States are worker/artifact records, not independent continuous process-liveness checks.
