# Source-generalization run status — 8 November 2026, 02:03 IST

Two authorized seed-17 TRAIN-only arms are running in parallel on gpu02 H100s. They use the full native 50-epoch P50 architecture and exact frozen configs; no TEST files or scores are read.

| Arm | Progress | Best development epoch so far | Best development AUC so far |
|---|---:|---:|---:|
| S0 original-only BCE | 50/50, verified | 45 | 86.692% |
| S1 same recipe + fixed nuisance bank | 50/50, verified | 35 | 82.547% |

Both full 50-epoch schedules completed and passed native validation replay. Selected-checkpoint development FPR/FNR were 9.665%/34.565% for S0 and 9.665%/47.465% for S1. The supported three-group gender×race F_FPR was 0.3251 for S0 and 0.2065 for S1. S1 loses 4.145 AUC points and raises FNR by 12.90 points against S0. The fold contains only 269 real images from 41 recorded components and supports only three of eight gender×race groups; these F_FPR values are descriptive, not evidence of population fairness. No TEST was read. See the full [source-generalization aggregate](https://github.com/hardikbha/CFES-BeatDing/blob/proposal/lh-source-generalization-20261105/CES_FL/results/lh_objective_priority_20261105_r1/SOURCEGEN_EXTENSION_RESULTS.json).

## Bound protocol and resources

- Scientific trainer source: `54c19fae5bc1d55569448d299c31d2c58f43e8c47dfde478282bec755d7226bb`.
- S0 config SHA256: `d8fec9fbb17b6c24b1362bcd82254cd71950b6f45cce7d4211b26ca3d109db3e`.
- S1 config SHA256: `a809f207d0faddecc1ddb6c4e7868608d679992fceb36be4a7283449f88ec397`.
- Fold fit/development manifest hashes: `eb4fd630a4964f337678f54c1b7b7059ba30ee5019b66e15aeba2fb00685c54f` / `83cbb49977444a7f0fa3a7d5ed183d9badd291c3d30bfe87cf7a9939539f161`.
- GPU bindings: S0 `GPU-bff97893-385b-6f3f-831b-113284c2e1f1`; S1 `GPU-65f63cc5-9366-5c2d-196a-6e7a0a62c1e8`. Both are separate gpu02 H100 devices with shared-pool leases and 16% process memory caps.
- Authorized cap: 23.8 aggregate GPU-hours, raised by the user. Actual measured use including smokes and replay was about 0.885 GPU-hours; no reservation remains. The full report and checkpoint hashes are in the linked aggregate. No FairForge-2 budget was used.

The source-generalization protocol and compute extension are documented under the `proposal/lh-source-generalization-20261105` branch. The Table1/2 L_A/L_H test summary is in [LATEST_TABLES_SUPPLEMENT_20261105.md](LATEST_TABLES_SUPPLEMENT.md). The full Ding-format Table1–3 PDF remains the older Nov. 2 snapshot; its ResNet Table3 does not contain a current ResNet L_H run.

No fairness or Ding-superiority claim is made from these in-progress runs.
