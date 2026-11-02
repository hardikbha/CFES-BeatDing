# Corrected-source forensic-response study: actual progress

Observed management UTC: 2026-11-02T21:57:51.013024+00:00

Execution commit `c332fd34fb603b8919b15fee08e8d022fb9f57c8`; script SHA256 `89530025a00b9b0405fba7026d3eed5f2749d86620174ad3ffde93a44f8b770b`.

User-authorized direct SSH with explicit shared pool-v2 GPU leases; not PBS allocations. Existing Ding source/jobs preserved. No TEST inference.

New-study task-held GPU time: **1.2625/24TOTAL GPU-hours**. Unspent conservative reservations: 5.8139hours. Count both GPUs at full task-held wall time, even when shared; not kernel-only/exclusive-H100 utilization. CPU preflight/review are not GPU time.

| Dataset | Arm | Verified process/completion state | Complete epochs | Selected so far | VAL AUC (%) | gender×race F_FPR (fraction) | FPR gap (pp) | Worst AUC (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ffpp | A | PENDING | 0/10 | — | — | — | — | — |
| ffpp | B | PENDING | 0/10 | — | — | — | — | — |
| ffpp | C | RUNNING | 4/10 | 1 | 98.3077 | 0.480534 | 30.9848 | 93.6205 |
| ffpp | D | RUNNING | 1/10 | 1 | 98.2526 | 0.379345 | 23.8391 | 93.8519 |
| ff2 | A | PENDING | 0/10 | — | — | — | — | — |
| ff2 | B | PENDING | 0/10 | — | — | — | — | — |
| ff2 | C | PENDING | 0/10 | — | — | — | — | — |
| ff2 | D | PENDING | 0/10 | — | — | — | — | — |

All values above are VAL. Selected-so-far values remain preliminary until ten epochs and native independent replay pass. No fairness/model/threshold selection against TEST. Single-seed results do not establish statistical superiority.

Full per-axis/group supports, FPR/FNR, calibration, coverage, exact configuration hashes, GPU leases/PIDs, checkpoint hashes and errors are in LIVE_STATE.json. Age/skin labels are inherited proxies; recorded contributor IDs do not certify biological identities.

FF++ reviewed88 procedural cases across22 originals/18recorded identity-source IDs. The review is an explicitly named AI visual screen, not human annotation, per-view universal approval or proof of actual forgery validity. Source-bound qualification receipt stays private.

The original six-key audit is unchanged; two TRAIN/VAL declarations checked, four TEST-related declarations DEFERRED. Known163TRAIN/VAL contributor overlaps remain disclosed.

The requested hardiksharma6555 fork rejected the evidence push. This separate branch is published in the previously authorized hardikbha upstream; no automatic merge.

Finite observer: every10minutes for at most48hours, with transient SSH/push errors reported/retried. It cannot launch/retry training, generate reviews or access TEST. Separate explicit study_coordinator.py handles only the already-authorized A-D sequence and hash-matched documented budget resumes.
