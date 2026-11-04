# Corrected-source forensic-response study: actual progress

Observed management UTC: 2026-11-04T15:44:16.392722+00:00

Execution commit `c332fd34fb603b8919b15fee08e8d022fb9f57c8`; script SHA256 `89530025a00b9b0405fba7026d3eed5f2749d86620174ad3ffde93a44f8b770b`.

C/D: authorized direct SSH shared GPU leases. A/B: actual PBS ngpus=2 resource-count allocation, distinct pool-v2 UUID bindings under user authorization; scheduler physical mapping unavailable unless recorded. No exclusive GPUs claimed. Existing Ding source/jobs preserved. No TEST inference.

New-study task-held GPU time: **22.9666/24TOTAL GPU-hours**. Unspent conservative reservations: 0.0000hours. Count both GPUs at full task-held wall time, even when shared; not kernel-only/exclusive-H100 utilization. CPU preflight/review are not GPU time.

| Dataset | Arm | Verified process/completion state | Complete epochs | Selected so far | VAL AUC (%) | gender×race F_FPR (fraction) | FPR gap (pp) | Worst AUC (%) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ffpp | A | COMPLETED_VERIFIED | 10/10 | 4 | 98.4264 | 0.505265 | 33.8391 | 93.9595 |
| ffpp | B | COMPLETED_VERIFIED | 10/10 | 2 | 98.4068 | 0.468638 | 30.4167 | 93.9677 |
| ffpp | C | COMPLETED_VERIFIED | 10/10 | 1 | 98.3077 | 0.480534 | 30.9848 | 93.6205 |
| ffpp | D | COMPLETED_VERIFIED | 10/10 | 1 | 98.2526 | 0.379345 | 23.8391 | 93.8519 |
| ff2 | A | COMPLETED_VERIFIED | 10/10 | 7 | 99.8085 | 0.290986 | 11.6609 | 99.6003 |
| ff2 | B | COMPLETED_VERIFIED | 10/10 | 1 | 99.8075 | 0.295459 | 11.0284 | 99.5864 |
| ff2 | C | BLOCKED_INPUTS | 0/10 | — | — | — | — | — |
| ff2 | D | BLOCKED_INPUTS | 0/10 | — | — | — | — | — |

All values above are VAL. Selected-so-far values remain preliminary until ten epochs and native independent replay pass. No fairness/model/threshold selection against TEST. Single-seed results do not establish statistical superiority.

Full per-axis/group supports, FPR/FNR, calibration, coverage, exact configuration hashes, GPU leases/PIDs, checkpoint hashes and errors are in LIVE_STATE.json. Age/skin labels are inherited proxies; recorded contributor IDs do not certify biological identities.

FF++ reviewed88 procedural cases across22 originals/18recorded identity-source IDs. The review is an explicitly named AI visual screen, not human annotation, per-view universal approval or proof of actual forgery validity. Source-bound qualification receipt stays private.

FairForge-2 operator review inspected88cases across22originals:72approved,8rejected,8uncertain. Rotation padding, hat/mask/hand occlusion confound some face-region probes. Qualification rejected the review; C/D remain BLOCKED_INPUTS. No sources silently removed or approvals invented.

The original six-key audit is unchanged; two TRAIN/VAL declarations checked, four TEST-related declarations DEFERRED. Known163TRAIN/VAL contributor overlaps remain disclosed.

The requested hardiksharma6555 fork rejected the evidence push. This separate branch is published in the previously authorized hardikbha upstream; no automatic merge.

Job43937 was stopped at the user request; it had8CPUs and no ngpus request. New PBS job45202 requests12CPUs/ngpus=2 for A/B in parallel with C/D. Exact grant, device UUIDs and overhead accounting are in LIVE_STATE.json.

Finite observer: every10minutes until the original48hour deadline, with transient SSH/push errors reported/retried. It cannot launch/retry training, generate reviews or access TEST. Separate explicit study_coordinator.py handles only the already-authorized A-D sequence and hash-matched documented budget resumes.
