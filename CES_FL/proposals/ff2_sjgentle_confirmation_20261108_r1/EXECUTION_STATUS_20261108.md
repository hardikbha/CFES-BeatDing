# RF-SJg confirmation status — 8 Nov 2026

Observed at 20:23 IST on 8 Nov 2026. Point-in-time status.

## Stage state

| Stage | State | Evidence |
|---|---|---|
| Proposal/config | Pushed | Branch experiment/ff2-sjgentle-confirm-20261108; active launch commit 91106b01ab28124c207319707ba841bfd32921d7. |
| Config control | PASS | Seed-7 and seed-27 configs are exact copies of SJg seed17 except the seed field; config SHAs are in README. |
| TRAIN/SELECT inputs | PASS | Fresh seed-specific receipts; 218,266 TRAIN / 31,107 SELECT; inventory SHA 9f5d4b58... matches the prior complete decode; zero split content overlap; test_read=false. Seed-7 receipt digest 6fd38864...; seed-27 receipt digest 71ad47f7.... |
| First preflight attempt | FAILED adapter only | Base head-ablation CLI rejected extra RFA config keys. No training or TEST read. See logs/PREFLIGHT_SCHEMA_FAILURE_20261108.json. |
| Redundant rehash attempts | ABORTED, no data mutation | A full-image scan and a repeated manifest-rescan were stopped as redundant; the old decoded inventory matched the exact manifest/config except seed. See PREFLIGHT_FULL_REHASH_ABORTED and PREFLIGHT_REBIND_RESCAN_ABORTED logs. |
| First GPU launch attempt | FAILED before binding | gpu02 lacks git on PATH. No training/checkpoint and zero GPU-hours; reservation released. See LAUNCH_FAIL_SJ_gentle_s7_ssh_git_missing.json. |
| Seed-7 RF-SJg | RUNNING | Direct shared-GPU lease on gpu02, PID 179980, visible device 7 / UUID GPU-65f63cc5-9366-5c2d-196a-6e7a0a62c1e8. At 20:23 IST: epoch 2, step 1,900 of 6,821. CUDA binding JSON exists. Log: logs/SJ_gentle_s7.log. |
| Seed-27 RF-SJg | PENDING | Will launch sequentially after seed 7 completes and actual GPU hours are reconciled. |
| TEST | SEALED | No TEST manifest, data, labels, scores or threshold selection were used. |

The launcher checks the exact proposal commit on the login node and verifies model/config SHA256 on the GPU node (git is not installed on gpu02). It uses the shared lock/idle-device check. It does not claim exclusive device ownership and does not stop or modify unrelated jobs.

## Budget and scheduler

The shared 72 GPU-hour ledger is at 61.8905 charged plus 4.0 reserved for seed 7, leaving 6.1095 unreserved. The total two-run cap is 8 GPU-hours. The existing qstat jobs 45924–45927 remain queued at the scheduler's user running-job limit; their target older output folders already have DONE receipts, so they are duplicate wrappers and are not being counted as new training. Four generic running PBS jobs were identified by the user as unrelated; all were left untouched.

## Scientific interpretation

Seed-17 RF-SJg SELECT results: AUC 99.7549%, FNR 0.4500%, gender×race F_FPR 27.5259%, versus matched RF-S at 99.8032%, 0.2930%, and 34.1222%. The predeclared Ding-reference target of 12.44% gender×race F_FPR was not met. The new seeds test whether the moderate validation reduction repeats; they cannot establish a Ding win or validate the full counterfactual CES-FL method.
