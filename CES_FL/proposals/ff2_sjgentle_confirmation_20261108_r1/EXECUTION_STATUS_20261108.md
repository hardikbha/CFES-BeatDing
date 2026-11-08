# RF-SJg confirmation status — 8 Nov 2026

Observed 20:26 IST. Point-in-time execution note.

## Stage state

| Stage | State | Evidence |
|---|---|---|
| Proposal/config | Pushed | Branch experiment/ff2-sjgentle-confirm-20261108; launch source commit 91106b01ab28124c207319707ba841bfd32921d7. |
| Config control | PASS | Seed-7 and seed-27 configs are copies of seed17 SJg with only the seed field changed. |
| TRAIN/SELECT inputs | PASS | Seed-specific rebind receipts: 218,266 TRAIN / 31,107 SELECT, source 8c25ba8..., identical prior decoded inventory SHA 9f5d4b58..., zero recorded TRAIN/SELECT byte overlap, test_read=false. |
| First preflight attempt | FAILED adapter only | Base head-ablation CLI rejected RF-SJg-only config keys; no training started. Preserved at logs/PREFLIGHT_SCHEMA_FAILURE_20261108.json. |
| Redundant preflight attempts | ABORTED with receipts preserved | A full 249,373-image byte rehash was stopped after 274 s / 607 MB; a repeated manifest path-existence scan was stopped after 177 s with no image-byte reads. The same prior full decoded/hash inventory was already PASS for SJg17/S7/S27. The replacement rebind checks source/config/manifest hashes and inventory digest; training/validation loaders enforce each byte SHA on actual reads. |
| First GPU launch attempt | FAILED before binding | gpu02 had no git executable on PATH; no CUDA binding, checkpoint, or GPU time. The 4-hour reservation was released. See logs/LAUNCH_FAIL_SJ_gentle_s7_ssh_git_missing.json. |
| Seed 7 RF-SJg | RUNNING | Direct shared-GPU lease on gpu02, PID 179980, visible device 7 / H100 UUID GPU-65f63cc5-9366-5c2d-196a-6e7a0a62c1e8. At 20:26: epoch 3 around step 100/6,821; epoch 2 selected-so-far as feasible. Logs: logs/SJ_gentle_s7.log. |
| Seed 7 current feasible checkpoint | PROVISIONAL | Epoch 2 SELECT AUC 99.7603%, FNR 0.4465%, gender×race F_FPR 30.8859%; against RF-S seed7: F_FPR −3.596 points, AUC −0.039 points, FNR +0.171 points. Later feasible epochs may replace it under the fixed selector. |
| Seed 27 RF-SJg | PENDING | Launch sequentially only after seed7 completes and actual budget is reconciled. |
| TEST | SEALED | No TEST manifests, data, labels, or scores were read. |

The successful launcher verifies exact model/config SHA256 on the GPU node; git is checked at the login node because it is not installed on gpu02. It uses a shared-GPU lock and does not claim exclusive device ownership. No other jobs were stopped or modified.

## Budget and scheduler

The shared 72 GPU-hour ledger is 61.8905 charged plus 4.0 reserved for seed7, leaving 6.1095 unreserved. The two-run maximum is 8 GPU-hours. Four generic PBS jobs were identified by the user as unrelated and left untouched. Four rf2 wrappers remain queued because the scheduler reports the user GPU running-job limit; their target older output directories already have DONE receipts.

## Scientific interpretation

Seed17 RF-SJg SELECT result: AUC 99.7549%, FNR 0.4500%, gender×race F_FPR 27.5259%, versus matched RF-S at 99.8032%, 0.2930%, and 34.1222%. The predeclared Ding corrected-GDA target of 12.44% gender×race F_FPR was not met. This seed confirmation tests reproducibility of a moderate validation change, not a Ding win or the full counterfactual CES-FL method.
