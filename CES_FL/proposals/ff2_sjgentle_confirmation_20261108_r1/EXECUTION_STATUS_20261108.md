# RF-SJg confirmation status — 8 Nov 2026

**Superseding observation: 9 Nov 2026, 01:43 IST (login clock).** Both required extra downstream seeds are complete in the original RFA tree. Do not launch another seed27. The timestamped 8 Nov observation below is retained as history.

- Original confirmation seed7: completed ten epochs, selected epoch2, independent replay verified. Checkpoint SHA256 `a20a32a4f3497a9bdef08717167a3334774293b6d6ae821c8acff8b83870b742`; run `/home/sachin.chaudhary/ces_fl_runs/ff2_sjgentle_confirmation_20261108_r1/SJ_gentle_s7/out`.
- A second seed7 and seed27 completed under `/home/sachin.chaudhary/ces_fl_runs/ff2_rfa_20261107/{SJ_gentle_s7,SJ_gentle_s27}/out`. Config dictionaries match this proposal's intended runs. The duplicate seed7 has identical selected VAL metrics, but different checkpoint bytes. It is not another independent seed.
- The existing upstream TEST freeze uses the RFA-tree seed7 SHA256 `d38610fcfbaa3100b729c683752011948756cc3f61748a3237f4743ddbeed449`, and seed27 SHA256 `09f37f9d732842a559d9c6e253f5fffb3b930e64f9a6e6b28ef8a01f260d3e9b`. Use that fixed list for both VAL and TEST reporting; do not substitute after seeing results.
- Three downstream seeds (7/17/27), sharing the same seed17 pretrained encoder: SELECT AUC `99.7556±0.0045%`, FNR `0.4674±0.0333%`, gender×race `100F_FPR=29.03±1.71`. SD is not a CI.
- Existing TEST exports were found at upstream commit `30a6b263aaa63677cf8d3d9f4ee51bdb472ae284`: RF-SJg AUC `98.60±0.06%`, primary `100F_FPR=47.69±3.91`. This is not an overall win over the corrected Ding-style comparator. No inference was rerun for this update.
- Shared ledger reconciled the original seed7 at `0.754444 GPU-h`: charged `64.601102/72`, reserved `0`, recorded balance `7.398898`. TEST/sanity runtimes still need reconciliation; that balance is not a complete compute accounting.
- At 01:43 IST no live `ces_fl_realfpr.py` Python process was found on gpu01/gpu02. Existing unrelated scheduler jobs are preserved. No new training or TEST evaluation launched.
- Current paper/evidence work continues on `paper/ff2-processing-fairness-20261109`. The historical claims below that seed27 is pending or TEST is sealed have been superseded.

Observed 20:33 IST. Point-in-time execution note.

## Stage state

| Stage | State | Evidence |
|---|---|---|
| Proposal/config | Pushed | Branch experiment/ff2-sjgentle-confirm-20261108; launch source commit 91106b01ab28124c207319707ba841bfd32921d7. |
| Config control | PASS | Seed-7 and seed-27 configs are copies of seed17 SJg with only the seed field changed. |
| TRAIN/SELECT inputs | PASS | Seed-specific rebind receipts: 218,266 TRAIN / 31,107 SELECT, source 8c25ba8..., identical prior decoded inventory SHA 9f5d4b58..., zero recorded TRAIN/SELECT byte overlap, test_read=false. |
| First preflight attempt | FAILED adapter only | Base head-ablation CLI rejected RF-SJg-only config keys; no training started. Preserved at logs/PREFLIGHT_SCHEMA_FAILURE_20261108.json. |
| Redundant preflight attempts | ABORTED with receipts preserved | A full 249,373-image byte rehash was stopped after 274 s / 607 MB; a repeated manifest path-existence scan was stopped after 177 s with no image-byte reads. The same prior full decoded/hash inventory was already PASS for SJg17/S7/S27. The replacement rebind checks source/config/manifest hashes and inventory digest; training/validation loaders enforce each byte SHA on actual reads. |
| First GPU launch attempt | FAILED before binding | gpu02 had no git executable on PATH; no CUDA binding, checkpoint, or GPU time. The 4-hour reservation was released. See logs/LAUNCH_FAIL_SJ_gentle_s7_ssh_git_missing.json. |
| Seed 7 RF-SJg | RUNNING | Direct shared-GPU lease on gpu02, PID 179980, visible device 7 / H100 UUID GPU-65f63cc5-9366-5c2d-196a-6e7a0a62c1e8. At 20:33: epoch 4 around step 4,700/6,821; epoch 2 remains the best feasible checkpoint so far. Logs: logs/SJ_gentle_s7.log. |
| Seed 7 current feasible checkpoint | PROVISIONAL | Epoch 2 SELECT AUC 99.7603%, FNR 0.4465%, gender×race F_FPR 30.8859%; against RF-S seed7: F_FPR −3.596 points, AUC −0.039 points, FNR +0.171 points. Later feasible epochs may replace it under the fixed selector. |
| Seed 27 RF-SJg | PENDING | Launch sequentially only after seed7 completes and actual budget is reconciled. |
| TEST | SEALED | No TEST manifests, data, labels, or scores were read. |

The successful launcher verifies exact model/config SHA256 on the GPU node; git is checked at the login node because it is not installed on gpu02. It uses a shared-GPU lock and does not claim exclusive device ownership. No other jobs were stopped or modified.

## Budget and scheduler

The shared 72 GPU-hour ledger is 61.8905 charged plus 4.0 reserved for seed7, leaving 6.1095 unreserved. The two-run maximum is 8 GPU-hours. Four generic PBS jobs were identified by the user as unrelated and left untouched. Four rf2 wrappers remain queued because the scheduler reports the user GPU running-job limit; their target older output directories already have DONE receipts.

## Scientific interpretation

Seed17 RF-SJg SELECT result: AUC 99.7549%, FNR 0.4500%, gender×race F_FPR 27.5259%, versus matched RF-S at 99.8032%, 0.2930%, and 34.1222%. The predeclared Ding corrected-GDA target of 12.44% gender×race F_FPR was not met. This seed confirmation tests reproducibility of a moderate validation change, not a Ding win or the full counterfactual CES-FL method.
