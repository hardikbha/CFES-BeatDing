# Operator status: Celeb-DF audit and Table 3 preparation

Observed 4 November 2026. Supervisor source `43c605bcd55395dce79650c1f342429c090553a8`; ResNet implementation `352e837b98bfb73aeb6db3f754604d950ca8b580`; active P50 runner `486d1e305977f13681cab41532acb9e1773abfe3`; downstream runner `f777275`; Ding-control adapter source SHA256 `9d480b3bf018d7fff46c840044011122b953b456df04d154f42f173ceeace324`. This file records aggregate evidence only. Private images, row-level predictions and checkpoints remain on the cluster.

## Celeb-DF cache and annotation audit

- Reproduced `PUBLISHED_AGGREGATES_RECONCILED` in a new output directory from the committed aggregate inputs. No new TEST inference or calibration was performed.
- The L_A and L_H private caches each contain 28,458 rows and 28,458 logits. Their rows are identical and match the frozen manifest order, labels and inherited groups exactly.
- Manifest SHA256 is `5b4c9b3af2589b90e6e3fe9825857eacac9d4c2517f3226feb80fa7d4e0f4873`. L_A/L_H checkpoint SHA256 values are `aee9670ca011937d457846fbf13717445c5607715064dd6a54da5a82bda580ae` and `6ef2caad3979d101fd94d7350aa3c1a50f95ee8307673cfaae63ac9d89af36f0`. Cache hashes match their inference receipts.
- Every derived row matches the archived Ding/Lin `img_path`, binary label and six-way `intersec_label` in the same order. The transformation is `0/1/2 -> Male-White/Black/Others` and `3/4/5 -> Female-White/Black/Others`.
- The upstream CSV contains only `img_path,label,ismale,iswhite,isblack,intersec_label`. It does not contain person identity, donor/target convention, age or skin-tone labels. The current manifest therefore uses video-name proxies; true person separation and fake donor/target demographic truth remain unverified. Age, skin and ids8 are unknown for all 28,458 rows.
- Frame support is highly concentrated: 21,503/22,745 fakes (94.5395%) are Female-White; all 1,212 male fake frames are Male-Black. Male-White and Male-Others are real-only; Female-Black is absent. Video-proxy support is also uneven. This is an upstream inventory property, not a new relabeling decision.
- A deterministic private 16-image sheet sampled distinct video proxies from errors and same-group correct cases at fixed probability 0.5. All samples are 380x380 crops. Blur/compression and facial motion occur in both errors and correct cases; some errors include closed eyes or speech motion. These are observations, not causal explanations. No demographic labels were inferred from appearance and no samples were removed.

Private audit root: `/home/sachin.chaudhary/ces_fl_runs/lh_finalization_celebdf_20261104_43c605b/private_celebdf_audit_v2/`.

## ResNet-50 implementation and checks

- Original Xception downstream source remains SHA256 `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`.
- New P50 source: `CES_FL/src/ces_fl_resnet50_p50.py`, SHA256 `2d66966b1733feec222fe3a80ff9a6cd7181a6ba28cb9f767ad980968d337098`.
- New downstream source: `CES_FL/src/ces_fl_head_ablation_resnet50.py`, SHA256 `72d368a439658c601a5fe4d717237b1561190e85c13c70f6db3a441bc5cdbe78`.
- P50 preserves the native 50-epoch, seed-17, every-original-once exposure, inverse label×ids8 weighting, AdamW learning rates, validation-AUC selection and BN-update policy. The declared substitutions are ResNet-50 and ImageNet mean/std.
- Downstream L_A/L_H use the same future verified P50 encoder, identical fresh seed-17 linear heads, layer4 fine-tuning, frozen BN running buffers, batch 32, ten epochs, bf16 TRAIN/fp32 validation, and the original L_A/L_H losses and rates.
- Parameter counts measured from executable code: 23,510,081 total; 14,966,785 trainable; 23,508,032 encoder; 14,964,736 trainable encoder layer4; 2,049 linear-head parameters.
- CPU P50 lifecycle smoke: PASS, 7 check families, source-bound receipt under `/home/sachin.chaudhary/ces_fl_runs/lh_finalization_celebdf_20261104_43c605b/cpu_smoke_p50/RESULT.json`.
- CPU downstream lifecycle smoke: PASS. It verifies finite gradients, frozen BN buffers, CAM/logit decomposition, identical initialization, wrong-backbone rejection, native save/reload/resume, fixed validation threshold and TEST-label-flip isolation. Receipt: `/home/sachin.chaudhary/ces_fl_runs/lh_finalization_celebdf_20261104_43c605b/cpu_smoke_downstream_v2/SMOKE_RESULT.json`.
- Production config preflight reads 76,139 TRAIN and 25,386 validation originals and reproduces the declared 163 identity/source proxy overlaps. No TEST manifest is read.

## Compute gate

On 4 November the user authorized an additional 24 GPU-hours for this study; the separate FairForge-2 budget was untouched. The shared ledger ceiling is48GPU-hours. Current verified accounting:20.065168GPU-hours consumed, zero reserved, and27.934832unreserved. This includes P50 at2.778227hours and the two failed CUDA-binding guards at a combined0.000327hours. The separate FairForge-2 budget was not used.

The prior P10 checkpoint cannot replace P50. The downstream templates intentionally retain `BIND_AFTER_VERIFIED_P50_COMPLETION` and are not runnable configs until a completed P50 hash is inserted and the source-bound preview/review, CUDA smoke and real-input checks pass.

The native Ding seed-17 classification epoch-49 checkpoint and TRAIN-only 2% SFD channel list were found and hash-checked. The connected-gradient ten-epoch adapter is prepared. Native group files match all76,139manifest paths, labels and gender×race groups. CPU smoke confirms a connected fairness gradient, a nonzero SGD update change when λ changes, strict classifier reload, frozen BN and the fixed ten-channel mask. A real-input CPU backward dry-run passed with zero persistent updates. CUDA dry-run remains after the ResNet priority arms. Native timings project roughly4.58GPU-hours for the specified2,380updates and96forward images/update, before validation passes, checks and replay. Ding-10 remains after ResNet L_A/L_H and its locked evaluation.

The local artifacts include `DING_base_s17/49.pth` (SHA256 `b2240f157d4f94895f0908ad1a177832d258ab3c0f71ac9e09acc87a333b4e32`) and `DING_full_s17/SFD_channels.txt` (SHA256 `ef9ac134b7564f1d9e78b641291479638c9246bd40772df0d73a8448125bb039`). The saved log binds that ten-channel mask to the seed-17 base checkpoint and `--prc 2.0`. The connected-gradient adapter is `CES_FL/src/ding10_connected_control.py` (SHA256 `9d480b3bf018d7fff46c840044011122b953b456df04d154f42f173ceeace324`); its locked config SHA256 is `3e668b4d29a2f5c369adb39e5ed5319775c248f7fc93c824c40cb8b89c179673`.

## Current stage states

| Stage | State |
|---|---|
| New CPU aggregate audit | COMPLETED |
| Private cache/annotation provenance audit | COMPLETED WITH LIMITATIONS |
| Private deterministic error sheet | COMPLETED |
| ResNet P50/downstream implementation | COMPLETED |
| CPU software smoke | COMPLETED, PASS |
| Source-bound CUDA smoke | COMPLETED, PASS on gpu01 GPU1 |
| Real-input CUDA backward check | COMPLETED, PASS, zero optimizer updates |
| ResNet P50 training | COMPLETED, seed17, 50/50 epochs, validation replay PASS |
| ResNet L_A/L_H training | BOTH COMPLETED 10/10; selected checkpoints independently replay-verified; L_A epoch5, L_H epoch1 |
| Table 3 TEST evaluation | COMPLETED, 8/8 one-pass inference receipts; 8/8 bootstrap analyses and 4/4 paired analyses completed |
| Ding-10 controlled adaptation | ADAPTER CPU-CHECKED; direct shared runner prepared; fresh CUDA smoke/backward and training pending |

P50 completed 50/50 epochs. Earliest maximum validation AUC selected epoch12 at96.534759%; epoch50 AUC was95.836585%. The selected checkpoint SHA256 is `c28bbb9cfcd06a11894027dc9369c5fd674f640de6c182c28b6d6f0aa8d77f55`; native validation replay matched with zero metric difference. P50 used2.778227GPU-hours. The threshold remains validation-real empirical10% and no TEST was read. These are FF++ validation results only.

## GPU execution record

PBS job `45387.mgmt01` was canceled while queued, before it acquired a GPU; its reservation was released and it consumed zero GPU-hours. Two direct-runner guard failures also occurred before training: one rejected the already-recorded P50 reservation, and one found `git` missing from the non-login SSH `PATH`. Neither performed an optimizer step or used GPU compute. The corrected runner uses an absolute Git executable and adopts the existing reservation. The replacement direct-SSH run uses gpu01 GPU1, UUID `GPU-6cc0a5b3-99ca-13e3-5508-a201a06e82d6`, through shared-pool lease `lh_resnet50_P50_s17_20261104`.

The P50 controller PID was `3906763`; its log is `/home/sachin.chaudhary/ces_fl_runs/lh_resnet50_table3_s17_20261104_P50_attempt3.log`. Outputs are under `/home/sachin.chaudhary/ces_fl_runs/lh_resnet50_table3_s17_20261104/P50/`. Both source-bound CUDA smoke receipts and the real-input pretrain dry-run passed; the dry-run used16 TRAIN images and performed zero optimizer updates. It ran on gpu01 GPU1, UUID `GPU-6cc0a5b3-99ca-13e3-5508-a201a06e82d6`, for2.778227GPU-hours.

At 06:55 IST the two downstream runners bound L_A/L_H to the same P50 best checkpoint and passed full 76,139 TRAIN +25,386 VAL image hash/decode preflight. Config SHA256s are `be851ef7f8ccbbdb1a9fe261035fbe4478215ca8211a276852fa8f10a145ccad` (L_A) and `858b7359fa56de1468d04f86633e1fb73306559b4c905c4fac17c7cad1bc183a` (L_H); configs differ only in `error_control`. Both then failed the CUDA binding guard before any backward or training: PyTorch had been imported before setting `CUDA_VISIBLE_DEVICES`, and `nvidia-smi` mapped each process to gpu01 GPU0 (`GPU-410a5796-06b3-dbe6-5371-5d06a1a53416`) rather than the requested UUID. Each briefly allocated a CUDA context on GPU0 (522MiB reported) and exited without an optimizer step; no existing job was terminated or changed. The failures and their GPU time are preserved in `L_A.direct.log`, `L_H.direct.log`, and each `ALLOCATION_END.json` (0.0001704h and0.0001570h). L_H also produced an exact-preview reuse receipt for 88 procedural cases; reviewer provenance is explicitly `Codex AI ... NOT human approval`, not represented as human approval.

The refreshed shared ledger is48GPU-hours total and21.771425consumed, with no open reservations;26.228575hours are presently unreserved. This includes P50, both downstream arms, the two brief PBS binding failures, and the two shared-GPU TEST inference sessions.

A second L_A startup (executor commit `9294083`) verified the expected gpu01 GPU1 UUID successfully, but stopped because the runner looked for the pre-existing CUDA smoke receipt under `P50/checks/`; the actual checked receipt is under the run-root `checks/synthetic_downstream/`. It performed no backward/training update and consumed0.0001175GPU-hours. Preserve the failed startup output at `/home/sachin.chaudhary/ces_fl_runs/lh_resnet50_table3_s17_20261104_cudaenv2/`. The fresh `cudaenv3` L_A run verified the correct UUID and passed its real-input CUDA backward check; L_H passed the same checks on GPU5.

Interim best through the latest completed epochs, selected by earliest maximum validation AUC:

| Arm | Completed | Best epoch | AUC | FPR | FNR | Gender F_FPR | Race F_FPR | Age F_FPR | Skin F_FPR | Gender×race F_FPR | Worst gender×race AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L_A | 10/10 | 5 | 96.6525% | 9.9951% | 9.6318% | 5.8398% | 12.0151% | 14.5474% | 8.5857% | 39.2534% | 90.4245% |
| L_H | 10/10 | 1 | 96.4214% | 9.9951% | 9.9934% | 5.6438% | 9.9922% | 12.2680% | 5.0148% | 33.3735% | 90.2108% |

All rates above are fractions converted to percent and use each checkpoint's fixed FF++ validation-real10% threshold (achieved validation FPR9.9951%). These are final selected-checkpoint validation results for the ten-epoch runs, not TEST results or evidence of beating Ding.

L_A and L_H both passed all ten epochs and independent selected-checkpoint validation replay. L_A selected epoch5, checkpoint SHA256 `0571a220faa63ad61c3b2d5815d73f25a077ecf59a1a8e30c379a620e31c0555`, GPU time0.675815h. L_H selected epoch1, checkpoint SHA256 `b08c2b5180d999f81e044872a8228f7e876d0250a6cb29034d59b11b1a897f10`, GPU time0.769700h. Both receipts declare `test_evaluated=false`.

The locked Table3 TEST evaluation completed8/8 passes and all aggregate/paired bootstrap analyses. The point table is `CES_FL/results/lh_finalization_celebdf_20261104_r1/TABLE_3_RESNET_POINT_ESTIMATES_20261104.md`; full uncertainty, including F_FPR/F_DP/es-AUC and worst-group metrics, is in `TABLE_3_RESNET_UNCERTAINTY_ALL_20261104.md`; hash-verified per-group FPR/FNR/AUC and supports are in `TABLE_3_RESNET_GROUP_METRICS_20261104_VERIFIED.csv`. Test AUC point estimates (L_A/L_H) are FF++96.2103/96.0537%, DFDC59.5355/60.0680%, Celeb-DF73.4969/73.8106%, and DFD73.6546/73.4718%. The paired bootstrap is conditional on seed17 and recorded contributors, not independent training-seed or true-person uncertainty. The report flags groups below the prespecified900/1000 valid-replicate floor; e.g., Celeb-DF race609/1000 and gender×race587/1000, DFDC gender×race608/1000, and FF++ age675/1000 have unavailable intervals. Age/skin labels are unavailable on several cross-domain manifests.

## Execution runner safety update

The original direct L_A/L_H runner update was committed at `b47caea5637612f6b1281eb80fcf298d84399230`; the ResNet Table3 protocol/wrapper commit is `dd3242fd8bd477a6016b810669efe5cb8ee59275`. CUDA initialization order was corrected at `92940837529f5b0a908adc1eb65e0adaf290c7de`; the smoke-receipt path and binding receipt were corrected at `0f6900ab3af1bea6ec4abc78d9a4dcb93a0d5a20`. The detached arm executors use `0f6900ab3af1bea6ec4abc78d9a4dcb93a0d5a20`. They support fresh output roots/attempt IDs, exact preflight/review reuse, the configured two-hour cap and process-to-UUID receipts. The scientific downstream source remains SHA256 `72d368a439658c601a5fe4d717237b1561190e85c13c70f6db3a441bc5cdbe78`. No training update has been overwritten.

The ResNet Table3 evaluation protocol is source-controlled at `configs/resnet50_table3_eval_protocol.json`, with wrapper `CES_FL/scripts/lh_resnet50_locked_table3.py`. It derives four manifest/overlap bindings, threshold views and bootstrap limits from the existing Tables1-2 locked plan (SHA256 `6dffee3b9b309975d7c3844dc1052427885dc41b2375e77c15f38ae33f76c5ea`). The wrapper requires both completed selected checkpoints before binding; its single-pass inference uses no target-domain calibration or TEST-based selection. Five CPU policy-guard tests passed using synthetic inputs only; no TEST manifest/image was opened by those tests. The first test invocation exposed a mistyped protocol hash; it was corrected to match the existing locked plan and the suite then passed.

The direct evaluator `CES_FL/cluster/run_resnet50_table3_eval_direct.py` completed one pass per model/dataset under shared-pool reader leases on gpu02. Shared-active mode (executor commit `8685c715e568731787662858d6a01e29ec51a81d`) requires at least24GiB free, no direct lease conflict, utilization≤20%, a 16GiB reader reservation, a 16% per-process memory cap and PID-to-UUID verification. The locked plan is `/home/sachin.chaudhary/ces_fl_runs/lh_resnet50_table3_s17_20261104_cudaenv3/table3_eval/LOCKED_PLAN.json` (SHA256 `a234331c21c5539fb350f011737c993dd9bf84feb3d1ea65132a4fc36e98a788`); it binds selected checkpoints, four test manifests, overlap audits, fixed0.5/source-validation thresholds and1000connected-contributor bootstrap. Test score caches remain private on the cluster; public reports contain aggregate metrics only.

The direct reader lease selected gpu02 GPU5 for L_A (UUID `GPU-bff97893-385b-6f3f-831b-113284c2e1f1`,26,369MiB free,0% utilization at claim) and GPU4 for L_H (UUID `GPU-ee335748-8273-2d67-42a5-d45463027c1b`,26,351MiB free,0% utilization at claim). Both had an existing process using about54.7GiB; each evaluator used a 16GiB reader lease and16% per-process allocator cap. The user's screenshot's GPU3 reading did not match current GPU3 state on either host. No co-tenant was stopped.

PBS job `45437.mgmt01` received `ngpus=1` on gpu02 but exited before Python because `CUDA_VISIBLE_DEVICES` was absent. Diagnostic PBS job `45438.mgmt01` confirmed `CUDA_VISIBLE_DEVICES` and `PBS_GPUFILE` are missing and `/var/spool/pbs/aux/45438.mgmt01` only names the host. Neither PBS attempt initialized CUDA or accessed test data; both are accounted in the shared ledger. The successful direct shared-pool path used actual CUDA UUID verification. No existing job was preempted.

After Table3 inference, all direct leases were released. Existing jobs remain untouched. Ding-10 remains the next authorized arm, pending fresh CUDA/backward qualification and an available shared lease.

Ding-10 source/config are under `CES_FL/src/ding10_connected_control.py` (SHA256 `9d480b3bf018d7fff46c840044011122b953b456df04d154f42f173ceeace324`) and `CES_FL/proposals/lh_finalization_celebdf_20261104_r1/configs/ding10_connected.json` (SHA256 `3e668b4d29a2f5c369adb39e5ed5319775c248f7fc93c824c40cb8b89c179673`). Private CPU smoke/preflight receipts are at `/home/sachin.chaudhary/ces_fl_runs/lh_finalization_celebdf_20261104_43c605b/ding10_cpu_smoke_v3/`; a real-input CPU backward receipt is at `/home/sachin.chaudhary/ces_fl_runs/lh_finalization_celebdf_20261104_43c605b/ding10_real_input_cpu_dry/DRY_RUN.json`. These are software/data checks only, not a new Ding run or CUDA qualification. The direct shared-pool runner is `CES_FL/cluster/run_ding10_connected_direct.py`; it checks the exact config, performs fresh CPU preflight and requires fresh CUDA smoke plus real-TRAIN backward PASS before training. It caps shared memory at16% and reserves6GPU-hours for the unchanged ten-epoch adaptation. First CPU preflight attempt at `.../ding10_connected_s17_20261104_shared1/` returned PASS for 76,139 TRAIN /25,386 VAL and exact native-row membership, but a runner-side key-name assertion failed before GPU lease; no CUDA or optimizer step occurred and no budget was reserved. The assertion is corrected in a new runner commit; a fresh output root will be used. The prior L_H procedural review remains explicitly AI-reviewed, not human-approved.

The fixed four-dataset ResNet test protocol is `CES_FL/proposals/lh_finalization_celebdf_20261104_r1/configs/resnet50_table3_eval_protocol.json`; the dynamic checkpoint/manifest/overlap binding completed after both selected runs passed validation replay. The public-facing table index and Table3 point/uncertainty outputs are `CES_FL/results/lh_finalization_celebdf_20261104_r1/TABLES_1_2_3_CURRENT_20261104.md`, `TABLE_3_RESNET_POINT_ESTIMATES_20261104.md`, `TABLE_3_RESNET_UNCERTAINTY_ALL_20261104.md`, and `TABLE_3_RESNET_GROUP_METRICS_20261104_VERIFIED.csv`. Table3 test inference and analysis are complete; Ding-10 remains the next authorized arm.
