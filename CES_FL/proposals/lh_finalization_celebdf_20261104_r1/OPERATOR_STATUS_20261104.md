# Operator status: Celeb-DF audit and Table 3 preparation

Observed 4 November 2026. Supervisor source `43c605bcd55395dce79650c1f342429c090553a8`; ResNet implementation `352e837b98bfb73aeb6db3f754604d950ca8b580`; active P50 runner `486d1e305977f13681cab41532acb9e1773abfe3`; Ding-control adapter source SHA256 `9d480b3bf018d7fff46c840044011122b953b456df04d154f42f173ceeace324`. This file records aggregate evidence only. Private images, row-level predictions and checkpoints remain on the cluster.

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

On 4 November the user authorized an additional 24 GPU-hours for this study; the separate FairForge-2 budget was untouched. The shared ledger is now 48 total GPU-hours. At this checkpoint it reports 18.170183 GPU-hours consumed, 9.116431 still reserved for P50 and 20.713387 available beyond that reservation. P50's first13 complete epochs took2,921.68 seconds, projecting roughly3.13 GPU-hours for50 epochs at current throughput; this is an early estimate and will be refreshed from later epoch timings. The prior ResNet P10 ran slower at1.72965 GPU-hours for10 epochs.

The prior P10 checkpoint cannot replace P50. The downstream templates intentionally retain `BIND_AFTER_VERIFIED_P50_COMPLETION` and are not runnable configs until a completed P50 hash is inserted and the source-bound preview/review, CUDA smoke and real-input checks pass.

The native Ding seed-17 classification epoch-49 checkpoint and TRAIN-only 2% SFD channel list were found and hash-checked. The new connected-gradient ten-epoch adapter and exact configuration are now prepared. The native group files match the 76,139-row manifest path/label/gender×race inventory exactly. CPU synthetic smoke, mask/checkpoint reload checks, and a real-input CPU backward dry-run pass; CUDA dry-run remains pending until the ResNet priority stages release a suitable device. Native Ding epoch timing projects roughly4.58 GPU-hours for the specified 2,380 updates and 96 forward images/update, before validation passes, checks and replay. Ding-10 remains after ResNet L_A/L_H and its locked evaluation.

The local artifacts do include `DING_base_s17/49.pth` (SHA256 `b2240f157d4f94895f0908ad1a177832d258ab3c0f71ac9e09acc87a333b4e32`) and `DING_full_s17/SFD_channels.txt` (SHA256 `ef9ac134b7564f1d9e78b641291479638c9246bd40772df0d73a8448125bb039`). The mask file has ten channel indices, and the archived run log shows seed-17 SFD used that base checkpoint with `--prc 2.0`. These are reusable candidate dependencies; the connected-gradient ten-epoch adapter and source-bound CUDA checks remain unimplemented. From the completed native seed-17 full run, 49 inter-epoch intervals average 1,827.35 seconds at 1,976 updates/epoch. Scaling to the requested 2,380 updates/epoch gives about 6.114 GPU-hours for ten training epochs alone, before the new adapter's checks/replay. That does not fit with adequate verification margin in the 6.213 GPU-hours currently left after the queued CUDA reservation, so Ding-10 is also held.

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
| ResNet P50 training | RUNNING, seed17, epoch14/50, step4700/4759 at last observation; 13 epochs recorded |
| ResNet L_A/L_H training | PENDING verified P50 |
| Table 3 TEST evaluation | PENDING both selected checkpoints; no TEST read |
| Ding-10 controlled adaptation | ADAPTER CPU-CHECKED; PENDING ResNet priority and CUDA dry-run |

P50's best observed checkpoint through epoch13 is epoch12, with validation AUC96.5348%; epoch13 validation AUC was96.1778%. Epoch1 was91.3360%. Validation selection remains AUC-only, and calibration uses validation real images. These are interim validation observations, not final results or TEST metrics. No threshold tuning, new loss search or TEST evaluation was performed.

## GPU execution record

PBS job `45387.mgmt01` was canceled while queued, before it acquired a GPU; its reservation was released and it consumed zero GPU-hours. The replacement direct-SSH run uses gpu01 GPU1, UUID `GPU-6cc0a5b3-99ca-13e3-5508-a201a06e82d6`, through shared-pool lease `lh_resnet50_P50_s17_20261104`.

The controller PID is `3906763`; its log is `/home/sachin.chaudhary/ces_fl_runs/lh_resnet50_table3_s17_20261104_P50_attempt3.log`. Run outputs are under `/home/sachin.chaudhary/ces_fl_runs/lh_resnet50_table3_s17_20261104/`. Both synthetic CUDA smoke receipts and the real-input pretrain dry-run receipt passed. The real-input backward used16 TRAIN images and no optimizer update. P50 is running on gpu01 GPU1, UUID `GPU-6cc0a5b3-99ca-13e3-5508-a201a06e82d6`; epoch14 is in progress. No L_A/L_H or Ding training has started.

Ding-10 source/config are under `CES_FL/src/ding10_connected_control.py` and `CES_FL/proposals/lh_finalization_celebdf_20261104_r1/configs/ding10_connected.json`. Private CPU smoke/preflight receipts are at `/home/sachin.chaudhary/ces_fl_runs/lh_finalization_celebdf_20261104_43c605b/ding10_cpu_smoke_v3/`; a real-input CPU backward receipt is at `/home/sachin.chaudhary/ces_fl_runs/lh_finalization_celebdf_20261104_43c605b/ding10_real_input_cpu_dry/DRY_RUN.json`. These are software/data checks only, not a Ding run or CUDA qualification.
