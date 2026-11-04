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

On 4 November the user authorized an additional 24 GPU-hours for this study; the separate FairForge-2 budget was untouched. The shared ledger is now 48 total GPU-hours. At this checkpoint it reports 19.559451 GPU-hours consumed, 7.727162 still reserved for P50 and 20.713387 available beyond that reservation. Thirty-nine P50 epochs took8,026.66 seconds, projecting roughly2.86 GPU-hours for50 epochs at current throughput; this estimate will be refreshed from later epoch timings. The prior ResNet P10 ran slower at1.72965 GPU-hours for10 epochs.

The prior P10 checkpoint cannot replace P50. The downstream templates intentionally retain `BIND_AFTER_VERIFIED_P50_COMPLETION` and are not runnable configs until a completed P50 hash is inserted and the source-bound preview/review, CUDA smoke and real-input checks pass.

The native Ding seed-17 classification epoch-49 checkpoint and TRAIN-only 2% SFD channel list were found and hash-checked. The connected-gradient ten-epoch adapter is prepared. Native group files match all 76,139 manifest paths, labels and gender×race groups. CPU smoke confirms a connected fairness gradient, a nonzero SGD update change when λ changes, strict classifier reload, frozen BN and the fixed ten-channel mask. A real-input CPU backward dry-run passed with zero persistent updates. CUDA dry-run remains after the ResNet priority arms. Native timings project roughly4.58 GPU-hours for the specified2,380 updates and96 forward images/update, before validation passes, checks and replay. Ding-10 remains after ResNet L_A/L_H and its locked evaluation.

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
| ResNet P50 training | RUNNING, seed17, epoch42/50, step1300/4759 at 06:27 IST; 40 epochs recorded |
| ResNet L_A/L_H training | PENDING verified P50 |
| Table 3 TEST evaluation | Evaluation plan frozen; pending both selected checkpoints; no TEST read |
| Ding-10 controlled adaptation | ADAPTER CPU-CHECKED; PENDING ResNet priority and CUDA dry-run |

P50's best checkpoint through epoch40 remains epoch12 with validation AUC96.5348%; epoch40 validation AUC was96.2303%. Epoch1 was91.3360%. Selection remains validation AUC-only and calibration uses validation real images. Epoch41/42 had not yet reached validation at the latest observation. These are interim validation observations, not final results or TEST metrics. No threshold tuning, new loss search or TEST evaluation was performed.

## GPU execution record

PBS job `45387.mgmt01` was canceled while queued, before it acquired a GPU; its reservation was released and it consumed zero GPU-hours. Two direct-runner guard failures also occurred before training: one rejected the already-recorded P50 reservation, and one found `git` missing from the non-login SSH `PATH`. Neither performed an optimizer step or used GPU compute. The corrected runner uses an absolute Git executable and adopts the existing reservation. The replacement direct-SSH run uses gpu01 GPU1, UUID `GPU-6cc0a5b3-99ca-13e3-5508-a201a06e82d6`, through shared-pool lease `lh_resnet50_P50_s17_20261104`.

The controller PID is `3906763`; its log is `/home/sachin.chaudhary/ces_fl_runs/lh_resnet50_table3_s17_20261104_P50_attempt3.log`. Run outputs are under `/home/sachin.chaudhary/ces_fl_runs/lh_resnet50_table3_s17_20261104/`. Both synthetic CUDA smoke receipts and the real-input pretrain dry-run receipt passed. The real-input backward used16 TRAIN images and performed zero optimizer updates. P50 runs on gpu01 GPU1, UUID `GPU-6cc0a5b3-99ca-13e3-5508-a201a06e82d6`; at 06:27 IST the controller was active, with epoch42 at step1300/4759 and the latest completed validation at epoch40. The refreshed ledger reports 19.6595 GPU-hours consumed and 10 hours reserved for P50; 18.3405 hours remain unreserved under the 48-hour ceiling. No L_A/L_H or Ding training has started. The matched downstream runner is prepared to bind L_A/L_H from verified P50 and claim another GPU only after a fresh lease check.

## Execution runner safety update

The direct L_A/L_H runner is being updated without changing the scientific trainer, configs, losses, selection, calibration or output paths. It now stages CPU binding/preflight before claiming a GPU, uses the configured two-hour per-arm execution cap, and ensures the direct GPU lease is acquired before CUDA checks and released on failure. The downstream source remains SHA256 `72d368a439658c601a5fe4d717237b1561190e85c13c70f6db3a441bc5cdbe78`. A source syntax check passed; actual arm preflight, CUDA backward checks and training remain gated on completed, replay-verified P50.

Ding-10 source/config are under `CES_FL/src/ding10_connected_control.py` and `CES_FL/proposals/lh_finalization_celebdf_20261104_r1/configs/ding10_connected.json`. Private CPU smoke/preflight receipts are at `/home/sachin.chaudhary/ces_fl_runs/lh_finalization_celebdf_20261104_43c605b/ding10_cpu_smoke_v3/`; a real-input CPU backward receipt is at `/home/sachin.chaudhary/ces_fl_runs/lh_finalization_celebdf_20261104_43c605b/ding10_real_input_cpu_dry/DRY_RUN.json`. These are software/data checks only, not a Ding run or CUDA qualification. The procedural-review rebinding helper reuses recorded decisions only if new and old preview sheet bytes match exactly; it has not yet been run because the ResNet P50 source-bound preview needs the verified P50 checkpoint. No new review or approval is claimed.

The fixed four-dataset ResNet test plan is `CES_FL/proposals/lh_finalization_celebdf_20261104_r1/configs/resnet50_table3_eval_plan.json`; its manifest hashes, thresholds and uncertainty procedure were frozen before new TEST inference.
