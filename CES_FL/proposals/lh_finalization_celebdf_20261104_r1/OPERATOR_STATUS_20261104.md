# Operator status: Celeb-DF audit and Table 3 preparation

Observed 4 November 2026 from supervisor source `43c605bcd55395dce79650c1f342429c090553a8`. This file records aggregate evidence only. Private images, row-level predictions and checkpoints remain on the cluster.

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

The shared original ledger is unchanged at 17.2866134967/24 GPU-hours used, leaving 6.7133865033. Historical ResNet-50 P10 on the same 76,139-row exposure, batch 16 and one H100 consumed 6,226.7448 seconds (1.72965 GPU-hours). A linear 50-epoch projection is 8.6483 GPU-hours before replay, checks or downstream L_A/L_H. This makes the required P50 alone larger than the remaining allowance. The full ResNet cohort and subsequent Ding-10 control must therefore not launch under the present budget.

The prior P10 checkpoint cannot replace P50. The downstream templates intentionally retain `BIND_AFTER_VERIFIED_P50_COMPLETION` and are not runnable configs until a completed P50 hash is inserted and the source-bound preview/review, CUDA smoke and real-input checks pass.

The native Ding seed-17 classification epoch-49 checkpoint exists. An independently bound TRAIN-only 2% mask and connected-gradient ten-epoch adapter have not been certified in this execution. Ding-10 remains pending behind both the priority ResNet cohort and the shared budget gate.

## Current stage states

| Stage | State |
|---|---|
| New CPU aggregate audit | COMPLETED |
| Private cache/annotation provenance audit | COMPLETED WITH LIMITATIONS |
| Private deterministic error sheet | COMPLETED |
| ResNet P50/downstream implementation | COMPLETED |
| CPU software smoke | COMPLETED, PASS |
| Source-bound CUDA smoke | PENDING ALLOCATION |
| Real-input CUDA backward check | PENDING ALLOCATION |
| ResNet P50 training | BLOCKED_BUDGET |
| ResNet L_A/L_H training and Table 3 TEST | BLOCKED_P50_AND_BUDGET |
| Ding-10 controlled adaptation | BLOCKED_PRIORITY_DEPENDENCY_AND_BUDGET |

No TEST tuning, new loss search, threshold change, full training, benchmark inference or automatic extension was performed in this milestone.
