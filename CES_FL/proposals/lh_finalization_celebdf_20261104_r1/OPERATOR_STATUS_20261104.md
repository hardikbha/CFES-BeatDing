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
| ResNet L_A/L_H training | FIRST STARTUP ATTEMPTS FAILED CUDA UUID GUARD; corrected fresh attempts pending |
| Table 3 TEST evaluation | Protocol frozen and CPU policy guards PASS; pending both selected checkpoints; no TEST read |
| Ding-10 controlled adaptation | ADAPTER CPU-CHECKED; PENDING ResNet priority and CUDA dry-run |

P50 completed 50/50 epochs. Earliest maximum validation AUC selected epoch12 at96.534759%; epoch50 AUC was95.836585%. The selected checkpoint SHA256 is `c28bbb9cfcd06a11894027dc9369c5fd674f640de6c182c28b6d6f0aa8d77f55`; native validation replay matched with zero metric difference. P50 used2.778227GPU-hours. The threshold remains validation-real empirical10% and no TEST was read. These are FF++ validation results only.

## GPU execution record

PBS job `45387.mgmt01` was canceled while queued, before it acquired a GPU; its reservation was released and it consumed zero GPU-hours. Two direct-runner guard failures also occurred before training: one rejected the already-recorded P50 reservation, and one found `git` missing from the non-login SSH `PATH`. Neither performed an optimizer step or used GPU compute. The corrected runner uses an absolute Git executable and adopts the existing reservation. The replacement direct-SSH run uses gpu01 GPU1, UUID `GPU-6cc0a5b3-99ca-13e3-5508-a201a06e82d6`, through shared-pool lease `lh_resnet50_P50_s17_20261104`.

The P50 controller PID was `3906763`; its log is `/home/sachin.chaudhary/ces_fl_runs/lh_resnet50_table3_s17_20261104_P50_attempt3.log`. Outputs are under `/home/sachin.chaudhary/ces_fl_runs/lh_resnet50_table3_s17_20261104/P50/`. Both source-bound CUDA smoke receipts and the real-input pretrain dry-run passed; the dry-run used16 TRAIN images and performed zero optimizer updates. It ran on gpu01 GPU1, UUID `GPU-6cc0a5b3-99ca-13e3-5508-a201a06e82d6`, for2.778227GPU-hours.

At 06:55 IST the two downstream runners bound L_A/L_H to the same P50 best checkpoint and passed full 76,139 TRAIN +25,386 VAL image hash/decode preflight. Config SHA256s are `be851ef7f8ccbbdb1a9fe261035fbe4478215ca8211a276852fa8f10a145ccad` (L_A) and `858b7359fa56de1468d04f86633e1fb73306559b4c905c4fac17c7cad1bc183a` (L_H); configs differ only in `error_control`. Both then failed the CUDA binding guard before any backward or training: PyTorch had been imported before setting `CUDA_VISIBLE_DEVICES`, and `nvidia-smi` mapped each process to gpu01 GPU0 (`GPU-410a5796-06b3-dbe6-5371-5d06a1a53416`) rather than the requested UUID. Each briefly allocated a CUDA context on GPU0 (522MiB reported) and exited without an optimizer step; no existing job was terminated or changed. The failures and their GPU time are preserved in `L_A.direct.log`, `L_H.direct.log`, and each `ALLOCATION_END.json` (0.0001704h and0.0001570h). L_H also produced an exact-preview reuse receipt for 88 procedural cases; reviewer provenance is explicitly `Codex AI ... NOT human approval`, not represented as human approval.

The refreshed shared ledger is48GPU-hours total and20.065168consumed, with no open reservations;27.934832hours are presently unreserved. The corrected runner sets the UUID before loading the model/PyTorch module and uses fresh attempt IDs/output paths. It reuses the just-produced source/config-bound preflight and recorded L_H qualification only after their hashes pass again. No L_A/L_H optimizer update has occurred yet.

## Execution runner safety update

The original direct L_A/L_H runner update was committed at `b47caea5637612f6b1281eb80fcf298d84399230`; the ResNet Table3 protocol/wrapper commit is `dd3242fd8bd477a6016b810669efe5cb8ee59275`. The latest runner patch sets the requested CUDA UUID before importing the model module, supports explicit fresh output roots/attempt IDs, and reuses only source/config-hash-matching preflight/review receipts. It retains the configured two-hour cap and process-to-UUID verification. The scientific downstream source remains SHA256 `72d368a439658c601a5fe4d717237b1561190e85c13c70f6db3a441bc5cdbe78`; runner `py_compile` and `git diff --check` pass. The first CUDA guard caught the initialization-order bug before training, which is now corrected and will be retried from new output directories.

The ResNet Table3 evaluation protocol is source-controlled at `configs/resnet50_table3_eval_protocol.json`, with wrapper `CES_FL/scripts/lh_resnet50_locked_table3.py`. It derives four manifest/overlap bindings, threshold views and bootstrap limits from the existing Tables1-2 locked plan (SHA256 `6dffee3b9b309975d7c3844dc1052427885dc41b2375e77c15f38ae33f76c5ea`). The wrapper requires both completed selected checkpoints before binding; its single-pass inference uses no target-domain calibration or TEST-based selection. Five CPU policy-guard tests passed using synthetic inputs only; no TEST manifest/image was opened by those tests. The first test invocation exposed a mistyped protocol hash; it was corrected to match the existing locked plan and the suite then passed.

Ding-10 source/config are under `CES_FL/src/ding10_connected_control.py` and `CES_FL/proposals/lh_finalization_celebdf_20261104_r1/configs/ding10_connected.json`. Private CPU smoke/preflight receipts are at `/home/sachin.chaudhary/ces_fl_runs/lh_finalization_celebdf_20261104_43c605b/ding10_cpu_smoke_v3/`; a real-input CPU backward receipt is at `/home/sachin.chaudhary/ces_fl_runs/lh_finalization_celebdf_20261104_43c605b/ding10_real_input_cpu_dry/DRY_RUN.json`. These are software/data checks only, not a Ding run or CUDA qualification. The procedural-review reuse helper did run on L_H's new preview and verified all88case sheet hashes identical; the resulting qualification explicitly preserves an AI-reviewer record marked “NOT human approval.” It does not assert human review or forensic validity.

The fixed four-dataset ResNet test protocol is `CES_FL/proposals/lh_finalization_celebdf_20261104_r1/configs/resnet50_table3_eval_protocol.json`; manifest/overlap bindings are inherited from the hashed existing Tables1-2 plan, while selected checkpoint hashes are bound only after the ResNet runs complete. No TEST inference has occurred.
