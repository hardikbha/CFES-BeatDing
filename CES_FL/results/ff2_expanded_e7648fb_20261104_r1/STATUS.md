# FairForge-2 expanded baseline readiness — 4 November 2026

## Current outcome

**No expanded model training or TEST evaluation was launched.** The exact source commit is `e7648fb25aeb16d49bbef33ad998e1b1cf5a5016`, checked out detached at `/home/sachin.chaudhary/ces_fl_sources/ff2-expanded-e7648fb`. The proposal README, method matrix, protocol, source lock and code are present. The specifically named `CES_FL/proposals/ff2_expanded_baselines_20261104_r1/GPU_OPERATOR_PROMPT.md` is absent from this commit; the README contains the execution protocol and is linked to that missing path.

The prior FairForge-2 BCE, DAG-FDD and DAW-FDD jobs were verified complete at50epochs with `COMPLETED_VERIFIED`, `DONE` and selected-checkpoint validation replay. They were reused; no baseline was repeated. Their TEST sets remain unread.

All nine expanded-suite CPU tests passed with no skips after checking the pinned AI-Face source; all six shared-budget CPU tests passed. Six executable new adapters loaded their bound local pretrained weights on CPU and produced parameter counts. Six method configs are bound to the same fixed public TRAIN/VAL split and source commit, with preflight receipts reusing the identical, fully image-hashed inventory from the completed baseline preflights. No dataset labels were added or remapped.

GPU work is blocked by the site allocation binding. PBS assigned one GPU for job45394, but supplied neither `CUDA_VISIBLE_DEVICES` nor `PBS_GPUFILE`; the launch guard stopped before model code. A separate allocated diagnostic job45396 confirmed PyTorch could see all eight node GPUs. No GPU smoke or 20-step profile ran, and no physical GPU was guessed. The failed profile attempt and diagnostic outputs are preserved on the cluster.

The existing72GPU-hour ledger records27.887102 hours charged,0 reserved and44.112898 uncommitted. The two short PBS diagnostic allocations are included. The three prior FF2 baseline jobs are complete; three other user jobs remain running on gpu01 and were preserved.

## Existing FF2 reference rows

Seed17; fixed public split:218,266 TRAIN and31,107 VAL. AUC/FPR/FNR are percent; F_FPR is the native fraction-sum below. Each method uses its earliest maximum-validation-AUC checkpoint.

| Method | State | Epochs | Selected epoch | VAL AUC | VAL FPR | VAL FNR | Gender F_FPR | Race F_FPR | Gender×race F_FPR | Age F_FPR | Skin F_FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Xception BCE | COMPLETED_VERIFIED | 50/50 | 40 | 99.804621 | 9.967186 | 0.299976 | 0.034285 | 0.195413 | 0.405237 | 0.087404 | 0.145671 |
| DAG-FDD | COMPLETED_VERIFIED | 50/50 | 50 | 99.813559 | 9.967186 | 0.219750 | 0.042760 | 0.183245 | 0.378911 | 0.104475 | 0.127781 |
| DAW-FDD | COMPLETED_VERIFIED | 50/50 | 41 | 99.836011 | 9.967186 | 0.244166 | 0.034285 | 0.128659 | 0.306953 | 0.104531 | 0.067956 |

All three use the same FF2 VAL-real threshold targeting10% FPR; achieved VAL FPR is243/2,438=9.967186%. Age/skin labels are automatic proxies. These are validation values, not Ding paper or TEST scores.

The private training logs remain on the cluster: `/home/sachin.chaudhary/ces_fl_runs/ff2_baseline_suite_s17_20261103T110324Z/bce/sessions/ff2_bce_gpu01_20261103T124210Z/train.log`, with the analogous `dag` and `daw` session paths. Their aggregate histories and checkpoint replay receipts are the existing 50-epoch records; full private score rows remain off GitHub.

[Full per-group counts/rates and all implemented fairness metrics](BASELINE_GROUP_METRICS.csv) · [bootstrap intervals](UNCERTAINTY.md). Contributor bootstrap uses1,000 draws. The primary gender×race F_FPR has only490 usable draws for each method, below the predeclared900 minimum; do not claim a statistically supported intersection-fairness difference. AUC has1,000 usable draws. Only one training seed was run.

The table above uses the selected validation-real10%FPR thresholds. A separate [fixed logit-zero validation report](BASELINE_FIXED05.json) is provided; thresholds change FPR/FNR/disparities, not AUC. Both views are calculated from the same selected checkpoints and saved validation logits.

## Expanded methods

| Method | Readiness | Parameters (total / trainable) | Next gate |
|---|---|---:|---|
| Existing Xception BCE | Reuse completed | 20,809,001 / 20,809,001 | None for this wave |
| EfficientNet-B4 | Local ImageNet weights load PASS | 17,550,409 / 17,550,409 | CUDA smoke and20-step TRAIN profile |
| F3Net AI-Face FAD-only adaptation | Local Xception weights load PASS | 21,313,353 / 21,012,297 | CUDA smoke and20-step TRAIN profile |
| SPSL spatial-FFT adaptation | Local Xception weights load PASS | 20,809,289 / 20,809,289 | CUDA smoke and20-step TRAIN profile |
| SRM 224px adaptation | Local Xception weights load PASS | 53,137,310 / 53,129,660 | CUDA smoke and20-step TRAIN profile |
| CORE paired original/flip | Local Xception weights load PASS | 20,811,050 / 20,811,050 | CUDA smoke and20-step TRAIN profile |
| UnivFD frozen CLIP ViT-L/14 | Local CLIP weights load PASS | 303,966,977 / 769 | CUDA smoke and20-step TRAIN profile |
| ViT-B/16 | BLOCKED_WEIGHT | Architecture count85,799,425; pretrained tensors not loaded | No confirmed local torchvision-compatible checkpoint; cached JX/timm state is not accepted as a strict match |
| UCF | BLOCKED_LABELS_AND_ADAPTER | — | No provenance-verified TRAIN Specific sidecar or native paired reconstruction/contrastive adapter |
| PG-FDD | BLOCKED_LABELS_PAIRED_ADAPTER_AND_SAM | — | No verified Specific labels, paired training adapter or SAM/BatchNorm parity |
| DAG-FDD | Reuse completed | Xception baseline | No rerun |
| DAW-FDD | Reuse completed | Xception baseline | No rerun |

The parameter numbers describe instantiated detector architecture only. FLOP coverage remains partial for adapters with unsupported operations; see the pinned source complexity receipts. F3Net is explicitly the AI-Face FAD-only implementation, not the complete dual-branch paper model. SPSL and SRM include the documented spatial/224px adaptations.

The `counted_G_FLOPs_1FMA` column comes from the proposal's one-FMA inference-graph counter at1×3×224×224 using architecture-only/random weights. Every method is marked PARTIAL because unsupported nonlinear, padding, FFT or attention operations remain. These counts are not complete FLOPs and are not training-step timing profiles.

## Budget and next gate

The shared72-hour FF2 ledger has43.989128GPU-hours uncommitted after the direct-SSH smoke/profile batch. No hours were transferred from the separate L_H study. Five methods passed CUDA smoke and20-step real-TRAIN profiles; SRM's profile failed with OOM. The PBS binding defect is bypassed using the explicitly user-authorized SSH route and a UUID lease from the existing pool-v2 allocator.

The training selection is frozen in [TRAINING_SELECTION.md](TRAINING_SELECTION.md): F3Net and CORE,50epochs each, seed17, using42.00GPU-hours total cap. Both are now running on distinct direct SSH shared-pool-v2 UUID leases: F3Net on gpu02/GPU7, CORE on gpu01/GPU5. At the10:40IST snapshot, both training processes were active but neither had completed an epoch. The shared ledger shows28.010872GPUh charged,42reserved,2 active GPUs and1.989128GPUh uncommitted. Full runs retain the exact configs; no test scores or images are read.

The source README links both `GPU_OPERATOR_PROMPT.md` and `CHECKS.md`, but neither file is present in this commit. This report followed the available README/protocol/method matrix and records these missing files as input defects. The exact source/config hashes, test logs, source clone proof, preflight provenance and PBS binding failures are in this directory's evidence folder. No TEST result exists for the expanded adapters.
