# FairForge-2 expanded detector setup

User request,4November2026: prepare Xception,EfficientNet-B4,ViT-B/16,F3Net,SPSL,SRM,UCF,UnivFD,CORE,DAW,DAG,PG on the fixed public FairForge-2 split, with detection/fairness metrics and complexity. This is a source/setup publication, not a new set of trained results. Existing experiments are preserved.

[Operator prompt](GPU_OPERATOR_PROMPT.md) · [Method cards](methods.json) · [Exact protocol](protocol.json) · [Paper/source audit](PAPER_AND_METHOD_AUDIT.md) · [Checks](CHECKS.md).

## What is ready

| Method | Category | Implementation / status |
|---|---|---|
| Xception | CNN | Existing50epoch BCE run; preserve/reuse; expanded adapter also checked for exact native initialization/forward parity |
| EfficientNet-B4 | CNN | New executable adapter; preserves pretrained stem |
| ViT-B/16 | Transformer | New executable torchvision adapter |
| F3Net | CNN / frequency | New executable AI-Face FAD-only implementation; not a full dual-branch paper reproduction |
| SPSL | CNN / frequency | New executable adapter with explicit spatial-only FFT correction |
| SRM | CNN / frequency | New executable native two-stream/AM-softmax adapter with224pixel attention grid |
| UCF | CNN / generalization | Source cloned; Specific-target provenance and paired training adapter remain required; execution refuses BCE fallback |
| UnivFD | VLM | New executable frozen CLIPViT-L/14 visual encoder plus769trainable head parameters |
| CORE | CNN / generalization | New executable paired-view consistency adapter; same-source original/flip views |
| DAW-FDD | Fairness | Existing50epoch run COMPLETED_VERIFIED; reuse |
| DAG-FDD | Fairness | Existing50epoch run COMPLETED_VERIFIED; reuse |
| PG-FDD | Fairness | Source cloned; Specific targets, TRAIN group counts, paired batches and SAM adapter/checks remain required |

The seven new executable adapters and Xception reference pass CPU synthetic224pixel forward/backward and evaluation batch-independence checks. Fresh local pretrained weights, actual FF2 input checks, CUDA checks and compute projections remain cluster prerequisites. These receipts do not claim scientific performance or production-weight loading.

All twelve named methods have pinned implementation source coverage in the official AI-Face benchmark. Six repositories were cloned as source-only sparse Git worktrees: AI-Face-FairnessBench,DeepfakeBench,UniversalFakeDetect,Fairness-Generalization,FairnessDetection and an anonymous IIL release whose README matches the requested paper. The anonymous author's identity was not independently verified. FairForensics official implementation was not verified; its paper is used for table organization, not invented results. Sources and Git blob hashes are in `sources.lock.json`; no model checkpoints/datasets were downloaded.

## Matched protocol and declared adaptations

Use public TRAIN218,266 / VAL31,107 / TEST87,564, seed17. Existing BCE/DAG/DAW and new runnable models use50epochs,batch128originals,SGD0.001,momentum0.9,weight_decay0.005,float32/TF32off. Every TRAIN original once per epoch. TRAIN direct bilinear224; VAL/TEST shortest-side bilinear224 plus center crop. Preserve architecture-specific input normalization and pretrained assets in each config. CORE adds a second view per original; UnivFD freezes the encoder. These differences, parameter counts and forward costs are disclosed.

This is a controlled common-recipe adaptation, not each original paper's optimal/native training recipe. Do not copy AI-Face/GenFace checkpoint scores or paper parameter/FLOP numbers into FF2 rows. Do not claim all methods have identical optimization or pretraining. UCF/PG's native real/fake paired sampling repeats real examples and changes exposure; they need a declared paired control/protocol before becoming comparable runnable rows. Do not guess Specific targets from filenames or replace their objectives with BCE.

Model adapters load only hashed upstream architecture/loss definitions; they never execute the upstream top-level trainers, test selection, generation code or automatic downloads. Unused original-mode Xception512adaptation is removed, consistent with the existing native FF2 baseline. F3Net DCT resolution256→224 and SRM attention16×16→14×14 are explicit geometry adaptations. The released SPSL FFT includes the batch dimension; this adapter uses H/W only and tests image independence. EfficientNet's released adapter resets its pretrained stem randomly; ours retains the actual pretrained stem. CORE pairs the same source with its horizontal flip instead of unrelated batch halves. Preserve these names/notes in publications.

## Initialization and execution

`configs/*.json` are exact training templates. Xception-based methods use the existing pinned local ImageNet checkpoint. EfficientNet-B4,ViT andUnivFD require locally supplied compatible pretrained weights and SHA256 binding through `ff2_expanded_runner.py bind`. Use EfficientNet-PyTorch standard ImageNetB4,torchvisionViT-B/16ImageNet1K-V1,and OpenAI CLIPViT-L/14, respectively; document asset origin/hash. Do not use a detector already trained on AI-Face/FF2 as an undeclared ImageNet initialization.

The new runner imports the unchanged tested FF2 native training engine in a fresh module and replaces only explicit model/step/image adapters. Deterministic per-update RNG supports dropout/stochastic-depth resume; the independent uninterrupted-versus-resumed test is exact. Existing source/checkpoints are never modified. New records include real config/source hashes and exact selected validation replay.

Selection is earliest strict maximum validation AUC after50epochs. Calibrate only on selected VAL-real scores at empirical10%FPR; also retain fixed probability0.5. TEST is a separate explicit frozen-checkpoint command, with membership/contributor/byte-overlap checks and no target calibration. The evaluator accepts completed legacy BCE/DAG/DAW or new expanded checkpoints; it does not restart old training. A fixture checkpoint cannot produce a benchmark table row.

## Scores and tables

`CES_FL/scripts/ff2_expanded_metrics.py` reports AUC,accuracy,AP,EER,FPR,FNR plus F_FPR,F_DP,F_MEO,F_EO,F_OAE,es-AUC and worst-group AUC for gender,race,gender×race,age,skin,ids8 and gender×existing-skin-bin where supported. Unknown labels remain unknown. Minimum20real/20fake per eligible group; print denominators and excluded groups. Gender×race is the primary intersection. FF2's existing skin bins are not silently converted to AI-Face's six gender×skin groups.

F_MEO=max(FPRgap,TPRgap); F_EO=sum_g(|FPR_g−FPR|+|TPR_g−TPR|); F_OAE=maxaccuracy−minaccuracy. F_FPR is Ding's summed absolute FPR deviation. EER is a descriptive ROC statistic, never a chosen deployment threshold. All JSON values are fractions. Optional1000-draw connected-contributor intervals cover native Ding metrics; extended EO/OAE/utility metrics are explicitly point estimates. Retain900usable-draw minimum and one-seed/proxy-label limitations.

`render_table.py` groups measured rows into CNN,Transformer,VLM,Fairness categories and emits separate VAL/TEST and threshold tables. It rejects duplicated model/split rows and synthetic results. New scores are unavailable until actual training/evaluation finishes; pending rows must not receive paper numbers.

## Parameters and FLOPs

`profile_models.py` measures the instantiated inference graph at1×3×224×224. Report total deployed,trainable and frozen parameters. UnivFD omits the unused CLIP text tower; training auxiliaries/text costs from other papers are not directly comparable. FLOPs use fvcore's one-FMA convention; explicit attention handlers count QK/AV and fused MHA projections. Unsupported operations remain listed and measurements are labeled PARTIAL—especially FFT and nonlinear operations. Do not present counted operations as a complete exact FLOP total or compare them unqualified with another paper's two-operation FMA convention.

## Compute and current results

Reuse the EXISTING additional72GPU-hour FF2 ledger; this request grants no new72hours. Maximum3allocated GPUs including active baseline jobs. The pinned latest observation `091a91d` has DAG/DAW50/50completed and BCE49/50recorded. Refresh actual completion/reservations before launching. Actual CUDA checks,profiling,training,evaluation and held-idle time all count. `with_gpu_budget.py` refuses a missing/new ledger and enforces the remaining/concurrency limits. Profile all remaining50epoch costs before deciding the execution order; the remaining budget may not fit every requested model. Report an exact cost blocker rather than shortening epochs or claiming partial runs are complete.

CPU preparation/reporting requires no GPU reservation. Never take GPUs assigned to active Ding,ResNet,L_H or other jobs. Do not transfer the separate original24hour study budget. No unattended monitor, new seed sweep or hyperparameter search is installed by this package.
