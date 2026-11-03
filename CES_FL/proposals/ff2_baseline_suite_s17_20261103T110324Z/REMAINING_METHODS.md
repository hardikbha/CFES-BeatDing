# Conditional work after the runnable BCE/DAG/DAW wave

The user authorizes preparing/running additional FairForge-2 paper baselines within the shared72 additional GPU-hours and maximum3 allocated GPUs. These work orders are not launch-ready configurations. Finish CPU inspection/adaptation, declare exact configs and verify parity before spending remaining GPU time. Publish blockers instead of relabeling an approximation as an official method.

## PG-FDD

Official [Purdue-M2/Fairness-Generalization](https://github.com/Purdue-M2/Fairness-Generalization), pinned `e552a5422987897ab747e3b91b4b1b38d6531446`. Inspect `training/detectors/fair_df_detector.py`, its dataset/trainer and loss modules. It has three encoders, forgery-specific labels, hardcoded FF++ class/group counts and SAM training. A pre-existing CLIPFAIR loss port is not this native pipeline.

1. Read actual FF2 TRAIN metadata and verify whether the manipulation-specific target required by `spe_label` is available with trustworthy provenance. Confirm canonical original membership and8 demographic cells. A broad `family` category is not automatically the original method's specific-generator target; never invent labels or silently train on a subset.
2. Replace the FF++ `cls_num_list=[2475,25443,1468,4163,8013,31281,1111,2185]` with counts computed from the complete authorized FF2 TRAIN only, preserving the official index/order and class semantics. Record the resulting mapping/count/hash evidence. Do not derive any training weights from VAL/TEST.
3. Retain the three-encoder feature splits, loss terms, SAM two-pass optimization, original BatchNorm behavior and parameter-sharing policy. Verify these with source references, values/gradients on deterministic fixtures, optimizer traces and actual nonzero gradients for every intended branch. Do not replace PG-FDD with DAW plus another regularizer.
4. Add a new dated request/config with exact initialization files/hashes, source pins, preprocessing, label mapping, objective coefficients, effective batch, optimizer/schedule, selection/calibration and expected GPU cost. Declare any deliberate adaptation and why. Use seed17 and the fixed public split. No automatic diffusion or extra dataset creation.
5. CPU save/reload/resume test, then separately ledgered one-GPU CUDA/actual-TRAIN backward and timing checks. Launch only if the full declared schedule plus overhead fits the shared remaining budget. Otherwise record BLOCKED_METADATA, BLOCKED_IMPLEMENTATION or BUDGET_PENDING. Keep its row pending.

## FairAdapter

Official [AppleDogDog/FairnessDetection](https://github.com/AppleDogDog/FairnessDetection), pinned `b81e1cc0205e79a0393513e72aa0208021452704`. README is sparse; multiple `training/train_clipFairness.py`, `train_clipfairness_loss.py` and `train_clip_batchsize.py` variants exist. Determine the paper-specific one from actual source/evidence before attributing results.

1. Preserve and disclose the native CLIP/ViT-L/14 architecture and paper adapter/losses. It is not an Xception-matched row. Audit frozen/trainable parameters, optimizer membership, BN/dropout behavior and train/eval normalization. Confirm every required view, target and group annotation is available from the authorized FF2 TRAIN.
2. Require existing local compatible CLIP weights with a recorded hash. Resolve missing prerequisites explicitly. **Do not execute the repository's top-level `main.py`: it contains model loading and BLIP/Pix2PixZero image generation, not a safe benchmark entrypoint.** Do not automatically start generation or heavyweight downloads as part of inspection.
3. Create a minimal data adapter retaining the official method, one immutable exact config and source/config/data hashes. Test official objective/gradient parity, actual data membership, intended branch gradients, save/reload and selected validation replay. Publish adaptations and blockers before launch.
4. Measure memory/throughput inside a new ledgered allocation, then train only if the full fixed schedule fits the shared budget. Otherwise record the concrete blocker. Multiple ambiguous upstream trainers are not license to choose whichever gives the best TEST result.

## RSEF-FDD

Paper: [Redundant Semantic Environment Filling via Misleading-Learning for Fair Deepfake Detection](https://arxiv.org/abs/2405.15173). No verified official implementation was found in this inspection. The method also requires redundant samples/environment construction. Keep BLOCKED_SOURCE/DATA until a verifiable implementation and data recipe are available. Do not substitute an invented loss under its name. Do not contact authors unless separately instructed.

## What belongs in the comparison

Ding's [Tables1–2 and AppendixE](https://arxiv.org/html/2511.10150v5) distinguish methods and data requirements. The native first wave tests two objectives against its own matched BCE baseline. Later PG/FairAdapter rows must describe their actual native architecture, training budget and dependencies. The current Ding50+50 FF2 run stays separate and is not resubmitted. Additional appendix detectors/ResNet Table3 are not silently added to this72h queue.

For every completed method: one selected checkpoint, fixed validation policy, all supported group metrics, resource totals and uncertainty. Never copy the paper's FF++/cross-domain numbers into an FF2 experiment row. Do not compare VAL scores with TEST scores or use missing rows as zeros.
