# BD on Xception: design for Amendment 4b (26 Oct 2026, ~17:00-18:30)

R = `~/ICLR_2027/research/cvpr_readiness/r20261009`, A = `R/agents/BEAT_DING`, D = `R/agents/DINGREPRO_LIN`.

**Status.** Design only. No Xception BD run exists. Nothing was launched, and `run/`, `configs/` and `cfes/` were not
touched. The only computation was a CPU probe on FF++ **validation** images (section 2.4). It read no test data and
trained nothing. This file is the draft of the "Amendment 4b" implementation note that the plan requires before any
Xception BD run. Three user decisions are needed first (section 8).

## 0. Recommendation (one paragraph)
Start each seed from **Ding's own phase-1 checkpoint**, `D/results/DING_base_s{7,17,27}/49.pth`. These are our runs
of Ding's official `train_ori_model.py`, which is the same start Ding's SFD+GDA uses. Use Ding's network, its 224 px
input and its 0.5/0.5 normalisation unchanged.
- **Step 0: BN re-estimation.** Re-estimate every BatchNorm statistic once, on clean 32 real + 32 fake batches
  (weights untouched), then freeze all BN statistics. Ding's GDA does the same thing implicitly: it runs
  `model.train()` with a frozen backbone, and its fairness loss has zero gradient.
- **Trainable scope.** Ding's GDA trains only `last_linear`. BD trains `last_linear` plus the BD heads:
  - the patch head and its tanh gate;
  - component C's multi-stage heads and their gate.
- **Optimiser.** Ding's phase-2 optimiser: SGD 1e-3, momentum 0.9, weight decay 5e-3, constant LR, fp32. The budget
  is the BD budget of 10 × 1976 steps.
- **Arms.** The winning CLIP components on an "Arm E on Xception" base, built like `configs/bd_resnet`.
- **Comparison.** Against Ding's published Tables 1-2, and against our reproduction of Ding's code, at the same
  matched operating point (val TPR 94.18 %).

This design isolates the one thing that differs: Ding's phase 2 (SFD + GDA) versus BD's phase 2, on the same
phase-1 network. It also avoids all three earlier failure modes. **Fallback** if the user keeps the 16 Oct rule
"never load Ding's checkpoints": a new phase-1 trained by our code with Ding's exact phase-1 recipe (section 3.1,
option X1). Do **not** use the existing `CFES/runs/xception_s*_p1` checkpoints (section 2.2).

## 1. Ding's official Xception recipe (read from `D/code/Ding_mod`, not from the paper)
| Item | Phase 1 `train_ori_model.py` | Phase 2a `SFD.py` | Phase 2b `GDA.py` |
|---|---|---|---|
| Network | `XceptionDetector`, `utils/xception.py`, mode `adjust_channel`: Xception trunk → bn4 (no ReLU) → `adjust_channel` = Conv1×1 2048→512 + BN + ReLU → avg-pool → `last_linear` 512→2 | same | same; SFD mask = 10 of 512 `adjust_channel` output channels zeroed by `prune.custom_from_mask` |
| Init | ImageNet `xception-b5690688.pth`. Ours is the substitute `xception-43020ad28` (same architecture). Its pointwise weights are already 4-D, which is a disclosed patch. Missing keys: `adjust_channel`, `last_linear` (new) | phase-1 `49.pth` | phase-1 `49.pth` + SFD mask |
| Input | `resnet_default_data_transforms['train']`: **Resize((224,224))**, ToTensor, Normalize(0.5, 0.5). **No augmentation, no flip** | 224 | 224 |
| Batch | `CombinedDataset`: 64 fakes (each once per epoch) + 64 random reals = **128 per step**, 988 steps per epoch | 64 | 32 fake + 32 random real = 64 per step, 1976 steps per epoch. Plus a 32+32 "group" batch from one random intersection group under `no_grad` |
| Loss | 2-class CE on `last_linear` | per-group channel SNNL | CE + 0.005 · Sinkhorn(cls, group cls). **The Sinkhorn term is computed inside `torch.no_grad()`**, so its gradient is 0 (`D/results/GDA_GRAD_CHECK.json`). GDA reduces to CE |
| Trainable | everything | none | **`last_linear` only** (1,026 parameters) |
| BatchNorm | train mode (batch stats + EMA, momentum 0.1) | — | **`model.train()`: BN uses batch stats and keeps updating its running stats** from the clean combined batches and the group batches, although every BN parameter is frozen |
| Optimiser | SGD lr 1e-3, momentum 0.9, wd 5e-3, constant LR, no clip, fp32 | — | the same. The `StepLR(60)` is created but never stepped, so the LR stays constant |
| Epochs | 50 (49,400 steps) | 1 pass | 50 (98,800 steps). Our s7 rerun took 24 h, mostly DataLoader re-creation every step |
| Checkpoint | final epoch `49.pth` (no validation selection) | — | final epoch `49.pth` |
| Test | `fair_df_default_data_transforms['test']`: Resize(224) + CenterCrop(224), which is a no-op crop on the 380×380 square crops; Normalize 0.5. Softmax P(fake) ≥ 0.5 | | |

Our earlier `CFES/configs/ffpp_ding_xception.yaml` says "Ding's training transform: Resize((256,256))". **That is
wrong.** 256 px is the transform of the `FairDetector` (`fair_df_default_data_transforms`). Both scripts that
actually train Xception, `train_ori_model.py` and `GDA.py`, import `resnet_default_data_transforms`, which is
224 px.

## 2. Why the earlier CFES-Xception attempts failed
### 2.1 Phase 1 (`CFES/runs/xception_s{7,17,27}_p1`): trained unlike Ding, ended on bad epochs
- **Deviations from Ding's phase 1.**
  - Train at 256 px, evaluate at 224 px. The scale differs between train and test, and the feature grid changes
    from 8×8 to 7×7.
  - 32 real + 32 fake per step × 988 steps, i.e. half of Ding's 128 images per step. That is half the fakes per
    epoch and half the total exposure.
  - The fused patch path is present, although α ended at about 3e-4, so it is inert.
- **Late-training instability.**
  - Val AUC by epoch is flat at about 92 up to epoch 30, then oscillates. Examples: s7 95.9 → 78.6 → 95.7 → 90.6;
    s27 96.1 → 70.2 → 91.3.
  - `last.pt` is a bad epoch for every seed: val AUC 90.6 / 93.4 / 91.3, against `best.pt` 95.9 / 96.1 / 96.1.
- **Ding's own phase 1 does the same.** `DING_base` at epoch 49 has val AUC 89.1 (s7), 96.1 (s17) and 96.6 (s27).
  Its test FF++ AUC is 89.0 for s7, against 92.69 for the "Ori" row Ding published.
- **Mechanism (observation plus interpretation).** With SGD at a constant 1e-3 and weight decay 5e-3, the
  BN-normalised conv weights shrink, so the effective learning rate rises over training.
  - In our s27 `last.pt`, `bn4.running_var` has mean 1.9e-7. That is **below BN's eps of 1e-5**, so bn4 no longer
    normalises.
  - The running statistics, an EMA over the last batches, then track a moving network. Evaluation-mode outputs
    swing from epoch to epoch.

### 2.2 Phase 2 failures (all seed 27, from the bad `last.pt` with val AUC 91.25; 16 Oct)
| Run | Scope / BN | What happened (train_log, checkpoints) | Root cause |
|---|---|---|---|
| `BROKEN_xception_s27_p2_bnstats` | `last_block` (adjust_channel + last_linear + patch head); trunk BN frozen, **adjust_channel BN in train mode** | val AUC by epoch: 92.6, **50.0**, 92.9, 92.2, **50.0**. adjust_channel conv norm 1.2 → 6.1. Running-variance range widened from [7e-5, 1.3e-4] to [1.8e-5, 2.4e-3] | The BN running stats were estimated on our **mixed** step batch: 64 clean + 64 nuisance views + 16 pair images, one forward. Evaluation then used clean images, a different distribution, and the scores shifted. AUC = 50.0 means every val score was identical: all scores saturated, and the pre-F3 code computed AUC from the float32 sigmoid (Codex F3, 25 Oct), where saturated scores tie. |
| `DIVERGED_xception_s27_p2_lastblock` | `last_block`, **all BN stats frozen** | det loss 0.2 → 12 → 6,191; stab 1.8 → 85,000; grad norm 13 → 7,000. adjust_channel conv norm 1.2 → **28.5**, BN γ max 0.14 → **12.6**, `last_linear` norm 2.1 → 33 | With frozen statistics, BN is a fixed affine map. Nothing bounds the scale of the trainable conv. The CE loss on separable training data, and the CFES terms on raw logits (stab Huber, sens margin), reward scale growth. Clipping the gradient at 5 does not stop drift in one direction. This is the same scale-inflation mechanism as the ResNet head-only divergence (`CFES/diagnosis_p2/DIAGNOSIS.md` §2) |
| `STOPPED_xception_s27_p2_headonly_ep5` | head only, all BN stats frozen | val AUC 91.252 → 91.273 over 7 epochs. F_FPR got worse | It started from the bad `last.pt` and froze the bad BN statistics. A linear head on frozen, mis-normalised features cannot recover. Ding's GDA, in contrast, re-estimates them (section 2.3) |
| `xception_s{7,17}_p2` | — | `TRAINED` = skip marker only (decision rule after the s27 head-only run) | never trained |

### 2.3 What Ding's phase 2 actually does
- **The fairness loss is inert.** Its gradient is 0, so GDA is head-only CE plus train-mode BN on clean 32+32
  batches. The SFD mask removes 10 of 512 channels.
- **Reproduction s7, FF++ test AUC by stage.** DING_base epoch 49 = 89.0. After GDA: **97.45 at epoch 6**, 97.41 at
  epoch 25, 97.44 at epoch 49. Celeb-DF goes 56.9 → 71.9 at epoch 6, and DFD 57.2 → 73.3. These test curves were
  scored on 19-20 Oct, before this design; they choose nothing here.
- **Ding's own paper shows the same jump.** "Ori" 92.69 against "Ours" 97.71 on FF++.

### 2.4 CPU probe (validation only): BN re-estimation alone
`scratchpad/bn_probe.py`, output `bn_probe_224.json`.
- **Setup.**
  - Val subsample: 800 real + 1,600 fake FF++ val frames, fixed seed 0, at 224 px with Ding's transform.
  - Re-estimation: 40 clean batches of 32 real + 32 fake train frames.
  - Weights and head unchanged. Score = logit_fake − logit_real.

PROBE_TABLE

## 3. Design
### 3.1 Start checkpoint
| Option | What | For | Against |
|---|---|---|---|
| **X0 (recommended)** | `D/results/DING_base_s{7,17,27}/49.pth`: Ding's official phase 1, run by us; 224 px, 64+64, 50 epochs, final epoch | The same phase-1 network that Ding's SFD+GDA starts from. Per seed, the only difference is the phase-2 method, so the comparison with our Ding reproduction is paired (DING_full s7 = DING_base s7 + SFD + GDA). No extra GPU time. All 3 seeds exist | Reverses the 16 Oct CFES rule "never load Ding's trained checkpoints", so it needs the user's go. The s7 start is a bad epoch (val 89.1), as it is for Ding's own GDA. Step 0 repairs it, and the probe measures by how much |
| X1 (fallback) | New phase 1 by **our** code with Ding's exact phase-1 recipe: `trainable: all`, 224 px, batch 128 (64 real + 64 fake), 988 steps × 50 epochs, SGD 1e-3 / 0.9 / 5e-3 constant, fp32, `score_path: cls` (no patch head), `grad_clip: .inf` (`train.py` always calls `clip_grad_norm_`; inf
makes it a no-op), `last.pt` at epoch 49 | Keeps the 16 Oct rule. Same recipe as Ding | About 2.5-3.3 h per seed of GPU time (0.105 s/step measured at 64 images and 256 px; Ding's s7 took 3 h 19 min). The phase-1 endpoint is noisy (section 2.1), so the comparison is no longer paired with Ding's reproduction at phase 1 |
| X2 (rejected) | existing `CFES/runs/xception_s*_p1/last.pt` or `best.pt` | exists | 256 px training against 224 px everywhere else; half of Ding's exposure; `last.pt` is in the oscillation. `best.pt` is val-selected, which Ding's phase 1 does not do. It would be a third, non-Ding phase-1 recipe |

Loading under X0: `model.xception.init_from = DING_base_s<seed>/49.pth`. This already exists in
`cfes/model_xception.py` and loads Ding's detector state dict strictly. Under X1: `model.init_ckpt = <X1 run>/last.pt`.

### 3.2 Network, input and precision (all as Ding)
- **Network.** Ding's `Xception(mode='adjust_channel', num_classes=2, inc=3, dropout=False)`.
  - `s_cls = logit_fake − logit_real`, so sigmoid(s_cls) is exactly Ding's softmax P(fake).
  - 2-class CE on (l0, l1) equals BCE on s_cls, value for value and gradient for gradient.
- **Vendoring for the pulid environment.** pulid (torch 2.4.1, used by `run/bd_run.sh`) cannot import Ding's
  `detectors` package (`efficientnet_pytorch` is missing; checked 26 Oct). So the classes `SeparableConv2d`, `Block`
  and `Xception` of `Ding_mod/utils/xception.py` are copied **verbatim** into a new `cfes/ding_xception.py`, without
  the registry or networkx imports. This mirrors how `model_resnet.py` rebuilds Ding's ResNet. A test asserts:
  - identical `state_dict` keys and shapes;
  - an identical ImageNet load: same pointwise patch, missing keys `adjust_channel.*` and `last_linear.*` only;
  - identical `DING_base_s7` load;
  - a bit-identical forward on CPU against Ding's own module (run under qwen38), on 8 fixed images.
- **Input.** `data.input_size = eval_size = 224` (not 256), `resample: bilinear`.
  - Our loader calls PIL `resize((224,224), BILINEAR)`, which is exactly torchvision `Resize` on a PIL image. Pixels
    are scaled to [0,1], then (x − 0.5)/0.5 inside the model.
  - A test checks bit-identity against Ding's transform on 16 train and 16 test frames.
- **Precision.** fp32 (`optim.amp: false`), as Ding and the ResNet BD arm.
- **Grid.** 7 × 7 at 224 px, so `model.xception.grid: 7`. The old configs said 8.

### 3.3 BatchNorm handling (fixes all three failures)
1. **Step 0: clean re-estimation**, done once, before the epoch −1 validation, and skipped on `--resume`.
   - Reset every BN's running statistics. Run the network in train mode under `no_grad` over **K = 200 clean
     batches** of 32 real + 32 fake FF++ train frames (12,800 images; about 1-2 min on a GPU).
   - Use cumulative averaging (`momentum=None`), Ding's transform, no augmentation and no views.
   - Sample with `BalancedRealFakeSampler` and the run seed + 999 (a separate stream, so the training data order is
     unchanged).
   - Weights are untouched. The new statistics are saved in every checkpoint (`ckpt.full: true`).
   - This is a deterministic version of what GDA's train-mode forward does with an EMA. The probe in section 2.4
     shows what it recovers.
2. **Then every BN statistic stays frozen** (`freeze_all_bn_stats: true`; the BN affine parameters are frozen too,
   since they are trunk parameters).
   - The mixed BD step batch (clean + augmented + nuisance views + pairs + stratified, 208 images, one forward)
     therefore never touches BN statistics. That fixes the BROKEN failure.
   - Evaluation uses exactly the training-time normalisation. Running BN in train mode as GDA does is not possible:
     here it would estimate statistics on processed and augmented images.
3. **No trainable layer feeds a frozen-statistics BN.** The trunk and `adjust_channel` are frozen (section 3.4), so
   the DIVERGED failure (unbounded conv scale under a fixed affine BN) cannot occur.
4. **Scale guards** carried over from the ResNet fix:
   - tanh-bounded fusion weight (`alpha_param: tanh`, |α| ≤ 1);
   - background-evidence BCE off and sensitivity map terms off (`bg_evidence_weight: 0`, `map_terms: false`);
   - 10-epoch budget, gradient clip 5.
   - The training log already records `s_cls` and patch scale. Abort rule: stop if mean |s_cls| on the clean batch
     exceeds 200, or the loss is non-finite. Report it as a failure; do not retune.
5. **A test before the first run** (CPU, qwen38 or pulid):
   - the step-0 statistics are finite;
   - the BN buffers do not change during 20 BD steps;
   - eval-mode scores equal train-mode scores of the BD model on a clean batch (no BN mode mismatch), within 1e-5.

### 3.4 Trainable scope
`model.xception.trainable: head`. Trainable parameters:
- `net.last_linear` (1,026, Ding's GDA scope);
- the patch head LN → 512→128 → GELU → 128→1 (66,817);
- the fusion gate α;
- component C (when it is in the winning set): three 1×1 convs and their gate (256 + 728 + 1024 + 3 + 1 = 2,012).

Everything else is frozen, with BN as in section 3.3. That gives about 70k trainable parameters against Ding's 1,026,
which is listed as a deviation. The scope is chosen because:
- it is Ding's phase-2 scope plus BD's own heads;
- it is the only scope without a recorded failure on this network once BN is handled;
- it is the scope of the ResNet BD arm, so Tables 2 and 3 are built the same way.

Wider scopes (`last_block` with clean-batch BN) are **not** run. They would need a separate amendment with their
own selection rule.

**Start equivalence.** Under X0 the patch head is new. Its last Linear is initialised to zero, so the starting score
is exactly the step-0 network's s_cls.
- Gradients still reach the zero layer (∂L/∂W = hidden × ∂L/∂t ≠ 0).
- Test: max |Δscore| = 0 against Ding's model plus the same step-0 statistics.
- The C gate starts at 0, as on ResNet.

### 3.5 Optimiser and budget
- **Optimiser.** SGD lr 1e-3, momentum 0.9, wd 5e-3, constant LR, fp32. Same as Ding's GDA and the ResNet BD arm.
  The 1-step warm-up of `ffpp_ding_*` is kept: a factor of 1 at step 0, so it has no effect.
- **Gradient clip.** 5.0 (Ding: none). Kept because the ResNet head-only arm needed it as a guard.
- **Budget.** **10 × 1976 steps**, the BD budget of plan section 3 for CLIP and ResNet. Ding's GDA uses 50 × 1976.
  Disclosed. Supporting fact: Ding's reproduction is already at its final level by GDA epoch 6 (section 2.3).
- **Ordinary batch.** 32 real + 32 fake = Ding's GDA batch.

### 3.6 Arm E base and components A-D on Xception
**Base `configs/bd_xception/base_E.yaml`.** It is built like `configs/bd_resnet/base_E.yaml`: the CFES-ResNet arm A
recipe on Xception, plus Arm E's group-FPR-gap.
- **Detection terms:** det + pair_det + strat_det.
- **Stability:** stab 0.5 on nuisance views (bg_blur 0.6 / photometric 0.4), `bg_evidence_weight: 0`.
- **Sensitivity:** sens 0.5 on 8 (fake, own source) pairs per step, shared intervention, `map_terms: false`.
- **Stratified batch:** 2 per (label × group) cell.
- **Group-FPR-gap:** `fprgap` w\* = 1, q 0.9, τ 0.5, n_min 2, include_views, window 4096.
- **Other settings:** `alpha_param: tanh`, `decoupled_rng: true`, `eval: {prob_float64, rank_metrics}`.
- **Like ResNet:** no real-group risk term, no local norm.
- **Selection:** `val_auc_plus_inter_es_auc`, `include_init: true`.

BD0-BD4 inherit from the base exactly as `configs/bd_resnet/BD*.yaml` do. Only the screen's winning set is trained:
the promoted BD arm, else `base`, through the same `resnet_components()` rule.

| Component | CLIP (screen) | Xception mapping |
|---|---|---|
| A, processing augmentation | `cfes/bd_aug.py` at the 224 px input, p 0.6, one of 6 ops + hflip 0.5, both classes, all three loaders, pair-consistent | **Identical**: architecture-free, same 224 px input, same ranges. BN statistics are not affected (section 3.3). Ding uses none |
| B, ESA | class-conditional multi-quantile alignment on asinh(score), gender/race/intersection, n ≥ 4, pool 208 = 64 + 64 views + 16 pairs + 64 stratified (4 per cell) | **Identical** (score-level). Same pool and sampler |
| C, multi-layer evidence | CLIP blocks 8/12/16/20 → MLP patch maps → top-8 + local norm, gate 0 | **Multi-stage, as on ResNet (layer2/3/4 = strides 8/16/32):** forward hooks on Xception `block2` (256 ch, 28×28, end of the stride-8 entry stage), `block11` (728 ch, 14×14, last middle-flow block, the end of stride 16) and `block12` (1024 ch, 7×7, exit block, stride 32). Per stage a 1×1 conv (C→1) gives an evidence-logit map. Each map is adaptive-avg-pooled to 7×7; the maps are averaged into P2. score += tanh(g)·top4(P2) with g = 0 at the start; no local norm (as ResNet). Hooks are safe: each tapped output is consumed by a non-inplace op (`block3.rep[0]` and `block12.rep[0]` are `ReLU(inplace=False)`; `conv3` is not in-place). Heads come from their own seed (`init_seed: 0`), so the global RNG is unchanged |
| D, group-AUC floor | soft pairwise AUC per group, weight 0.5, same pool | **Identical** (score-level) |
| Patch head | CLIP patch tokens | `adjust_channel` 7×7×512 map → LN-MLP patch logits → top-4 (k = 4 of 49, as ResNet), tanh gate |

### 3.7 Selection and operating points (plan section 4, unchanged)
- **Within-run selection.** Val AUC + val intersectional es-AUC, threshold-free (F3). The start checkpoint after
  step 0 is eligible (epoch −1). Ding uses the final epoch; disclosed.
- **Operating point (primary).** One global threshold per model, set on FF++ val so that val TPR = **94.1768 %**,
  the Xception reproduction's val TPR at 0.5 (`D/results/audit_scores/full_s7_ep49_ffval.npz`). For Xception this
  is a genuine same-backbone target, not a proxy as on ResNet.
- **Also reported:** 0.5 / native class-balanced calibration and A1, with TPR/FPR, the F_DP oracle and the FF++
  i-F_FPR video-permutation noise floor. All are produced by `scripts/bd_eval.py`, which already maps backend
  `xception` to Tables 1-2.

### 3.8 Evaluation, like for like
- **Data and code.** The same test CSVs `D/data/{ff++,dfdc,celebdf,dfd}/test.csv`, row counts equal to Ding's
  Table A.1: 25,401 / 22,857 / 28,458 / 9,385. The same 8 and 6 group schemes, the same metric code
  `cfes/metrics.py` (diff 0 against Ding's released formulas), frame-level as Ding.
- **E1. Against Ding published Tables 1-2** (40 columns, `refs/Ding_CVPR2026.txt`).
  - The 3-seed mean ± SD at the matched point.
  - The plan section 5 success rule, unchanged: ≥ 25/40; AUC and all three es-AUC better on all four sets;
    i-F_FPR better on ≥ 3/4.
  - Report: `results/bd_final_xception.{json,md}`. This needs a third `--family xception` in `bd_report.py final`
    (additive; recipe prefix `X_`).
- **E2. Against our reproduction of Ding's official code, paired.**
  - DING_full s7 epoch 49 against X-BD s7: the same phase-1 network under X0, the same rows. Video-cluster paired
    bootstrap per column (5,000 reps, seed 2026), using the DINGREPRO `analyze.py` scheme.
  - **Gap:** DING_full s17 and s27 were stopped by the user (22 Oct), so E2 is s7-only unless they are re-enabled
    (section 8, D2). The released GDA takes about 24 h per seed. A faithful but faster rerun (the same released step,
    only reusing the group DataLoader iterators) would take ≈ 3-5 h, but it is a code change to Ding's script and
    would need disclosure.
- **E3. Controls** (cheap; recommended; the user decides, section 8 D3).
  - **X-recal:** step 0 only, no training. This is epoch −1 of every BD run, already logged and scored for free. It
    shows how much of any gain is the BN re-estimation alone. Ding gets this for free too.
  - **X-base:** the base without A-D, s7. It mirrors the screen's role of `base`.
- **Reporting caveat, stated before the runs.** Our reproduction of Ding is itself 5.5-8.9 AUC points below Ding's
  published cross-domain AUC: DFDC 56.05 vs 61.86, Celeb-DF 71.22 vs 76.75, DFD 72.54 vs 81.46. FF++ is 97.44 vs
  97.71.
  - On the same backbone, beating every published AUC therefore needs gains of that size over the reproduction.
    The "AUC on all four sets" criterion is likely to fail on Xception even if BD beats the reproduction.
  - E2 is the like-for-like test of the method. E1 stays the pre-registered claim test and is reported as is.

### 3.9 Compute (estimates, from measured neighbours)
- **Measured neighbours.**
  - CFES-Xception head-only phase 2: 176 images per step at 256 px, measured 0.080 s/step, peak 5.5 GB.
  - ResNet BD4 smoke: head-only, 208 images, A on: 0.42 s/step, 2.9 GB.
  - Xception phase 1, trainable all, 64 images at 256 px: 0.105 s/step, 11.0 GB.
- **Expected for BD-Xception** (208 images at 224 px, frozen trunk, A on):
  - **0.4-0.6 s/step**, bound by augmentation in the loaders, as on ResNet;
  - **about 5 GB**;
  - 10 × 1976 steps ≈ **2.2-3.3 h per seed** plus about 15-25 min of evaluation (val + 4 test sets = 111,487
    frames; the leak-free variants reuse the same scores).
  - About 7-10 GPU-h for 3 seeds, or about 3.5 h wall time on 3 GPUs.
- **Step 0:** about 2 min. **X1** (if chosen): plus about 3 h per seed.
- The first BD-Xception task must be a 20-step smoke, like `bd_smoke`. It records s/step and peak memory, checks
  that the BN buffers are unchanged and that start equivalence holds.

## 4. Every deviation from Ding's recipe
| # | Item | Ding (official code) | BD-Xception | Why |
|---|---|---|---|---|
| 1 | ImageNet weights | `xception-b5690688.pth` (Baidu link, unavailable) | substitute `xception-43020ad28` copied under that name (inherited from the reproduction; 0 unexpected keys) | official file unreachable |
| 2 | Phase-1 start | own phase-1 run | X0: our run of Ding's official phase-1 code (identical recipe); X1: our code, same recipe | X0 is as close as possible |
| 3 | SFD channel decoupling | 10 of 512 adjust_channel channels masked | none | BD replaces SFD + GDA |
| 4 | Fairness loss | Sinkhorn ×0.005 (inert, gradient 0) | BD terms: Arm E fprgap + winning B/D | the method under test |
| 5 | BN statistics | train mode throughout GDA: EMA over clean 32+32 and group batches | one-shot cumulative re-estimation on 200 clean 32+32 batches, then frozen | the BD step batch mixes processed views; train-mode BN on it broke the s27 run |
| 6 | Trainable parameters | `last_linear` (1,026) | `last_linear` + patch head + α (+ C heads/gate) ≈ 70k | BD's heads; trunk frozen as in Ding |
| 7 | Score | softmax(last_linear) | s_cls + tanh(α)·top4(P) (+ tanh(g)·top4(P2)) | CFES/BD fusion; starts exactly at s_cls |
| 8 | Step batch | 32+32 (+32+32 group batch, no grad) | 32+32 clean + 64 nuisance views + 8 pairs (16) + 64 stratified (4 per label × intersection cell) = 208 | Arm E / ESA pools |
| 9 | Augmentation | none (Resize only) | A: p 0.6 processing op + hflip 0.5, if A wins | component A |
| 10 | Other losses | CE | + pair_det, strat_det, stab 0.5, sens 0.5 (Arm E base) | BD base = Arm E |
| 11 | Budget | 50 × 1976 steps | 10 × 1976 | BD budget (plan §3); Ding's curve flat after epoch 6 |
| 12 | Gradient clip | none | 5.0 | ResNet head-only guard |
| 13 | Checkpoint | final epoch | best val (AUC + i-es-AUC), start eligible | plan §4 |
| 14 | Decision threshold | P(fake) ≥ 0.5 | matched val TPR 94.18 % (primary); 0.5 / native / A1 also reported | plan §4 |
| 15 | Real sampling | random real with replacement per fake | `BalancedRealFakeSampler`: reals without replacement, recycled | our loader |
| 16 | Software | Ding env (our repro: qwen38, torch 2.10) | pulid torch 2.4.1 + vendored Xception, bit-identity tested | same tree as the CLIP/ResNet BD arms |
| 17 | Groups / data / test transform / metric code | — | identical | — |

## 5. Implementation plan (after approval; nothing below is done)
All of this is additive. The CLIP screen is running, and ResNet tasks are queued that import `cfes/engine.py`.
1. **New `cfes/ding_xception.py`** (verbatim Ding classes) and **new `cfes/model_xception_bd.py`** (`CFESXceptionBD`):
   - vendored network, `init_from` / `init_ckpt`;
   - `trainable` and BN switches as in `model_xception.py`;
   - zero-init last patch layer under X0;
   - C hooks (`_keep_block_output`) on `block2` / `block11` / `block12`;
   - `bn_recal()`.
   `cfes/model_xception.py` is **not** edited.
2. **`cfes/engine.build_model`:** a new backend `xception_bd`, imported lazily inside the branch, as `fsvfm` is.
   - Old backends stay bit-identical; checked with `tests/identity_rev4_EF.py` and a ResNet BD4 step identity.
   - Edit by writing a new file, then `py_compile`, then an import test in pulid, then an atomic `mv`.
   - **Running processes have already imported `engine`. Queued ResNet tasks will import the new file, so it must
     be import-safe.**
3. **`scripts/train.py`:** call `model.bn_recal(...)` after the model is built, when `--resume` is off and
   `model.xception.bn_recal.batches > 0`. Otherwise a no-op.
4. **Configs:**
   - `configs/ffpp_ding_xception_bd.yaml` (network/input/optimiser as section 3.2/3.5) → `configs/xception_bd_p2.yaml`
     (head scope, BN frozen, step 0) → `configs/bd_xception/{base_E,BD0..BD4}.yaml` (inheritance as `bd_resnet`).
   - CFES configs stay untouched.
5. **`scripts/bd_report.py`:** `--family xception` → `results/bd_final_xception.*`, recipe `X_<set>`, plus an E2 paired
   section against `D/results/audit_scores/full_s7_*`.
6. **`run/bd_run.sh`:** a new kind `xception <s>` (and `xception_smoke`, `xception_final`) that uses
   `init_of xception <s>` = `D/results/DING_base_s<s>/49.pth` (X0).
   - **Written as `run/bd_run.sh.new`, tested with `BD_DRY=1`, then `mv` over the live file** (never edited in
     place).
7. **`run/bd_tasks.json`:** add `bd_xception_smoke` → `bd_xception_s7/s17/s27` → `bd_xception_final`, ordered after
   `bd_screen` and **before** `bd_resnet_*` (Amendment 4 order).
   - Written to a temp file, then `os.replace`, after `normalise_external` / `validate_spec` pass.
   - `bd_winner_s17/s27` stay last.
8. **Tests:** `tests/bd_checks.py` gains `x1`-`x6`:
   - vendored-network bit-identity;
   - transform identity;
   - start equivalence;
   - BN frozen after step 0, and no BN mode mismatch;
   - C hooks against stage shapes (N,256,28,28) / (N,728,14,14) / (N,1024,7,7);
   - a dry run of the task chain.
   Plus a GPU smoke with 20 steps.
9. **Amendment 4b** appended to `R/decisions/2026-10-26-beat-ding-plan.md` with sections 3-4 of this file,
   **before** any Xception BD run.

## 6. Order and timing
After `bd_screen`, which is expected at about 21:00-22:00 today, from current epoch progress: BD1 6/10, BD2 5/10,
BD3 5/10, BD4 2/10:
1. `bd_xception_smoke` (about 10 min);
2. X-BD s7/s17/s27 in parallel where GPUs allow (about 3 h);
3. `bd_xception_final` (CPU);
4. ResNet (Amendments 2-3);
5. CLIP winner s17/s27 (deferred).
X1, if chosen, adds about 3 h of phase 1 before step 2.

## 7. Expected outcome (stated before any run; not a target)
- **What favours BD.**
  - On Xception, BD's scope equals Ding's GDA scope.
  - Ding's fairness term contributes nothing (gradient 0), so any fairness gain must come from BD's terms.
  - FF++ in-domain columns are where Ding's reproduction is strongest: AUC 97.44, i-F_FPR 24.2, below the
    permutation null.
- **Hard to win.**
  - Cross-domain AUC and es-AUC against the *published* values (section 3.8 caveat).
  - FF++ AUC against 97.71.
- **Possible on FF++.** F_FPR and F_DP columns at the matched point, and gains over the reproduction (E2).
- **No prediction** of the 25/40 rule.

## 8. Decisions needed from the user before Amendment 4b is written
- **D1. Start checkpoint.**
  - X0, recommended: Ding's phase-1 `DING_base_s*/49.pth`. This reverses the 16 Oct "never load Ding's checkpoints"
    rule for this arm only.
  - X1: re-train phase 1 with our code at Ding's exact recipe (+ about 3 h per seed).
- **D2. Re-enable DING_full s17/s27 for a 3-seed paired reproduction reference?**
  - Option (a): the released GDA unchanged, about 24 h per seed.
  - Option (b): the same released training step with only the per-step DataLoader re-creation removed (≈ 3-5 h per
    seed; disclosed code change).
  - Option (c): no; E2 stays s7-only.
- **D3. Run the X-base control (s7, about 3 h)?** The X-recal control costs nothing.

---
**STOPPED by the user, 26 Oct ~17:52.** Workflow `wefftufgl` completed the Design stage only. Implement, Verify and Fix
did not run to completion. Checked: no file in configs/, cfes/, run/, tests/ was created or changed, bd_tasks.json is
unchanged since 15:26, and no Xception task is queued. Amendment 4b is NOT written. The next step after resuming:
user decisions D1-D3 (section 8), then implement.
