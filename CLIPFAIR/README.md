# CLIPFAIR: four matched-backbone fairness baselines on CLIP (Amendment 3)

Pre-registration: `decisions/2026-10-25-cesel-plan.md`, Amendment 3 (25 Oct 2026, user: "yes add the four CLIP
baselines"; written before any baseline run or CESel result). R = `~/ICLR_2027/research/cvpr_readiness/r20261009`.

The question these rows answer: is a published fairness method's gain the method or the backbone? So every baseline
runs on the SAME CLIP ViT-L/14 + rank-1 SVD adapters, data, samplers, start checkpoint, budget, selection rule and
evaluation as CESel B0. Ding's published Xception rows stay as the reference.

## Status (25 Oct 2026)

- Code + CPU checks: `tests/clipfair_checks.py` prints `ALL_PASS` (18/18, including one real CLIP ViT-L/14 pass from
  `CFES/runs/clip_s7_p1/last.pt`). The checks run the official loss code, taken from the official files with `ast`,
  against these ports. Values and gradients are equal for DAW-FDD, DAG-FDD, the whole PG-FDD training loss, Ding's
  SNNL and the SAM step.
- CPU dry run of the whole pipeline (stub backbone, scratch output): SFD, 4 trainings, evaluation, A1 and the report
  all ran with rc 0.
- GPU smoke: PBS 44073 (gpu01, GPU 3) finished with exit 0 (SMOKE OK). Each run was 20 bf16 steps of 176 images,
  then evaluation, calibration, A1 and the report on 256 items per set. Logs are in `work/gpu_smoke_44073.mgmt01/`
  (the checkpoints were deleted).
  - SFD on 20 batches picked 20 channels.
  - Results per baseline:

    | Baseline | s/step | Peak allocated |
    |---|---|---|
    | DAW-FDD | 0.495 | 47.1 GB |
    | DAG-FDD | 0.488 | 47.1 GB |
    | PG-FDD (SAM, two passes) | 1.36 | 53.7 GB |
    | Ding | 0.19 | 3.5 GB (frozen backbone) |

  - Ding: the GDA Sinkhorn value was 2.19 with requires_grad 0, as released.
  - Estimated full runs: about 3 h of training for DAW/DAG and about 7.5 h for PG-FDD, plus validation.
- The pipeline jobs are queued:
  - gpu01 runner: PBS 44074, `-W depend=afterany:44067`.
  - gpu02 runner: PBS 44075 (GPUs 6 and 7 only), `-W depend=afterany:44068`.
- **No baseline number exists yet.**

The CESel code in `R/agents/CESEL` and `R/agents/CFES` was NOT modified. `cfes/` and `scripts/{train,train_ces,
evaluate_ces,calibrate,fair_a1,ces_posteval,cue_stats}.py` and `configs/ces_ffpp.yaml` are byte-identical copies
(sha256 list in `work/COPIED_FROM_CESEL.sha256`). The CLIPFAIR code is only the new files listed below.

## Common protocol (identical to CESel B0; checked in tests/clipfair_checks.py section 1)

| Item | Value |
|---|---|
| Backbone | FairReL CLIP ViT-L/14, frozen, rank-1 SVD residual adapters on every attention projection. `feat` = 16x16x1024 patch tokens after ln_post; `z` = CLS after ln_post |
| Start | `CFES/runs/clip_s{seed}_p1/last.pt`, backbone keys only (872, strict). New heads are randomly initialised (as B0). Ding also takes p1's CLS head (its method starts from the trained base, see below) |
| Data | FF++ Lin/Ding split: `DINGREPRO_LIN/data/ff++_ding/train.csv` (76,139); val = realval + fakeval |
| Images per step | 176 in one backbone pass, the B0 stream (same seed, so the same images): 64 balanced (32 real + 32 fake), 8 (fake, own-source real) pairs (16 images), 96 stratified (4 per label x group x family cell) |
| Budget and optimiser | 10 epochs x 1976 steps. AdamW: adapters lr 1e-4, new modules lr 1e-3, wd 4e-3, warmup 200, cosine, grad clip 1.0, bf16 |
| Adapter regulariser | FairReL orthogonality loss, weight 0.1, as in B0 / CFES (not for Ding, whose backbone is frozen) |
| Selection | Amendment-9 `val_oppoint` rule (`scripts/train.py` selection_value). The start checkpoint is eligible. Validation only |
| Evaluation | `evaluate_clipfair.py`: the logic of `evaluate_ces.py`, with full test sets FF++ / DFDC / Celeb-DF / DFD and leak-free variants. Then `calibrate.py` (unchanged), then `clipfair_posteval.py`, which is the A1 part of `ces_posteval.py` unchanged (`fair_a1.py`, the verbatim CFES fair_oppoint rule; reference points t = 0 and t = -b) |
| Seeds | 7, 17, 27 |

**Optimiser.** Every baseline uses the shared CLIP-adapter recipe, not the official SGD settings:
- AI-Face (DAW-FDD, DAG-FDD): SGD lr 5e-4.
- PG-FDD: SAM(SGD lr 5e-4, momentum 0.9, wd 5e-3).
- Ding GDA: SGD lr 1e-3, momentum 0.9, wd 5e-3.

Those SGD settings were tuned for full Xception fine-tuning. The amendment fixes the backbone, budget and recipe. The
*method* hyper-parameters (CVaR alphas, loss weights, LDAM margins, SAM rho, SFD ratio, Sinkhorn blur and lambda) are
the official defaults. Nothing is tuned.

**Loss pool.** Each method's loss is computed over the 176-image pool of the step. B0 instead sums three balanced BCEs,
one per stream. The DRO losses (DAW, DAG, PG-FDD's fused CVaR) need one set of per-sample losses, and the pool holds
at least 12 images of every intersectional group.

Notation: N = 176 pool images, y in {0,1} (1 = fake), g in 0..7 (Lin's intersec_label), l_i = per-sample loss,
CVaR_a(l) = min_lam lam + (1/a) mean_i relu(l_i - lam). The official code solves the dual with
`scipy.optimize.fminbound` on [min l - 1000, max l]. Here the same search runs on a detached CPU float32 copy, and
lam* then enters the differentiable loss exactly as in the official code.

## 1. DAW-FDD (Ju et al., WACV'24)

Source: `CFES/third_party/ff2id/AI-Face-FairnessBench/training` (the ff2id patches touch only `train_test.py` I/O).
- Loss: `detectors/daw_fdd.py:73-94`. Defaults inner_alpha 0.9 and outer_alpha 0.5 (`:75-76`).
- Dual search: `:55-65`.
- Per-sample BCE: `loss/daw_bce_loss.py:10,24`.
- Classifier: `:36-38` (Xception, `num_classes` 1).

```
s_i = w^T z_i + b                                   (Linear(1024, 1) on the CLS feature; official: relu-avgpool-Linear(2048, 1))
l_i = BCEWithLogits(s_i, y_i)
L_g = CVaR_0.9({l_i : g_i = g})        for every group g present in the pool
L   = CVaR_0.5({L_g}) + 0.1 L_orth
```

Port: `cfes/clipfair_losses.py` (`daw_fdd_loss`). Equal in value and gradient to the official `get_losses`, as checked.

## 2. DAG-FDD (Ju et al., WACV'24)

Source: `detectors/dag_fdd.py:74-84` in the same repo, default dag_alpha 0.5 (`:76`).

```
L = CVaR_0.5({BCEWithLogits(s_i, y_i)}_i) + 0.1 L_orth          (no group labels)
```

Port: `dag_fdd_loss`. Equal in value and gradient to the official code, as checked.

## 3. PG-FDD (Lin et al., CVPR'24)

Source: `CFES/third_party/ff2id/Fairness-Generalization/training`.
- `detectors/fair_df_detector.py`: modules `:34-105`, forward `:336-431`, loss `:193-282`.
- Losses: `loss/bal_loss.py` (LDAM), `loss/contrastive_regularization.py`, `loss/bi_level_CE.py`, `loss/l1_loss.py`.
- Optimiser: `sam.py` (SAM).
- Training loop: `train.py:223-242` (SAM two-pass step), `utils/bypass_bn.py` (BN momentum 0 in the perturbed pass).
- The ff2id patch to `cls_num_list` is for FF2 only. The official FF++ value is used here, and it equals our FF++ train
  counts: [2475, 25443, 1468, 4163, 8013, 31281, 1111, 2185].

Architecture (`cfes/clipfair_models.py` PGFDDDetector):

```
T = CLIP token map (1024 x 16 x 16)
f = E_f(T), c = E_c(T), d = E_fair(T)                   E_* = Conv1x1(1024->512)-BN-ReLU   (the Xception adjust_channel analog)
f_spe = B_spe(f), f_sha = B_sha(f), f_fair = B_fair(d)  B_* = Conv2d1x1 (512->256->256)     (:86-100, :161-166, :345)
fused = AdaIN(f_fair, f_sha)                            (:346)
heads: H_sha (2), H_spe (6), H_fair (8), H_fused (2)    Head = avgpool-Linear(256,512)-LeakyReLU-Linear-Dropout(0.2)
reconstruction on the [real; fake] halves R, F:          cd = AdaIN(d, c)
   G(f_F, cd_F) ~ x_F, G(f_R, cd_R) ~ x_R, G(f_R, cd_F) ~ x_F, G(f_F, cd_R) ~ x_R       (G = conditional UNet, :479-538)
score = H_fused[1] - H_fused[0]                         (inference = fused head, :348-381)
```

Loss (`:220-272`, weights unchanged):

```
L = CE(H_sha, y) + 0.1 CE(H_spe, spe) + 0.3 L_rec + 0.05 L_con + 0.1 LDAM(H_fair, g) + L_fuse + 0.1 L_orth
  L_rec  = sum of the 4 L1 terms against the image in [-1, 1]
  L_con  = triplet contrastive regulariser on the pooled sha / spe features, margin 3.0
  LDAM   : max_m 0.5, s 30, margins n_g^-1/4 rescaled to max 0.5
  L_fuse = smoothed CVaR_0.5 over groups (tau1 1e-3, tau2 1e-4) of the per-group CVaR_0.9 of CE(H_fused)
SAM: g = grad L(w); w' = w + 0.05 g / ||g||; second forward and backward at w' (BN momentum 0); restore w;
     clip; AdamW step with grad L(w')
```

spe = Lin's `spe_label` column: 0 real, 1 Deepfakes, 2 Face2Face, 3 FaceSwap, 4 NeuralTextures, 5 FaceShifter. These are
the 6 classes of `:63`.

Adaptations to CLIP (each one a reviewer could question):
1. **One shared adapted CLIP trunk, three necks.** The official model has three separate Xception encoders
   (`encoder_f`, `encoder_c`, `encoder_fair`, `:44-46`). Here those are three adjust_channel necks on one shared adapted
   CLIP trunk. Three ViT-L/14 passes of 176 images do not fit one 80 GB GPU (one CESel pass of 176 images peaked at 47 GB in PBS 44066). The
   content/forgery/fairness disentanglement therefore separates only above the shared trunk.
2. **Decoder output size.** The decoder's three x2 stages give 128 px from the 16x16 grid. Its last upsampling
   resamples to 224 px (official: a fixed x4 from 8x8 to 256 px). The output is compared with the 224 px input mapped
   to [-1, 1] (official Normalize 0.5/0.5).
3. **Real/fake halves.** Reconstruction and the contrastive term use the 40 real + 40 fake images of the balanced batch
   plus the pairs, which gives the official [real; fake] chunk layout. The sha, spe, fair and fused terms use the whole
   176-image pool.
4. **SAM base optimiser.** The base optimiser is AdamW (the shared recipe), not SGD. SAM is the official two-pass
   rule; the check compares it with `sam.py` with an SGD base. Gradient clipping is applied to the perturbed gradient.
5. **Float32 statistics under bf16.** AdaIN statistics and the heads run in float32; the trunk, necks, blocks and
   decoder run in bf16 autocast.

Faithfully reproduced released behaviour:
- `block_fused` is constructed but never used (`:101-105`).
- Dropout acts on the head logits.
- The smoothed outer CVaR's inf guard limits lam to at least max_g L_g - ~88 tau1. In practice this puts almost all
  of the outer weight on the worst group of the batch (noted earlier in `fairforge2/src/sota_pgfdd_full.py`).

## 4. Ding et al. CVPR'26: SFD + GDA

Source: `DINGREPRO_LIN/code/Ding_mod` (= ywh1093/Fairness-Optimization @ 059ec37, with the disclosed DINGREPRO_LIN
patches).
- SFD: `SFD.py:15-58` (SoftNearestNeighborLoss), `:61-81, 136-188` (per-channel SNNL over the training set, lowest
  `prc` %).
- GDA: `GDA.py:63-74, 151-158` (out-channel pruning), `:121-124` (only `last_linear` trains), `:191` (Sinkhorn,
  blur 5e-4), `:202-239` (step), `:86` (lambda 0.005).

```
SFD (scripts/ding_sfd.py; once per seed on the start checkpoint p1, the analog of Ding's CE-trained base):
  for every training batch (64, shuffled, whole train set) and channel c of the 1024 ln_post channels:
      SNNL_c = soft-nearest-neighbour loss of the channel's 16x16 token map (256-d), cos distance, T = 1, label = g
  decoupled = the round(2% x 1024) = 20 channels with the lowest mean SNNL
GDA (train_clipfair.py, method ding):
  z' = z * keep  (keep_c = 0 on decoupled channels);  logits = Linear(1024, 2)(z'), initialised from p1's cls_head
  L  = CE(logits, y) + 0.005 * Sinkhorn_{p=2, blur 5e-4}(logits_pool, logits_{pool, group k})   k ~ U{0..7}
  RELEASED: the Sinkhorn term (and the total) is computed inside torch.no_grad() (GDA.py:232-235)
```

**The released GDA fairness term is inert.** It contributes no gradient. `DINGREPRO_LIN/results/GDA_GRAD_CHECK.json`
shows this: with the released step the parameter change from lambda = 0 to lambda = 1000 is exactly 0. Here the term is
implemented AS RELEASED by default (`clipfair.ding.gda_grad: false`; the check confirms the term has no gradient), so
the Ding row is SFD + classifier fine-tuning with CE. The gradient-enabled variant (`gda_grad: true`) is implemented and
tested but refuses to run unless `CLIPFAIR_ALLOW_GDA_GRAD=1`, which is to be set only after the decision file is
amended.

Adaptations to CLIP:
1. **Channels.** The 512 Xception adjust_channel conv channels (8x8 maps) become the 1024 CLIP ln_post channels (16x16
   token maps). 2% gives 20 channels, not 10.
2. **Decoupling.** "Pruning" = zeroing those channels of the CLS feature fed to the classifier. The official code
   zeroes the conv out-channel weights, which leaves a constant (bias -> BN -> ReLU) channel; with a linear
   classifier that difference is only a bias shift.
3. **Base model.** The base is the CFES phase-1 checkpoint p1. It is CE-trained on the same FF++ split, but with a fused
   CLS + patch score (Ding: Xception CE for 50 epochs). GDA starts from p1's CLS-path classifier, rewritten exactly as
   an equivalent 2-way head (rows -w/2, +w/2). The check confirms the head reproduces p1's CLS logit.
4. **Trainable parameters.** As released, only the classifier trains (`train_backbone: false`), so the backbone and its
   adapters stay at p1 and no orthogonality term applies. This is faithful, but it gives Ding fewer trainable
   parameters than the other three baselines.
5. **Group logits.** The group sample for the Sinkhorn term is the pool images of the drawn group. The official code
   forwards a separate 32 + 32 batch of that group under no_grad, which in train mode also updates Xception's BN
   running statistics (a side effect with no analog: CLIP has no BN). Because the term is inert, this changes only
   the logged value.
6. **Pool and sampler.** The main CE is over the 176-image pool. Official: 32 real + 32 fake, which is the balanced
   part of the B0 stream.

Geomloss: the training env (pulid) has no geomloss, and installs are not allowed, so geomloss 0.3.1 (pure Python, MIT)
is vendored unchanged in `third_party/geomloss`, copied from the qwen38 env.

## Files

| File | Role |
|---|---|
| `cfes/clipfair_losses.py` | the four methods' losses, CVaR duals, LDAM, contrastive, SNNL, SFD selection, Sinkhorn (file:line in every docstring) |
| `cfes/clipfair_models.py` | LinearHeadDetector (DAW/DAG), DingDetector, PGFDDDetector (+ AdaIN, ConditionalUNet, Conv2d1x1, Head), `build_model` (p1 warm start) |
| `cfes/clipfair_engine.py` | pool assembly (B0's 176 images), `compute_losses` per method, active terms, SAM helpers, BN bypass |
| `configs/clipfair_ffpp.yaml` | `_base_: ces_ffpp.yaml` (the CESel FF++ config, copied) + `clipfair.*` (every method hyper-parameter, with its source line) |
| `configs/ffpp/{daw,dag,pgfdd,ding}.yaml` | method switch |
| `scripts/train_clipfair.py` | trainer (train_ces.py logic: validation, Amendment-9 selection, checkpoints; SAM for PG-FDD) |
| `scripts/ding_sfd.py` | Ding SFD pass -> `runs/ffpp_ding_sfd_s<seed>/sfd_channels.json` |
| `scripts/evaluate_clipfair.py`, `scripts/clipfair_posteval.py` | evaluation (full test sets, leak-free variants) and A1 |
| `scripts/clipfair_pipeline.py`, `run/clipfair_run.sh`, `run/pick_gpu.sh` | CESel runner pattern: shared RUNLOCKs, GPU free check (>= 60 GB free, < 10 % util, not locked by CLIPFAIR or CESel), <= 4 concurrent (gpu02: 2), STOP sentinel `work/STOP_CLIPFAIR` |
| `run/pbs_clipfair_smoke.sh`, `run/pbs_clipfair_ffpp.sh`, `run/pbs_clipfair_ffpp_gpu02.sh` | PBS jobs |
| `scripts/clipfair_report.py` | `results/clipfair_ffpp.md` (+ .json) |
| `tests/clipfair_checks.py` | CPU checks |

## Commands

```
~/.conda/envs/pulid/bin/python tests/clipfair_checks.py                          # ALL_PASS
CLIPFAIR_DRY=1 CLIPFAIR_OUT=<scratch> SEEDS=7 ~/.conda/envs/aiface/bin/python scripts/clipfair_pipeline.py   # CPU dry run
qsub run/pbs_clipfair_smoke.sh                                                   # GPU smoke (20 steps per baseline)
qsub -W depend=afterany:44067 run/pbs_clipfair_ffpp.sh                           # gpu01 runner, after CESel gpu01
qsub -W depend=afterany:44068 run/pbs_clipfair_ffpp_gpu02.sh                     # gpu02 runner (GPUs 6, 7), after CESel gpu02
touch work/STOP_CLIPFAIR                                                         # stop new launches
~/.conda/envs/aiface/bin/python scripts/clipfair_report.py                       # report at any time
```

Outputs:
- `runs/ffpp_<method>_s<seed>/`: config, provenance, train_log, best.pt / last.pt, TRAINED, and `eval/` (scores,
  summary, calibration, a1.json).
- `runs/ffpp_ding_sfd_s<seed>/`: the SFD channels.
- Logs: `logs/clipfair_pipeline_<host>.log` and `logs/clipfair_runs/<task>.log`.
