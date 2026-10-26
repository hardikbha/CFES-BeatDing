# BEAT_DING: the "Beat-Ding" recipe (BD), built 26 Oct 2026

Plan, pre-registered before any BD run: `R/decisions/2026-10-26-beat-ding-plan.md`.
- Amendment 1 sets the base weight w\* = 1.
- Amendment 2 adds ResNet-50 (Table 3).

R = `~/ICLR_2027/research/cvpr_readiness/r20261009`.

**Status (26 Oct ~04:30).** Code, configs, run script, task list and CPU tests are done. **No BD run exists yet.** All
GPUs are busy. The runs start through the CESEL_MA priority runner (section "Task registration").

`code/` and `results/{gap_analysis,matched_op}.json` hold the 26 Oct diagnosis (section 1 of the plan). They are
unchanged.

## What this tree is
This tree is a copy of the Arm-E code path of `R/agents/CFES`. CFES, CESEL, CESEL_MA and CLIPFAIR are read only; none
of them was modified. The copy covers:
- the `cfes/` package;
- `scripts/{train,evaluate,calibrate,validate,protocol_eval,ding_tables}.py`;
- the config chain `configs/{base,ffpp_ding,clip_p2}.yaml` → `study/clip_A` → `abl/clip_{Pdet,Pboth}` →
  `rev4/clip_{A,B,D,E}`;
- the ResNet chain `configs/{ffpp_ding_resnet,resnet_p2}.yaml` → `study/resnet_{A,B}` → `rev4/resnet_A`;
- `scripts/fair_a1.py`, from CESEL with its F7 fix;
- `refs/Ding_CVPR2026.txt`, `third_party/fairrel`, and the tests `identity_rev4_EF.py` and `test_metrics.py`.

**Codex-review fixes kept in the copy** (`R/agents/CESEL/CODEX_REVIEW_20261025.md`). The code matches CESEL's fixed
versions.

| Fix | Where | What it does |
|---|---|---|
| F3 | `scripts/train.py validate` | Validation AUC, group AUCs and ding@0.5 are computed from `rank_prob(logit)`. The val-TPR threshold and the operating point use logits. |
| F4 | `scripts/evaluate.py`, `cfes/metrics.evaluate_scores(thr_scores=)` | The validation-TPR threshold is chosen and applied in logit units. |
| F8 | `cfes/data.cluster_key` | Leak-free cluster keys are dataset-aware. |

**With every BD flag off the copy trains exactly rev4 Arm E.** `configs/bd/base_E.yaml` equals rev4 Arm E w1 apart
from two things: the selection metric and the BD blocks, which are off. Checks:
- `tests/bd_checks.py e1` checks the config equality.
- `tests/identity_rev4_EF.py` gives a bit-identical CPU training step against the CFES tree; see "Tests".

## Components (`configs/bd/*.yaml` flags; plan section 2)
| Arm | Components | Config |
|---|---|---|
| base | rev4 Arm E (CFES-D + group-FPR-gap), `loss.fprgap.weight = 1` | `bd/base_E.yaml` |
| BD0 | base + B (no augmentation, isolates A) | `bd/BD0.yaml` |
| BD1 | base + A | `bd/BD1.yaml` |
| BD2 | base + A + B | `bd/BD2.yaml` |
| BD3 | base + A + B + C | `bd/BD3.yaml` |
| BD4 | base + A + B + C + D | `bd/BD4.yaml` |

### A. Processing augmentation
`data.aug`, implemented in `cfes/bd_aug.py`.
- **Sampling.** With p = 0.6, one op per image, drawn uniformly from the 6 plan ops (JPEG, bilateral, Gaussian blur,
  down-up, photometric, unsharp) with the plan's ranges. Then a horizontal flip with p = 0.5.
- **Coverage.** Both classes, on all three training loaders: the ordinary batch, the stratified batch and both images
  of every pair.
- **Image format.** The image is uint8 at the model input size (224, after the dataset's resize). The output stays
  float32 in [0, 1]. FF++ crops are 380 px natively.
- **Per-item RNG.** Seeded from `(seed, epoch, stream, index)`. `EpochTagged` samplers pass the epoch to persistent
  DataLoader workers. The wrapped sampling order is unchanged.
- **Pairs.** One draw per (fake, own source) pair gives the same op, the same parameters and the same flip. The
  evidence target `frac` comes from the unprocessed pair and is flipped with it.
- **Stability pairs.** The CFES stability view is made from the already-augmented image, so (image, view) pairs share
  the augmentation.
- **Choices not fixed by the plan.**
  - Unsharp mask: σ ~ U[0.5, 1.5] px, amount ~ U[0.5, 1.5], no threshold.
  - Brightness and contrast are multiplicative factors U[0.75, 1.25]. Contrast is applied around the grey mean.
  - Saturation is a factor U[0.7, 1.3], blended with grey.
  - Gamma, brightness, contrast and saturation are applied as one "photometric" op, in that order.
  - JPEG quality is an integer U{30..90} (cv2, 4:2:0).

### B. ESA
`loss.esa`, implemented as `cfes/losses.esa_loss`.
- **Quantiles.** On u = asinh(score), with inverted-CDF empirical quantiles Q(q) = x\_(⌈qn⌉). Identical empirical
  distributions therefore give zero, whatever the sizes.
- **Cell.** A cell is one (class, level, group), for levels gender, race and intersection. A cell needs ≥ 4 items of
  that class in the step pool.
- **Loss.** Loss = mean over cells of Σ_q |Q_g(q) − sg Q_pool,c(q)|.
  - Reals use q ∈ {.5, .75, .9, .95}; fakes use q ∈ {.05, .1, .25, .5}.
  - The weight is 1.0.
- **Pool ("all images in the step").** The pool has 208 images:
  - the ordinary batch (64);
  - its nuisance views (64);
  - both images of the 8 pairs, each with its own group label from the training manifest (16);
  - the stratified batch, at **4 per (label × intersection) cell = 64**.
- **Groups with n < 4.** Such a group has no cell and gets no gradient, but its images still enter the pooled
  quantiles.

### C. Multi-layer evidence
`model.multilayer`, implemented in `cfes/model.py`.
- **Blocks.** Forward hooks on CLIP blocks **8, 12, 16 and 20, numbered from 1**, i.e. `resblocks[7, 11, 15, 19]`. The
  FairReL wrapper exposes only the last block, hence the hooks.
- **Head.** Per block: LN → Linear(1024→256) → GELU → Linear(256→1) gives a patch-logit map. The 4 maps are averaged
  into P2.
- **Fusion.** score = s_cls + α·t̃ + β·t̃2, with t2 = top-8 mean of P2 and β = tanh(raw), raw = 0. Adapters are shared
  and unchanged. The heads use their own seed, so the global RNG stream is unchanged.
- **Local norm.** t̃2 = t2·min(1, c0/r2), where r2 is t2's own running RMS.
  - **c0 is the existing map's phase-1 reference `loc_c0`** (`multilayer.c0: base`; ≈ 44.8 logits for s7).
  - Reason: a c0 measured on the randomly initialised heads would be ≈ 0.25 logits. That would cap C's contribution at
    noise level against scores in the tens to hundreds, so C would be inert by construction.
  - `c0: own` gives the literal alternative. This is an implementation choice; please flag it if unwanted.
- **Start.** β = 0 at the start, so every starting score equals the base's. The measured max |Δ| is **0.0** on the
  stub, and **< 1e-4** on real CLIP with the phase-1 checkpoint; see "Tests". At β = 0 the gate gets gradient and the
  heads do not; once β ≠ 0, the heads and adapters do.

### D. Group-AUC floor
`loss.gauc`, implemented as `cfes/losses.group_auc_floor_loss`.
- Soft pairwise AUC sigmoid((u_f − u_r)/1) on asinh scores.
- Loss = Σ_g relu(sg AUC_pool − AUC_g), weight 0.5.
- The plan names no levels. This build uses the gender, race and intersection groups (the levels of Ding's three
  es-AUC columns), each with ≥ 4 fakes and ≥ 4 reals. The pool is the same as B's.

### Selection, operating points, evaluation (plan section 4)
- **Within-run selection.** `select.metric = val_auc_plus_inter_es_auc`, i.e. val AUC + val intersectional es-AUC.
  Both are threshold-free and computed with F3. The start checkpoint is eligible, as in Arm E.
  - Each epoch also logs the val Ding-score as a diagnostic only.
- **Per-run evaluation.** `run/bd_run.sh` runs three scripts:
  1. `evaluate.py --leak_free` scores FF++ test, DFDC, Celeb-DF and DFD.
  2. `calibrate.py` fits the native class-balanced intercept.
  3. `bd_eval.py` writes `eval/bd_eval.{json,md}`.
- **What `bd_eval.py` reports.** Three thresholds, all chosen on FF++ val:
  - **matched** (val TPR = **94.1768 %**, the Ding reproduction's val TPR at 0.5, recomputed from
    `DINGREPRO_LIN/results/audit_scores/full_s7_ep49_ffval.npz`; the `code/matched_op.py` quantile rule);
  - **native**;
  - **A1**.

  For each threshold and test set it reports Ding's 10 columns, TPR and FPR, and win marks against Ding published.
  Tables 1–2 apply to CLIP and Table 3 to ResNet; the values are checked against `refs/Ding_CVPR2026.txt`. It also
  reports:
  - the **val Ding-score**: FF++ val columns at the matched threshold against the reproduction's val columns at its 0.5;
  - the **F_DP oracle** (5.1 / 43.2 / 59.6 / 33.1, intersection);
  - the **FF++ i-F_FPR noise floor**: the video-permutation null at the matched threshold and at real FPR 12.5 %.
- **Validation of `bd_eval.py`.** On Arm E w3 s7 it reproduces the diagnosis: 19/40 at the matched point, with
  identical win marks. The noise floor gives 34.5 observed against a null of 40.2 [24.7, 61.5].
- **Reports (`scripts/bd_report.py`).**
  - `refs` gives the reference rows in `results/refs/`. Done: CFES-D s7 4/10; E w1 6/10; E w3 5/10; E w10 6/10; Ding
    reproduction 0 (it is the reference).
  - `screen` writes `results/bd_screen.{json,md}`:
    - winner = the highest val Ding-score, ties → val AUC;
    - promoted iff its val Ding-score is ≥ the best reference's (6/10);
    - otherwise the recipe is the best reference (E w1).
  - `final` writes `results/bd_final.{json,md}`:
    - 3-seed mean ± SD against Ding published, with win marks;
    - the success rule: ≥ 25/40, AUC and all 3 es-AUC on all 4 sets, i-F_FPR on ≥ 3 sets;
    - the F_DP oracle and the noise floor.
  - `final --family resnet` writes the same report against Table 3, in `results/bd_final_resnet.*`.

### w\* (plan step 3, Amendment 1)
`scripts/bd_wstar.py` writes `results/wstar.json`. Validation only; val Ding-scores:

| Run | val Ding-score | val AUC |
|---|---|---|
| E w1 | 6/10 | 96.883 |
| E w3 | 5/10 | 97.014 |
| E w10 | 6/10 | 96.561 |
| CFES-D (reference) | 4/10 | — |

**w\* = 1.** It ties w10 and wins on val AUC. The plan discloses that this rule postdates seeing the test results.

## ResNet-50 (Amendment 2; `configs/bd_resnet/`, `cfes/model_resnet.py`)
- **Base.** CFES-ResNet (rev4 arm A: Ding's ResNet50 `adjust_channel`, head-only phase-2 scope with a frozen trunk and
  BN, SGD, fp32) + Arm E group-FPR-gap at w\* = 1. Phase-1 checkpoints: CFES `runs/resnet_s{7,17,27}_p1/last.pt`.
  - **No risk term.** This follows the coordinator's wording "CFES-ResNet + group-FPR-gap", whereas CLIP Arm E carries
    risk 0.3 from CFES-D.
  - **No local norm.** The ResNet model has none.
- **C on ResNet.** Hooks on layer2, layer3 and layer4, i.e. `net.resnet[5, 6, 7]`: (N,512,28,28), (N,1024,14,14),
  (N,2048,7,7).
  - Each stage gets a 1×1 conv evidence map, adaptive-avg-pooled to the 7×7 grid.
  - The maps are averaged, then reduced by top-4 (as in the existing ResNet map), with a tanh gate initialised at 0.
  - Start-equivalence is tested on the real network with the s7 phase-1 checkpoint.
- **Component set.** The CLIP screen's: the promoted BD arm, else `base`.
- **Operating point and comparison.** The same selection rule and the same matched point (94.18 %, a disclosed proxy),
  compared with Table 3.

## Run script and tasks
`run/bd_run.sh <kind>` is resumable and quotes every path. Its RUNLOCK functions are copied **verbatim** from the fixed
`CESEL/run/cesel_run.sh` (token + atomic-rename compare-and-swap, heartbeat, fencing; exit 4 = held). Kinds:

| Kind | What it does | Output |
|---|---|---|
| `smoke` | 20 bf16 steps of CLIP BD4 (batch 208) + 20 fp32 steps of ResNet BD4, then eval / calibrate / bd_eval on 256 items per set | done marker `runs/SMOKE_OK.json` (timing, memory, terms) |
| `arm BDk <s>` | CLIP arm | `runs/BDk_s<s>` |
| `refs`, `screen` | reference rows and the screen | |
| `winner <s>` | the screen's recipe for seed s: a promoted BD arm, or the best reference — reusing CFES `runs_rev4/clip_E_w<w>_s<s>` if trained, else training rev4 Arm E w here | `runs/WINNER_s<s>.TRAINED` |
| `winner_eval <s>` | evaluation of that run | `runs/WINNER_s<s>.EVALED` |
| `final` | 3-seed CLIP report | `results/bd_final.*` |
| `resnet <s>` | ResNet run with the CLIP screen's component set | `runs/R50_<set>_s<s>`, marker `runs/RESNET_s<s>.DONE` |
| `resnet_final` | 3-seed ResNet report | `results/bd_final_resnet.*` |

Stop file: `work/STOP_BD`. Environment:
- `BD_WORKERS`: workers per loader, default 3, for 3 loaders.
- `BD_DRY=1`: CPU stub dry run, for tests only.

**`run/bd_tasks.json`**, in run order. Every task depends on `bd_smoke`, transitively; if it fails, nothing else runs.

| Task(s) | Kind | Waits for |
|---|---|---|
| `bd_smoke` | GPU | — |
| `bd_BD0_s7` … `bd_BD4_s7` | GPU, train + eval | `bd_smoke` |
| `bd_screen` | CPU | all 5 arms |
| `bd_winner_s17`, `bd_winner_s27` | GPU | `bd_screen` |
| `bd_winner_eval_s17`, `bd_winner_eval_s27` | GPU | their winner run |
| `bd_final` | CPU | both winner evals |
| `bd_resnet_s7`, `_s17`, `_s27` | GPU | `bd_screen` (queued after the CLIP winner seeds) |
| `bd_resnet_final` | CPU | the 3 ResNet runs |

Expected cost:
- **Memory.** Arm E peaked at 47 GB for 176 images per step. BD4 uses 208 per step, so ≈ 56 GB is expected; the smoke
  measures it.
- **Time.** Arm E took 3.3–8 h for 10 × 1976 steps.

## Task registration (step 6)
`R/agents/CESEL_MA/run/tasks.yaml` already includes this file:
`include: {file: '{R}/agents/BEAT_DING/run/bd_tasks.json', priority: 5, optional: true, reserve: true}`.
- Priority 5 is the highest; every other task is ≥ 10.
- The runner re-reads the file on change.
- `run/bd_tasks.json` is validated with the runner's own `normalise_external` / `validate_spec`: 16 tasks, 0 errors.

**Coordinator.** "Only after the CESel screen ends" is NOT encoded in `bd_tasks.json`. It relies on the runner's
`HOLD_UNTIL_FILES` (the PBS job's env), which the coordinator sets.

## Tests (CPU)
- `tests/bd_checks.py` → **ALL_PASS** (log `logs/bd_checks_*.log`). It covers A, B, C (stub + real CLIP + real
  ResNet-50), D, the configs, selection, F3/F4/F8, Ding's Tables 1–3 against the PDF text, the winner rule, and a stub
  dry run of the whole task chain through `run/bd_run.sh`:
  smoke → BD0/BD4 → screen → winner s17/s27 + evals → final → ResNet s7/s17/s27 → final.
- `tests/test_runlock_bd.sh` → ALL_PASS (8 cases, CESel lock test on `bd_run.sh`).
- Copied existing tests:
  - `tests/test_metrics.py` → 10/10. It needs a shim for `efficientnet_pytorch`, which is missing in pulid/aiface and
    used only to register a backbone.
  - `tests/identity_rev4_EF.py` gives a bit-identical step: CFES tree, against this tree (rev4 Arm E, then
    `bd/base_E`); see `logs/identity_*.json`.
