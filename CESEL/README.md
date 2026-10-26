# CESEL: Counterfactual Evidence Selectivity (CES) detector

Status (25 Oct 2026): code and CPU smoke tests only. **No training run exists, on GPU or otherwise** (a standing
user stop order covers CFES-family GPU runs). `tests/ces_checks.py` prints `ALL_PASS` (15/15 checks, including one real
CLIP ViT-L/14 forward and backward pass on 2 FF++ images). Every number in this file is either a property of the code
or a CPU measurement. Decisions and open questions are in `DESIGN_NOTES.md`.

R = `~/ICLR_2027/research/cvpr_readiness/r20261009`. The CFES-D code in `R/agents/CFES` was copied here and **not
modified there**. The only change to the copied files is that `cfes/config.py` now accepts a list `_base_`.

**Codex review fixes (25 Oct, decisions/2026-10-25-cesel-plan.md Amendment 4; CODEX_REVIEW_20261025.md F1-F11):**
RUNLOCK = token + atomic-rename compare-and-swap protocol (run/cesel_run.sh header); validation AUC from logits under
eval.rank_metrics (scripts/train.py validate); validation-TPR threshold in logit units (evaluate_ces.py); A1 over every
configured test set (fair_a1.py); dataset-aware leak-free cluster keys (cfes/data.py); exact deletion curves
(ces_posteval.intervene); fail-closed promotion and atomic monotone report publication (cesel_report.py); FF2-public
provenance guard (cfes/public_guard.py) + real CFES-D / phase-1 FF2-public configs. Checks: tests/ces_checks.py and
tests/test_*.py, tests/test_runlock_f1.sh, tests/rerun_codex_repros.py.

## Algorithm as implemented

Notation:
- K = 4 cues in a fixed order (tex, freq, bound, sem). d = 128.
- y ∈ {0,1}, with 1 = fake. y' = 2y − 1. g is the group (8 Ding/Lin intersectional groups).
- m is the manipulation condition: 0 = real, then 1.. for the fake's family.
- sg = stop-gradient.

```
feat (N,G,G,D), z (N,D) = backbone(x)                          CLIP: 16x16x1024 + CLS; ResNet/Xception: map + avg pool
c_tex   = LN( MLP( AttnPool( LN(feat[core cells]) ) ) )         46 cells (FF++), 52 (FF2) at 224 px
c_bound = LN( MLP( AttnPool( LN(feat[ring cells]) ) ) )         44 cells (FF++), 50 (FF2)
c_sem   = LN( MLP( LN(z) ) )
c_freq  = LN( FreqNet( log(1 + |fftshift(fft2_ortho(crop_w(x)))|) ) )   pixel space, separate CNN, 131 k parameters
s       = sum_k w_k^T c_k + b         (head: additive, default)    |  s = MLP([c_1;..;c_4])   (head: mlp)
```

- **Attention pooling:** a = softmax over cells of (qᵀh_c), and the pooled vector is Σ_c a_c·h_c.
- **MLP:** Linear(D, 256), then GELU, then Linear(256, 128).
- **FreqNet:**
  - 4 blocks of Conv3×3 with stride 2, each followed by GroupNorm and GELU (channels 3→16→32→64→128);
  - then global average pooling;
  - then an MLP (128 → 128 → 128).
- **crop_w:** the bounding box of the face ellipse r ≤ 1.
  - Each channel is mean-removed and multiplied by the window clamp((1 − r)/0.05, 0, 1).
  - The crop is zero-padded, centred and without resampling, to P×P. P = 176 for FF++ and 192 for FF2 at 224 px.

**Cell masks.** The face template is `cf.face_mask`: FF++ uses cx .5, cy .48, ax .30, ay .38; FF2 uses cx .5, cy .578,
ax .3125, ay .4297. The elliptical radius is r(p) = sqrt(((x − cx·S)/(ax·S))² + ((y − cy·S)/(ay·S))²). For each cell,
take the share of its pixels inside a region; this share is computed by adaptive average pooling, so any S and G work.
- core (texture) = share of pixels with r ≤ 0.78 is at least 0.90;
- ring (boundary) = share of pixels with 0.80 ≤ r ≤ 1.00 is at least 0.30.

The two masks are disjoint, and this is asserted. `CueGeometry.describe()` prints both masks; they are also logged at
training start.

FF++, 224 px, 16×16 (core / ring):
```
core                ring
................    ................
................    .......##.......
................    .....######.....
.......##.......    ....##....##....
......####......    ....#......#....
.....######.....    ...##......##...
.....######.....    ...#........#...
.....######.....    ...#........#...
.....######.....    ...#........#...
.....######.....    ...##......##...
.....######.....    ....#......#....
......####......    ....##....##....
................    .....######.....
................    ......####......
................    ................
................    ................
```

**Cue removal.** It acts at head level only; there is no second backbone pass. The replacement R is always detached.
- A fake whose paired source is in the same forward (the 8 pairs per step) gets R_k = c_k(source).
- Any other fake gets R_k = μ_real,k.
- A real gets R_k = μ_fake,k.
- The μ are EMA class means of the detached cues: momentum 0.02, initialised from the first batch, and updated after
  each step's losses.
- Control `ces.replace: random`: R_k = c_k(j), where j is a random other image of the step.

```
s^(-k)  = H(C with c_k := R_k)          additive head: s - w_k^T c_k + w_k^T R_k (closed form, exact)
m(s,y)  = y' * kappa * asinh(s / kappa)  (kappa = 1; ces.margin: linear gives y' s)
u_{i,k} = m(s_i, y_i) - m(s_i^(-k), y_i)
L_sens  = sum_{k validated} mean_i relu(tau_k - u_{i,k})
L_floor = sum_k sum_{(g,c,m): n >= 4} relu( sg(ubar_{k,c,m}) - ubar_{g,k,c,m} - 0.1 )^2      k over validated cues
L_inv   = mean_{i valid} Huber_1( asinh s_i, asinh s'_i ),  s'_i = H(C'_i)
          C'_{i,k} = c_{i,k} - P_k c_{i,k} + P_k sg(c_{j,k}),  j = random image of the step with same label, other group
          P_k = U_k U_k^T, U_k = top-2 right singular vectors of {mu_{y,g,k} - mean_g mu_{y,g,k}} (EMA group means)
          valid_i = probe(g_j | C'_i) > probe(g_i | C'_i)    (online linear probe on detached [c_1..c_4], own Adam)
L_det   = BCE(batch) + BCE(pair images) + BCE(stratified batch)      (balanced BCE, the CFES-D exposure)
L_total = L_det + 0.5 L_sens + 0.5 L_inv + 1.0 L_floor (+ orth 0.1, the FairReL adapter regulariser, CLIP only)
```

**Pools.**
- The CES pool is every clean labelled image of a step:
  - 64 ordinary images (32 real + 32 fake);
  - 16 pair images;
  - 96 stratified images: 4 per (label, group, condition) cell, which is 8×4 reals + 8×2×4 fakes on FF++.
- In total that is 176 images, the same number CFES-D forwards with gradient.
- ubar_{k,c,m} pools every group of that class and condition. Each of its cells needs n ≥ n_min images.
- L_inv starts after 200 state steps (`loss.inv.warmup_steps`), once a subspace is usable.

**Validated cues and targets.** `scripts/ces_validate_cues.py` runs the starting model (the B0 checkpoint) on the
validation split. `cfes/ces_validation.decide_validated` then applies this rule:
- valid_k holds when the bootstrap 2.5% bound of the class-balanced mean utility is above 0;
- τ_k = clamp(median_i u_ik, 0.1, 2.0).

If no JSON is given, all four cues are valid and τ = 0.5 (asinh units).

## Files

| File | What it does |
|---|---|
| `cfes/` (copied) | the CFES-D package, unchanged except `config.py` (list `_base_`) |
| `cfes/ces_geometry.py` | `CueGeometry`: elliptical radius, core and ring cell masks, frequency crop/window, `describe()` |
| `cfes/ces_backbones.py` | swappable backbones that return (feat N×G×G×D, z N×D): `clip` (FairReL rank-1 adapters), `resnet`, `xception` (Ding nets), `stub` (tests) |
| `cfes/ces_model.py` | `CESDetector`: 4 cue branches (TokenCue, FreqNet), cue LayerNorms, `CueHead` (additive or mlp), param groups, `load_cfes_backbone` (CFES checkpoint → backbone) |
| `cfes/ces_state.py` | `CESState`: EMA class and group means, demographic subspaces U_k, online group probe (state in the checkpoint) |
| `cfes/ces_losses.py` | margin, replacement, utilities (closed form or general), L_sens, L_floor, demographic partner/swap/validity, L_inv, matched-set L_inv |
| `cfes/ces_validation.py` | validated-cue rule and JSON loader |
| `cfes/ces_data.py` | manipulation condition codes, `CondImageTable`, `CESPairTable` (pairs restricted to TRAIN, optional family exclusion), `StratifiedCondBatchSampler` |
| `cfes/ces_engine.py` | `active_terms`, `build_model`/`build_state`, loaders, `compute_ces_losses` (one forward; the CES pool; all terms; state update) |
| `scripts/train_ces.py` | trainer: CFES validation, Amendment 9 selection and checkpoint format, plus the CES state |
| `scripts/ces_validate_cues.py` | validation pass → validated-cue JSON |
| `scripts/build_ff2_manifest_pairs.py` | FF2 pairs from `FAKE_MANIFEST.csv real_path` (TRAIN-only) → `data/pairs_ff2full_manifest_train.csv` (464,517 pairs; report next to it) |
| `scripts/train.py`, `scripts/calibrate.py` | copied CFES trainer and intercept fit (imported by `train_ces.py`) |
| `configs/ces_ffpp.yaml`, `configs/ces_ff2.yaml` | datasets plus every CES flag with its default |
| `configs/arms/*.yaml` | B0, sens, inv, floor, full, C_randcue, C_shufgroup, C_novalid, B0_cfesD (CFES-D stab/sens/risk on the CES architecture) |
| `configs/{ffpp,ff2}/<arm>.yaml` | `_base_: [../ces_<ds>.yaml, ../arms/<arm>.yaml]` |
| `tests/ces_checks.py` | CPU smoke tests (below) |

## Flags (defaults in `configs/ces_ffpp.yaml`)

- **Backbone:** `model.backend` = clip, resnet, xception or stub. `model.init_backbone_ckpt` is a CFES checkpoint, of
  which only the backbone is loaded; this was checked once with `CFES/runs/clip_s7_p1/last.pt` (872 keys).
  `model.init_ckpt` is a CES checkpoint (model plus state).
- **Cue branches:**
  - `ces.dim`, `ces.hidden`;
  - `ces.pool` = attn or mean;
  - `ces.cue_norm` = layernorm or none;
  - `ces.freq_widths`, `ces.freq_hidden`;
  - `ces.geometry.*` (core_r, core_frac, ring, ring_frac, freq_r, freq_feather).
- **Head, margin and removal:**
  - `ces.head` = additive or mlp;
  - `ces.margin` = asinh or linear, plus `ces.kappa`;
  - `ces.replace` = counterfactual or random (control);
  - `ces.ema`.
- **Validated cues:** `ces.validated_json`, `ces.default_target`, `ces.validation.*`.
- **L_inv:**
  - `loss.inv.*`: weight, rank, subspace_every, warmup_steps, probe_lr, huber_delta;
  - stopgrad = none, clean or cf;
  - detach_donor;
  - validity_mask: false is the control;
  - source = feature or matched (a MatchedSetTable from a validated `mfer.csv`).
- **L_floor:** `loss.floor.*`: weight, n_min, delta, cues (validated or all), reduce (sum or mean), shuffle_groups
  (control).
- **CFES-D terms**, all with weight 0 by default:
  - `loss.stab` (needs `cf.enabled`);
  - `loss.sens` (pairs; `pairs.shared_intervention`);
  - `loss.risk`;
  - `loss.fprgap` (arm E).
- **Data:** `data.families` and `data.family_map` (the manipulation condition), `pairs.exclude_families`,
  `strat.per_cell`, `strat.by_cond`.
- **Optimiser:** `optim.lr` is the backbone adapters' rate (1e-4); `optim.lr_new` is for the new modules (1e-3).

## Run order (not executed)

```
python scripts/train_ces.py --config configs/ffpp/B0.yaml --out runs/ffpp_B0_s7 --set seed=7 \
       model.init_backbone_ckpt=../CFES/runs/clip_s7_p1/last.pt
python scripts/ces_validate_cues.py --config configs/ffpp/B0.yaml --ckpt runs/ffpp_B0_s7/best.pt \
       --out runs/ffpp_B0_s7/validated_cues.json
python scripts/train_ces.py --config configs/ffpp/<arm>.yaml --out runs/ffpp_<arm>_s7 --set seed=7 \
       model.init_ckpt=runs/ffpp_B0_s7/best.pt ces.validated_json=runs/ffpp_B0_s7/validated_cues.json
```

The matched reference is B0 continued with the same budget: `configs/ffpp/B0.yaml` with the same `model.init_ckpt`.

## Smoke tests (`tests/ces_checks.py`, CPU, result 15/15, ALL_PASS)

1. All 18 arm configs load. Their active terms, control flags, data and pair paths, and face template are checked.
2. Shapes, and s = Σ_k s_k + b.
3. Additive head:
   - Σ_k u_k equals the utility of full replacement, with the linear margin (max error 5e-7);
   - the closed form equals head re-evaluation;
   - with the asinh margin, the logit drops still add up.
4. A cue replaced by itself has utility 0, for both the additive and the mlp head.
5. Gradients reach every branch, the head and the backbone adapters. Each CES term alone is non-zero and
   back-propagates.
6. L_floor:
   - it is 0 for equal groups;
   - it is exactly K(0.75 − δ)² for one deficient group of 4;
   - its gradient only raises the deficient group;
   - the shuffled-group control changes it.
7. Demographic swap on planted group directions:
   - probe accuracy is 1.00 on clean cues;
   - after the swap, the probe predicts the partner's group 100% of the time;
   - the projection and its complement are exact (2e-6);
   - the gap between the estimated and the planted subspace is 0.026;
   - with no planted signal, validity is 0.48 (chance).
8. Controls (random-cue, shuffled-group floor, no-validity inv) run, and each changes only its own terms, because every
   CES random draw has its own stream. B0 and B0+CFES-D also run.
9. The validated-cue rule and its JSON round trip.
10. Parameter counts and timing: the stub runs at 0.06 s per step.
11. End to end on CPU with the stub backbone and 64 synthetic images: `train_ces.py` B0, then `ces_validate_cues.py`,
    then `train_ces.py` full from the B0 checkpoint.
12. One real CLIP ViT-L/14 forward and backward pass on 2 FF++ images. CPU timing: load 31 s, forward 0.9 s, backward
    1.2 s. All rank-1 adapters receive gradient, and the geometry has 46 core and 44 ring cells.

**Parameter counts (CLIP, d = 128).**
- Total: 204.43 M.
- Trainable: 1.223 M, made up of:
  - adapters: 196,704;
  - texture: 298,625;
  - frequency: 131,200;
  - boundary: 298,625;
  - semantic: 297,600;
  - head: 513.

**GPU step time is not measured.** A step forwards 176 images with gradient, the same as CFES-D's 0.48 s/step step
(measured in `CFES/results/cfea/2026-10-24-cfea-design.md`). The branches cost a few MFLOP against the ViT-L/14
forward, and the K head re-evaluations are negligible. The estimate is therefore about 0.5 s/step on FF++. On FF2 the
step has 304 images (estimate about 0.85 s/step).

## Not implemented

- A DCT variant of the frequency cue (only the FFT log-amplitude exists).
- Group-conditional removal baselines.
- Symmetric paired replacement for the paired real source (a source real always uses μ_fake).
- An automatic B0 → validation → arm pipeline script.
- Test-set evaluation for CES runs. `CFES/scripts/evaluate.py` should work with `model(x)["score"]`, but it was not
  tested with a CES model.
- The matched-set L_inv (`loss.inv.source: matched`) is coded but has no validated matched-set file. `mfer.validated`
  is false for every existing file, so it cannot be enabled.
- No intervention parity audit by group, beyond the probe's per-group validity rate that is logged every step
  (`inv_valid_g*`).
