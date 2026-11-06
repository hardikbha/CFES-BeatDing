# Proposal: FairForge-2 L_H with a padding-safe probe (L_H-PS) and its matched L_A control

**Date:** 6 Nov 2026 · **Status:** APPROVED by the user on 6 Nov 2026, including amendment PS-2. Human reviewer of the 88 cases: Ashish (PhD
researcher, named by the user). TEST not read.
Parent: `ff2_lh_priority_20261104_r1` (blocked at qualification on 4 Nov).

## 1. Why

L_H on FF2 never trained. Its qualification needs every one of 88 preview cases (22 TRAIN real photos × 2 operators ×
2 strengths) to pass five review flags. On 4 Nov the AI pre-screen left 16 cases unapproved, all failing the same flag,
`face_crop_suitable`, on 4 source photos. The other four flags passed on all 88.

| Source | Problem | Cases |
|---|---|---:|
| case_216279 | rotated crop; the probe warps the black canonicaliser padding (bright border line in the difference map) | 4 |
| case_206861 | hood / face covering | 4 |
| case_209182 | hand over nose and mouth | 4 |
| case_217989 | hat over the eyes | 4 |

## 2. Changes

### PS-1 (code): the probe never touches padding

`intervention()` now multiplies the probe mask by `1 − dilate(padding, 6 px)`. Padding = near-black pixels
(max channel ≤ 6/255) connected to the image border; 6 px covers the largest warp shift. Group-blind, computed from the
input pixels only. Source `CES_FL/src/ces_fl_head_ablation.py`: sha256 `08a9c5e5…` → `3705ab85…`; nothing else in the file
changed.

Checks (all done, CPU only):

- **Tests:** 12/12 pass. New `test_padding_safe_intervention`: on unpadded images the output is bit-identical to the frozen
  reference implementation for all 4 operator/strength cases; on padded images the mask is exactly 0 on padding + margin
  and sham = treated there. Two guard tests were updated on purpose: `intervention` left the AST-identity list (covered
  numerically instead) and historical proposal configs stay bound to the legacy source hash.
- **New preview (same 22 sources, same seeds):** 78/88 sheets byte-identical to 4 Nov; the 10 changed sheets all
  belong to the 3 sources with padding in the probe region. case_216279 no longer shows the border artefact.
- **TRAIN census (16,777 reals):** padding reaches the probe region in 7.0% of FairFace and 5.0% of FFHQ reals; clipping
  removes on average 0.34–0.68% of probe area per gender×race group (lowest for Female-Black 0.34% and Male-Black 0.42%),
  so it does not disproportionately remove dark-skinned faces. It also clips pure-black natural background or hair that
  touches the border. Aggregates: `evidence/PADDING_AND_PARSING_AGGREGATES.json`.

### PS-2 (review rule): needs your decision

The three occlusion sources are unchanged by PS-1. Proposed clarification of `face_crop_suitable`: *partial occlusion
(hat, hand, hood, mask) does not by itself make a crop unsuitable, provided the probe lies on the head region and inside
valid image content.*

- **Why:** the probe is applied to every TRAIN real in arm D (main batch and auxiliary draws), so occluded photos are
  part of the real distribution the loss sees. Dropping them would need an automatic occlusion detector. We tested one
  (facexlib ParseNet face parsing): it does not recognise the hat or the hand (face fraction 0.92 for the hand and 0.91 for the
  hat, inside the 0.78–0.99 range of approved photos), so it cannot separate them. An exclusion could also correlate with group (headwear).
- **Cost, disclosed:** on occluded photos the probe also perturbs the occluder's texture, so the response terms are
  partly an occluder-sensitivity signal there.
- **Without PS-2:** a human reviewer judges those 12 cases under the original wording. If any is rejected, L_H-PS
  cannot qualify and stays blocked.

This changes how a qualification flag is read, so it is reported as an amendment, not as a fix. It was decided before
any L_H model was trained, and no model result has been seen.

### Config

`L_PS_config.json` = the 4 Nov `L_H.json` with two edits: `source_sha256` → `3705ab85…` and `max_gpu_hours` 2.0 → 4.0
(per-arm cap, set before profiling). Digest `20c869c9…`. Everything else is unchanged: Xception plain linear head;
encoder transferred from the verified FF2 BCE50 checkpoint (epoch 40); `exit` encoder policy, frozen BN; 10 epochs,
batch 32, seed 17; head LR 1e-3, encoder LR 1e-5; λ_fair = λ_sens = λ_align = 0.1; margin 0.5; `hierarchical_real`
error control on the 5 axes; VAL-AUC checkpoint selection; VAL-real 10% FPR threshold. Preflight reuses the identical
TRAIN/VAL inventory (no re-decode), rebound to the new source and config.

## 3. Arms

| Arm | CLI | Needs qualification | What it is |
|---|---|---|---|
| L_A | `--arm A` | **no** | BCE only, same encoder, head, seed and config: the matched control |
| L_H-PS | `--arm D` | yes (88-case human review) | hierarchical-real error control + probe sensitivity/control + response alignment |

L_A does not use the probe, so it can start as soon as this proposal is approved and the CUDA checks pass.

## 4. Run plan and gates

1. **User approval** of this proposal and a decision on PS-2.
2. **Human review** of the 88 cases. Private files: `REVIEW_SHEETS_PRIVATE.pdf` (22 pages, one per source) and
   `REVIEW_TEMPLATE_WITH_AI_PRESCREEN_PRIVATE.json`. The reviewer sets all five flags, their name and a note for every
   case. The AI pre-screen is advisory and is not a review. Then `qualify` must exit 0.
3. **CUDA checks:** smoke test and a dry run per arm.
4. **Profile:** 20 steps per arm. Stop and report if the projected 10-epoch cost exceeds 4 GPU-hours.
5. **Train L_A**, then **L_H-PS** once qualified (in parallel if two GPUs are free).
6. **Report** VAL results into the FF2 tables PDF and push aggregates to GitHub.

GPUs: gpu01 {0,1,2,3,4,6,7} or gpu02 {6,7} only; a GPU is free if it has ≥60 GB free and <10% utilisation; at most 2
used at once. Budget: the FF2 ledger shows 48.95 of 72 GPU-hours charged (23.05 left); this proposal may charge at most
8. Each run is charged to that ledger.

## 5. Pre-declared reading (VAL only, one seed, screening)

L_H-PS counts as improving fairness over L_A only if all three hold on FF2 VAL at the VAL-real 10% threshold:

- gender×race F_FPR at least **2.0 points** lower;
- AUC at most **0.5 points** lower;
- miss rate (FNR) at most **2×** that of L_A.

Every other metric in the FF2 tables (gender, race, age, skin, within-source race, per-group FPR) is reported either way.
One seed supports no state-of-the-art claim; multi-seed confirmation and a single TEST run come later, after freezing.

**Prediction, stated before training:** the FF2 race and age gaps are mostly an FFHQ-vs-FairFace source effect, and
L_H's groups are demographic, not source. We therefore do not expect L_H-PS to come close to Ding-style GDA's
gender×race F_FPR (12.44); a reduction against L_A of a few points is the plausible best case.

## 6. Self-check

| Check | Result |
|---|---|
| All 16 failures explained | yes: 1 padding source (fixed by PS-1), 3 occlusion sources (PS-2 decision) |
| PS-1 changes only padded images | yes: bit-identical on unpadded inputs (test); 78/88 sheets byte-identical |
| PS-1 group-neutral | yes: probe-area loss 0.34–0.68% per group, lowest for Black groups |
| Automatic occlusion filter possible | no: face parsing cannot detect the hat or hand; not used |
| Any other code changed | no: only `intervention()` and two guard tests |
| Thresholds or rules relaxed after seeing model results | no: no L_H model exists; PS-2 is a disclosed amendment |
| L_A blocked by review | no: arm A does not call `check_qualification` |
| TEST read | no |
| Budget | 23.05 GPU-hours left; cap 8 for this proposal |
| Residual risks | weak per-batch fairness term; source confound not targeted; 4 GPU-hour cap may be too low for arm D (the profile decides) |

## 7. Commands (after approval)

```bash
SRC=~/ces_fl_sources/ff2-lh-ps-v2-20261106/CES_FL/src/ces_fl_head_ablation.py
R=~/ces_fl_runs/ff2_lh_ps_20261106_r1; PY=~/.conda/envs/aiface/bin/python
$PY $SRC qualify --preview $R/preview --review $R/review_SIGNED_PRIVATE.json --out $R/qualification.json
$PY $SRC smoke --device cuda --head-type linear --out $R/smoke_cuda
for A in A D; do $PY $SRC dry-run --config $R/L_PS.json --preflight $R/preflight --arm $A --device cuda --out $R/dry_$A; done
$PY $SRC train --config $R/L_PS.json --preflight $R/preflight --arm A --device cuda --out $R/L_A \
  --software-receipt $R/smoke_cuda/SMOKE_RESULT.json --dry-run-receipt $R/dry_A/DRY_RUN.json --budget-hours 4
$PY $SRC train --config $R/L_PS.json --preflight $R/preflight --arm D --device cuda --out $R/L_H_PS \
  --software-receipt $R/smoke_cuda/SMOKE_RESULT.json --dry-run-receipt $R/dry_D/DRY_RUN.json --budget-hours 4 \
  --qualification $R/qualification.json
```

## 8. Decision log

- 2026-11-06: user approved this proposal and accepted PS-2 (partial occlusion does not by itself make a crop
  unsuitable). Human review assigned to Ashish. L_A may start on the first free allowed GPU; L_H-PS waits for
  `qualify` to pass on Ashish's signed review.
- 2026-11-06 ~12:00: first L_A attempt on gpu01 ran at 2.86 steps/s (projected ~6.6 h for 10 epochs, over the 4 h cap)
  and was stopped by the operator after 0.27 GPU-hours, per section 4. Cause: image loading from network storage on a
  CPU-saturated node (gpu01 load ~100/104); a read probe gave 271 img/s with 4 readers vs 835 with 16.
- 2026-11-06: user approved `workers` 4 → 16 (data-loading parallelism only; batches, seed, model, losses and epochs
  unchanged). Config re-bound (digest `266057b1…`); preflight rebound to the same inventory; preview regenerated with
  all 88 sheets byte-identical, so the human review kit is unchanged. L_A restarted from scratch (no checkpoint existed).
- 2026-11-06 ~15:00: workers=16 attempt ran at 3.55 steps/s (projected 5.3 h > 4 h cap); stopped after ~0.2 GPU-hours.
  Losses at steps 400–600 matched the first attempt exactly. Cause found in the training loop: every 4 steps the main
  process builds the auxiliary batch (64 image reads from network storage + 32 probe renders), even for arm A.
- 2026-11-06: user approved amendment **PS-3 (speed only)**: `AuxiliaryPlan` / `auxiliary_loader` build the same
  `auxiliary(rows, active, c, event, epoch)` batches in DataLoader workers (in order). New test
  `test_auxiliary_loader_is_bitwise_identical_to_direct_calls` (13/13 pass) and the CPU smoke lifecycle (incl. bitwise
  resume) pass. Source `8c25ba8c…`, config digest `143cb13b…`; preview regenerated with all 88 sheets identical.
