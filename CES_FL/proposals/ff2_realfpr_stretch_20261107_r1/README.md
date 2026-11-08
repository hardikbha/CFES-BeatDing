# Stretch experiment: real-only false-alarm equalisation + JPEG augmentation on FF2 (pre-registered 7 Nov 2026)

**Status:** approved by the user (option 2 after a Codex gpt-6-astra review that answered NO to the original plan; this is
the revised version with Codex's required changes, run explicitly as a stretch experiment). TEST is never read.

## Design (same Xception, 20,809,001 parameters, same 10-epoch recipe from the FF2 BCE50 encoder, seed 17)
Trainer `CES_FL/src/ces_fl_realfpr.py` (reuses the unchanged head-ablation model/data/metric code). Probe terms removed.
- **Arm R:** BCE (same inverse-cell weights) + λ·mean_g |r_g − stopgrad(Σ_h π_h r_h)| on a group-balanced microbatch of
  16 unchanged TRAIN reals per gender×race group (128) every 4 steps; r_g = mean sigmoid((z − t)/τ); π = natural TRAIN-real
  priors. λ = 0.1 with a 2-epoch linear ramp; τ 1.0 → 0.25 linearly; t refreshed 4× per epoch on a fixed natural 2,048-real
  TRAIN pool. Per-group EMA (0.9) is monitored only.
- **JPEG augmentation** (classification stream only, same for every label/source/group): 50% unchanged, 25% q75 4:2:0,
  25% q∈[70, 95] with 4:2:0 or 4:4:4. Emulated at source scale on the 256 px crop (resample by 1/s, JPEG, resample back), s
  drawn from the 13,601 FairFace TRAIN-real source-to-crop scales (median 1.158). Fairness microbatches and VAL are unchanged images.
- **Selection:** minimum hard gender×race F_FPR (VAL-real 10% threshold) among epochs with AUC ≥ 99.5 and FNR ≤ 0.5651%;
  ties by AUC then earlier epoch. No feasible epoch = failure. The same rule is applied to L_A's per-epoch history.

## Gate Q-J (passed before training)
Crop-level q75 4:2:0 emulation must reproduce the source trace: L_A's FFHQ-real VAL flag rate 67.8% → **0.0%** (pass ≤ 20%);
FFHQ fakes 99.3% → 0.7%; q75 4:4:4 and q95 4:2:0 instead push FairFace reals to 97% / 80% flagged (`gate_qj.json`).

## Runs (vs matched L_A, arm A without JPEG)
| Run | Arm | JPEG | Encoder |
|---|---|---|---|
| J | A | yes | exit (last blocks), LR 1e-5 |
| S | R | no | exit |
| SJ | R | yes | exit |
| SJF | R | yes | full, LR 1e-6 (BN frozen) |

Cap 4 GPU-hours per run (epoch-boundary stop). FF2 ledger before launch: ~54 of 72 GPU-hours.

## Pre-declared reading
- **Screening success (stretch):** SJ or SJF reaches gender×race F_FPR < 12.44 (Ding) with AUC ≥ 99.5 and FNR ≤ 0.5651% on VAL.
  This is development evidence only (selection and reporting both use VAL). A claim needs 3 seeds that all meet it.
- **Ablation reading:** effect of the loss = S vs L_A; of JPEG = J vs L_A; combined = SJ; encoder = SJF vs SJ. All other FF2
  table metrics, source FPRs and per-family FNRs are reported either way.
- **Expectation (Codex):** beating 12.44 needs ~67% less disparity than L_A; credible stretch, not an expected win.

## Results (8 Nov 2026; VAL, seed 17)

All four runs completed 10/10 epochs (4.88 GPU-hours in total; FF2 ledger now 58.58 / 72). Per-epoch VAL metrics:
`EPOCH_HISTORY_VAL.json`.

| Run | Feasible epochs | Selected (rule) | AUC | FNR | g×r F_FPR | Lowest F_FPR in any epoch (AUC / FNR) |
|---|---:|---|---:|---:|---:|---|
| L_A (rule re-applied to its history) | 10/10 | epoch 2 | 99.81 | 0.28% | 37.37 | 37.37 |
| **S** (loss only) | 10/10 | epoch 2 | 99.80 | 0.29% | **34.12** | 34.12 |
| J (JPEG only) | 0/10 | none | — | — | — | 21.47 (99.28 / 1.07%) |
| SJ (loss + JPEG) | 0/10 | none | — | — | — | 20.92 (99.27 / 1.11%) |
| SJF (SJ, full encoder) | 0/10 | none | — | — | — | 27.82 (99.30 / 1.25%) |
| Ding-style GDA (reference) | — | — | 99.06 | 1.07% | 12.44 | — |

**Pre-declared reading: screening target not met.** No run reached g×r F_FPR < 12.44 under the AUC ≥ 99.5 / FNR ≤ 0.5651%
constraints. The real-only loss alone (S) lowers F_FPR by 3.25 points against L_A with unchanged detection, and leaves the
source gap (FFHQ real FPR 64.5% vs FairFace 4.0%). JPEG augmentation attacks the shortcut (F_FPR 21–24) but costs detection
(AUC 99.1–99.3, FNR 1.1–1.5%), the same kind of trade-off Ding-style GDA makes, while still staying above Ding's 12.44.
Training the full encoder (SJF) recovered some AUC (up to 99.59) but never brought FNR under 0.5651% and lost fairness.

## Round 2 (8 Nov 2026): more seeds + shortcoming-fix attempts

Pre-registered before launch. Budget: 13.42 of 72 GPU-hours left; cap 4 h/run as before.

| Run | Purpose | Design |
|---|---|---|
| S_s7, S_s27 | confirm RF-S (the only round-1 feasible design) across seeds 7 and 27, alongside the existing seed 17 | identical to S, seed changed only |
| SJ_gentle | shortcoming fix attempt 1: round 1's JPEG augmentation (50% of images, mixed quality) broke detection immediately. Try far milder: 15% of images, exact q75/4:2:0 only (no random-quality branch) | arm R, `jpeg_p_any=0.15, jpeg_p_q75=1.0, jpeg_ramp_epochs=0`, seed 17 |
| SJ_ramp | shortcoming fix attempt 2: same 50/25/25 mix as round 1's SJ, but phased in over 2 epochs (matching the fairness-loss ramp) instead of full strength from step 1 | arm R, `jpeg_p_any=0.5, jpeg_p_q75=0.5, jpeg_ramp_epochs=2`, seed 17 |

Code change: `ces_fl_realfpr.py` — `jpeg_plan` now takes `p_any`/`p_q75` (defaults reproduce round 1's fixed 50/25/25 mix,
statistically identical, RNG draw count differs so not bit-replayable); new `jpeg_p_any_effective` ramps `jpeg_p_any`
linearly over `jpeg_ramp_epochs` completed epochs (0 = no ramp). 21/21 tests pass, including 2 new ones for the ramp and
the custom-probability mix. Same selection rule as round 1 (min hard gender×race F_FPR s.t. AUC ≥ 99.5, FNR ≤ 0.5651%).

**Reading plan:** report S's 3-seed mean ± std regardless of outcome (this is our one design with a feasible checkpoint).
SJ_gentle/SJ_ramp count as meeting the original stretch target only if they beat Ding (12.44) under the same AUC/FNR gates;
otherwise they are reported as a further (still negative) data point on whether the shortcut can be removed without an
unacceptable detection cost. Single seed each; no further seeds without a positive gate result.

## Round 2 results: S across 3 seeds (8 Nov 2026)

S (real-only false-alarm equalisation, no JPEG) now has 3 seeds, all selecting epoch 2:

| Seed | AUC | FNR | g×r F_FPR |
|---|---:|---:|---:|
| 17 (round 1) | 99.80 | 0.29% | 34.12 |
| 7 | 99.80 | 0.28% | 34.48 |
| 27 | 99.81 | 0.30% | 34.12 |
| **mean ± std** | **99.80 ± 0.01** | **0.29 ± 0.01%** | **34.24 ± 0.21** |

**Confirmed, 3-seed claim:** S lowers gender×race F_FPR by 3.1 points vs L_A (37.37, single seed) with no AUC/FNR cost,
very low seed variance. **It does not meet the stretch target** (Ding-style GDA 12.44). Per-epoch histories:
`EPOCH_HISTORY_VAL_round2.json`.

SJ_gentle (shortcoming-fix attempt 1) and SJ_ramp (attempt 2) are still training; results follow separately.

## Round 2 final: SJ_ramp (8 Nov 2026)

**NO_FEASIBLE_CHECKPOINT.** All 10 epochs exceeded the FNR limit (range 0.68%–2.08%, never ≤ 0.5651%), despite the
2-epoch ramp. Lowest F_FPR among any epoch: 22.37 (epoch 8, AUC 99.34, FNR 1.05%). Full history:
`EPOCH_HISTORY_VAL_SJ_ramp.json`.

## Round 2: final summary (both shortcoming-fix attempts)

| Attempt | Design | Outcome |
|---|---|---|
| SJ_gentle | 15% of images, exact q75/4:2:0 only, no ramp | **Feasible (epoch 2): F_FPR 27.53**, AUC 99.76, FNR 0.45% |
| SJ_ramp | 50/25/25 mix (round-1 strength), phased in over 2 epochs | Infeasible every epoch; best F_FPR 22.37 at FNR 1.05% |

Milder, cleaner augmentation (fewer images, the one causally-verified edit only) solved the detection-loss problem that
sank round 1's JPEG runs; simply ramping in the stronger round-1 mix did not. RF-SJg (27.53) is now the best FF2
fairness result of this whole project, confirmed only on seed 17.

## Experiment status: complete

6 runs total across 2 rounds (10.49 GPU-hours; FF2 ledger 61.89/72, 10.11 left). Final standing vs the pre-declared
target (g×r F_FPR < 12.44, AUC ≥ 99.5, FNR ≤ 0.5651%): **not met by any run.** Best result: RF-SJg at 27.53, roughly
2.2× Ding-style GDA's 12.44. RF-S (loss only) is confirmed across 3 seeds at 34.24 ± 0.21. Next step, if pursued: 2
more seeds of SJ_gentle's exact config to confirm 27.53 is not a lucky seed, within the ~10 GPU-hours remaining.

## Round 3 (8 Nov 2026): RF-SJg 3-seed confirmation, ahead of a TEST freeze

Pre-registered before launch. Purpose: RF-SJg (seed 17) is currently our best VAL fairness result (g×r F_FPR 27.53) but
is single-seed and was noisy across epochs within that one seed. Before including it in a frozen TEST-evaluation list,
confirm it across seeds 7 and 27 (identical config to SJ_gentle, seed changed only), the same practice used for RF-S.

Budget: 10.11 of 72 GPU-hours left at launch. Same selection rule and cap as rounds 1–2.

## Round 3 results: RF-SJg across 3 seeds (8 Nov 2026)

All three seeds selected a feasible epoch:

| Seed | Selected epoch | AUC | FNR | g×r F_FPR |
|---|---:|---:|---:|---:|
| 17 (round 2) | 2 | 99.76 | 0.45% | 27.53 |
| 7 | 2 | 99.76 | 0.45% | 30.89 |
| 27 | 6 | 99.75 | 0.51% | 28.67 |
| **mean ± std** | | **99.755 ± 0.005** | **0.467 ± 0.034%** | **29.03 ± 1.71** |

**Confirmed, 3-seed claim:** RF-SJg lowers gender×race F_FPR by 8.3 points vs L_A (37.37) and by 5.2 points vs RF-S's
3-seed mean (34.24), with essentially unchanged AUC and only a small FNR increase. Seed variance is larger than RF-S's
(1.71 vs 0.21) but the direction is consistent across all three seeds — none approached RF-S's or L_A's level. RF-SJg
seed 17 is the checkpoint carried into the TEST evaluation freeze (`ff2_test_eval_20261108_r1`); seeds 7/27 are
confirmatory and also included in that freeze's 3-seed TEST mean.
