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
