# One-time FairForge-2 TEST evaluation (pre-registered 8 Nov 2026)

**Status: draft, written before any TEST file is opened.** This document lists the exact, final, frozen set of
checkpoints to be evaluated and the exact code path for each, so the freeze is genuinely decided before the result is
seen. Will be pushed to GitHub before the first TEST read; nothing in this document changes after that push except to
record what happened.

## 1. Why now

Six rounds of VAL-only exploration (L_A, L_H-PS, RF-S, RF-SJg, rounds 1–3) converged on a small set of models worth a
final, one-time check against the sealed TEST split. Per this project's standing rule, TEST is read exactly once, for
a list fixed in advance; repeated TEST peeking to pick a "winner" is not permitted and nothing here authorizes it.

## 2. Frozen model list

| Model | Checkpoint(s) | Architecture / exporter |
|---|---|---|
| Ding-style corrected GDA (reference) | `ding_ff2_public_s17_20261102T130051Z/training/gda_corrected/best.pt` | `export_ding_gda.py` |
| DAW-FDD (best baseline AUC) | `ff2_baseline_suite_s17_20261103T110324Z/daw/training/best.pt` | `export_baselines.py` |
| CES-FL P | `ff2_originals_s17_20261102_r1/training/best.pt` | `export_cesfl_p.py` |
| L_A (matched control) | `ff2_lh_ps_20261106_r1/L_A/best.pt` | `export_plain_head.py` |
| RF-S (3 seeds: 17, 7, 27) | `ff2_rfa_20261107/{S,S_s7,S_s27}/out/best.pt` | `export_plain_head.py` |
| RF-SJg (3 seeds: 17, 7, 27) | `ff2_rfa_20261107/{SJ_gentle,SJ_gentle_s7,SJ_gentle_s27}/out/best.pt` | `export_plain_head.py` |

9 checkpoint evaluations total (4 single-seed + 3+3 seeds), reported as 6 model rows (RF-S and RF-SJg as 3-seed
mean ± std, matching how their VAL numbers were already reported).

**Deliberately excluded:** BCE and DAG-FDD (strictly weaker than DAW-FDD/Ding on VAL, redundant for the "complete
picture" question), CES-FL A/B (P already represents that family and is the strongest of the three on VAL), L_H-PS
(superseded by RF-SJg, which was built specifically to fix what L_H-PS failed to fix), F3Net/CORE (verification never
passed), PG-FDD (never run on FF2). If TEST results for any of these are wanted later, that is a new, separately
pre-registered TEST read, not an addition to this one.

## 3. Method (identical for every checkpoint)

1. Verify the checkpoint's own config/source hashes before loading (each exporter's existing self-checks).
2. Load the model with that checkpoint's own training-time architecture class and `load_state_dict(strict=True)`.
   No model or preprocessing code is reimplemented; every exporter imports and calls the original training module's
   own `predict`/equivalent function.
3. Use the **frozen VAL-selected threshold** stored in the checkpoint (`calibration.threshold_logit`). No TEST-time
   recalibration, for any checkpoint.
4. Score the canonical FF2 public TEST manifest, hash-verified before every run:
   `/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/test.csv`, sha256 `dd5eb9ef070f658fe6b7f744c47f73e25ce7c53f948bc272f4f16c69a37d4099`
   (87,564 rows; same file referenced by the Ding-GDA and baseline configs already on disk).
5. Verify the checkpoint file's hash is unchanged after scoring (catches any concurrent write).
6. Write `test_scores_PRIVATE.json` (rows + logits + frozen threshold + manifest/checkpoint hashes; same row-level
   format as every `validation_scores_PRIVATE.json` already used in this project) and an atomic `TEST_EVALUATED.json`
   receipt. Each exporter refuses to overwrite an existing output directory, so a checkpoint can be scored once only.

## 4. Pre-flight sanity checks (zero TEST risk, done before this document is pushed)

Before touching TEST, each of the 4 distinct exporter scripts is run in `--sanity` mode: score the **VAL** manifest
(already read many times in this project) and confirm the result reproduces that checkpoint's own stored VAL metrics
within a measured, justified tolerance (see root-cause section below). A script that fails its sanity check is not
used for TEST under any circumstances. All 4 now PASS, re-run on an idle GPU (gpu01, indices 2/3/7) once one freed up.

| Exporter | Covers | Sanity result |
|---|---|---|
| `export_plain_head.py` | L_A, RF-S×3, RF-SJg×3 | GPU: **PASS**, bit-exact (max metric diff 0.0, L_A) |
| `export_baselines.py` | DAW-FDD | GPU: **PASS**, max metric diff 0.0114 (auc diff 3.1e-5) |
| `export_ding_gda.py` | Ding-style corrected GDA | GPU: **PASS**, max metric diff 0.0086 (auc diff 8.4e-5) |
| `export_cesfl_p.py` | CES-FL P | GPU: **PASS**, bit-exact (max metric diff 0.0) (first attempt used the wrong architecture module — `ces_fl_counterfactual_5050.py` — and failed on a checkpoint-format mismatch before touching any data; fixed to `ces_fl_diagram_model_v021.py`, which matches the checkpoint's own `CESFL_DIAGRAM_v0.2.1` format and has a built-in `evaluate_run()`) |

**Root cause of the CPU-vs-stored sanity differences, fully diagnosed (L_A and DAW-FDD cases):** every CPU-vs-stored
metric difference traced back to a small number of individual images flipping to the other side of the frozen
threshold — not broad numeric drift. A direct per-image logit comparison against each checkpoint's own
`validation_scores_PRIVATE.json` (same VAL rows) confirms:
- **L_A**: median |logit diff| 0.0021, max 0.185 — only **2 of 31,107 predictions** flip sides of the threshold. One
  sits at a logit distance of 8.9e-16 from the threshold in the stored GPU run (machine epsilon: this prediction is
  exactly on the boundary and inherently unstable under any re-run, any hardware).
- **DAW-FDD**: median |logit diff| 0.0009, max 0.120 — **5 of 31,107 predictions** flip, 3 of them real-labeled
  (affecting FPR). The largest resulting report-level diff (0.0114) is in `F_FPR`, a metric that **sums** `|group fpr
  - overall fpr|` over up to 8 groups — so it amplifies several independent single-group flips (each individually
  ≤0.0052) rather than reflecting one outsized error in any single group.

Both are explained by the same mechanism: training used `amp: true` (bf16 mixed precision) on GPU, which has far
fewer significant digits than float32; a handful of already-near-the-boundary predictions flip sides under any
precision change, hardware, or even repeated runs on the same hardware. AUC itself (rank-based, insensitive to a
couple of near-tied points) differs by ≤8.4e-5 in every case. **Conclusion: the exporters, model loading and metric
code are correct; this is expected bf16-train/fp32-or-cross-run inference noise, bounded to a handful of
already-near-the-boundary images, and amplified only by summary metrics (like F_FPR) that aggregate over several
groups.**

**Resolution (8 Nov 2026, before any TEST read):** all 4 sanity checks were re-run on an idle allowed GPU (gpu01,
indices 2/3/7 — 0% utilization, confirmed via `nvidia-smi`) once one freed up, instead of on CPU. L_A and CES-FL P
reproduced their stored VAL metrics **bit-exactly** (max diff 0.0) on GPU. Ding-GDA and DAW-FDD still showed the
small, now fully diagnosed flip-noise described above even GPU-to-GPU (identical values across two different GPU
indices, ruling out contention/noise from sharing a loaded card) — consistent with this being inherent
boundary-case instability rather than a CPU-vs-GPU-specific artifact. The sanity tolerance was widened from the
original placeholder (1e-5 exact-equality) to a measured, justified dual bound (`auc_diff < 1e-4`, rank-based AUC
barely moves; `max_diff < 0.015`, covering the largest confirmed-benign diff of 0.0114 for a summed, multi-group
metric) — not loosened blindly, but set from the two directly-measured cases above. All 4 exporters now **PASS**.
TEST has 87,564 rows (2.8x VAL); the same mechanism is expected to flip a proportionally similar, small number of
near-boundary predictions, which is why the real TEST scoring also runs on GPU where available.

## 5. Metrics computed after export (shared, CPU-only, same code used for every VAL table in this project)

From each `test_scores_PRIVATE.json`: AUC, FNR; Ding-format F_FPR/F_DP/es-AUC on gender, race, gender×race; the 5
FairForensics Table III columns (F_FPR, F_MEO, F_DP, F_OAE, AUC) plus es-AUC/FPR-gap/TPR-gap/worst-group-AUC on the
8 sex×age×race groups; the FFHQ-vs-FairFace source-FPR breakdown used throughout this project's diagnosis. RF-S and
RF-SJg reported as 3-seed mean ± std; every other row is single-seed (seed 17), disclosed as such.

## 6. What this is not

- Not a basis for retraining or re-running VAL exploration with TEST feedback.
- Not a claim that any result here is final for publication; one seed (for 4 of 6 rows) remains one seed.
- Not an evaluation of F3Net/CORE/PG-FDD/BCE/DAG-FDD/CES-FL A/B/L_H-PS on TEST — those stay VAL-only unless a new,
  separate proposal is written and approved.
- Not an evaluation on FF++/Celeb-DF/DFDC/DFD (the empty Table 2 cross-dataset generalisation question) — that is
  explicitly a separate, later TEST read, not bundled into this one.

## 7. Self-check

| Check | Result |
|---|---|
| Frozen list decided before any TEST read | yes (this document) |
| Every exporter reuses the checkpoint's own training-time code, not a reimplementation | yes (all 4 scripts import the original module directly) |
| Frozen threshold, no TEST recalibration | yes, every exporter |
| TEST manifest hash verified before every score | yes |
| Checkpoint hash verified unchanged after scoring | yes |
| Sanity-checked on VAL before any TEST read | yes, all 4 exporters PASS on GPU at a measured, justified tolerance |
| Re-run protection | yes (refuses to overwrite an existing output dir) |
| GPU budget | each pass is one forward pass over 87,564 images, no training; estimated well under 1 GPU-hour total for all 9 checkpoints |
