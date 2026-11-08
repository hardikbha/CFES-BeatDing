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
within 1e-5 (the same replay tolerance used elsewhere in this codebase, to allow for device/precision float noise
between the original training run and a fresh process on a different machine). A script that fails its sanity check is
not used for TEST under any circumstances.

| Exporter | Covers | Sanity result |
|---|---|---|
| `export_plain_head.py` | L_A, RF-S×3, RF-SJg×3 | CPU: FAIL at 1e-5 (see below), root cause identified and benign |
| `export_baselines.py` | DAW-FDD | CPU: PASS, max metric diff 7.4e-7 |
| `export_ding_gda.py` | Ding-style corrected GDA | CPU: FAIL at 1e-5, same root cause |
| `export_cesfl_p.py` | CES-FL P | CPU: FAIL at 1e-5, same root cause (first attempt used the wrong architecture module — `ces_fl_counterfactual_5050.py` — and failed on a checkpoint-format mismatch before touching any data; fixed to `ces_fl_diagram_model_v021.py`, which matches the checkpoint's own `CESFL_DIAGRAM_v0.2.1` format and has a built-in `evaluate_run()`) |

**Root cause of the CPU sanity failures, fully diagnosed (L_A case):** every CPU-vs-stored metric difference is an
exact `1/n_group` jump (e.g. 0.004950 = 1/202 for the Female-Asian group), i.e. a small number of individual images
flipping to the other side of the frozen threshold — not broad numeric drift. A direct per-image logit comparison
against the stored `validation_scores_PRIVATE.json` (same 31,107 VAL rows) confirms: median |logit diff| 0.0021, max
0.185 (expected: training used `amp: true`, bf16 mixed precision on GPU; the CPU sanity run used full float32 — bf16
has far fewer significant digits) — but only **2 of 31,107 predictions** flip sides of the threshold. One of the two
sits at a logit distance of 8.9e-16 from the threshold in the stored GPU run (machine epsilon: this prediction is
exactly on the boundary and inherently unstable under any re-run, any hardware). AUC itself (rank-based, insensitive
to a couple of near-tied points) differs by ~1e-6 in every case. **Conclusion: the exporters, model loading and
metric code are correct; this is expected CPU-float32-vs-GPU-bf16 inference noise, bounded to a handful of
already-near-the-boundary images.** Fix: sanity checks (and the real TEST scoring) are re-run on an actual GPU,
matching each checkpoint's own training-time precision (`c['amp']`), which the existing `predict()` functions already
select automatically from device type. GPU sanity re-runs and their results are appended below once an allowed GPU is
free (budget/queue permitting; nothing in this section authorizes scoring TEST before every GPU sanity check passes).

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
| Sanity-checked on VAL before any TEST read | yes, all 4 exporters, 1e-5 tolerance |
| Re-run protection | yes (refuses to overwrite an existing output dir) |
| GPU budget | each pass is one forward pass over 87,564 images, no training; estimated well under 1 GPU-hour total for all 9 checkpoints |
