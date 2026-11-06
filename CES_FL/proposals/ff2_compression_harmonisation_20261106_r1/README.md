# Proposal: remove the source-compression shortcut from FairForge-2 (FF2-H) and re-run the FF2 tables

**Date:** 6 Nov 2026 · **Status:** draft, awaiting user approval · nothing launched · TEST not read.
Evidence: `CES_FL/results/ff2_fairness_tables_20261105/source_diagnosis_20261106/REPORT.md` (results branch).

## 1. Problem

FF2 real photos come from two sources with different compression histories:

- **FairFace** reals start as 448 px JPEGs (quality 75, 4:2:0 chroma). The v2 crop barely resamples them, so the trace survives.
- **FFHQ** reals start as lossless 1024 px masters, and fakes are generated as lossless outputs, so neither carries that trace.

FF2-trained detectors learn "q75 4:2:0 trace → real". Measured on FF2 validation with L_A: FFHQ reals are flagged fake 67.8%
of the time vs 3.6% for FairFace. Re-making the FFHQ reals with FairFace geometry plus one q75 4:2:0 round-trip drops this to
2.1%, and the same treatment lets 94% of FFHQ-sourced fakes pass as real. Because "Others" is 100% FairFace, every FF2
race/age fairness number so far is confounded with this shortcut. The dataset should not be released, and the FF2 tables
should not be finalised, until it is removed.

## 2. Goal and success criteria (pre-declared)

Build **FF2-H** (FairForge-2 v1.1, harmonised): same images, same splits, same labels, with one common compression history.
It counts as fixed only if all three gates pass on FF2-H VALIDATION:

| Gate | Test | Pass if |
|---|---|---|
| G1 trace probe | logistic regression on the 46 compression/sharpness statistics of the diagnosis, trained on FF2-H TRAIN, real vs fake **within each source** | VAL AUC ≤ 0.60 in both sources |
| G2 counterfactual | re-run the diagnosis arm A4 (add one more q75 4:2:0 round-trip) on FF2-H VAL with the G3 model | flag rate of reals and of fakes each changes by ≤ 10 points |
| G3 retrained control | train L_A on FF2-H TRAIN (identical recipe to the 6 Nov L_A) | FFHQ-real FPR / FairFace-real FPR ≤ 2.0 (now 18.8), with AUC ≥ 99.0 |

If any gate fails, stop and report. Thresholds will not be changed after seeing results.

## 3. Recipe R1 (primary): one common compression history for every image

For **every** FF2 image, real or fake, from any source and in every split (TRAIN, SELECT, CAL, TEST):

1. Start from the lossless master (FFHQ 1024 px masters; generator outputs; FairFace original files).
2. Resize so the face is at FairFace scale (the median interocular distance of FairFace sources), using INTER_AREA.
3. JPEG round-trip at quality 75 with 4:2:0 chroma and standard IJG tables (libjpeg via OpenCV).
4. Canonicalise with **one** profile (v2) for all images: SCRFD landmarks, 5-point alignment to the ArcFace template, 256 px,
   final JPEG q95 4:4:4 (unchanged).

FairFace reals already passed through q75 4:2:0 at the same grid. A second identical pass is nearly idempotent, and applying the
rule to them too keeps it symmetric (no real/fake or source exceptions). Labels, person IDs, splits, manifests and hashes of the
source items stay the same; only pixels change.

**Fallback R2 (only if G1 fails):** randomise the step-3 quality in {70, 75, 80, 85, 90} with 4:2:0 or 4:4:4, drawn by a fixed
per-item seed, identically for reals and fakes, so no single compression signature remains.

**Not chosen:** a test-time transform (the diagnosis shows it also lets fakes pass), dropping FFHQ (loses all FFHQ identities, the FFHQ-only
generators and every TEST-only generator, which are FFHQ-sourced), or a source-aware threshold (needs the source label at deployment).

## 4. Phases

| Phase | What | Compute | Gate |
|---|---|---|---|
| 0 Inventory (CPU) | master available for every item in all splits (path existence only; no TEST pixels opened); FairFace originals on disk; measure FairFace interocular median and the step-2 scale | minutes | 100% masters, or list the missing |
| 1 Build FF2-H (CPU, gpu01 cores) | apply R1 to all ~399k public images; write to a new root `ICLR_2027/fairforge2_h/`; new manifests with new SHA-256s; old files untouched | est. 2–4 h, ~12 GB | identity of labels/splits/persons vs FF2 manifests |
| 2 Probe (CPU) | G1 on FF2-H TRAIN/VAL | < 1 h | G1 |
| 3 Retrained control (GPU) | L_A on FF2-H, same recipe, 10 epochs, seed 17 | ~2.6 GPU-h | G2, G3 |
| 4 Re-run the FF2 tables (GPU) | retrain every table row on FF2-H TRAIN, seed 17, VAL selection | see §5 | none (results reported as they are) |
| 5 Seeds and TEST (later) | seeds 7 and 27 for every row, then one frozen FF2-H TEST evaluation | later proposal | — |

Phases 0–3 are the decisive test (~2.6 GPU-hours). Phase 4 starts only after G1–G3 pass and after you approve its budget.

## 5. Compute for Phase 4 (estimates, to be confirmed by profiling)

FF2 ledger: 51.97 of 72 GPU-hours charged (~20 left). Speeds measured on gpu01 this week: BCE50 ~9.8 GPU-h; L_A with the
PS-3 loader ~2.6 GPU-h.

| Row | Recipe | Estimate |
|---|---|---:|
| L_A, L_H-PS | 10 epochs, plain head | ~2.6 + ~3.5 |
| BCE, DAG-FDD, DAW-FDD | 50 epochs, batch 128 (as before) | ~9–10 each |
| Ding-style corrected GDA | classification 50 + GDA 50 | ~10–12 |
| CES-FL P / A / B | as before | ~6–8 total |
| **Total, seed 17** | | **~55–60** |

That exceeds the remaining 20 GPU-hours, so Phase 4 needs a new allocation or a cheaper uniform protocol. **Option P4-lite:**
every row trained for 10 epochs from the same FF2-H BCE warm start, which is closer to a controlled comparison and costs
~25–30 GPU-hours. You choose at the Phase 4 gate.

## 6. What this changes

- **FF2 release:** becomes v1.1 (FF2-H) with a changelog entry: "all images share one compression history (R1); v1.0 crops
  carried a source-JPEG shortcut". Datasheet §4 and the paper's preprocessing text get updated. v1.0 is not released.
- **Current FF2 tables:** kept as v1.0 results and labelled "confounded by the source-compression shortcut"; the paper tables
  use FF2-H only.
- **L_H-PS proposal:** its probe operates on FF2-H TRAIN reals after this change. The preview must be regenerated. FFHQ-sourced
  sheets will change, so Ashish's review would need a short top-up for those cases (padding/occlusion judgements carry over;
  `qualify` checks image hashes). Recommendation: let Ashish finish the current review; re-review only changed sheets.
- **Ding-format FF++ tables (Tables 1–3) and the FairForensics-style FF++→X table:** unaffected (FF++ data).

## 7. Self-check

| Check | Result |
|---|---|
| Cause measured, not assumed | yes: counterfactual 67.8% → 2.1% on all 242 FFHQ VAL reals (one model) |
| Fix applies equally to reals and fakes | yes: R1 has no label or source exceptions |
| Masters available | TRAIN/VAL: 226,068 fake records, all with `master_path`; 200/200 spot-checked exist. TEST and FairFace originals: checked in Phase 0 |
| Gates fixed before data is built | yes (§2) |
| TEST read | no; TEST is rebuilt with the same rule but not scored until Phase 5 |
| Risks | R1 lowers FFHQ sharpness (by design); generator families whose outputs are small may be upsampled at step 2; double JPEG on FairFace reals may leave a faint signature (G1 detects it; R2 is the fallback); the shortcut may not be the only source cue (G3 bounds the residual) |
| Inferred, not yet measured | that the other six FF2 models share the shortcut (similar gaps); Ding-style GDA's smaller gap (14.9% vs 9.4%) is unexplained |

## 8. Decisions requested

1. Approve Phases 0–3: build FF2-H with R1, run the G1 probe, retrain L_A on FF2-H (~2.6 GPU-hours from the FF2 ledger).
2. Confirm that the L_H-PS run waits for FF2-H rather than training on the v1.0 crops.
3. Phase 4 budget (full ~55–60 GPU-hours, or P4-lite ~25–30) can be decided after G1–G3.
