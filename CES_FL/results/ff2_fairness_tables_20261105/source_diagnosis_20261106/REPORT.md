# Why FFHQ real photos are flagged as fake on FairForge-2 VALIDATION

6 Nov 2026. FF2 VALIDATION and TRAIN only; no TEST, CAL or external data opened. CPU only (gpu01). One checkpoint scored:
L_A (plain-head BCE, matched control of proposal `ff2_lh_padding_safe_20261106_r1`). Aggregates only. Diagnosis run by a
delegated agent; key counterfactual numbers re-checked against `recrop_summary.json`.

## Verdict

The cause is the **compression history of the source photos before cropping**, not faces or demographics.

- FairFace source files are 448 px JPEGs at quality 75 with 4:2:0 chroma subsampling. Profile v2 aligns them with a slight
  upsampling warp (median align_scale 1.16), so every FairFace-derived crop keeps that q75 4:2:0 trace.
- FFHQ masters are lossless 1024 px images. Profile v1 downsamples them (median align_scale 0.65), so FFHQ-derived crops
  never carry such a trace.
- FairFace is 81% of TRAIN reals. L_A learned "FairFace-type compression trace present → real".

**Decisive test (measured, L_A, all 242 FFHQ VAL reals):** re-made from their masters with FairFace-like geometry plus a q75
4:2:0 JPEG round-trip, the flag-as-fake rate falls from **67.8% to 2.1%** (95% CI 0.4–3.7; 162 flagged→passed, 3 the other
way). The same treatment lets **94% of FFHQ-sourced fakes pass as real** (detection 99.7% → 5.8%; 0–20% per method), so the
trace overrides forgery evidence.

**Measured vs inferred:** only L_A was scored. That BCE, DAG-FDD, DAW-FDD and CES-FL P/A/B share the cause is inferred from
their similar gaps. Ding-style GDA (FFHQ 14.9% vs FairFace 9.4%) behaves differently and was not examined.

## 1. Observation (each model at its frozen 10% VAL-real FPR threshold; VAL reals are 90.1% FairFace)

| model | FFHQ real FPR (n=242) | FairFace real FPR (n=2,196) |
|---|---:|---:|
| BCE | 72.7% | 3.1% |
| L_A | 67.8% | 3.6% |
| DAG-FDD | 59.9% | 4.5% |
| DAW-FDD | 47.1% | 5.9% |
| CES-FL A / B | 45.0% | 6.1% |
| CES-FL P | 44.2% | 6.2% |
| Ding-style GDA | 14.9% | 9.4% |

For L_A, detection is 99.4% (FFHQ-sourced fakes) and 99.8% (FairFace-sourced); within-source real-vs-fake AUC is 0.949 (FFHQ)
and 0.999 (FairFace). The gap is where the reals score, not detection.

## 2. Preprocessing and image statistics

- v1 (FFHQ) and v2 (FairFace) end in the same encoder (q95, 4:4:4): all 37,914 sampled TRAIN/VAL crops share one
  quantisation table, so the final file shows no re-save signature. The trace is in the pixels.
- The profile follows the source dataset for fakes too: FFHQ-sourced fakes are 100% v1, FairFace-sourced 100% v2.
- VAL medians, FFHQ vs FairFace reals: file size 31.5 kB vs 19.5 kB; log Laplacian variance 6.07 vs 3.18; noise sigma 2.19 vs
  0.39. A logistic regression on 46 statistics separates the two sources' reals with VAL AUC 1.000; 78% of FFHQ-sourced vs 10%
  of FairFace-sourced fakes "look like FFHQ reals".
- Global sharpness alone does not explain the gap: a statistics-only real/fake model gives FFHQ-real FPR 19.8% vs 8.9% (2.2×)
  against L_A's 19×, and L_A's FFHQ flag rate is flat across sharpness terciles (67.9 / 67.5 / 67.9%).

## 3. Counterfactual re-crops scored with L_A (threshold logit −4.838)

Sanity: scores of the stored crops match the stored VAL logits within 0.047; the v1 re-crop agrees with stored flags 94%
(reals) / 100% (fakes).

| arm | FFHQ reals flagged (95% CI) | FFHQ-sourced fakes flagged (n=360, 24 methods) |
|---|---:|---:|
| A0 stored crop | 67.8% [62.0, 73.6] | 99.7% |
| A1 v1 re-crop | 68.6% [62.4, 74.4] | 99.7% |
| A2 v2 profile from 1024 master | 58.7% [52.5, 64.9] | 98.3% |
| A3 FairFace geometry, lossless | 98.3% [96.7, 99.6] | 100% |
| **A4 = A3 + q75 4:2:0 source JPEG** | **2.1% [0.4, 3.7]** | **5.8%** |
| A5 q75 4:2:0 on 1024 master, then v1 | 40.1% [34.3, 46.7] | 90.8% |
| A6 = A4 with 2× upsampled source | 2.5% [0.8, 4.5] | 5.0% |

JPEG follow-up (A3 geometry): q75 4:4:4 → 99.6% flagged, q95 4:2:0 → 100%, q50 4:4:4 → 93.0%. Only **q75 together with 4:2:0**,
the exact FairFace source format, flips the decision. The profile itself is not the cause; geometry alone makes FFHQ reals look
more fake.

## 4. Composition (manifests)

TRAIN fakes per real: 13.9 (FFHQ) vs 11.6 (FairFace), only 1.2×. FFHQ-only generators: arc2face, InstantID, PhotoMaker (4.3% of
FFHQ-sourced fakes); FairFace-only: ProjGAN. Composition is a contributing factor (inferred): only 19% of TRAIN reals lack the
trace, so the shortcut is cheap to learn, and the VAL threshold is set 90% by FairFace reals.

## 5. Implications (inferred, not tested)

- FF2 reals come from two acquisition domains with different compression histories. Per-race and per-source real FPRs are
  confounded with that history ("Others" is 100% FairFace).
- **Dataset fix (needed before release):** give both sources the same compression history before canonicalisation, applied to
  reals and to fakes alike (e.g. FairFace-like q75 4:2:0 round-trip on FFHQ masters and on FFHQ-sourced generator outputs, or a
  common re-encoding of all sources), then retrain. A test-time transform will not work, because A4 also makes FFHQ fakes pass.
- At minimum, report source-stratified FPRs.
- Any FF2 fairness result trained on the current crops, including all rows of the FF2 tables, partly measures this shortcut.

## Files (this directory)

Image statistics `feat_stats.py`, `analyse_stats.py` → `stats_summary.json`; fake profiles `fin_profile.py`; counterfactual
re-crops `recrop_score.py`, `aggregate_recrop.py` → `recrop_summary.json`; JPEG follow-up `recrop_jpeg_split.py` →
`recrop_jpegsplit_summary.json`. `_*_PRIVATE.pkl` hold per-image data and stay private.
