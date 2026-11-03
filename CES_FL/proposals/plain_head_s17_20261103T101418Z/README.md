# Plain Xception head ablation — seed17, FF++

User-authorized question: does the global/local gated head help enough to retain it? This proposal changes the prediction head and preserves the existing loss/training recipes. It does not replace active H, launch anything on publication, or claim a fairness gain.

Base reviewed: `58ed0cb64341ec1413000eb2266a5846215582b1`. New branch: `experiment/cesfl-plain-head-s17-20261103T101418Z`. Exact execution commit must be recorded from the immutable checkout, along with source/config hashes and launch receipts. See [operator prompt](GPU_OPERATOR_PROMPT.md), [commands](commands.sh), [project memory](../../PROJECT_MEMORY.md).

## Model and exact comparisons

New standalone source: [ces_fl_head_ablation.py](../../src/ces_fl_head_ablation.py). Xception produces 2048-channel spatial features; global average pooling and `Linear(2048,1)` produce one BCE logit. No nonlinear global/patch classifiers, gates or fusion remain. A diagnostic linear CAM is computed from the SAME classifier weights; it introduces no parameters or separate loss. Its spatial mean plus bias reconstructs the full logit. It is not a validated forgery mask.

| Head | Total | Head only | Trainable with fixed exit policy |
|---|---:|---:|---:|
| Existing gated | 21,594,159 | 787,207 | 7,575,591 |
| Plain linear | 20,809,001 | 2,049 | 6,790,433 |

Encoder: 20,806,952 total; 6,788,384 trainable. Reduction: 785,158 parameters, 3.64% of the full gated model. Frozen-encoder linear training would update 2,049 parameters, but is NOT this proposal. Neither speed nor accuracy improvement has been measured.

| New run | Native CLI arm | Matched existing reference | Active objectives |
|---|---|---|---|
| L_A | A | Completed gated A | Weighted original BCE |
| L_H | D | Completed gated H, once available | BCE + H hierarchical soft error control + procedural sensitivity + sham stability + response alignment |

Only TWO new runs. `head_type=gated` exists for software/reference replay parity; no gated retraining is authorized. No new Ding-model computation. H's native CLI name remains D; do not mistake L_H for the older D fairness objective. `error_control=hierarchical_real` distinguishes it.

Hypothesis: the simpler head may retain detection/fairness performance, showing that the head is not needed for the observed objective effect. A difference in the opposite direction is equally informative. Matched settings are not necessarily separately optimal for each architecture. This tests removing the entire gated head package; it cannot assign any difference solely to gates versus nonlinear heads or local aggregation.

Both arms inherit the SAME original-only P50 encoder, which was pretrained using the gated architecture. Thus this is a downstream head ablation conditional on that encoder; it is not a comparison of two architectures pretrained independently end to end.

## Fixed protocol and dependencies

Exact bound configurations: [L_A](L_A_plain_bce.json), [L_H](L_H_plain_hierarchical.json). [Machine-readable config comparison](evidence/config_comparison.json) lists every changed key. Legacy A's previously implicit `error_control=legacy` and zero warmup are now explicit. H differs only in head type, new checkpoint format and source binding.

Keep FF++ TRAIN76,139 / VAL25,386, exact hashes and six-key overlap audit. TRAIN/VAL have 163 declared contributor exceptions; do not call this identity-held-out. TEST-related declarations are deferred without opening TEST. P50 SHA `3174e5b01a84c91cc70afae538505663ca96454d897e6b793c0672826a8f08b5` is required. Cluster input bytes are not available locally and must be reverified there.

Ten downstream epochs, seed17, batch32, workers4, size224, bilinear/0.5 normalization, bf16 TRAIN and float32 VAL. Fine-tune block12/conv3/bn3/conv4/bn4, freeze BN running buffers. AdamW head1e-3 / exit1e-5 / decay1e-4, clipping5. Original weighted BCE visits every original once/epoch; same auxiliary schedule, calibration subset and deterministic intervention draws. Head widths retained as unused config fields in linear mode for recipe compatibility. All loss coefficients remain fixed.

Select earliest strict highest VAL AUC after ten epochs. Calibrate one global threshold using the selected checkpoint's VAL reals at the same finite-sample10% FPR budget; preserve each reference's stored threshold. No group-specific thresholds, fairness-selected epochs or test tuning. Fixed probability0.5 diagnostics are separate.

L_H requires fresh source/config-bound qualification. The new source invalidates old receipts. [reuse_review.py](reuse_review.py) may preserve existing named H reviewer decisions ONLY if every case/setting/metadata/source byte/contact-sheet byte agrees with fresh previews. It records reuse explicitly, does not invent a new visual review, and calls the strict qualifier. Otherwise retain the failure and perform a real new named review, or leave L_H blocked. AI review must stay labeled AI; it does not establish identity, demographic or real-forgery validity. L_A is independent of this gate.

## Required checks and reporting

CPU checks and actual outcomes: [CHECKS.md](CHECKS.md). CUDA synthetic smoke, production-input hashing/decoding, and actual TRAIN-threshold backward checks are pending on the cluster. Verify finite nonzero classifier/exit gradients, frozen BN and H fairness gradient; the CLI refuses stale/wrong-head/device receipts. No downloads occur automatically.

Preserve old H source/checkpoints. Use [replay_reference.py](replay_reference.py) only for completed gated A/H if private validation predictions are absent or need verification. It checks exact config/source/receipt, ten-epoch history, earliest maximum-AUC selection, checkpoint/data bytes, strict gated weights and exact stored validation metrics. It does not train, recalibrate or read TEST. Local tests cover its guards; real checkpoint replay remains pending.

Report four rows (gated A, linear A, gated H, linear H), selected epoch, AUC, global FPR/FNR, each axis's F_FPR/F_DP/es_AUC/gaps/worst-group AUC, per-group denominators and actual compute. Fairness values are fractions internally; clearly mark any100x display. Pairwise deltas are linear minus gated. Report H-minus-A within each architecture as descriptive objective effects, preserving differences in auxiliary exposure.

Run1000 paired connected-recorded-contributor bootstrap replicates, seed17, fixed thresholds/eligible groups. The inherited FF++ membership yielded only746 usable primary replicates, below the existing900 requirement. Do not relax that criterion or claim formal noninferiority from nonsignificance. This single-seed diagnostic has no new binary success gate: report effects, intervals, sparse support, AUC/fairness tradeoffs and uncertainty. No new SOTA claim follows from VAL-only results.

Budget: up to6 additional aggregate GPU-hours for both runs, CUDA checks and any necessary VAL replay, subject to the REMAINING original24h ledger. At the pinned H observation10.6674h remained, but the operator must refresh it. Each config retains2h native epoch-boundary allowance. Same-source/config resume may finish the SAME ten epochs only within the aggregate cap; record any explicit cumulative allowance increase. No extra epochs, seeds, loss tuning, new Ding runs, FF2, Qwen generation or TEST inference.

Do not reinterpret retained historical gates. Results may justify simplifying Part C, retaining it, or calling the evidence inconclusive. Broader backbone/TEST/seed confirmation is a future separately authorized study.
