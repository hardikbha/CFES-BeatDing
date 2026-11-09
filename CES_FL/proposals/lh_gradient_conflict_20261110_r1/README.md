# Provenance review, repeated gradients, and one conditional L_H update test

User request, 10 November 2026: push the proposed experiments and give the operator an executable handoff. This package prepares P0 (CPU provenance), G0 (zero-update TRAIN gradients) and one conditional H_GP candidate. Publishing does not launch a job. No new overall GPU budget is granted.

## New evidence checked before preparation

- The [source-tail cohort](https://github.com/hardikbha/CFES-BeatDing/blob/aa074b2024a9730a9d66f6020440d2137e7cd6f1/CES_FL/results/source_tail_training_20261109_r1/REPORT.md) is COMPLETED: treatment/control selected joint100F_FPR34.8206/34.8501, a reduction of only0.0295points, with primary uncertainty inconclusive. It is a separate frozen-encoder/head-only FF2 method, not L_H. Do not rerun or retune it.
- The [source-stratified audit](https://github.com/hardikbha/CFES-BeatDing/blob/70916d030c9da5fba6063f83e4bf3af1b59bc6f5/CES_FL/results/ff2_source_stratified_audit_20261109_r1/REPORT.md) is completed. Missing FFHQ-real Others cells block full source-standardization, not native demographic F_FPR or the global eight-cell hierarchy. Do not exclude Others or reset the bootstrap rule.
- The [three-epoch component study](https://github.com/hardikbha/CFES-BeatDing/blob/fee7c863c3052261dffd3f1fb93f5884dfc9dada/CES_FL/results/lh_objective_priority_20261105_r1/REPORT.md) already includes A0/B0/C0/H0. H0−C0 improves the joint fairness point while losing detection; primary uncertainty746/1000fails900. Reuse these results rather than rerunning the ladder.
- The [nuisance diagnostic](https://github.com/hardikbha/CFES-BeatDing/blob/e993315a853aa278ec649010fa11daef1e601e44/CES_FL/results/lh_nuisance_pilot_20261104_r1/REPORT.md) found alignment/BCE cosine−0.498 on one H0 TRAIN batch and−0.600 on H1. These are local observations, not proof that conflict causes validation harm. This package repeats measurement with native uniform main sampling before any new candidate.

This branch starts from the already tested diagnostic/source package `6ce942a83d22bd7e3ad1b4fcca32ae9ba5c4731f`. Historical source/configuration/results remain unchanged. Later source-tail/audit commits above are references, not code silently merged into this experiment.

## Stages and scope

| Stage | Dataset | Work | Stop condition |
|---|---|---|---|
| P0 | FF2 TRAIN/SELECT metadata only | Trace category/identifier construction; inventory missing source×group cells and dependency links | Untraceable semantics stay unresolved; no relabeling or graph splitting |
| G0 | FF++ TRAIN only, existing H0 checkpoint |20fixed native-main batches covering5axes×4operator/strength cases; no optimizer updates | Stop candidate if repeated-conflict trigger fails |
| H_GP | FF++ TRAIN/VAL |Same full L_H as H0, but project only the opposing alignment-gradient component | Requires complete G0, unchanged reviews/inputs, fresh CUDA check and affordable full profile |
| Report | Saved FF++ validation caches |H_GP−H0, all metrics and paired uncertainty | Stop after reporting; no automatic extension |

P0 and G0 answer separate dataset questions. An unresolved FF2 source-standardization estimand does not invalidate a scoped FF++ native-metric diagnostic. Conversely, the FF++ pilot cannot clear FF2 provenance gaps. This is a procedural optimization study, not a validated forensic or causal-fairness experiment.

## Exact mathematical change

Let b be the gradient of weighted original BCE and a the **weighted** response-alignment gradient over all trainable parameters. When a·b<0, use

```text
a_projected = a - (a·b / ||b||²) b
g_new       = g_native_total + (a_projected - a)
```

Otherwise preserve the native total gradient exactly. A zero/tiny BCE norm is unchanged. The projection is global over trainable parameters, not independently per layer. Original classification, hierarchical fairness, sensitivity and sham-control gradient contributions stay in the native total. The alignment coefficient already includes aux_every×lambda_alignment; no ramp is introduced. Native clip-norm5 and AdamW follow the combination.

This is a one-sided **PCGrad-style** adaptation of established [gradient surgery](https://arxiv.org/abs/2001.06782), not a novelty claim, the complete symmetric/random-order PCGrad algorithm, or a guarantee that classification loss/AUC/FNR improves. Other objective gradients, clipping, AdamW preconditioning/momentum and held-out distribution shift prevent that guarantee. Some conflict may be necessary to improve fairness. The displayed native scalar losses are monitored but the modified direction is not generally the gradient of their simple sum.

Code: `CES_FL/src/alignment_gradient_projection.py`. The unchanged native implementation is `CES_FL/src/ces_fl_nuisance_resnet50.py`, SHA1997eebf093aa3e37de1cbbe58302a1539ac2e517f966a4607aae89dc695a8e0. The wrapper records its rule/fingerprints in every new checkpoint and its sidecar; a historical native source hash alone must not identify the candidate as ordinary H0.

## What stays fixed

`native_config.json` is byte-identical to H0's SHA34800f42f2cc81df32508b7a732700e45b4df5f6ddfebca1a109aad1db1407ef. ResNet50+linear; same original P50 encoder/fresh seed17head; same TRAIN76,139/VAL25,386 membership; batch32;3epochs; original inverse label×ids8 weights; layer4/head adapted; frozenBN; AdamW head1e-3/encoder1e-5/decay1e-4; same pair/auxiliary schedule; no added nuisance augmentation; same probe masks and strengths; same loss coefficients. Selection remains earliest strict max clean-validation AUC and the selected model's global validation-real10%FPR threshold. Fixed0.5 is supplementary.

The control is the already completed H0, not the frozen teacher, RF-S, a differently augmented arm, a new seed or a ten-epoch baseline substituted after results. Binding verifies completion, exact checkpoint/config hashes, max-AUC selection and stored-score replay. If that control cannot be verified, stop; do not invent a substitute or automatically retrain it.

## G0 diagnostic and gate

Use H0's selected checkpoint and20fixed TRAIN events. Each event has a32-image main batch from a deterministic uniform permutation without replacement across the events, retaining the native BCE weights. Auxiliary sampling retains the native balanced groups. This corrects the interpretation limits of a single class/group-balanced main diagnostic batch.

Measure all five weighted components' gradient norms/cosines, response/margin summaries and score magnitudes. Calibrate the TRAIN threshold once on the existing fixed2048real subset. The complete model state must remain bitwise unchanged. Report all20events; never select favorable batches.

As in the reused helper, this diagnostic uses float32 forwards/backwards; it is not a bitwise replay of a bf16 optimization step. Only the conditional candidate retains the original configured bf16 learning path.

The prospective **engineering** trigger requires at least16valid events, at least60%negative alignment/BCE cosines, and median weighted alignment/BCE norm ratio≥0.25. Only the original identity-processing condition and H0 govern the gate. These thresholds are not significance criteria or proof of harmful conflict. If false, write the unsupported concern and stop without H_GP. The older H1/resize90 C1 proposal remains separate and is not additionally authorized by this handoff.

## Input/review validity and privacy

Binding rechecks exact H0 preflight/initialization and the complete canonical88source/operator/strength cases. It also recomputes their procedural tensors under the current environment and requires exact equality to all352existing contact-sheet panels, without writing new images or granting approval. Empty, missing, duplicate, changed or rejected cases fail. Reuse existing decisions only for unchanged bytes; retain the AI-only scope and `real_forgery_validity_proven=false`. No new review approval is generated and no difficult source is dropped. Deterministic procedural tensors are generated by the existing recipe during diagnosis/profile/learning; no Qwen, demographic edit bank or new saved preview images are authorized.

P0 reads only the two pinned FF2 CSVs and writes aggregate inventories. It compares the full donor-linked graph with a real-only graph **as a diagnostic**, never as permission to change bootstrap units. Original raw identifier and provenance strings are not exported. The operator must inspect original taxonomy/identifier builder source read-only, record paths/hashes/line evidence privately, and publish a sanitized semantics assessment. A hash match alone does not validate label meaning or biological identity. No code automatically repairs labels or source membership.

## Resources, check status and outcome rules

At most1verified GPU and2aggregate GPU-hours, including allocation idle, diagnostics, software checks, profile, candidate and replay, within a freshly verified remainder of the **original L_H study ledger**. Do not borrow the FF2 ledger (latest source-tail publication records3.769462remaining there, not a new grant). Use one fixed cohort deadline; no per-stage clock reset. The explicit runtime candidate allowance is at most1.5hours. The copied native config retains its historical0.75hour field for byte identity; the new bound CLI allowance overrides execution time only, never epochs or updates. Profile20warmup+40measured actual steps, validation and I/O before launch. If the full estimate cannot fit the native or cohort cap, stop without a partial-budget success claim. No extra epochs, seeds, full tuning, CLIP, additional source-tail work or Ding-model computation.

An idle GPU is not an allocation. The operator/controller must verify actual PBSngpus or an explicitly authorized existing pool lease and bind the exact physical UUID. Receipt checks do not authenticate scheduler ownership. An external deadline/ledger wrapper must cover calibration, loading, validation, I/O and all held idle, not just optimizer steps. Use a shared self-contained detached checkout; a .git link into login-node /tmp is not a node-readable source.

Primary comparison: H_GP−H0 jointF_FPR. A favorable **point screen** requires lower primary, AUC no lower and FNR no higher; it is not a noninferiority test. All gender/race/age/skin/ids8, worst-group, FNR and other regressions are retained. Use the unchanged1000paired connected-contributor draws and900valid floor. Withhold under-supported intervals and retain INCONCLUSIVE even if points improve. Native subgroup estimates remain defined where their denominators exist; no source-standardized estimate is fabricated. Existing TEST exposure is disclosed and no TEST/CAL command exists.

See [CHECKS.md](CHECKS.md), [GPU_OPERATOR_PROMPT.md](GPU_OPERATOR_PROMPT.md) and [commands.sh](commands.sh). The operator may execute these user-requested bounded stages once their gates pass; publication itself does not execute them. Private inputs, qualification files, runtime and allocation must be verified on the node. No full experiment or GPU work was launched locally.
