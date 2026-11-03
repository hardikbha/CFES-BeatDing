# L_H completion handoff and descriptive Celeb-DF audit

User request, 4 November 2026: provide a prompt/commit for the GPU operator to complete the remaining comparisons and investigate Celeb-DF. This package performs only CPU arithmetic on already-published aggregates. It does not train, infer on benchmark images, select thresholds, change historical protocols, or declare a final model/SOTA result.

Start with [the GPU operator prompt](GPU_OPERATOR_PROMPT.md), [Celeb-DF findings](CELEBDF_DIAGNOSIS.md), [exact request](request.json), and [checks](CHECKS.md). The original scientific source remains `CES_FL/src/ces_fl_head_ablation.py`, SHA256 `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`.

## Actual readiness

| Work | State |
|---|---|
| Original Xception L_A/L_H Tables 1–2 | Completed, result commit `916999c8e54d0eeaa86bd2b09d9d3ceaadea0bb5`; preserve |
| Three-epoch lower-LR experiment | Completed, `40cbe7da6e1cf2564bb52293a746f5a3bd4e56ff`; no overall improvement |
| SHAM-FIX and NO-ALIGN | Completed, `08726ed2ed21b38099d1ca00f27392405535cb5b`; diagnostic results, no replacement selected |
| This Celeb-DF aggregate audit | Executed locally; arithmetic reconciles with published reports |
| Celeb-DF private row/image/annotation audit | Operator task; unavailable locally |
| Current ResNet-50 L_H and matched ResNet L_A | Adapter, native pretraining configuration binding, smoke and training still required |
| Connected-gradient Ding ten-epoch control | Adapter/checkpoint/SFD binding, smoke and training still required |

This commit contains an executable CPU analysis plus an implementation-and-run handoff. It does **not** contain ready-to-launch ResNet L_H or Ding-10 trainers. Do not launch the historical Xception CLI with a fabricated ResNet field, or archived GDA merely with a changed epoch flag.

## Reproduce the aggregate audit

From the repository root, with Python 3.9+:

```bash
python CES_FL/proposals/lh_finalization_celebdf_20261104_r1/analyze_published.py --out /tmp/cesfl_celebdf_audit_new
```

Add `--plot` if numpy/matplotlib are already installed. Choose a new output directory; existing outputs are preserved. The committed evidence includes the actual summary and [PNG](evidence/analysis/celebdf_group_diagnostics.png)/[PDF](evidence/analysis/celebdf_group_diagnostics.pdf). No network or GPU is used by this command.

`evidence/published_inputs.json` contains aggregate excerpts, not image paths or individual predictions. Full original input-file hashes and the immutable GitHub result commit identify the extraction sources. The script reconciles partition counts, integer error counts, pooled error rates, main fairness formulas, A/H group membership and threshold-independent AUC. It does not independently certify the unpublished row-level caches or labels.

## Scope and budget

The original candidate remains Xception L_H, epoch1 selected after ten epochs, checkpoint `6ef2caad3979d101fd94d7350aa3c1a50f95ee8307673cfaae63ac9d89af36f0`. No new loss or hyperparameter candidate is introduced. The ResNet and Ding-10 work orders existed before the locked TEST evaluation; this package makes their remaining dependencies explicit. The prospective cosine run is deferred after the completed lower-LR/ablation studies; do not add it automatically.

Latest published original ledger:17.286613/24 aggregate GPU-hours used,6.713387 remaining. Re-read the actual shared ledger and existing allocations before using any GPU. At most two GPUs, aggregate charging including checks/idle/replay. The separate FairForge-2 72-hour allowance is not available to this task. Profile the complete remaining stage costs, reuse verified existing stages, and do not launch a cohort whose full cost cannot fit. Lack of remaining compute is a reported blocker, not permission to truncate a 50-epoch pretraining protocol or relabel an old ten-epoch model.

Celeb-DF is already-exposed TEST data. This audit describes errors; it must not select a new threshold, checkpoint, data filter, augmentation or loss. If a metadata/code defect is established, preserve historical artifacts and publish the scope and correction under a new version; do not silently overwrite labels or recalculate a more favorable main table.
