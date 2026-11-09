# Preparation checks — 10 November 2026

These are local CPU software checks, not real-data results or CUDA certification.

| Check | Actual result | Evidence |
|---|---|---|
| New projection/provenance/gate tests |12/12 PASS |`evidence/unit_cpu_published.log` |
| Reused diagnostic/helper regressions |9/9 PASS |`evidence/reused_helpers_cpu.log` |
| Full-width ResNet50,224-pixel backward under new update hook |PASS |`evidence/software_cpu_published.json`, `.log` |
| Projection-disabled optimizer matches native update |Exact parameters/state match in unit check; full-width exact state match |Same receipts |
| Opposing-gradient projection |Independent2D vector reference and actual AdamW moment checks PASS |New tests |
| Synthetic native-D lifecycle |Two epochs, exact saved-score replay, original-real pair coverage, checkpoint update-rule metadata PASS |New tests; synthetic metrics are not scientific results |
| Diagnostic and allocation gates |Reject missing/duplicate events, changed sampling plan, unsupported conflict, wrong ledger, expired/reset window and synthetic allocation |New tests |
| Metadata missing cells and donor dependencies |Zero source/group cell preserved; full dependency graph not replaced by real-only diagnostic; raw identifiers omitted |New tests |
| Python syntax / shell syntax |PASS |`evidence/RELEASE_CHECKS.json` |

The full-width CPU receipt's native synthetic alignment gradient was non-conflicting, so it records `projection_applied=false`. The negative-dot branch is exercised explicitly by independent numerical and optimizer-moment tests. This distinction is retained rather than claiming the fixture proves real-data conflict.

Parameter counts:23,510,081total,14,966,785trainable,2,049head parameters. Native scientific source and H0 config hashes remain1997eebf... and34800f42..., respectively. All exact file hashes are in `FROZEN_FILES.json` and the CPU receipt.

Commands actually run locally:

```bash
python -m unittest discover -s CES_FL/tests -p test_lh_gradient_conflict.py -v
python -m unittest discover -s CES_FL/tests -p test_lh_alignment_followup.py -v
python CES_FL/scripts/lh_gradient_conflict.py software-check --device cpu --out CES_FL/proposals/lh_gradient_conflict_20261110_r1/evidence/software_cpu_published.json
python -m py_compile CES_FL/src/alignment_gradient_projection.py CES_FL/scripts/lh_gradient_conflict.py CES_FL/scripts/ff2_provenance_inventory.py CES_FL/tests/test_lh_gradient_conflict.py
bash -n CES_FL/proposals/lh_gradient_conflict_20261110_r1/commands.sh
```

Production private manifests/checkpoints/review sheets were unavailable locally. P0 production inventory, H0 binding, repeated real TRAIN gradients, actual allocation, fresh CUDA check, cost profile and H_GP learning/reporting have **not run here**. Old CUDA receipts cannot certify the new hook. The operator must use the documented explicit stages and preserve blocked/incomplete outcomes. No TEST/CAL, Ding-model work, new generator or scientific result was produced by preparation.
