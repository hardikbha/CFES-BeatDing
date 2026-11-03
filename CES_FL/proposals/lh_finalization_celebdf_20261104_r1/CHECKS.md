# Actual checks, 4 November 2026

Executed on the local CPU. No dataset images, model weights, benchmark inference, training or scheduler submission were used. These are audit-software checks, not model/CUDA certification.

```bash
python CES_FL/proposals/lh_finalization_celebdf_20261104_r1/analyze_published.py --out CES_FL/proposals/lh_finalization_celebdf_20261104_r1/evidence/analysis --plot
```

Result: `PUBLISHED_AGGREGATES_RECONCILED`, source result commit `916999c8e54d0eeaa86bd2b09d9d3ceaadea0bb5`. Input/script hashes are in `evidence/analysis/SUMMARY.json`. The generated heatmap was visually inspected for labels, missing-class handling, annotation values and scale interpretation.

Additional executed checks, recorded in [checks.log](evidence/checks.log):

- All eight source reports reconcile partition/class counts, integer false-positive/false-negative counts, pooled rates, fairness formulas and unchanged AUC across the two frozen thresholds.
- Independent reconstruction from the original public group CSV gives5,713real/22,745fake,2,384false positives/6,231false negatives for L_H fixed0.5.
- Every observed Celeb-DF class×intersection count agrees with the older frozen `upgrade_tests_s17_20261101T040407Z/protocol.json` inventory.
- Deliberately altered class counts, fractional error counts and a threshold-dependent AUC are rejected.
- Reusing the existing output directory is rejected and its summary hash remains unchanged.
- Original `src/ces_fl_head_ablation.py` SHA remains `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`.

The model source, completed results/configs and historical preregistration are unchanged. The ResNet/Ding-10 training adapters, bound native pretraining configuration, actual cluster input/CUDA checks, private annotation provenance and scheduler allocation are **not certified by this package**. Their implementation and verification steps are in the operator prompt. No new training result is implied by publishing this commit.
