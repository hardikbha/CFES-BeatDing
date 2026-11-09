# FairForge-2 source-standardized frozen-score audit

[Findings and explicit stop](REPORT.md) · [Primary result](PRIMARY_RESULT.json) · [Source/group support](SOURCE_GENDER_RACE_FPR.md) · [Commands and checks](COMMANDS.md) · [Frozen plan](FREEZE.json)

**CPU audit completed; primary estimand BLOCKED; relevant fairness intervals INCONCLUSIVE. Stop after reporting.**

The full race/gender×race standardization requires FFHQ-real Others cells that are absent in both TRAIN and SELECT. Their FPRs cannot be inferred from existing predictions. No groups/sources were removed, imputed or renormalized. Gender-only point standardization is identified, but only 1,871/2,000 paired component draws retain its required denominators, below the fixed 1,900 requirement. Native joint intervals have 999 valid draws; FairFace joint residual intervals have 202.

Six existing selected RF-S/RF-SJg runs (seeds 17/7/27) were checked and analyzed at their stored thresholds. The 31,107 unchanged-image SELECT score exports align with the manifest. Actual checkpoint hashes and historical source/config bindings match. Dependency components are formed from every recorded identity/master/donor link:45 SELECT-bearing components, not 45 people. Private outcome rows were first parsed after TRAIN source weights were frozen at 18.9306789% FFHQ / 81.0693211% FairFace.

This is an exploratory SELECT diagnostic, not a new detector, an improvement, causal decomposition or independent confirmation. It uses recorded acquisition provenance, not inferred appearance. Individual processing/master-byte histories are absent in permitted caches, so processing-history-specific claims are blocked. Biological identity independence is unverified.

No TEST/CAL access, new predictions, GPU/CUDA checks, model/training/calibration changes, label/approval generation, Ding-model computation, or automatic next experiment. Any training/data-revision proposal requires separate review.

## Outputs

| File | Role |
|---|---|
| `SOURCE_GENDER_FPR.md`, `SOURCE_RACE_FPR.md`, `SOURCE_GENDER_RACE_FPR.md` | TRAIN/SELECT real support and all six source-specific FPRs; empty cells retained |
| `SOURCE_BY_GROUP_REAL_FPR.csv` | All 168 real source/group/model cells with support, component counts, rates and interval-validity denominators |
| `NATIVE_UTILITY.csv` | Native AUC/FPR/FNR and joint disparity, same selected checkpoints and thresholds |
| `GROUP_FAMILY_SOURCE_UTILITY.csv` | Native group/family/reference-source AUC/FNR/FPR, support and uncertainty status |
| `DISPARITIES.csv`, `PAIRED_CONTRASTS.csv` | Native/standardized/residual quantities, exact matched contrasts and uncertainty |
| `METRICS_AND_INTERVALS.csv` | Complete aggregate metric/interval records |
| `COVERAGE.json`, `COHORT_SUPPORT.json`, `METADATA_CHECKS.json` | Support, provenance limits and dependency structure |
| `BINDINGS.json`, `ALIGNMENT.json` | Source/config/checkpoint/input SHA256s and cache alignment |
| `STOP_STATUS.json` | Explicit primary blocked status and stop instruction |
| `audit.py`, `audit_stats.py`, `test_audit.py` | Scoped CPU helper and pre-analysis checks; trainer unchanged |

**Units:** raw CSV `estimate`/interval values are fractions (AUC0–1; F_FPR is a sum and can exceed 1). Markdown tables multiply them by100. Thresholds are logits; counts are integers. No “percentage explained” is computed.
