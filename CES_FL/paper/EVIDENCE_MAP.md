# Evidence map and exact versions

## Reviewed source and frozen artifacts

- Current audited upstream: `30a6b263aaa63677cf8d3d9f4ee51bdb472ae284` on `experiment/ff2-realfpr-stretch-20261107`.
- Existing TEST freeze draft: `8d06aa09de29f3a6a7f666e0ab41c0cc3e7048ab`; sanity-tolerance publication: `8233f4c1bf1a52847f44eb555fc897ec8f90952b`.
- RF source: `CES_FL/src/ces_fl_realfpr.py`, SHA256 `2d99295cc0b8190a571695b581ef5ba73e7b5fc18994461bdf1a21c1d83089d1`.
- Model/core: `CES_FL/src/ces_fl_head_ablation.py`, SHA256 `8c25ba8cb844b059adbbce75b57785a156333182afa2d5d576985997f2d36733`.
- Original RF-S seed17 receipt records earlier RF SHA `5ee7c8b92c0b5def8219d100439c9ba8cb4541e604af79fe41b7dff7a281b8c8`; source extension concerned JPEG sampling/ramp. No claim of bitwise replay under a different source version.
- Shared pretrained encoder SHA256 `741a8eb5b60ac8efe2ca34be985b423de12facac053acac7f05803df23248a23`.
- Exact downstream dictionaries/configuration digests: [CONFIGURATIONS.json](evidence/CONFIGURATIONS.json). Original proposal configurations remain under `CES_FL/proposals/ff2_realfpr_stretch_20261107_r1/configs*`.
- Existing export code/protocol: `CES_FL/proposals/ff2_test_eval_20261108_r1/`. Its historical nine-checkpoint count is an arithmetic typo; actual four singletons plus six RF checkpoints are ten.

## Manuscript assertions → saved evidence

| Assertion | Public evidence | Original cluster artifact |
|---|---|---|
| Ten actual checkpoint-byte hashes and TEST manifest binding | `evidence/TEST_RECEIPTS.json`, `verification/evidence_collection_final.log` | `/home/sachin.chaudhary/ces_fl_runs/ff2_test_eval_20261108/out/*/TEST_EVALUATED.json` |
| Selected RF epochs, feasibility, replay, threshold unchanged | `evidence/VALIDATION_SELECTED.json`, `EPOCH_HISTORY_AGGREGATE.json`, `PROVENANCE.json` | `/home/sachin.chaudhary/ces_fl_runs/ff2_rfa_20261107/{S,S_s7,S_s27,SJ_gentle,SJ_gentle_s7,SJ_gentle_s27}/out/{RUN_VERIFIED.json,history.json}` |
| Full TEST metrics on eight reporting views | `evidence/METRICS_LONG_TEST.csv`, `GROUP_SUPPORT_TEST.csv` | `/home/sachin.chaudhary/ces_fl_runs/ff2_test_eval_20261108/metrics/` |
| Native frozen/0.5 reports where already available | `evidence/TEST_NATIVE_AGGREGATE.json` | Existing `out/*/metrics.json`; corrected GDA lacks the fixed-0.5 view, and no new view is invented |
| Source-processing probe | `evidence/PROCESSING_PROBE.json` | `/home/sachin.chaudhary/ces_fl_runs/ff2_source_diag_20261106/recrop_summary.json` |
| Duplicate seed7 disclosure | `evidence/DUPLICATE_RUN_AUDIT.json` | `/home/sachin.chaudhary/ces_fl_runs/ff2_sjgentle_confirmation_20261108_r1/SJ_gentle_s7/out` |
| Ledger balance, incomplete TEST overhead | `evidence/BUDGET_SNAPSHOT.json` | `/home/sachin.chaudhary/ces_fl_runs/ff2_baseline_suite_s17_20261103T110324Z/additional72_gpu_hours.json` |
| Every summary fairness formula recomputes from group aggregates | `verification/public_snapshot_checks.log`, `tools/verify_public_snapshot.py` | No private scores or inference needed |

All rate/AUC inputs are explicitly fractions or multiplied-by-100 values. `F_MEO_max` is max(FPR gap, TPR gap), not mean equalized odds. Equity-scaled AUC and F_FPR use eligible groups, with overall rates computed across the full evaluation set.

Canonical public TEST manifest SHA256: `dd5eb9ef070f658fe6b7f744c47f73e25ce7c53f948bc272f4f16c69a37d4099`. The paper snapshot hashes checkpoint bytes; it does not decode/re-hash the full image bank or certify true-person separation.

## Temporal and numerical caveats

The 01:43 IST login observation corresponded to 01:32 on both GPU nodes. Do not independently certify freeze order using those uncorrected timestamps. The prior exporter README claims pre-export publication, but current evidence cannot fully prove the claim. Representative-family sanity checks and their tolerances are documented; they are not ten individually bit-exact replays. Small DAW/GDA differences persisted GPU-to-GPU. The earlier report's bf16 explanation is a hypothesis, not a proven root cause.

Private row-level scores, all face images, private reviews and checkpoint bytes remain on cluster storage. This snapshot only includes approved aggregate categories; it does not create new demographic annotations or counterfactual approvals.
