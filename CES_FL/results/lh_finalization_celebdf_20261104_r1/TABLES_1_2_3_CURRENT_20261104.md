# Tables 1–3: current comparison status (4 November 2026)

## Table 1 — FF++ intra-domain, Xception

- [Ding-format fairness and detection table, source-validation-real 10% FPR](https://github.com/hardikbha/CFES-BeatDing/blob/results/lh-tables-ee73f2f-20261103/CES_FL/results/lh_tables_s17_20261103T133000Z_cluster/full_paper_tables_20261103T1405Z/TABLE_1_SOURCE_VAL10.md)
- [Same table at fixed probability 0.5](https://github.com/hardikbha/CFES-BeatDing/blob/results/lh-tables-ee73f2f-20261103/CES_FL/results/lh_tables_s17_20261103T133000Z_cluster/full_paper_tables_20261103T1405Z/TABLE_1_FIXED05.md)

These completed rows compare Ding's published reference values with the single-seed-17 Xception L_A and L_H test results. Our selected thresholds come from FF++ validation and are not recalibrated on TEST.

## Table 2 — Cross-domain, Xception

- [Ding-format DFDC, Celeb-DF and DFD table, source-validation-real 10% FPR](https://github.com/hardikbha/CFES-BeatDing/blob/results/lh-tables-ee73f2f-20261103/CES_FL/results/lh_tables_s17_20261103T133000Z_cluster/full_paper_tables_20261103T1405Z/TABLE_2_SOURCE_VAL10.md)
- [Same table at fixed probability 0.5](https://github.com/hardikbha/CFES-BeatDing/blob/results/lh-tables-ee73f2f-20261103/CES_FL/results/lh_tables_s17_20261103T133000Z_cluster/full_paper_tables_20261103T1405Z/TABLE_2_FIXED05.md)

These use the same frozen Xception L_A/L_H checkpoints as Table 1. DFDC's native duplicate-row multiplicity is preserved. Training and preprocessing differ from Ding's paper protocol, so the rows are contextual comparisons rather than a protocol-matched reproduction.

## Table 3 — ResNet-50

- [Existing full ResNet Table 3](https://github.com/hardikbha/CFES-BeatDing/blob/run/lh-finalization-resnet50-20261104/CES_FL/results/ding_format_tables_s17_20261102/TABLE_3.md) — Ding published references plus historical CES-FL ResNet A/B/C test rows.
- [New locked Table 3 protocol](https://github.com/hardikbha/CFES-BeatDing/blob/run/lh-finalization-resnet50-20261104/CES_FL/proposals/lh_finalization_celebdf_20261104_r1/configs/resnet50_table3_eval_protocol.json)
- [New ResNet L_A/L_H Table 3 point estimates](TABLE_3_RESNET_POINT_ESTIMATES_20261104.md)
- [Connected-contributor uncertainty and paired L_H−L_A analysis](TABLE_3_RESNET_UNCERTAINTY_ALL_20261104.md)
- [Per-group TEST FPR/FNR/AUC and support by axis](TABLE_3_RESNET_GROUP_METRICS_20261104_VERIFIED.csv)

The new plain-head ResNet L_A/L_H runs are complete and validation-replay verified. Their locked TEST passes completed once per dataset on shared gpu02 H100s under verified pool-v2 reader leases. PBS job `45437.mgmt01` did receive `ngpus=1` but had no CUDA binding; probe `45438.mgmt01` confirmed missing `CUDA_VISIBLE_DEVICES` and `PBS_GPUFILE`, so PBS stopped before TEST inference. No existing job was preempted.

| Dataset | Ding published ResNet-50 AUC (paper TEST) | New CES-FL L_A TEST | New CES-FL L_H TEST |
|---|---:|---:|---:|
| FF++ | 95.53% | 96.2103% | 96.0537% |
| DFDC | 62.54% | 59.5355% | 60.0680% |
| Celeb-DF | 77.35% | 73.4969% | 73.8106% |
| DFD | 78.63% | 73.6546% | 73.4718% |

These point estimates use Ding's published values as references under Ding's protocol; they do not establish a protocol-matched win. Fairness metrics at both frozen thresholds are in the point-estimate link; group-wise FPR/FNR/AUC and per-group support are in the CSV supplement. Confidence intervals and paired differences are in the uncertainty link.

### Current selected-checkpoint validation snapshot

| New arm | Selected epoch | FF++ validation AUC | FPR | FNR | Gender F_FPR | Race F_FPR | Age F_FPR | Skin F_FPR | Gender×race F_FPR | Worst gender×race AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet L_A | 5/10 | 96.6525% | 9.9951% | 9.6318% | 5.8398% | 12.0151% | 14.5474% | 8.5857% | 39.2534% | 90.4245% |
| ResNet L_H | 1/10 | 96.4214% | 9.9951% | 9.9934% | 5.6438% | 9.9922% | 12.2680% | 5.0148% | 33.3735% | 90.2108% |

This validation snapshot uses the selected checkpoint's FF++ validation-real 10% FPR threshold; achieved validation FPR is 9.9951%. The TEST table above is separate and was generated once under the locked plan, with no target-domain calibration or threshold search.

## Current source and checkpoints

- P50 ResNet-50 original-only pretraining: 50/50 epochs, selected epoch12, validation AUC96.5348%; checkpoint `c28bbb9cfcd06a11894027dc9369c5fd674f640de6c182c28b6d6f0aa8d77f55`.
- ResNet L_A selected checkpoint: `0571a220faa63ad61c3b2d5815d73f25a077ecf59a1a8e30c379a620e31c0555`.
- ResNet L_H selected checkpoint: `b08c2b5180d999f81e044872a8228f7e876d0250a6cb29034d59b11b1a897f10`.
- Locked dynamic evaluation plan SHA256: `a234331c21c5539fb350f011737c993dd9bf84feb3d1ea65132a4fc36e98a788` (cluster-private; no test scores included).

The locked plan binds four test manifests, overlap audits, selected checkpoint hashes, fixed-0.5 and source-validation threshold views, and connected recorded-contributor uncertainty. Inference used gpu02 GPU5 for L_A and GPU4 for L_H, with shared reader leases and per-process UUID verification. The PBS binding failure was preserved. Private scores remain on the cluster; only aggregate metrics were published.

All results are seed17. The Xception tables, historical ResNet A/B/C results, Ding's published rows and current ResNet L_A/L_H validation results use distinct protocols/checkpoints. No result here establishes that we beat Ding under a matched protocol.
