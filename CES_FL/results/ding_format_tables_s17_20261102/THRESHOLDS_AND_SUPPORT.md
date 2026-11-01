# Frozen thresholds and measured support

Requested validation FPR is 10%. Every selected checkpoint achieved 409/4,092 = 9.995112% on validation real images. Test FPR below is measured; the target is not guaranteed on test. No threshold was recalibrated here.

| Arm | Backbone | Dataset | Stored logit threshold | Test real n | Test fake n | Measured test FPR (%) | Eligible gender/race/intersection groups |
|---|---|---|---:|---:|---:|---:|---|
| A_imagenet_frozen | Xception | ffpp | 4.606697559 | 4538 | 20863 | 12.6928 | 2 / 4 / 8 |
| A_imagenet_frozen | Xception | dfdc | 4.606697559 | 14699 | 8158 | 39.4585 | 2 / 4 / 8 |
| A_imagenet_frozen | Xception | celebdf | 4.606697559 | 5713 | 22745 | 25.2932 | 2 / 3 / 3 |
| A_imagenet_frozen | Xception | dfd | 4.606697559 | 3092 | 6293 | 29.5602 | 2 / 2 / 4 |
| B_detection_frozen | Xception | ffpp | -0.631477594 | 4538 | 20863 | 10.5333 | 2 / 4 / 8 |
| B_detection_frozen | Xception | dfdc | -0.631477594 | 14699 | 8158 | 32.9274 | 2 / 4 / 8 |
| B_detection_frozen | Xception | celebdf | -0.631477594 | 5713 | 22745 | 64.6946 | 2 / 3 / 3 |
| B_detection_frozen | Xception | dfd | -0.631477594 | 3092 | 6293 | 42.6261 | 2 / 2 / 4 |
| C_detection_invariance | Xception | ffpp | -1.543892145 | 4538 | 20863 | 10.3570 | 2 / 4 / 8 |
| C_detection_invariance | Xception | dfdc | -1.543892145 | 14699 | 8158 | 40.0435 | 2 / 4 / 8 |
| C_detection_invariance | Xception | celebdf | -1.543892145 | 5713 | 22745 | 64.5370 | 2 / 3 / 3 |
| C_detection_invariance | Xception | dfd | -1.543892145 | 3092 | 6293 | 40.5563 | 2 / 2 / 4 |
| A_resnet50 | ResNet-50 | ffpp | 3.054235458 | 4538 | 20863 | 8.7924 | 2 / 4 / 8 |
| A_resnet50 | ResNet-50 | dfdc | 3.054235458 | 14699 | 8158 | 32.7165 | 2 / 4 / 8 |
| A_resnet50 | ResNet-50 | celebdf | 3.054235458 | 5713 | 22745 | 33.8526 | 2 / 3 / 3 |
| A_resnet50 | ResNet-50 | dfd | 3.054235458 | 3092 | 6293 | 21.3131 | 2 / 2 / 4 |
| B_resnet50 | ResNet-50 | ffpp | 2.392199039 | 4538 | 20863 | 12.7810 | 2 / 4 / 8 |
| B_resnet50 | ResNet-50 | dfdc | 2.392199039 | 14699 | 8158 | 35.8392 | 2 / 4 / 8 |
| B_resnet50 | ResNet-50 | celebdf | 2.392199039 | 5713 | 22745 | 58.4281 | 2 / 3 / 3 |
| B_resnet50 | ResNet-50 | dfd | 2.392199039 | 3092 | 6293 | 39.4567 | 2 / 2 / 4 |
| C_resnet50 | ResNet-50 | ffpp | 2.278280258 | 4538 | 20863 | 12.5165 | 2 / 4 / 8 |
| C_resnet50 | ResNet-50 | dfdc | 2.278280258 | 14699 | 8158 | 34.1996 | 2 / 4 / 8 |
| C_resnet50 | ResNet-50 | celebdf | 2.278280258 | 5713 | 22745 | 57.4829 | 2 / 3 / 3 |
| C_resnet50 | ResNet-50 | dfd | 2.278280258 | 3092 | 6293 | 38.7775 | 2 / 2 / 4 |
| R0_frozen_bce | Xception | ffpp | -0.631477594 | 4538 | 20863 | 10.5333 | 2 / 4 / 8 |
| R0_frozen_bce | Xception | dfdc | -0.631477594 | 14699 | 8158 | 32.9274 | 2 / 4 / 8 |
| R0_frozen_bce | Xception | celebdf | -0.631477594 | 5713 | 22745 | 64.6946 | 2 / 3 / 3 |
| R0_frozen_bce | Xception | dfd | -0.631477594 | 3092 | 6293 | 42.6261 | 2 / 2 / 4 |
| R1_partial_bce | Xception | ffpp | -1.791873455 | 4538 | 20863 | 10.3349 | 2 / 4 / 8 |
| R1_partial_bce | Xception | dfdc | -1.791873455 | 14699 | 8158 | 31.5668 | 2 / 4 / 8 |
| R1_partial_bce | Xception | celebdf | -1.791873455 | 5713 | 22745 | 65.4997 | 2 / 3 / 3 |
| R1_partial_bce | Xception | dfd | -1.791873455 | 3092 | 6293 | 42.5938 | 2 / 2 / 4 |
| R2_partial_dro | Xception | ffpp | -0.535319269 | 4538 | 20863 | 10.0926 | 2 / 4 / 8 |
| R2_partial_dro | Xception | dfdc | -0.535319269 | 14699 | 8158 | 28.8999 | 2 / 4 / 8 |
| R2_partial_dro | Xception | celebdf | -0.535319269 | 5713 | 22745 | 68.9130 | 2 / 3 / 3 |
| R2_partial_dro | Xception | dfd | -0.535319269 | 3092 | 6293 | 43.8551 | 2 / 2 / 4 |
| RN_partial_dro_nochange_pilot | Xception | ffpp | -0.432014644 | 4538 | 20863 | 9.4755 | 2 / 4 / 8 |
| RN_partial_dro_nochange_pilot | Xception | dfdc | -0.432014644 | 14699 | 8158 | 26.3623 | 2 / 4 / 8 |
| RN_partial_dro_nochange_pilot | Xception | celebdf | -0.432014644 | 5713 | 22745 | 61.3163 | 2 / 3 / 3 |
| RN_partial_dro_nochange_pilot | Xception | dfd | -0.432014644 | 3092 | 6293 | 33.8939 | 2 / 2 / 4 |
| RD_partial_dro_age_pilot | Xception | ffpp | -0.881052971 | 4538 | 20863 | 10.0705 | 2 / 4 / 8 |
| RD_partial_dro_age_pilot | Xception | dfdc | -0.881052971 | 14699 | 8158 | 24.9269 | 2 / 4 / 8 |
| RD_partial_dro_age_pilot | Xception | celebdf | -0.881052971 | 5713 | 22745 | 60.5461 | 2 / 3 / 3 |
| RD_partial_dro_age_pilot | Xception | dfd | -0.881052971 | 3092 | 6293 | 40.1358 | 2 / 2 / 4 |

Groups need at least 20 real and 20 fake rows. In Celeb-DF, Male-White and Male-Others have no fake rows and are excluded from eligible intersection aggregates. In DFD, Others has no real rows and is excluded. Full group supports and lists are preserved in source_snapshot.json.

Native DFDC duplicate excess rows: 1,476, preserved. Native frame counts are not independent-identity counts. Hidden donor links, proxy provenance and single-seed uncertainty remain limitations.
