# Actual D failure audit and completed E/F diagnosis

2026-11-03T09:17:01.797000+00:00 management UTC.

Source/configs/results were read directly: D c332fd34; E/F8d5f099; result snapshot206d5eed960bd2c727ec810e8ed0b92c6d825a29. Native E/F completions and selected checkpoint bytes were rehashed. No old run or protocol was changed.

The requested D predictions are bound to selected checkpoint7d9409e2cb6aa00fdd61cd3a072bd5473848be6dd86aafcc3b7b740da9f25d9d and frozen logit threshold−4.397897720336913. All25,386manifest rows/labels and image hashes match; full metric replay equals the stored checkpoint report. Adopted upstream validation group/membership mapping also matches. No recalibration/inference occurred in this failure export.

|Requested stratum|Errors / class rows|Error rate %|Recorded components|
|---|---:|---:|---:|
|male_white_real|193/1264|15.2690|69|
|female_black_real|9/30|30.0000|2|
|age60plus_real|24/63|38.0952|6|
|male_others_fake|112/1170|9.5726|19|
|age45to59_fake|220/2398|9.1743|24|

Five private sheets show errors beside same-group correct examples,40displaycells. Different recorded components are used where available. Actual visual review found faces in crops, variable broadcast softness/illumination/pose and cosmetics on both correct/error sides. Some similar frames from the same component differ in outcome. A missed FaceSwap sample visibly contains a blending discontinuity; some sharp/frontal fakes are also missed. This is descriptive visual evidence, not a causal attribution. Exact sample paths and case observations remain private.

Manifest demographics were not inferred or corrected from appearance. Age/skin proxies and inherited gender/race can be wrong; source videos/human ground truth are unavailable. PNG container type does not establish upstream compression. Known163TRAIN/VALrecorded contributor overlaps remain. No discovered new byte/provenance/decoding discrepancy invalidates the adopted-annotation-relative comparison; true demographic fairness/biological split independence remain unverified.

The supplied output/cesfl_failure_diagnostics_20261103.zip was unavailable at the named location, searched home/tmp locations and inspected published branches. The new exporter is independently authored and identified; no bundle contents or supplied exporter execution are claimed.

|Arm|Epochs complete|Selected epoch|AUC %|Intersection F_FPR fraction|Gender F_FPR fraction|Age F_FPR fraction|
|---|---:|---:|---:|---:|---:|---:|
|D|10|1|98.252564|0.379345|0.053498|0.303616|
|E|10|2|98.228795|0.468281|0.054478|0.312441|
|F|10|1|98.319331|0.495578|0.048598|0.296048|

E/F both rejected their frozen rules, not promoted. E primarydelta+0.088936,95%CI[−0.011646,0.154790]; F+0.116232,[−0.004303,0.144386]. Both fail primary/age, E also gender. Only746/1000primary draws valid, below900required. Analysis CPUjob45259 completed. Later epochs reduce training sensitivity/alignment but do not improve selected AUC; this suggests optimization/generalization tension, not a proven cause. E/F altered real and fake errors, used sampled-diversity discounts, and F ramped alignment. H instead preserves D fake errors and all other-axis penalties, changes only joint-event real risk, jointly adding TRAIN-weighted parent-gender risk without discount/ramp. It is not a renamed E/F config.

A new labels-only structural check proves exactly746usable primary draws with unchanged components/eligibility/seed, independent of model scores. PASS is therefore unavailable under the retained900rule; numerical success can only be INCONCLUSIVE. This is disclosed before launch, not used to loosen inference. This experiment still measures the explicitly requested numerical effect.

Fixed objective/criteria/config and prospective limits: README.md, decision_protocol.json and PRELAUNCH_LIMITATIONS.md. No TEST or Ding model computation. All subsequent comparisons must include supported-axis regressions, fixed-threshold uncertainty and fakeFNR.
