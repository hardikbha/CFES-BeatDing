# What is failing on Celeb-DF?

This is a descriptive audit of previously published TEST aggregates at commit `916999c8e54d0eeaa86bd2b09d9d3ceaadea0bb5`. No new benchmark inference, threshold search, training or private image inspection was performed. The source checkpoints are original L_A epoch3 and original L_H epoch1, both selected on FF++ validation after ten downstream epochs. All percentage values below are100 times native fractions.

## 1. The weakness is shared detection transfer, with a large operating-point shift

| Model / already frozen operating point | Celeb-DF AUC % | FPR % | FNR % |
|---|---:|---:|---:|
| L_A / fixed probability0.5 |70.7475|61.0887|15.7002|
| L_H / fixed probability0.5 |71.4039|41.7294|27.3950|
| L_A / stored FF++VAL10% threshold |70.7475|65.2547|13.5282|
| L_H / stored FF++VAL10% threshold |71.4039|64.9221|13.1106|

L_H FF++TEST AUC is98.4889%, so its Celeb-DF AUC is27.0850percentage points lower. Ding's paper reports76.75% on Celeb-DF, a5.3461point gap, under a different training/evaluation protocol. The archived local Ding GDA reproduction has71.2196%; it also struggles. This does not establish equivalence with that reproduction or identify which protocol difference matters.

L_H improves over the matched L_A control by0.6565AUC points. The existing paired recorded-contributor interval is[+0.2273,+1.0849]points,1000/1000usable. It is conditional on these two trained models, not a multi-seed population claim. Thus it would be incorrect to blame the entire Celeb-DF AUC deficit on L_H's fairness objectives. At fixed0.5, L_H reduces false alarms but misses more fakes than L_A; at their respective stored source-validation thresholds, the operating rates are much closer. Threshold adjustment cannot repair the underlying AUC ranking deficit.

The original [Celeb-DF paper](https://arxiv.org/abs/1909.12962) describes improved synthesis and high-quality fakes intended to challenge detectors. That supports domain shift as a plausible explanation for FF++-trained models' weakness. It does not establish which texture, compression, crop or manipulation cue these particular checkpoints use. Private image/score inspection is still required for those causal hypotheses. No new Celeb-DF-tuned training change is justified by this audit.

## 2. The worst supported groups have concrete error patterns

| Manifest group | Real / fake frames | AUC % | FPR % | FNR % |
|---|---:|---:|---:|---:|
| Female-White |3929 /21503|71.9842|42.3517|25.7499|
| Male-Black |710 /1212|59.5457|29.5775|56.6007|
| Female-Others |90 /30|36.5926|76.6667|26.6667|
| Male-White |924 /0|Undefined|43.8312|Undefined|
| Male-Others |60 /0|Undefined|60.0000|Undefined|

L_H fixed0.5 produces2,384false positives among5,713reals and6,231missed fakes among22,745fakes. Male-Black misses686of1,212fakes. Female-Others has69false positives among90reals, and only30fake frames; below-chance AUC there is descriptive, not proof of a labeling error. The aggregate male group AUC is55.4522%, versus71.3829% for the female group. These are inherited benchmark labels, not demographic identities inferred from appearance.

## 3. Sparse and imbalanced coverage strongly affects fairness summaries

21,503of22,745fake frames (94.5395%) are labeled Female-White. All1,212male fake frames are labeled Male-Black. Only three intersectional groups meet both-class support for the main fairness summary; Male-White and Male-Others are real-only. Age and skin labels are entirely unknown in this Celeb-DF manifest, so those fairness axes are unavailable.

The main intersectional F_FPR0.477116 decomposes as follows:

| Eligible group | Absolute group-versus-pooled FPR deviation | Share of reported total |
|---|---:|---:|
| Female-Others |0.349373|73.2260%|
| Female-White |0.006224|1.3044%|
| Male-Black |0.121519|25.4696%|

This explains much of the large intersectional number, but does not excuse the actual high errors. Repeated frames need not represent independent people. The existing connected bootstrap has587/1000usable intersectional draws and609/1000race draws, below the retained900requirement. Main values and limitations must remain visible. Do not remove the sparse group or relax eligibility to improve a score.

Demographic parity also reflects class prevalence. Even hypothetical perfect predictions on these SAME observed counts would have gender F_DP42.5645% and eligible-intersection F_DP59.5510%, because the fake/real proportions differ across groups. These are algebraic illustrations, not model results or replacement metrics. Report class-conditional FPR/FNR alongside F_DP; do not interpret all positive-rate disparity as a detector error difference.

## 4. Integrity checks passed; label truth and paper equivalence are still unverified

The locked evaluation reported28,458unique images, no duplicate-extra rows, zero recorded fit:TEST contributor overlap and zero byte overlap. The exact manifest SHA is `5b4c9b3af2589b90e6e3fe9825857eacac9d4c2517f3226feb80fa7d4e0f4873`. Counts agree with the older `upgrade_tests_s17_20261101T040407Z/protocol.json`; this is not a new L_H-only grouping discrepancy.

The historical local-Ding audit reports equal subgroup names/counts under its explicit male/nonmale→Male/Female mapping. That verifies consistency between local implementations, not biological demographic truth or the authors' exact hidden row inventory. Ding AppendixA lists six Celeb-DF gender×race names, whereas our manifest has five observed groups and three both-class-eligible groups; the appendix does not provide per-group counts. Resolve that denominator/label-convention comparison before claiming fully matched fairness metrics. No labels are changed here.

## Operator audit needed

Use existing private inference caches, not fresh model runs. Verify row order, labels, checkpoint and manifest hashes; trace raw gender/race encodings, the source/target face labeling convention, and all path-to-annotation joins to the canonical benchmark annotation source. Report missing/duplicate joins, per-video/contributor group counts and real-only/absent groups. If canonical annotations are unavailable, mark that limitation rather than fabricating a match.

Inspect small deterministic private sheets of false positives/negatives and same-group correct cases, with contributor diversity. Record crop/occlusion/blur/compression/frame-source issues; do not infer demographic attributes from faces, identify people, relabel samples, remove difficult cases or choose transforms from TEST performance. The sheets and row-level data stay private. Main TEST tables, thresholds and checkpoints remain frozen.

The CPU audit and heatmap reproduce the observed aggregate failure pattern. They cannot establish an image-level causal explanation or certify unseen-domain generalization.
