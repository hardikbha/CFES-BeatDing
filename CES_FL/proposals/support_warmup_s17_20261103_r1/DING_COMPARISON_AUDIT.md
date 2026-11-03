# Ding protocol comparison: reading only, no new Ding-model computation

Observed3November2026. Sources: current official [paper v5,6March2026](https://arxiv.org/pdf/2511.10150v5),§4.1,Tables1–3,AppendicesA/D; local archived official revision059ec37fc969825bab9dc5f79e01c9864d00eaec under DINGREPRO_LIN/code/Ding_mod. The older HTMLv1 was also checked; use pinnedv5 for the current reference. Exact archive file/result hashes are in evidence/COMPARISON_SOURCE_HASHES.json and HISTORICAL_DING_FILES_READ_ONLY.json. No Ding file was executed/imported, no Ding checkpoint loaded, and no new Ding training/scoring/calibration performed. Existing Ding jobs/outputs untouched.

## Published reference results only

The paper reports FF++ training/validation76139/25386 and tests FF++25401,DFDC22857,Celeb-DF28458,DFD9385. Gender/race/intersection are evaluated; FF++/DFDC have8intersectional groups and DFD/Celeb-DF6. Tables1/2 use Xception,Table3 ResNet-50. Its stated optimizer is SGD,lr0.001,batch64,50epochs,λ0.005,ε0.0005. AppendixD alternates channel decoupling and fairness optimization; this does not establish an unambiguous single flat end-to-end epoch budget. Exact compression, checkpoint-selection and threshold/calibration rules are not established by the inspected paper text. [Primary source](https://arxiv.org/pdf/2511.10150v5).

| Published reference, Xception | AUC (%) | Gender F_FPR (%) | Race F_FPR (%) | Intersection F_FPR (%) |
|---|---:|---:|---:|---:|
|FF++ TEST|97.71|0.53|9.29|20.18|
|DFDC TEST|61.86|1.76|17.93|37.37|
|Celeb-DF TEST|76.75|6.41|16.60|23.32|
|DFD TEST|81.46|4.72|2.09|27.32|

These are paper TEST references, never competitors for our VAL pilot. No superiority claim is based on the table. Paper formulas: F_FPR=sum_g|FPR_g−pooledFPR|; binary F_DP=max_gP(fake|g)−min_gP(fake|g); es-AUC=AUC/(1+sum_g|AUC−AUC_g|). Internal fractions must be multiplied by100 once for paper-style display. [Equations8–10 and Tables1/2](https://arxiv.org/pdf/2511.10150v5).

## Archived code/records versus our pilot

The following details come from executable archived source and historical verification records, not an assertion that the paper used that exact release.

| Issue | Archived official/reproduction evidence | Current CES-FL pilot |
|---|---|---|
|Membership|VERIFICATION.md: same Lin public-release counts; only DFDC has author-released row-by-row manifest proof. Others count/provenance verified, not author-row equivalence.|TRAIN/VAL canonical manifest hashes fixed; no new TEST row/image read. Do not claim exact author split from counts alone.|
|Compression|Public crop release; compression generation/provenance not fully documented in these inspected records.|Existing image bytes retained;224bilinear processing. Exact compression matching remains unverified.|
|Preprocessing|train_ori_model and GDA import resnet_default transforms:224resize,mean/std0.5; test.py uses224resize/centercrop branch. Other unused299/256definitions and256FLOP dummy must not be mistaken for the active loader.|224bilinear and0.5normalization. Archived test square crops make geometry similar, but complete interpolation/version equivalence still needs review.|
|Architecture|XceptionDetector with extra adjust_channel and2-output classifier; SFD masks ten512channel outputs at2%, not structural removal.|Our single Xception, global/local MLPs, learned gates and signed contributions; no Ding SFD/GDA algorithm.|
|Initialization|PREREG discloses ImageNet/timm replacement for inaccessible author weights; historical reproduction not exact initialization.|Verified original-only CES-FL P50 encoder checkpoint, selectedepoch29, SHA3174e5b01a84c91cc70afae538505663ca96454d897e6b793c0672826a8f08b5; fresh seed17heads.|
|Classification|Released code default50epochs,SGD0.001,momentum0.9,wd0.005,all model params trainable,model.train updatesBN.|Existing P50 already complete. New downstream AdamW10epochs,exit-only7,575,591trainable,allBNbuffers fixed.|
|Batch/exposure|CombinedDataset length=fake rows, random real partner; batch64pairs gives128images, not64images. Real rows are repeatedly sampled. GDA batch32pairs gives64main images plus group forwards.|Batch32originals,all76139 once/epoch,2380updates/epoch; inverse-label×ids8weights.12926real proceduralpairs/epoch and balanced auxiliary draws.|
|SFD|README classification→SFD→GDA; one archived SFD pass, while paper discusses3iterations. Channel-masking statistics use TRAIN.|None. These are different algorithms; no attempt to execute Ding stages.|
|GDA|Everything frozen except last_linear; model.train may still update BN. GDA.py lines232–235 build auxiliary fairness/total inside no_grad.|Our fairness/sensitivity/alignment are differentiable; source-bound gradient checks required.|
|Historical gradient defect|Saved GDA_GRAD_CHECK.json reports released fair_requires_grad=false, fair_grad_norm0, identical updates for λ0/0.005/1000. The connected variant is a separate archived diagnostic.|No rerun of any Ding diagnostic/corrected variant. Do not label historical released-GDA results as a validated differentiable GDA reproduction.|
|Selection|Historical PREREG finalepoch49; paper selection unspecified. Existing code's per-epoch TEST logging is not adopted.|Earliest highest VAL AUC after ten completeepochs; no fairness-based checkpoint switch.|
|Threshold|Archived test uses softmax0.5; paper threshold unspecified.|One global validation-real empiricalFPR10% threshold, conservative finite-sample ties; fixed0.5 separately reported. These operating points differ.|
|Groups|Gender/race/intersection benchmark flags; age/skin not main published table axes.|Same named gender/race mappings plus inherited age/skin proxies, not newly human-validated ground truth. Eligibility min20perclass disclosed.|
|Support/dependence|DFDC historically contains1476duplicate rows; old release has known cross-split contributor overlap.|Preserve split/audit exceptions, no dedup silently. VAL209recorded connected components; largest360/25386rows. Female-Black30real/120fake occupy only2components. Biological independence not established.|

## Historical reproduced numbers, not new baselines

Historical baseline epoch49 AUCs(seeds7/17/27): FF++89.01/96.32/96.99; DFDC54.41/55.60/54.04; Celeb-DF56.94/69.16/75.32; DFD57.19/68.01/75.21, rounded from existing JSON. Only the archived full released-SFD/GDA seed7 is complete: FF++97.444344,DFDC56.046779,Celeb-DF71.219557,DFD72.537352%. Its fairness term has the documented detached-gradient issue and auxiliary seed behavior. No new matched Ding baseline is claimed. Three classification seeds are not three full-method seeds.

## Development and final evaluation plan

New E/F use FF++ TRAIN/VAL only. Existing A/B/C/D controls share P50,heads,original/auxsampling,processing,10epochs,optimizer,selection and calibration, so objective-package comparisons are applicable; no completed control is retrained. E−oldD changes error control; F−E changes ramp. This does not isolate every internal algebraic modification within E and does not match the paper's staged schedule.

Before any later TEST inference: freeze named selected candidate/checkpoint hash, exact configurations, comparison table and calibration. Audit fit(TRAIN+VAL) versus TEST with the source's explicit fit:test exception schema; the historical six-key audit is not TEST authorization. Verify current TEST manifest membership/duplicate handling and input hashes then, not by peeking at scores. Report FF++ and external DFDC/Celeb-DF/DFD under BOTH frozen validation threshold and fixed0.5, with labels/support/formulas/uncertainty. External TEST is never development VAL. No TEST command is attached to this pilot. Prior broader-project R2/upgrade and historical Ding TEST exposure must be disclosed; this study is not untouched-benchmark discovery.

Table3 requires a separately specified CES-FL ResNet-50 study; the current standalone revision is Xception only. We cannot fill a matched ResNet claim or resolve paper initialization/staging ambiguities by running Ding under this request. No new FairForge-2 or Ding experiment is in scope. A paper-aligned final CES-FL schedule remains separately recorded/compute-gated; P50+10 is a pilot, not equivalence to a different method's50epochs or iterative stages.
