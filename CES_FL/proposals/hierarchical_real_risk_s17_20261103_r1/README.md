# One CES-FL hierarchical original-real risk experiment, seed 17

This is a new dated proposal under the user's3November2026 authorization. It does not amend E/F, the old CESel-MA gate or any completed protocol. Only H_joint_real may run, once, on the unchanged FF++ TRAIN/validation inputs. No Ding-model computation, TEST access, new dataset/backbone/seed, Qwen generation, sampler, warmup, gradient surgery or full edit bank is authorized. Existing jobs and immutable sources are preserved.

## Evidence reviewed before model revision

Reviewed completed D source c332fd34fb603b8919b15fee08e8d022fb9f57c8 / SHA89530025a00b9b0405fba7026d3eed5f2749d86620174ad3ffde93a44f8b770b; completed E/F source8d5f0996a3f1cc02fbeaff9d25c1d95f714e8e77 / SHA0eeafd838da3ca158336dcfe3b8195842dd15bc9fecd67ebbaf2ea03ba3c2680; latest published aggregate206d5eed960bd2c727ec810e8ed0b92c6d825a29. Current main67d2308 is an older snapshot. Actual native checkpoints, histories, configurations and final uncertainty results were inspected, not just README prose.

D validation predictions are the exact requested private file. Checkpoint7d9409e2cb6aa00fdd61cd3a072bd5473848be6dd86aafcc3b7b740da9f25d9d, selectedepoch1 after ten complete epochs; stored logit threshold−4.397897720336913 is unchanged. All25,386 image hashes, canonical manifest metadata/order, checkpoint/source/manifest hashes and full aggregate metrics pass. Upstream adopted Lin/Ding realval/fakeval membership and intersection labels also match row-for-row; this is not independent human truth or proof of exact author row membership. The supplied ZIP was absent in the inspected workspace/repository; the new exporter is explicitly independently written.

Private sheets contain40cells, four errors/four same-group correct frames per requested stratum, diverse recorded contributors where available. Counts: Male-White real193/1264FP; Female-Black real9/30FP; age60+real24/63FP; Male-Others fake112/1170FN; age45–59fake220/2398FN. Five sheets were actually inspected. Similar crop/blur/compression/cosmetics/pose variations occur on both correct and error sides, sometimes within the same recorded component; there is no verified single quality cause. PNG containers do not establish upstream compression. No gross empty/black crop or byte/threshold/provenance discrepancy was found. Demographics were never inferred or relabeled from appearance. Age/skin remain proxies. Identity/source records are not verified biological identities; the unchanged audit includes163TRAIN/VALexceptions. This supports a benchmark-annotation-relative experiment, not identity-held-out or true-demographic fairness claims.

E/F completed ten epochs and their authorized1000replicate paired analysis completed (CPUjob45259). Both reject under their frozen rules: primary F_FPR worsened, age safeguard failed, and E also failed gender. E−D primary delta+0.088936,95%CI[−0.011646,0.154790]; F−D+0.116232,[−0.004303,0.144386]. Only746/1000primary replicates are usable. Source/checkpoint hashes and exact safeguards are in evidence/E_F_COMPLETED_REVIEW.json. No duplicate E/F run is started.

## Hypothesis and exact single change

Matching procedural responses can cancel a group-specific additive original-image score offset. Joint control of original-real intersection and parent-gender risks may reduce false-alarm disparity while retaining sensitivity. This is a hypothesis, not a novelty or improvement guarantee.

The narrow implementation replaces ONLY the real fairness summand on existing gender_race auxiliary events. Other axes' real penalties, including the standalone gender event, stay exactly D. This intentionally preserves the other-axis constraints and sampling, but their two-sided gradients can still oppose the new term. Fake-image error control stays exactly D on every event, including attached-reference gradients. No E/F contributor discount or ramp is used.

Let j index the eight supported gender×race groups, s the two inherited parent-gender labels. z_jk are ORIGINAL REAL auxiliary logits, not sham/treated logits. Existing t_TRAIN is calibrated on the fixed TRAIN-real subset each epoch; T=1 is unchanged. Define:

```
r_j = mean_real_in_joint_j sigmoid((z-t_TRAIN)/T)
pi_j = N_j,real / sum_k N_k,real                         # TRAIN only
q_j|s = N_j,real / sum_k:parent(k)=s N_k,real            # TRAIN only
R_s = sum_j:parent(j)=s q_j|s * r_j
b = stopgrad(sum_j pi_j*r_j)

H = 0.5*(2*mean_j relu(r_j-b)) + 0.5*(2*mean_s relu(R_s-b))
F_j = mean_fake_in_joint_j (1-sigmoid((z-t_TRAIN)/T))
b_f = sum_j pi_j,fake*F_j                               # NOT detached
L_fair,new = 0.5*H + 0.5*mean_j abs(F_j-b_f)
```

Thus both hierarchy levels have equal0.5 coefficients within the real component; real/fake retain0.5/0.5 class mass. The factor2 matches one-sided versus two-sided deviation under equal group priors; with natural priors this is a declared normalization choice, not exact equality. Parent risks use actual TRAIN within-gender race proportions, never uniform race averaging or fake counts. Means over j and s are equal-group means. The real reference is detached; its direct gradients cannot raise lower-risk real outputs. A low-risk child can still be pushed down if its parent risk exceeds the reference. Shared model/other losses can still worsen a group's outcomes; there is no guarantee about hard FPR tails.

Full objective keeps lambda_fair0.1, auxiliary_frequency_compensation4, lambda_sensitivity0.1, lambda_alignment0.1, sham_weight0.5, margin0.5 and all existing procedural response/asinh/control definitions. Other-axis losses are bitwise unchanged in the tested native-step fixture. Fake-score gradients match the D legacy term. Metadata adds deterministic parent codes/real-count proportions; it adds no image, forward branch, sampler, random draw or trainable parameter. Complete TRAIN joint coverage and consistent inherited parent labels are checked; sparse/missing joint cells fail rather than being dropped.

## Exact unchanged training protocol

H_joint_real_template.json is derived from the actual D config. Differences are source/format binding, explicit error_control=hierarchical_real, and alignment_warmup_epochs=0 (the same constant alignment as D). P50 initialization is NOT the trained D model: it is the same original-only pretraining checkpoint SHA3174e5b01a84c91cc70afae538505663ca96454d897e6b793c0672826a8f08b5. Heads/gates/fusion are fresh identical seed17 tensors. Xception/global/local/gates/fusion unchanged:21,594,159total/7,575,591trainable, exit blocks, frozen BN buffers.

TRAIN76139(real12926/fake63213), VAL25386(real4092/fake21294). TRAIN SHA6ee9326fe8361dc3aee0f803c49b5eada169be5cf3bc3cb44ad3d934ba5ca657; VAL SHA52f4844cffe5a03adb08cecde8829244bb50e05eeead36f6d659599f1c24f5b1; original six-key audit SHA793db56c30c85dd654c6d68745afc926f4b1f8dc8bec749db01b07b3aa306c97. TEST declarations remain deferred; no TEST data is opened.

Exactly ten epochs,2380original updates/epoch,batch32,workers4,224bilinear/0.5normalization,bf16TRAIN/float32VAL,AdamW(head1e-3,exit1e-5,wd1e-4),clip5, original ids8 inverse-cell weighting; all original rows BCE once/epoch. Aux axes/order, every4steps,4perclass/group, operators/strengths/nuisances, original-real pair coverage and TRAIN-calibration2048 stay D. No validation image contributes gradients or group weights. No map-consistency objective is added. Gates are learned weights and maps signed local contributions.

Selection remains earliest strict highest validation AUC after all ten epochs, never a fairness-selected checkpoint. Each selected model uses the same conservative finite-sample global validation-real FPR10% calibration/tie rule; D's already-stored threshold is not recalibrated. Expected empirical allowedFP409/4092 is a finite-sample budget, not a guaranteed TEST operating point. Fixed0.5 diagnostics are reported separately. There is no TEST command or automatic progression.

## Frozen criteria and uncertainty

Exact decision_protocol.json is fixed before training. Gender F_FPR must be at most0.042798380714166834 (20% below D0.05349797589270854), stricter than the retained E/F gender ceiling0.04869892586415193. Intersection must be strictly below D0.3793453731153111. AUC must be at least0.9792641103497183 (oldA−0.005); age F_FPR≤0.286732979753881; pooledfakeFNR≤0.04209965248426786; worst gender/age/intersection AUC≥0.9654828015666387/0.9706360894352599/0.93120498515896. All retained safeguards are unchanged or stricter. Fractions versus percentages are explicitly labeled.

Use1000paired connected-recorded-contributor bootstrap draws,seed17,frozen thresholds and eligible groups; report all per-group sample counts, FPR/FNR/AUC, worst groups, gaps and intervals. No independent-frame or training-seed uncertainty claim. Minimum900usable primary replicates and paired primary95%CI entirely below0 remain required. The existing746/1000usable rate is membership/eligibility-driven; under the unchanged resampling it can prevent PASS regardless of improved point metrics. This limitation is declared BEFORE launch, not hidden or repaired by relaxing the rule. The single numerical hypothesis remains measurable; all-point-guard success with unusable uncertainty is INCONCLUSIVE, any numerical safeguard failure is FAIL. No automatic retry, third arm, weight tuning, threshold change or TEST evaluation follows.

## Checks, resource cap and launch conditions

Required: completed failure audit and E/F analysis, source/input-hash binding, fresh full CPU preflight, genuine source-bound procedural qualification, CPU unit/full smoke, CUDA full smoke and production-style TRAIN-threshold real backward checks. Verify exact fake-gradient preservation, parent weights, sparse failure behavior, original exposure, unchanged components/init and no VAL/TEST gradients. Older-source receipts cannot certify this version. No human or real-forgery validity is fabricated. AI recipe review is limited representative procedural qualification, not a validated demographic edit bank.

Original shared24TOTALGPU-hour ledger currently12.43608249used/0reserved. This proposal caps ALL new GPU checks/training/replay at4.0GPUhours, leaving at least7.56hours unspent if the cap is reached. Estimates: checks~0.1h, ten-epoch training~1–1.6h at observed E/F throughput, possibly~3h under contention. All task-held walltime is charged even on shared GPUs. No budget extension is requested. Same native2h epoch-boundary allowance is retained; a documented compatible same-source/config/qualification/output resume to cumulative3h may finish the SAME ten epochs within the4h cap. Incomplete/failed runs remain labeled; no new configuration/retry.

Use a legitimate allocation or the user's existing explicitly authorized pool-v2 shared UUID route, preserving other jobs and Ding device leases. Low utilization is not ownership. Before any new CUDA command verify source bytes, detached HEAD, clean-checkout evidence and actual single-device binding. Publish exact commits/config hashes/receipts/compute and only aggregate evidence. Private images, reviews, row scores and checkpoints stay on cluster. D-versus-H heatmaps and private H failures are generated only from fixed selected validation predictions after completion.
