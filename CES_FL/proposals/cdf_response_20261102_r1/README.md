# CES-FL: real-score distribution parity and controlled manipulation response

**Research release: DO NOT RUN the full study. Software checks are separate from scientific/data readiness.**

New proposal, 2 November 2026. Base main: `d744851e6fd0c13c2ab7095b47a8b19600008653`, observed at management UTC `2026-11-02T05:38:42Z`. Branch: `proposal/cesfl-tail-response-20261102-r1`. See [source/input hashes](SOURCE_MANIFEST.json), [observation evidence](evidence/), [protocol](PREREGISTRATION.md), [operator commands](GPU_OPERATOR_PROMPT.md), [data gate](DATA_GATE.md), [Ding audit](DING_BASELINE.md), and [primary literature](RELATED_WORK.md). Status pages on main are dated snapshots; this proposal does not certify their older RUNNING records.

## A. Supported by current evidence

The completed Xception seed-17 original-only study used 50 pretraining and 50 downstream epochs. Its selected **FF++ validation** results are:

| Metric | BCE | Existing fair | Fair minus BCE |
|---|---:|---:|---:|
| Overall AUC (%) | 98.448762 | 98.487114 | +0.038352 pp |
| Gender×race F_FPR (fraction, lower better) | 0.657819251 | 0.684208602 | +0.026389352 |
| Gender×race FPR gap (pp) | 47.935606 | 50.619890 | +2.684284 |
| Worst-group AUC (%) | 95.032007 | 94.049637 | −0.982370 pp |

This is **no primary fairness win**. Each model's own validation-real calibration achieved 409/4092 false positives (9.995112%); the test FPR is not guaranteed. The native selected-checkpoint replay passed; this task verified input/checkpoint compatibility rather than running another full validation inference. [Measured report, checkpoints and histories](../../results/manip_response_originals_s17_20261102/REPORT.md).

The earlier R2 Xception study used ten downstream epochs, not 50. Existing TEST reports give AUC 98.3962/59.5189/74.6706/82.1211% on FF++/DFDC/Celeb-DF/DFD. Against Ding's published 97.71/61.86/76.75/81.46%, detection is mixed and fairness is mixed or worse. These are historical report values, not fresh test inference; they did not select the new hyperparameters. [Historical comparison and mismatches](../../results/ding_reproduced_vs_cesfl_20261102/REPORT.md).

**Correction to the initial seed premise:** classification and SFD/GDA completion artifacts exist for seeds 7, 17 and 27. A complete four-domain native final-checkpoint evaluation export was found only for full seed 7. Three trained checkpoints do not provide three verified evaluation results. The archived GDA fairness path is detached; its results cannot stand in for a differentiable class-conditioned method. [Stage audit](DING_BASELINE.md).

The current 224-row Qwen review file contains zero recorded approvals and zero named reviewers. Generated images are not complete manipulation quartets or evidence of preserved authenticity. No paired response arm has run. Earlier six-source human approvals apply to their exact artifacts, not this proposed 320-source bank.

## B. Highest-priority gap and next experiment

The current surrogate equalizes real-image **mean** sigmoid scores at logit zero. Evaluation measures **tails** at one global, validation-calibrated threshold. Equal means need not yield equal FPRs. A TRAIN-only synthetic fixture demonstrates old mean loss = 0 while group FPRs at logit 1.5 are 0.5 and 0. The new smooth real-score CDF loss is 0.124998465 and produces nonzero gradients on that fixture.

Response alignment also has a separate blind spot: adding a group-specific constant to both scores of an operation leaves the response unchanged. The executed fixture has response loss = 0, demographic score-anchor loss = 0, yet new CDF loss = 0.25. Thus response alignment alone is not a group-fairness guarantee. [Loss source](../../src/ces_fl_cdf_objectives.py) and [executed fixtures](evidence/CPU_RESULT.json).

**Recommend one primary hypothesis:** with a valid matched bank, adding operation-response alignment plus sensitivity to the *same* CDF-and-score-consistency model improves gender×race F_FPR at the frozen global operating point, subject to prespecified AUC/FNR guards. Response versus score is the primary method ablation. First release only the valid non-counterfactual controls—BCE, complete group-DRO, existing mean-score fairness, and CDF fairness—on three matched seeds. This tests the objective mismatch without waiting for synthetic edits. Paired arms remain blocked; their hypothesis cannot currently be tested.

## C. Exact protocol gaps with Ding Tables 1–3

| Comparison | Preserved fact | Gap / required interpretation |
|---|---|---|
| Table 1, FF++ Xception | Supplied TRAIN/VAL frame membership matches the archived Ding protocol | Historical CES ten-epoch budgets, direct 224 resize, AdamW, heads/gates and validation selection differ from Ding classification256/evaluation resize+crop, two-logit head, SGD and native final checkpoint. Same row membership is insufficient for an apples-to-apples claim. |
| Table 2, cross-domain Xception | Existing reports cover DFDC, Celeb-DF, DFD | Same issues plus domain shift, differing calibration/support policies and one complete archived full-seed export. A numerical 0.5 cutoff does not impose the same FPR. No target-domain recalibration or hyperparameter choice. |
| Table 3, ResNet-50 | New configuration/software supports ResNet-50 | A complete local native Ding ResNet reproduction was not located in the inspected archive. Do not copy published numbers into a reproduced-results row. Exact stage wrapper, checkpoints and common evaluation remain prerequisites. |

Keep **paper**, **archive-faithful reproduction**, **Ding-CC-OT-r1 adaptation**, and **CES-FL measured results** in separate labeled rows. Report native preprocessing separately from the prospective common-protocol comparison; never silently replace it after seeing test scores. [Ding paper](https://arxiv.org/html/2511.10150v5), Tables 1–3 and Eq.6; [local comparison](../../results/ding_reproduced_vs_cesfl_20261102/REPORT.md).

## D. Gap analysis and severity

| Severity | Finding | Action / scientific consequence |
|---|---|---|
| HIGH | Detached archived GDA fairness; updating BatchNorm despite head-only gradients | Preserve archive; use explicitly named corrected comparator. A loss kernel is tested, its production detector/SFD wrapper is still a blocker. |
| HIGH | Mean-score objective and deployed FPR disagree | New real-CDF objective in separate files/configs; old studies unchanged. Smoothing and finite batches still prevent a fairness guarantee. |
| HIGH | Response alignment cancels score offsets | Add no claim of standalone fairness; compare response against the identical CDF+score arm. Test the cancellation explicitly. |
| HIGH | No approved matched manipulation bank; tiny historical edit pool | Hard TRAIN-only review/coverage gate. No bank generation authorized here. Six originals and many variants are not population coverage. |
| HIGH | Generative age editing may itself change the authenticity label | Define the task and independently defend nuisance-label/evidence preservation. Same identity or a realistic-looking Qwen output does not establish a valid counterfactual. |
| HIGH | Benchmark contributor dependence | Fresh preflight finds zero image-path overlap but 163 recorded TRAIN/VAL identity/source overlaps. Existing declaration records 174 TRAIN/TEST and 57 VAL/TEST overlaps. Those are source-video proxies, not verified people. No person-disjoint claim; new bank excludes held-out contributors; stronger confirmation needs independently locked provenance. |
| HIGH | Three seeds and small groups do not establish superiority | Three seeds for validation development; eight prespecified seeds for confirmation. Report component-aware paired uncertainty, invalid replicates and class/group support. With three seeds exact two-sided sign-flip p cannot be below 0.25. |
| HIGH | Confirmation support floor is not met by current VAL metadata | VAL has209recorded components overall, but Female-Black has2perclass, Female-Others real4 and Male-Black fake4. Do not drop these groups or lower the fixed5component floor. Current benchmark development remains exploratory; confirmation needs independently locked adequate support. |
| MEDIUM | AUC-only checkpoint choice need not improve fairness | Preserve AUC-only selection for all arms; fairness is an outcome and a preregistered go/no-go rule, not an after-the-fact checkpoint selector. |
| HIGH | Demographic proxies and missingness | Inventory by class/axis; human review separate from original labels. No inferred age/skin/race label is ground truth. Female-Black VAL has only 30 real frames: one FP changes FPR by 3.33 pp. |
| HIGH | Benchmarks have already been viewed historically | This is prospective code/protocol, not a previously unseen benchmark. Freeze future evaluation once; qualify claims and require independent locked replication. |
| HIGH | FairForge-2 full review remains blocked | Let only the already-authorized original-only PID2525928 finish, per user's explicit decision. No follow-on, test evaluation, Ding training or synthetic arm. Internal/full-split historical numbers and FF++→FF2 transfer are separate. |

Gender, race, gender×race, age and skin evaluation remain distinct. The proposed edit experiment is **age-direction only**; it does not demonstrate gender- or skin-edit validity. [TRAIN/VAL metadata inventory](evidence/TRAIN_VAL_metadata_inventory.json). A final per-group report must state unknown/excluded support, proxy definitions, real/fake counts, forgery families and recorded components.

## E. Readiness, code and novelty

The historical architecture is unchanged: one encoder, global and patch prediction heads, learned image/patch weights and signed local contributions. Gates are **learned weights**, not calibrated reliability. Maps are **signed local contributions**, not forgery masks. Map consistency remains disabled.

Measured parameter totals are Xception 21,594,159 and ResNet-50 24,295,239; both have 787,207 trainable parameters when the encoder is frozen. The new downstream proposal explicitly trains all parameters with frozen encoder BN buffers; P50 updates BN. Software tests cover gradients, frozen state, logit decomposition, reload, quota matching, missing-group rejection, class-conditioned OT and validation/test isolation.

**No claim of a novel backbone, first group alignment, first rebalancing, first FPR-aware detector, or first manipulation-aware individual fairness is supported.** A defensible possible contribution is the incremental effect of independently validated, matched demographic×operation response constraints beyond CDF fairness, editor no-change and score consistency under a single deployment threshold. It requires the bank and actual multi-seed effects; it is currently a hypothesis. [Prior-work positioning](RELATED_WORK.md).

Full-study verdict is **DO NOT RUN**. Non-counterfactual code has software PASS and one real-input preflight/backward check as recorded in evidence; release still needs supervisor review and per-seed P bindings. Paired arms additionally need the bank. A full corrected Ding comparison additionally needs the production stage wrapper/SFD binding and its own real-input checks. No new full training, TEST data reading or image generation was performed by this task.

During this audit the previously running FairForge-2 original-only job finished10epochs. Native DONE and RUN_VERIFIED are present; [final observation](evidence/FINAL_JOB_OBSERVATION.json). No downstream FF2 or TEST job was started. [VAL component support](evidence/VAL_COMPONENT_SUPPORT.json) is a fresh metadata-only calculation; contributor proxies do not prove true-person independence.

The previous CESel-MA gate remains FAIL (`CESEL_MA/results/gate.json`, spread2.06): Full or +floor must beat B0 by >2.06 and differ from **every** negative control by >2.06 in the registered validation selection value. Full−B0=−1.560196 and +floor−B0=−1.387797; both also fail separation from relevant negative controls. That older Amendment-9 `val_oppoint` selector is AUC−0.5FPRgap with its registered operating-point/reference guards, not the new proposal's AUC-only selector. This is a separate prospective study, not a reinterpretation, promotion or relaxation of that gate. See [execution evidence](SMOKE_AND_EXECUTION.md) for actual commands, failures, timings, hashes, compute limits and resource observations.
