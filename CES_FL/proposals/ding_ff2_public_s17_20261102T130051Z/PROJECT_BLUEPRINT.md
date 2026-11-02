# CES-FL / CESel-MA project blueprint and continuation memory

Recorded 2 November 2026. This is an evolving project memory, not a substitute for executable code, native receipts or timestamped observations. **Latest user instructions override earlier scoped holds; no approval or successful execution may be inferred from a proposal.** Update this file after every implementation, validation, launch, completion, failure or protocol decision, retaining the dated history and exact evidence.

## 1. Vision and motivation

Build a lightweight deepfake detector that preserves detection utility while reducing demographic error disparities across gender, race/intersection, age and skin tone. Target a rigorous CVPR-level paper, with matched comparisons against Ding et al., within FF++ and the team's FairForge-2 public dataset. High AUC alone is insufficient: false accusations on real faces and missed fake images must be measured separately by group. Never promise to beat Ding or infer novelty/generalization from software smoke tests.

Separate three questions: forensic encoder pretraining; fairness-objective effects with matched architecture/data exposure; and the incremental value of validated demographic/manipulation interventions. Fairness evaluation on original held-out images does not require synthetic demographic editing. Synthetic edits are additional training data requiring scientific validity, provenance and review.

## 2. Current authorized task — highest priority

The user explicitly authorized **training Ding's architecture on FairForge-2 public now**, correcting GDA, with the same image/global-batch settings as the archived executable Ding pipeline. Try two GPUs and sufficient CPU workers, provide a continuously readable absolute log path. Latest steering explicitly says **run directly through SSH on gpu01 or gpu02**. This supersedes the previous FF2 hold **only for this new original-only Ding baseline**. It does not authorize CES-FL counterfactual arms, new image banks, relaxed review thresholds or a blanket FF2 expansion.

Active request metadata: `/home/sachin.chaudhary/cesel_ma_audit_logs/DING_FF2_ACTIVE_REQUEST.json`.

- Request: `ding_ff2_public_s17_20261102T130051Z`.
- New source: `/home/sachin.chaudhary/ces_fl_sources/ding_ff2_public_s17_20261102T130051Z`.
- New branch: `run/ding_ff2_public_s17_20261102t130051z`, base `92bbd26938e42c8d74790ea65e7830a79a236c35`.
- New proposal: `CES_FL/proposals/ding_ff2_public_s17_20261102T130051Z`.
- Run/output: `/home/sachin.chaudhary/ces_fl_runs/ding_ff2_public_s17_20261102T130051Z`.
- Master log: `/home/sachin.chaudhary/ces_fl_runs/ding_ff2_public_s17_20261102T130051Z/logs/training.log`.
- Seed17; planned classification50 -> TRAIN-only SFD2% -> corrected class-conditioned GDA50. Architecture unchanged. Previous internal FF2 five-plus-five runs are not this run.
- **Actual status at this blueprint's creation: IMPLEMENTATION_AND_PREFLIGHT. No new Ding training process, CUDA smoke or training job has yet been launched.** The fresh worktree/request/log exist. Full production stage wrapper still needs implementation, tests, frozen source and actual GPU claim. Do not report RUNNING until verified.

The user also requested this exhaustive /compact project memory. Creating it does **not** cancel the authorized launch task; continue the actual work afterward.

## 3. Critical fresh correction: image size and batch semantics

Fresh executable inspection on2November corrects an earlier audit assumption. `train_ori_model.py` imports `src.transform.resnet_default_data_transforms`; GDA/SFD import the top-level equivalent. **Both actual dictionaries resize TRAIN to224x224**, ToTensor, normalization mean/std0.5. The **256x256 tensor in train_ori_model.py/GDA.py is only a FLOPs probe**, not the image loader. Older summaries saying Ding classification training256 are wrong and must not be repeated. Preserve history; publish a dated correction rather than rewriting a locked report.

- Classification CLI batch64 means64real/fake pairs -> **64real+64fake=128 original images per global update**.
- GDA CLI batch32 means **32real+32fake=64 main classification images per global update**, plus a selected group's32real+32fake auxiliary batch (64 more forward images).
- Each epoch visits each fake once; real images are sampled uniformly with replacement. `CombinedDataset.__len__=len(fake)`.
- GDA picks one group uniformly per step; `MergedDataset` independently samples real/fake examples with replacement within it. These are demographic group examples, not synthetic counterfactual pairs.
- Native TEST `test.py` imports `fair_df_default_data_transforms['test']`: aspect-preserving Resize224, CenterCrop224, normalization0.5. Direct224resize during training is a distinct operation.
- Paper states batch64,50epochs,SGDlr.001,lambda.005 and two RTX4090s; its text inspected did not specify224/256. Preserve/disclose actual archive batch semantics instead of silently equating a pair batch with total images.

Planned two-GPU wrapper should keep those **global** counts, not double them per GPU. DDP with SyncBatchNorm during classification can preserve global BN statistics. Corrected GDA head-only training should explicitly specify frozen BN (different from updating archive BN); disclose it. This is a named corrected implementation, not proof of equivalence to the unpublished paper runs.

## 4. Exact datasets and current split evidence

Image/frame-row counts, not videos or unique people:

| Dataset/split | Images | Real | Fake |
|---|---:|---:|---:|
| Ding FF++ TRAIN |76139|12926|63213|
| Ding FF++ VAL |25386|4092|21294|
| Ding FF++ TEST |25401|4538|20863|
| FairForge-2 public TRAIN |218266|16777|201489|
| FairForge-2 public VAL |31107|2438|28669|
| FairForge-2 public TEST |87564|5441|82123|

Adapter manifests: `/home/sachin.chaudhary/ces_fl_pipeline/manifests/ding_ffpp/{train,val,test}.csv` and `/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/{train,val,test}.csv`.
Native FF2 schema: `/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/data_ding_schema/fairforge2_public/`. It has `train.csv,realval.csv,fakeval.csv,test.csv,cal.csv,provenance.csv,BUILD_REPORT.json`; it does **not** currently provide `realtrain.csv/faketrain.csv` or per-group training CSVs. A wrapper must split original TRAIN rows by labels/groups rather than invoke the legacy launcher against absent files or substitute internal data.

Ding external test inventories: DFDC22857(real14699/fake8158), Celeb-DF28458(real5713/fake22745), DFD9385(real3092/fake6293). DFDC preserves1476duplicate excess rows,21381unique images. Their TRAIN/VAL are not used in this FF++ protocol.

Ding/Lin FF++ frame membership is preserved, including Others. Recorded contributor overlaps: TRAIN/VAL163,TRAIN/TEST174,VAL/TEST57. Source/video IDs are proxies, not verified true people. Public FF2 recorded path/contributor separation previously passed, but hidden donors/true-person independence and duplicate content cannot be proved from path checks.

Historical internal FF2 is **different**: TRAIN502035 /VAL74974 /TEST182884. Old57,665-row FF2 subset exports are another inventory. Never relabel these as public87564.

Fresh counts/hashes, native Ding cross-check and class counts: `/home/sachin.chaudhary/cesel_ma_audit_logs/dataset_fairness_tables_20261102T104319Z/SPLITS.json` and committed counterpart below.

## 5. Architecture evolution

### Initial CES idea (historical prototype discussion)

One pretrained encoder plus one binary classifier, frozen encoder initially; BCE; demographic consistency; forensic sensitivity; optional coupled selectivity. No demographic adversarial head or attribution branches. Intended initial separate objective:

- d_demo=|sigmoid(z)-sigmoid(z_demo)|.
- u_for=(2y-1)(z-z_for), correct-class margin reduction after a valid cue-removing intervention.
- BCE +lambda_d mean_valid(d_demo)+lambda_f mean_valid relu(tau-u_for).
- Coupled BCE+lambda_s mean_joint relu(tau+d_demo-u_for). Coupled loss alone does not force invariance.

Probability/logit scale mismatch in coupled comparisons was identified. Asinh score space is a distinct objective with a different margin interpretation, not a guarantee. Frozen encoder requires frozen BN and conditional no_grad; simply toggling requires_grad is insufficient. Historical user messages report CPU/CUDA indexing/calibration fixes passed, but do not infer exact current prototype paths or unseen receipts from those messages.

### Current CES-FL v0.2.1 family

One encoder; global classifier; patch classifier; learned image/patch gates; positive fusion weights plus bias. Signed local map contribution averaged spatially plus global contribution plus bias exactly equals the logit. Gates are **learned weights, not proven reliability**; maps are **signed local contributions, not forgery masks**. Map consistency stays off without spatial validation.

Verified counts: Xception encoder20806952; global head262401; patch head262401; image gate131201; patch gate131201; fusion3; total21594159; frozen-encoder trainable787207. ResNet50 encoder23508032, total24295239, same frozen trainable787207. End-to-end mode trains all parameters with explicitly controlled BN/no_grad/optimizer groups.

Sources: `CES_FL/src/ces_fl_diagram_model_v021.py` SHA0e001dd1251570ce43fb82bde307c519dea4045d9734506e1ce90d2ed014f8de; `ces_fl_counterfactual_5050.py` SHA3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab. Do not modify sources that running jobs read.

## 6. Completed controlled pilots and interpretation

Original seed17 Xception P/A/B/C used10pretraining/10downstream epochs. A=ImageNet-frozen encoder+BCE; B=detection-pretrained frozen encoder+BCE; C=same encoder+demographic invariance. B/C is a comparison, not another arm. Identical fresh heads isolate encoder pretraining or counterfactual effects. C is invariance-only, not full selectivity.

FF++ VAL: A67.9134%,B98.1949%,C98.0661%. Old TEST AUC: B98.3467/59.4712/74.3252/82.3578 onFF++/DFDC/Celeb-DF/DFD; C98.2216/59.6667/73.6992/81.2362. High VAL AUC is not Ding FF++ TEST AUC.

Five approved upgrade arms completed10epochs, source875bce025084871438ef799ca9086985c4375672: R0 frozen BCE,R1 partial BCE,R2 partial DRO,RN no-change diagnostic,RD age diagnostic. RN/RD used36variants of six reviewed Ding TRAIN originals, not broad demographic coverage. R2 TEST AUC98.3962/59.5189/74.6706/82.1211: higher than published Ding onFF++/DFD, lower onDFDC/Celeb-DF; mixed/worse fairness. They are not an apples-to-apples win. Source/dataset/optimizer/selection/BN/threshold differences remain disclosed. Twenty-five frozen-checkpoint evaluations finished; no test tuning/recalibration.

Evidence: `CES_FL/results/upgrade_tests_s17_20261101T040407Z/`, `CES_FL/results/ding_reproduced_vs_cesfl_20261102/`, `CES_FL/results/ding_format_tables_s17_20261102/` (Table1/2/3 andPDF). Old B/R0 identical scores are not independent replications.

Historical user-pasted BD0/CLIP summary reports seed17wins20/40, seed7/17/27counts18/20/16, mean18; preregistered pooled-metric count17/40. These are distinct aggregation rules and an older method, not present CES-FL. Treat that pasted summary as historical until its actual BEAT_DING refs are rechecked. Do not substitute it for current counterfactual results or infer statistical equivalence from three seed values.

## 7. Completed fifty-plus-fifty FF++ study

Source science commit58773bbef9d9fe79dd313610f694ddeea12c6037. Trainer `ces_fl_manipulation_response.py` SHA0e82a522a22e641dc5588b6a15317e6842896ab2bdcc5364833af25e207f257e. Shared original-only P50 selected epoch29, checkpoint SHA3174e5b01a84c91cc70afae538505663ca96454d897e6b793c0672826a8f08b5.

| Seed17 FF++ VAL | AUC% | F_FPR fraction | FPRgap pp | Worst-group AUC% | FNR% | Selected epoch |
|---|---:|---:|---:|---:|---:|---:|
|BCE|98.448762|0.657819251|47.935606|95.032007|3.841458|22/50|
|fair|98.487114|0.684208602|50.619890|94.049637|3.587865|37/50|

AUC+0.038352pp; primary fairness worse as point estimates. No consistent fairness gain or significance claim. Both achieved409/4092VALFP,9.995112%, via VAL-real calibration. Female-Black VAL has30real frames; one FP is3.33pp, only2recorded components/class. Old original sampling: all76139once/epoch, batch16, label×ids8 weighting. No quartets/paired draws. Don't fill the new quota64 matrix with these values.

Root `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/{P,bce,fair}`. Fresh hashes/history/checkpoint-integrity confirmation: `CES_FL/results/cdf_originals_readiness_20261102T095742Z/`. BCE checkpoint71d9b50377d3e0812c2721d9271ca3f4d59aaeedfa798a69853d9bff769bfb0b; fair21a9fedc42be28b261e59c8bcb6f969bb422744453a9e0217fd5a32f6ab6d974.

## 8. FairForge-2 measurements and their roles

New original-only Xception CES-FL P17 on public TRAIN finished10epochs, selected9, native replay PASS. **Public VAL**31107: AUC99.8105359%,FPR9.967186%,FNR0.3802016%. Checkpoint43152de1b98dce48801114efb2454cb0bc6aa7400445dd5f139759022229895a. Root `/home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1/training`. No TEST evaluation for this P17 model. It has no fairness/counterfactual training objective.

VAL fairness, percentage-scaled F_FPR /FPRgap pp /FNRgap pp /worst AUC%:
- Gender3.5980/3.5980/0.1180/99.8080.
- Race12.1565/7.5679/0.1260/99.6840.
- Gender×race28.3857/11.8517/0.2848/99.5827.
- Age9.2329/7.4848/0.1737/99.6982.
- ITA skin11.6827/7.9134/0.2700/99.6977.
- ids834.8666/18.1073/0.5583/99.5209.

Existing FF++->public FF2 TEST transfer (seed17, frozen FF++VAL thresholds), AUC/FPR/FNR%:
R0 41.6819/57.1402/56.8428;R1 41.8710/59.5663/52.5468;R2 42.8294/60.2463/51.5556;RN 38.8312/59.0700/56.6394;RD39.5290/58.2981/58.0434. Retain weak transfer; never invert scores or compare them with in-domain VAL as one experiment.

Historical public CFES-D CLIP ViT-L/14 seed7 TEST AUC98.2597622%, different model/protocol. It has raw0.5, stored VAL threshold and legacy VAL-intercept calibration exports; FPR/FNR trade-offs differ. Root `.../agents/CESEL_MA/runs_ff2public/cfesd_s7/eval`. Historical in-domain Ding base/full99.2619223/99.3572537% TEST is the internal182884build,5+5epochs, not public87564. Native FF++-trained Ding FF2 exports also use filtered57665 or internal182884inventories. Do not relabel any of them public in-domain.

Complete22-page standalone TeX/PDF, class counts, all group/axis metrics, stored CIs and distinct legacy records:
`/home/sachin.chaudhary/cesel_ma_audit_logs/dataset_fairness_tables_20261102T104319Z/ding_and_fairforge2_tables.{tex,pdf}`; repo `CES_FL/results/dataset_fairness_tables_20261102T104319Z/`. Publication92bbd26938e42c8d74790ea65e7830a79a236c35. No new model evaluation was performed for the export; sources are hashed.

## 9. Ding archive audit and corrected-baseline requirements

Archive upstream059ec37fc969825bab9dc5f79e01c9864d00eaec. `DINGREPRO_LIN/code/Ding_mod/` contains preserved sources. GDA.py SHA bef29fa69e46c729dc679356d0ac3904ebac6da4c00180350daf282fe94599db. Actual native Xception detector total21858090: shared ImageNet features plus random2048->512adjustment and512->2head. Classification all params trainable; SFD eval/no optimizer; GDA freezes all except last_linear1026parameters, but native model.train updates123BN buffers in the actual prior probe.

GDA lines232-235 construct group predictions/fairness/total under no_grad. Under GeomLoss0.3.1, CE can still backprop because of grad-mode side effects, but fairness is detached. CPU/CUDA exact AST fixtures confirmed identical head gradients forlambda0/.005/1000. Other dependency versions may fail backward entirely. Removing no_grad alone is not enough: native aligns mixed real/fake two-logit vectors; paperEq6 aligns class-conditioned probability distributions.

Three classification and three SFD/GDA trained checkpoint sets exist, seeds7/17/27. Complete four-domain native full final evaluation export was located only forseed7. Don't say only one full training seed exists. GDA's helper hardcodesseed121; disclose archive results and honor declared seed in corrected variant. Native ImageNet weights were substituted with cached standard weights because Baidu was inaccessible; conditional pointwise2D->4D loading is disclosed.

Pruning is a mask, not structural deletion: ten-of512mask zeroes20480elements and adds1048576mask-buffer elements; no registered parameters removed.

Fresh SFD caveats: native helper uses sequential children[:-2] plus adjust_conv, not literally backbone.features; native SNNL uses cosine distances, includes self in numerator but excludes diagonal in denominator; signed batch-mean ranking is used without abs. The paper equation describes different details. Preserve/disclose archive semantics for this corrected-GDA-only baseline rather than silently fixing extra components. Vectorization/device fixes need equality tests against the archived reference. Ranking n=max(1,round(.02*512))=10, strict<sorted[n]; ties may produce fewer, which must stop/report, not invoke fallback channels.

A prior proposed Ding-CC-OT-r1 loss kernel exists in `ces_fl_cdf_objectives.py`: class-conditioned fake probabilities, epsilon5e-4,GeomLoss0.3.1,p2,blur=sqrt(epsilon),diameter1,debiasFalse,tensorized. Native blur5e-4 means epsilon2.5e-7 and defaultdebiasTrue; these corrections are explicitly distinct. Default cost0.5squared distance and no KDE remain disclosed adaptations. Constant-score auto-diameter failed and was fixed to1. Full production wrapper was **not implemented** in that prior proposal. New user-requested native batches require a new config/variant, not silently reusing its quota/batch64classification spec.

## 10. New CDF/response proposal — prepared, not trained

Branch `proposal/cesfl-tail-response-20261102-r1`; scientific code commit1ffdce6d746482f5de8766dff5622c08d6b3fded. Proposal `CES_FL/proposals/cdf_response_20261102_r1/`. Code: `ces_fl_cdf_study.py` SHAed1ebf081ec96fd524088fa13de0af54aa44d905d1b7f7aab1b7815db3a59f20; objective3f0f47691667f58f3bba25ae17ec6aa3eb705db84beee06f0f8fbd5638b47175; paired uncertainty63e4b7e5fb823bfa48fbb8246124a7e597487f02f46bcf6810fe181925f96c31.

Original mean-soft-score parity at logit0 does not control real-score tails at a VAL-selected threshold. Synthetic equal-means fixture oldloss0 but FPR.5vs0 atlogit1.5; new smoothCDF loss.124998465, gradientnonzero. New realCDF uses detached TRAIN-real knots,temperature.25,fixed naturalTRAINreal priors,max over knots of mean absolute group-versus-pooled soft-survival gaps. No VAL threshold enters training. Finite smoothing/batches do not guarantee population FPR parity.

Response asinh(T(x)-x) can cancel group additive offsets; executed fixture response0,scoreanchor0,butCDF.25. Thus response alone is not group fairness. Combined formulation uses CDF+score consistency+response alignment/sensitivity, primary comparison response-versus-score with identical pair exposure. Resize control is diagnostic only; map consistencyoff. Margin.5 is transformed response, not an age threshold.

Matrix BCE,complete label×gender_race DRO,existing mean-fair,CDF-fair,nochange,score,response. Each backbone/seed shares verified P50, fresh identical heads, downstream50epochs/allparams/BNfrozen. Quota64=4draws×2labels×8groups,1190updates/epoch; it is replacement sampling, not each original once. Paired arms6views×4quartets per step. Xception/R50 configs for8declared seeds exist; initial development7/17/27,confirmation37/47/57/67/77. Only Xception P50seed17 is bound; otherbindingsnull. No new matrix arm trained.

Primary development rule response-vs-score: AUC drop<=.005,F_FPR improvement>=.02,overall and every eligible-group FNR increase<=.02,worst AUC drop<=.01,in each of3seeds. Full fairness/detection superiority requires paired AUCdelta95%lower>0 andF_FPRdelta95%upper<0 versus separately correctedDing,plus guards and multiplicity policy. Three-seed exacttwo-sided signflip minimum p=.25;8seeds don't guarantee power. Existing benchmark TEST has been viewed historically; stronger confirmation needs independent locked replication. Current FF++ VAL component support fails5perclass/groupfloor: FemaleBlack2each,FemaleOthersreal4,MaleBlackfake4. Don't drop groups or relax floor.

Final source-matched smokev6: CPU15tests,14pass+1CUDA-onlyskip;GPU15pass onH100. Real seed17TRAIN backward finite,zerooptimizerupdates. Includes BN/frozen-state, checkpoint reload, score-adapter,class-conditionedOT,lambda effects,offset counterexamples,quota RNG and pairedbootstrap tests. Native P50 compatibility initially failed KeyErrorcompleted_epochs because budget is in summary/history; strict adapter now checks50contiguous epochs/native replay. Private runpy harness errors were corrected; no model protocol change. A later additional check was blocked by pool reservations before CUDA; no false PASS or training claim.

## 11. Counterfactual/editor history and hard data gate

Earlier LLaDA skin edits reportedly passed identity only11.5%; their visual plausibility alone was insufficient. Age/gender editor survey found no2025-2026candidate meeting all release/license/edit requirements. A research-only2024re-aging smoke was authorized; blur/identity failures motivated other editors. User suggested mlarnes/face-editing and later Qwen. These are historical discussion/outcome claims unless their native reports are rechecked; do not invent missing files or approvals.

Qwen eight FairForge originals received explicit visual acceptance. Thirty-six Ding edits of six originals were separately reviewed for exploratory invariance. Those approvals are artifact-specific. The later Qwen32pilot produced224images; the recorded pending224row review file hadzeroapprovals/unnamed reviewers at the verified snapshot. It supplies x0/xa candidates, not matched T0/Ta manipulation quartets. No current bank can be inferred from these approvals.

Hard gate for proposed response: TRAIN-only real originals; unchanged source, actual-pipeline resize control, editor no-change, independent younger/older edits, matched T(x0)/T(xa). Same editor/model/settings/geometry/seed, same operator/donor/settings/seed and verified parent hashes.320originals,40per8groups,both directions ->640quartets;2validated operator families,20sources/group/operator;>=5components/group,<=8frames/component;two independent reviewers with substantive evidence and all six asset hashes. Heldout source/donor exclusions via metadata receipt; shared donor links cannot be split into fictitious independent components. Source-real unrelated pairs are not valid forensic interventions.

Generative age editing is itself synthetic manipulation and may change authenticity. Identity/age attainment is not sufficient; defend the task-specific label/evidence-preservation assumption independently or reject it. Model predictions are not demographic truth. Validity flags don't establish science. If a bank cannot be qualified, response hypothesis is untestable and only original-only baselines may proceed. Never scale generation or lower coverage to manufacture validity.

## 12. Historical preregistered gate

Original CESel-MA evidence-floor gate remains FAIL. `CESEL_MA/results/gate.json`, spread2.06. Full or+floor must beatB0by>2.06 and differ fromeverynegativecontrolby>2.06, using registered VALselection only. B085.623771,Full84.063575(diff-1.560196),floor84.235975(diff-1.387797); relevant negative controls remain withinspread. The +sens screening winner is not an eligible passing gatearm. Earlier Amendment9val_oppoint=AUC-.5FPRgap with operating/referenceguards differs from new AUC-only selection. Do not rewrite preregistration, relax gate or present CES-FL/new Ding as a passed old expansion. Pre-gate computation of some old test exports was documented; don't claim universal historical blindness.

## 13. Metrics, calibration, uncertainty and reporting rules

y1=fake,y0=real; Ding binary margin=z_fake-z_real, not fake logit alone. Overall AUC,gender/race/intersection F_FPR,F_DP,esAUC; FPR/FNR separately,worst-group AUC/errors,age/skin/ids8where defensible,forgeryfamilycounts. F_FPR=sum_g|FPR_g-FPR_all|,different frommax-minFPRgap. F_DP=max-minpositive prediction rate. esAUC=AUC/(1+sumabs groupAUC deviations). Native legacy JSON aggregate percentages differ from modern fraction dictionaries; never multiply twice. Undefined metrics remain null/dash.

Global VAL-real FPR-budget rule: nreal,k=floor(target*n),descending scores,threshold=nextafter(score[k],+inf),fakeifscore>=threshold. Tied blocks excluded together; achievedFP<=k may undershoot target. Freeze threshold/checkpoint beforeTEST. Report actualFP/n; no requested TESTFPRguarantee. Fixed probability.5 is separate, not substituted after results. No per-group threshold silently introduced.

Common eligible aggregates need20real+20fake/group; show all supports/unknowns. Recorded contributor connected components, not frames/editedvariants, are bootstrapunits. Existing single-seed1000rep intervals do not quantify seed variance/paired differences. New paired helper uses shared seed/component draws2000repsseed1729,intervalsconditionalonselectedmodels/calibration; <2componentsnoCI, <20global or<5classgroup or<90%validreplicates no stronginferentialclaim. FrameN is not peopleN. No test checkpoint/threshold/lossweight/method selection.

## 14. Resource and operational rules

No preemption/termination of existing jobs, no silent overwrite, no model downloads/largegeneration without scope. Fresh immutable source checkout for launches. Do not pull over active job sources. Persistent logs/receipts with exact commit/config/input/checkpoint hashes,PID/device/UUID,monotonic timings; node clocks differ from management by about11min. OldSTATUSfiles are snapshots.

User explicitly permits direct SSH. Do not claim a PBS ngpus allocation on that route. If usingPBS, request actualngpus and verify assignment/binding; both gpu/workq max_user_run2currently occupied in observed scheduler. Direct CPU16means a bounded worker/thread budget, not a fabricated exclusive reservation.

Existing cooperative GPU pool supports sharing **light/reader detector tasks** with editors when reservations fit. Read-only protocol source: `/home/sachin.chaudhary/TruFor/formatted_datadeclare/paper/transaction_paper/ICLR_Misinfo/scripts/factmux_pool_v1.py` around414-545. Registry `/tmp/scf_gpu_locks/gpuN`, shared ownerpool_owner.json, task_<name>.json, atomic host mutex`.pool_mutex`. Account free VRAM minus unconsumed reservations, need+4096MiB margin. Never join an exclusive/legacy lock. Two-GPU claims must be atomic under the same mutex. Own livecontrollerPID and torchrunPGID protect claims; cleanup only own entries. Existing workers preserve live unknown entries. Never drop others' leases or stale-break their mutex. A persistent resource waiter may wait/report; only claim RUNNING after real steps.

Planned2GPUbudget: ~20,000MiB reserved perGPU, per-rank Torch allocatorcap<=reservation,12DataLoaderworkers+2OMPthreads/rank (~16CPUconcurrency). Verify actual full globalbatch memory and NCCL/SyncBN using disposable real-TRAIN CUDA smoke before full training. Stop on failed checks, preserve failures. Recheck current resources; snapshots do not reserve devices.

## 15. Collaboration, GitHub and artifacts

Repository https://github.com/hardikbha/CFES-BeatDing. hardikbha account admin was previously verified; no access changes needed. User wants supervisor-visible code/results/log updates and a shareable handoff. Push exact scoped branches/aggregate evidence; no dataset faces, credentials, raw per-image predictions or binary checkpoints. PDFreports are permitted. No Slack/email messages or third-party notifications without explicit authorization. No new automation that starts unapproved arms. Status observers are read-only, finite, do not change config/kill/retry/train.

Key commits:875bce0controlledupgrade;58773bb50+50response architecture/trainer;1ffdce6prospectiveCDFstudy;36790feexistingresult/readinesssnapshot;92bbd26split/fairnessTeXexport. Main last fetched67d2308225e422a91c715c40427f40c3df6812cf on2November; onlyFF2live-status changed since task base d744851. This is a dated observation, not a claim main can never advance.

Logs and source maps: `/home/sachin.chaudhary/cesel_ma_audit_logs/`; `CES_FL/README.md,STATUS.md,WORKFLOW.md,AGENTS.md`; handoffs and proposals dated byrequest. Blueprint should be copied into the current new source handoff/proposal and updated with actual tests/launch rather than held as a chat-only summary.

## 16. Prior work and scoped novelty

Primary references: Ding CVPR2026 arxiv2511.10150v5/code ywh1093/Fairness-Optimization; Sagawa group-DRO arxiv1911.08731/code kohpangwei/group_DRO; JuWACV2024DAG/DAW/code littlejuyan/DF_Fairness; LinCVPR2024PG-FDD/code Purdue-M2/Fairness-Generalization; FairAlignTMLR2025githubMkolahdoozi/FairAlign; DAIDNeurIPS2025 officialproceedings/codeiLearn-Lab/NeurIPS25-DAID; HouMM2025individualfairness arxiv2507.14326v2; FairForensicsAug2026arxiv2608.01661 (notFairForge-2); FFTJune2026arxiv2606.09881; FYC27Oct2026arxiv2610.37882; Hardt1610.02413; Pleiss1709.02012; GeomLoss officialAPI blur^p=epsilon, defaultdebiasTrue.

Do not claim first group alignment, rebalancing,FPR-aware objective, manipulation-aware/individual fairness or a novel backbone. Possible contribution: incremental effect of independently validated matched demographic×manipulation-response constraints beyond strong CDF/no-change/score controls with one global threshold and matched exposure. It is an untested hypothesis. Strong superiority requires matched baselines,utility/FNRguards,paireduncertainty,adequate sources and independent replication. No promise of beating published numbers.

## 17. Roadmap, unresolved questions and immediate continuation

1. **Continue the newly authorized Ding public baseline now.** Implement the missing source-pinned native-architecture twoGPU classification/SFD/corrected-GDA wrapper,CPUtests,inputpreflight,CUDAglobalbatch/backward/BN/adaptertests,thenlaunchonSSHgpu01/02 with honestcooperativeclaims. ReportPID,UUIDs,exactlog,actualphase. Preserve original pipeline sources and all old outputs. Freeze a new versioned config; native image/batch user settings override the older unlaunched CDF comparator spec.
2. Record and publish every milestone. Final evaluation may only use a prespecified locked selected checkpoint/VALthreshold; never auto-select methods/test-best epochs. The recent user accepted the train-and-evaluate baseline workflow, but no TEST data may affect training/selection/calibration.
3. Later original-only CES controls on3seeds,otherPbindings,ResNetmatchingTable3; separate releases and no duplicate P17 rerun. Compare losses under matched encoder/head/sampler/budget.
4. Human-review demographics,labels/support; qualify small operation/edit pilot beforebank generation; no training invalid syntheticimages. Original real-image age/skin evaluation remains possible without synthetic agegeneration.
5. If matchedquartets pass,compare nochange/score/response under identicalpairpool/exposure. Test global offset blindspots, held-out manipulation families,operator confounds and preservationchecks. If not, reportresponsehypothesisuntested.
6. Establish adequately supported independently locked replication,atleast3developmentandprespecifiedconfirmationseeds,pairedstatistics. Separate paper/nativearchive/correctedbaseline/ourmethod rows and VALvsTEST vs transfer vs internal builds.

Open questions: Can nuisance edits preserve the defined authenticity task? Is the new CDF package helpful beyond temperature/priorchanges? Does responseaddanything beyondscoreconsistency? How much performance comes frompretraining/capacity versusfairnessloss? How sensitive are tails tosmallgroups/proxylabels? Does alteredSFDsemantics materially changebaseline? Can sameglobalFPRreducegroupgapswithoutinflatingfakeFNR? What is usable independently validated edit throughput perGPU-hour? All need prospective experiments, not assertions.

### Dated continuation log

-2Nov2026: completedsourceaudit/CDFproposal and CPU/CUDA softwarechecks; no newfullcontroltraining. PreservedGDAarchive and failedoldgate.
-2Nov2026: priorauthorizedFF2P17finished10epochs;lateruserallowednewcorrectedDingpublictraining specifically.
-2Nov2026: freshsplitcounts and22pageTeX/PDF published92bbd26; allprotocolrolesseparated, no newmodelinference.
-2Nov2026: userrequestedDingpublictraining,twoGPU,nativeframes/batches,persistentlog;thenexplicitlydirectSSH. Newrequest/worktree/logcreatedat13:01UTC;productionwrapper/testing/launch still outstanding as ofthisentry. Freshsourceinspectioncorrects256FLOPs-versus224loaderconfusion. Next agent must **do the authorized implementation/checks/launch**, not merely re-summarize or ask permission again.
