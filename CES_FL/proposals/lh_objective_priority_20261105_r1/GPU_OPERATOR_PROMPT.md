# Priority training and diagnosis after Tables1–3 review

Fetch the supplied supervisor commit from https://github.com/hardiksharma6555/CFES-BeatDing.git into a new detached worktree. Read CES_FL/AGENTS.md,WORKFLOW.md,this request and the pinned source. Preserve all active jobs/completed results. This is FF++ source-only development, not a new FairForge-2 baseline run.60/80 is an aspiration, never a selection objective against already-seen TEST values.

## First: original-image fairness ablation, then component attribution

Run B0 FIRST: native armB, three epochs, BCE plus original-image hierarchical error control. Then run C0: native armC, adding the original procedural sensitivity/sham terms but no response alignment. Reuse the verified A0 and H0 three-epoch controls from `/home/sachin.chaudhary/ces_fl_runs/lh_nuisance_pilot_20261104_r1/`, result commit e993315a853aa278ec649010fa11daef1e601e44. Do not substitute their10epoch counterparts or rerun already equivalent completed B/C arms.

The copied scientific source is byte-identical to84ad9838ac14f9e252d292146df5b1e1b1121d7c, SHA1997eebf093aa3e37de1cbbe58302a1539ac2e517f966a4607aae89dc695a8e0. B0/C0 config dictionaries are exactly the pinned H0 configuration; only the CLI arm differs. Same ResNet P50 encoder, fresh identical seed17heads,batch32,layer4training,frozenBN,head1e-3/encoder1e-5,all five axes,unchanged sampling and no nuisance augmentation. No new pretrained encoder, loss coefficient, mask or optimizer search.

```bash
export CES_PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
export CES_RUN_ROOT=/home/sachin.chaudhary/ces_fl_runs/lh_objective_priority_20261105_r1
export CES_CONTROLS_ROOT=/home/sachin.chaudhary/ces_fl_runs/lh_nuisance_pilot_20261104_r1
export CES_CMD=CES_FL/proposals/lh_objective_priority_20261105_r1/commands.sh
"$CES_PY" CES_FL/proposals/lh_objective_priority_20261105_r1/check_arm_losses.py
bash "$CES_CMD" prepare
```

Preparation independently replays cached control metrics, verifies hashes/config/selection and performs new config-bound TRAIN/VAL preflight. Inspect existing progress before repeat invocations. B0 has no procedural forward/loss and needs no intervention qualification; the native auxiliary loader may construct unused probe tensors, which are not forwarded by armB.

Use one or at most two properly allocated/bound GPUs. Refresh the actual L_H-finalization ledger; the last report says22.9666/48used,25.0334remaining, but verify live allowance and reservations. This request grants no additional hours. Cap the whole B0/C0 cohort,checks,replay and any held-idle time at3GPUhours within that remaining allowance; CPU reporting uses no GPU. Individual configs retain0.75h epoch-boundary limits. Measure realistic throughput and reserve the full stage costs; apply external job deadlines as well. Do not consume FF2's72h or interrupt active jobs.

After valid resource binding run `bash "$CES_CMD" checks B0`. Record the actual allocation, CUDA visibility, ledger reservation and full-stage timing estimate in a private receipt; set CES_ACCOUNTED_STAGE_RECEIPT to it. Then `bash "$CES_CMD" train B0`. The script does not itself allocate or reserve GPUs: use the existing verified ledger guard/site mechanism around GPU stages.

For C0, verify the existing H0 qualification's exact source/config/preview/review hashes through the native check_qualification function. The source/config/recipe are identical, so a valid unchanged qualification may be reused with recorded provenance; set CES_C0_QUALIFICATION to the actual file. If it fails, inspect and perform substantive review; do not fabricate approval. This is the FF++ procedural screen, never the rejected FF2 review. Run `checks C0`, then `train C0` only after the gate and budget pass. An unresolved C0 review must not block B0 or original-image diagnostics. Keep procedural-versus-real-forgery limitations explicit.

After native DONE and selected validation replay, run `bash "$CES_CMD" compare` on CPU. Report A0→B0(original fairness),B0→C0(sensitivity/sham package),C0→H0(alignment),B0→H0(full package comparison). Also calculate H0−A0 and all per-axis/group regressions. Select earliest maximum clean validation AUC for every arm; no change to historical checkpoint selection. Both frozenVAL10%and fixed0.5 reports are required; the native compare command's intervals use storedVAL thresholds, so compute a separately labeled fixed0.5 analysis from copies without changing original caches.1000paired recorded-contributor draws,900usable minimum; no training-seed uncertainty claim. Report AUC,FPR,FNR,worst-groupAUC and all fairness axes, not just favorable cells. Do not choose a threshold or checkpoint to meet60/80.

## Second: find where detector ability is lost

On fixed TRAIN samples and existing clean FF++VAL only, compare the complete native P50 classifier with the new A0/B0/C0/H0 checkpoints. Load the P50 through its actual native architecture/configuration. Keeping encoder weights while replacing or changing the classifier/ReLU/GAP computation does not preserve predictions. Check exact forward identity before calling any future protocol classifier-preserving. Measure per-class score distributions,calibration,group errors and the already fixed nuisance bank; report weighted gradient norms/cosines on several prespecified TRAIN batches across axes. Use a maximum0.25GPUh for these zero-update diagnostics, inside the same ledger. No new training from this diagnostic and no target-domain score reading.

## Third: prepare a source-only generalization experiment

Create a new dated protocol and fold manifest based ONLY on FF++TRAIN metadata. Use a deterministic seed17split of recorded contributor components, keeping a development subset independent of fit contributors. Choose the held-out manipulation family by a fixed metadata rule (lexicographically first supported known fake family), not by external TEST performance. Exclude that family from every detector fitting/pretraining stage and exclude the development contributors from fit. Development evaluation uses that held-out family's fakes plus the development contributors' reals. Report unused rows and all group/class/family support. If the provenance graph leaves insufficient support, report the limitation; never silently weaken independence.

Existing full-FF++P50 has seen the held-out family and cannot initialize a claimed unseen-family study. Bind two exact original-only BCE configurations from the same independently verified ImageNet initialization: native P50 recipe versus the same recipe plus the previously fixed identity/JPEG90/resize90/blur035 bank on BOTH classes. Copy/audit native P50 optimizer,exposure and selection settings before saving those configs; do not guess them or tune strengths. Do not alter the public benchmark split or mix these development-fold checkpoints into the main table. This is a robustness comparison, not another nuisance-H1 claim.

Implement and check that fold adapter, save the exact fold/config/source hashes and profile the complete cohort before launch. Stage3 is not implemented by this commit and must not be called launch-ready. User authorizes completing this preparation and running the bounded source-only comparison if feasible: maximum5aggregate GPUh, within refreshed existing allowance, with native full epoch schedule and verified local initial weights. If it cannot fit, publish the precise preparation/budget blocker without starting a truncated protocol. No parameter sweep, extra seed or external TEST evaluation follows automatically.

## Reporting and stopping

Push source/config/check receipts before any new implementation is launched, and aggregate progress/results after milestones. Keep native manifests, source/checkpoints and prior preregistration immutable; row scores and images stay private. Report completed/blocked stages separately. Primary new evidence is the original-image fairness contribution and source-only robustness, not a pooled win count. A future final-candidate evaluation needs a prospective frozen validation/utility/fairness policy applied to its controls and explicit acknowledgement of earlier TEST exposure. Do not reselect published models, loosen uncertainty requirements or claim60/80 as achieved without actual compatible frozen evaluation.
