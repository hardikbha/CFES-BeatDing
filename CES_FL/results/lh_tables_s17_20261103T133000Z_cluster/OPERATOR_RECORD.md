# ee73f2f cluster execution record

Pinned supervisor commit `ee73f2fdf5218e440852917e90281818bbe711bb`, fetched from `hardiksharma6555/CFES-BeatDing`, clean detached source `/home/sachin.chaudhary/ces_fl_sources/cesfl-lh-tables-ee73f2f`. Runner SHA256 `cc21328aed395a3f0e133658341b58ef341f610c71947852bcf5a6806e526a29`; historical plain-head scientific source SHA `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`. Neither source nor any accepted configuration was modified.

Private root `/home/sachin.chaudhary/ces_fl_runs/lh_tables_s17_20261103T133000Z`. Private inventories, overlap contributor values, image bytes, full scores and checkpoints stay on cluster. Aggregate evidence is published on upstream branch `results/lh-tables-ee73f2f-20261103` from a separate publication checkout.

## Completion and software gates

The existing plain-head allocation45295 finished with exit0. Both native arms completed exactly ten epochs, DONE, RUN_VERIFIED and selected validation replay. L_A selected epoch3, checkpoint `aee9670ca011937d457846fbf13717445c5607715064dd6a54da5a82bda580ae`; L_H selected epoch1, checkpoint `6ef2caad3979d101fd94d7350aa3c1a50f95ee8307673cfaae63ac9d89af36f0`. No checkpoint was forced or reselected. Completed four-row validation comparisons and1000draw paired intervals were published separately at `results/cesfl-plain-head-0d699d2-20261103`, `CES_FL/results/plain_head_s17_20261103T101418Z_cluster/COMPARISON_TABLES.md`. Plain versus gated BCE loses0.193787percentage-point AUC; plain versus gated H changes AUC by+0.002646pp with interval crossing0. The latter's primary intersection interval also crosses0 and has746usable draws, below900. This supports descriptive simplification discussion, not equivalence or a guaranteed improvement.

Executed with aiface Python:

```
python -m unittest discover -s CES_FL/tests -p test_lh_locked_benchmarks.py -v
python -m unittest discover -s CES_FL/tests -p test_matched_recipe.py -v
```

Actual results:4tests PASS in6.950s and6tests PASS in18.274s, no skips. Logs are in evidence. These guard tests are synthetic; they are not benchmark performance results. Separate native completed checkpoint/config/selection/cache/image integrity checks certified the actual models.

## Input audit and frozen plan

`prepare_integrity.py` performs CPU-only source/config/checkpoint/cache verification, current image hashing/decoding and overlap checks without any model forward. Each selected validation cache reproduces its native stored report at its original threshold. The exact old TRAIN/VAL memberships are equal across A/H. Every TEST manifest matches the previously pinned benchmark protocol before any scoring.

FF++ contributor declarations are the union of the untouched pinned six-key audit's TRAIN:TEST and VAL:TEST declarations:231recorded identity tokens and231source tokens across fit=(TRAIN+VAL). Cross-domain audits enforce zero recorded-contributor overlap against the fixed historical benchmark manifests. All four reject any fit:TEST content overlap; actual byte overlap is zero. No actual collision was converted into a new exception. Cross-domain provenance comes from the existing hash-pinned benchmark inventory and current metadata; this is not a claim that namespaced proxy IDs establish biological-person separation. Original audit remains unchanged.

All unique images decoded/hashed:FF++25,401; DFDC21,381unique/22,857rows; Celeb-DF28,458; DFD9,385. Native DFDC multiplicity (1,476extra rows) is preserved. `INPUT_INTEGRITY.json` reports hashes and counts without private rows.

Both checkpoints, configurations, all four manifests/audits, unchanged native calibration thresholds and helper/protocol hashes are bound in `LOCKED_PLAN.json` (SHA `6dffee3b9b309975d7c3844dc1052427885dc41b2375e77c15f38ae33f76c5ea`). `PRETEST_DESIGNS_FROZEN.json` hashes the accepted prospective Ding10/ResNet/schedule specifications before TEST inference. Missing implementation choices may not be selected using these TEST results. The pre-inference lock/prospective receipt was pushed as commit9e24619 before releasing the evaluation job.

Full validation operating curves at1%,2%,5%,10%requested FPR are published for A/H. They are exploratory and change neither the native10%threshold nor the fixed0.5view. For H, achieved FPRs are0.9775/1.9795/4.9853/9.9951%, with FNR13.1680/9.9746/6.2459/4.3064%. Lower false-alarm disparity at a stricter threshold is not free detection improvement; report both errors. Score-ranking AUC/es-AUC do not change with the decision threshold.

## Actual allocation and inference

PBS job `45314.mgmt01`, workq, gpu02, ngpus1,ncpus6,mem32GB, walltime02:05. Resource guard hard cap2GPUh. Scheduler resources verified inside the actual allocation. Site supplied no physical binding, so the previously explicitly authorized shared pool-v2 lease bound H100 GPU5 UUID `GPU-bff97893-385b-6f3f-831b-113284c2e1f1`. Exclusive physical ownership is not claimed. Actual one-visible-device binding:PyTorch2.7.0+cu126,CUDA12.6,H10080GB. ControllerPID1152316. Source checkout is independently verified clean/detached using the real shared Git binary on the node.

Original24GPU-hour ledger had15.196284h consumed, zero reservations. This study reserves2h there; it does not use or increase the separate FF2 baseline72h allowance. All held time, including integrity rechecks/native loading/idle overhead, is charged. No existing job was killed or modified.

Exact eight invoked stages are the pinned helper's `infer` command for L_A and L_H, each onffpp,dfdc,celebdf,dfd, with `--plan PRIVATE_ROOT/LOCKED_PLAN.json --device cuda`. Each output is fresh; the worker stops if an output exists or any subprocess/receipt fails. Successful TEST inference is never repeated. No training/calibration command is called. Exact argv and timestamps live in `L_A_ffpp_COMMAND.json` etc.; logs `L_A_ffpp_inference.log` etc. Native outputs are `PRIVATE_ROOT/inference/ARM/DATASET/`.

CPU-only PBS `45315.mgmt01` depends afterok on45314,4CPUs24GB,3h,ngpus0. It runs the exact pinned `analyze` helper for8model/dataset outputs, then paired H-minus-A1000draw connected-recorded-contributor bootstrap for each dataset under both threshold views. Original score files/threshold metadata are untouched. Eligibility stays fixed; usable counts and the900minimum are reported. This is one explicit analysis job, not a new automatic runner/retry service.

## Separate training blockers

- New Ding10 scope is authorized, but the supplied commit has no connected-gradient ten-epoch Ding trainer/config/smoke or verified SFD-bound parent launch. Archived detached GDA is not substituted. Required remaining work: verify existing classification50/SFD provenance, implement the separately named head-only differentiable adaptation, fix BN policy and additional group-forward accounting, test nonzero fairness gradients/update sensitivity, profile and assign a separate FF++training budget.
- ResNet L_H has no current verified backbone adapter/P50/A/H checkpoints. Table3 remains PENDING. Historical ResNet A/B/C are not reused as L_H. The documented ImageNet normalization/layer4/frozen-BN policy needs code and CPU/CUDA/resume checks before any new training.
- The one cosine-schedule candidate is frozen prospectively in the accepted README, but has no implemented/tested scheduler/resume configuration or assigned full-run allowance. It is not launched, and no TEST result may determine its settings.
- Matched50+50 recipe remains explicitly deferred by its own priority update. Its passing CPU tests do not authorize its full training.

The final tables distinguish archived paper references, historical local classification/GDA seeds/protocols, pending new Ding10, and the two fixed new CES-FL checkpoints. Table1/2 have separate fixed0.5 and source-VAL10%views; Table3 stays pending. No direct protocol-equivalent win, SOTA, biological-identity independence, demographic-counterfactual validity or causal forgery-mask claim is supported merely by successful software/evaluation execution.
