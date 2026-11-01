# P50 completed; matched original-only BCE and fairness arms launched

Observed management UTC: 2026-11-01T16:55:22.183611+00:00. This is a new execution milestone under the original authorized seed17 manipulation-response handoff, after the user requested direct SSH. Scientific source remains **58773bbef9d9fe79dd313610f694ddeea12c6037**. No model, loss, split, seed, selection, threshold or training budget was changed. Source worktree and existing artifacts are preserved.

## Verified P50 outcome

All **50 original-only epochs completed**, with native `DONE`, `summary.json=COMPLETED_VERIFIED` and `RUN_VERIFIED.json=PASS`. Independent selected-checkpoint validation replay differed by **0.0**. The selected epoch is **29**, fixed by maximum validation AUC, not fairness. Checkpoint bytes were independently rehashed and operator verification checked exact history/config/calibration/source/selected epoch and finite model tensors.

| Selected P checkpoint metric | Validation result |
|---|---:|
| Overall AUC | 98.405989% |
| Achieved overall FPR | 9.995112% |
| FNR | 4.221847% |
| gender_race worst-group AUC | 94.109001% |
| gender_race FPR max–min gap | 30.984848 percentage points |
| gender_race summed F_FPR | 0.5075766324 (fraction units) |

These are validation results from original-only pretraining, not TEST or results of the proposed counterfactual/response method. They do not establish fairness improvement or superiority over Ding. Small subgroups, including only30real Female-Black validation examples, limit precision. Epoch29's checkpoint must not be replaced by a different fairness-favorable epoch.

Selected checkpoint SHA256: `3174e5b01a84c91cc70afae538505663ca96454d897e6b793c0672826a8f08b5`.

## Exact original-only binding

The native binding command completed with exit0:

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb/CES_FL/src/ces_fl_manipulation_response.py bind --config /home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb/CES_FL/proposals/manipulation_response_20261101_r1/config.json --pretrain-run /home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/P --out /home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/bound_originals
```

Bound configuration SHA256: `b3d19e4937897ca568836dc67587f4d5f3b14222963b9bfb676a19be82627ece`. Only the actual P path/hash were bound; every other scientific setting matches the original config. `bank=null`. This cannot authorize a paired arm or claim quartets exist.

## Actual configured CUDA backward checks

Both checks completed with native `DRY_RUN.json=PASS` and operational PASS on their recorded physical devices. Each made **zero optimizer steps**, used original TRAIN images only, no quartet, and produced finite gradients under the current unchanged research code. Full commands, PIDs, bindings, hashes and native outputs are preserved in `checks/` and `backward_checks/`.

| Check | Host / GPU | Classification loss | Fairness term | Total loss |
|---|---|---:|---:|---:|
| dry_bce | gpu02 /5 | 0.4861428738 | 0 | 0.4861428738 |
| dry_fair | gpu02 /4 | 0.4861428738 | 0.0084448904 | 0.4869873524 |

Original classification losses are exactly equal for these matched initialized batches. The fair total matches classification+0.1×fairness. This is a software sanity check, not evidence that fairness improves after training. Native model-transfer/reset semantics remain unchanged.

Five operational guard-test groups actually passed before launch: paired-arm rejection, backward requirement, exact same host/device restriction, changed-config/existing-output rejection, and incomplete-P/nonfinite-loss rejection. They use temporary metadata fixtures and do not count as model-performance results. New operational helpers provide dry-only execution or one explicit arm submission; neither installs an automatic run-all executor. Training uses the already tested and previously published `direct_mr.py` unchanged.

## Matched arms launched separately after PASS

Each arm runs **50epochs**, full parameter fine-tuning with frozen encoder BN running buffers. Both share the same selected P encoder, identical fresh head/gate/fusion seed17, original order and fixed label×ids8 weighting, preprocessing, update budget, optimizer settings, validation-AUC selection and validation-only threshold policy. Fairness axis remains gender_race. No paired image is used.

Actual launcher commands (each returned exit0):

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/runtime/launch_baseline.py --arm bce --phase train --host gpu02 --device-uuid GPU-bff97893-385b-6f3f-831b-113284c2e1f1
/home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/runtime/launch_baseline.py --arm fair --phase train --host gpu02 --device-uuid GPU-ee335748-8273-2d67-42a5-d45463027c1b
```

| Arm | Actual status at observation | Host / physical GPU | Worker PID | Output |
|---|---|---|---:|---|
| P | COMPLETED | gpu02 /5 (finished) | 1390084 (finished) | `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/P` |
| bce | RUNNING | gpu02 /5 | 2961788 | `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/bce` |
| fair | RUNNING | gpu02 /4 | 2961831 | `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/fair` |
| nochange | BLOCKED_INPUTS | Not assigned | — | No run launched |
| score | BLOCKED_INPUTS | Not assigned | — | No run launched |
| response | BLOCKED_INPUTS | Not assigned | — | No run launched |

These are distinct output directories and verified device UUIDs. Source hashes and all external input hashes are rechecked by each worker. Before launch, both nodes were scanned for duplicate exact-config trainers; none was found. The resource policy is the previously user-authorized direct SSH route, cooperative dispatcher/project leases,20GiB headroom and15% Torch allocator cap. Existing processes continue. No PBS allocation/exclusive GPU ownership is claimed. Each training arm is restricted to its own successful backward check's exact host/UUID.

## Current limits and next dependencies

Qwen32 completed224/224outputs across32sources. This is generation completion, not per-pair semantic validity or a complete manipulation quartet bank. It remains separate; it was not rerun or extended. No nochange/score/response training was launched. Their missing operator adapter, matched t0/ta assets, independent review, group/direction coverage and per-arm backward checks remain required. Prospective inputs and cached SimSwap weights do not resolve those requirements.

No FairForge-2, TEST evaluation, new seed, model download, image supplementation, preregistration change or unrelated job termination occurred. Current training can compare the fairness baseline with BCE once both selected checkpoints pass native independent replay; it does not yet test the response hypothesis.

[Timestamped current status](../../../status/manip_response_ssh_20261101T115720Z_live.md) is published by the existing read-only observer; it cannot launch/retry/approve/cancel/chain jobs. Native per-arm epoch histories, subgroup metrics/support, selected calibration, pair draws (zero here), checkpoint hashes and failures appear as recorded. Do not claim a completed epoch or performance result for a newly launched arm from startup alone.

Actual logs: `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/logs/bce.log`, `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/logs/fair.log`; native outputs: the separate bce/fair paths above. Exact scientific source, original handoff and prior P50 receipts remain preserved. The detailed supervisor handoff's31-epoch state is an explicitly dated historical snapshot; use this newer execution record and current native artifacts for ongoing state.
