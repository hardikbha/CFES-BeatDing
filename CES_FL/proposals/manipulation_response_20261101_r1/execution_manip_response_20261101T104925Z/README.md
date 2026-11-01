# Priority manipulation-response execution — queued CUDA smoke

Observed management UTC: 2026-11-01T11:16:52.353553+00:00. Exact fork branch tip and detached worktree verified: `58773bbef9d9fe79dd313610f694ddeea12c6037`, `/home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb`. The exact source commit is also published on the canonical repository branch `source/manipulation-response-58773bb`; main contains byte-identical reviewed source/config/test snapshots. No running-job source was updated.

## Checks actually completed

Four source/config/test SHA256 values match the supplied manifest. Four external FF++ TRAIN/VAL/overlap/ImageNet input hashes match. The embedded P configuration equals the existing pinned base-trainer configuration. Five standard-library allocation/UUID/protected-device tests passed in aiface, and `bash -n` passed for the smoke launcher. These are **not** the six new ML/CUDA smoke groups. Their Mac CPU receipt was inspected but is not a target CUDA result.

Actual guard command: `/home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/ces_fl_runs/manip_response_20261101T104925Z/runtime/test_allocation_guards.py`; output `Ran 5 tests ... OK`, exit 0. Actual shell command: `bash -n /home/sachin.chaudhary/ces_fl_runs/manip_response_20261101T104925Z/runtime/smoke_one_gpu.pbs`, exit 0. Full command/stdout/stderr receipts are in checks/ and logs/.

## Actual allocated job request

**44963.mgmt01**, `mr17_smoke`, is **QUEUED**. It requests ngpus=1, eight CPUs, 48GB RAM, three-hour walltime on gpu01; rerun is disabled. The scheduler states: `Not Running: Insufficient amount of resource: ncpus (R: 8 A: 4 T: 208)`. A queued job has no allocated physical UUID yet. No quota or priority was changed, no existing job was terminated, and the Qwen pilot remains protected.

The wrapper first proves RUNNING PBS ngpus=1 allocation on the current single host. This site's existing mapping provides host/count rather than a UUID; the wrapper therefore waits for an empty physical device on the allocated host, respects any inherited CUDA restriction, holds both the shared-dispatch exclusion and project lease, rechecks occupancy/headroom, and verifies PyTorch's full UUID. It explicitly excludes the active Qwen UUID. If no eligible empty GPU appears within two hours, it fails without touching other processes. This is distinct from claiming PBS assigned a particular UUID.

The queued job runs **only**:

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb/CES_FL/tests/smoke_manipulation_response.py --device cuda --out /home/sachin.chaudhary/ces_fl_runs/manip_response_20261101T104925Z/smoke_cuda
```

The exact source/test/dependency hashes are checked again inside the allocation. Actual environment, binding, PASS/FAIL and logs will be published. No full handoff, recovery, P/BCE/fair submission, paired training or test evaluation is attached automatically.

## Separate current arm states

| Arm | Status | Dependency |
|---|---|---|
| P (50 original-only epochs) | BLOCKED_INPUTS | New CUDA suite, all-real decode/hash audit and exact configured backward check pending |
| bce (50 epochs) | BLOCKED_INPUTS | Verified shared P50, binding and BCE backward check |
| fair (50 epochs) | BLOCKED_INPUTS | Same P50, binding and fair backward check |
| nochange (50 epochs) | BLOCKED_INPUTS | P50 plus valid reviewed matched quartet bank and its backward check |
| score (50 epochs) | BLOCKED_INPUTS | Same bank/P50 and score backward check |
| response (50 epochs) | BLOCKED_INPUTS | Same bank/P50 and response backward check |

Original-only P/BCE/fair do not require quartet approval. The prepared `real_inputs.py` performs full TRAIN/VAL decode/hash readiness on the allocated node and then executes the exact base pretraining backward command, making zero optimizer updates. It preserves the declared benchmark contributor-overlap exception and reports exact byte overlap before training; it does not alter membership. This helper has been syntax-checked, not executed as a real-input PASS. No legacy ten-epoch checkpoint is substituted for P50. Before P submission, check completed/running exact-configuration P candidates and require all 50 epochs plus replay/hash receipts for any reuse.

Source recipe and commands for later milestones remain the original GPU_OPERATOR_PROMPT.txt. Operate milestone by milestone; the finite observer cannot advance them. Paired arms remain blocked until substantive quartet validity/coverage is established, not merely appearance or prompt compliance. [New additive quartet preparation](../../manipulation_response_quartets_20261101_r1/README.md).

## Artifacts and publication

Run root: `/home/sachin.chaudhary/ces_fl_runs/manip_response_20261101T104925Z`. Source: `/home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb`. CUDA job logs: `logs/smoke.pbs.log`, `logs/smoke.log`; output receipt: `smoke_cuda/RESULT.json`; job/binding/failure records: `checks/`. Prospective image/donor/review rows remain private. No image, binary checkpoint, credential or model weight is published.

[Actual timestamped status](../../../status/manip_response_20261101T104925Z_live.md) is published by a finite read-only smoke observer (at most 12 hours, about every ten minutes/on state change, stops on terminal result). It cannot train, approve, retry, cancel, download or evaluate test data. Queued is not running training. No SOTA/novelty/fairness result is claimed. FairForge-2 remains DO NOT RUN.
