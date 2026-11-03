# GPU operator prompt — FairForge-2 remaining baselines

You are the GPU operator for `CFES-BeatDing`. Execute the authorized FairForge-2 baseline work from the exact commit supplied with this prompt. Read `CES_FL/AGENTS.md`, `PROJECT_MEMORY.md`, and this proposal's `README.md`, `REMAINING_METHODS.md`, actual Python sources, configs and evidence first. This task is separate from the previous plain-head FF++ ablation and existing Ding FF2 run.

**Authorization: at most72 additional aggregate GPU-hours and at most3 concurrently allocated GPUs.** Use3 one-GPU jobs when legitimately available,2 or1 otherwise. Account CUDA checks, profiling, validation, checkpoint overhead and idle allocation time as well as training. Do not borrow GPUs or terminate existing jobs. Do not launch new Ding-on-FF++ experiments. Preserve existing Ding, CES-FL, H and plain-head jobs/results. No TEST inference, new seed, hyperparameter search, fabricated labels or generated-image pipeline is authorized by this handoff.

## 1. Fetch an immutable source and inspect existing progress

Published source is on the writable fork `https://github.com/hardiksharma6555/CFES-BeatDing.git`, branch `experiment/ff2-baseline-suite-s17-20261103T110324Z`. Upstream `hardikbha` was not writable from the preparation environment. Use the **full accepted commit from the accompanying handoff**, not a later moving branch head.

```bash
FF2_REVIEWED_COMMIT=PASTE_FULL_COMMIT_FROM_HANDOFF
FF2_REPO=/path/to/your/existing/CFES-BeatDing
git -C "$FF2_REPO" fetch https://github.com/hardiksharma6555/CFES-BeatDing.git experiment/ff2-baseline-suite-s17-20261103T110324Z
git -C "$FF2_REPO" cat-file -e "${FF2_REVIEWED_COMMIT}^{commit}"
FF2_SOURCE="/home/sachin.chaudhary/ces_fl_sources/ff2_baselines_${FF2_REVIEWED_COMMIT}"
git -C "$FF2_REPO" worktree add --detach "$FF2_SOURCE" "$FF2_REVIEWED_COMMIT"
FF2_PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
FF2_PROPOSAL="$FF2_SOURCE/CES_FL/proposals/ff2_baseline_suite_s17_20261103T110324Z"
FF2_WORK=/home/sachin.chaudhary/ces_fl_runs/ff2_baseline_suite_s17_20261103T110324Z
FF2_SOURCES=/home/sachin.chaudhary/ces_fl_dependencies/ff2_baselines_pinned_20261103
FF2_LEDGER="$FF2_WORK/additional72_gpu_hours.json"
```

Resolve the actual repository path; do not reset or clean anything. If the source/output already exists, verify its commit and receipts and inspect scheduler/process state before continuing. Completed methods must be reported, not repeated. Resume only incomplete compatible checkpoints; never delete previous failure evidence or steal a live method lock. Inspect the current Ding branch separately to avoid duplicate work; a past GitHub RUNNING message is not live allocation evidence.

## 2. CPU preparation before reserving GPUs

```bash
mkdir -p "$FF2_WORK"
"$FF2_PY" "$FF2_PROPOSAL/fetch_sources.py" --root "$FF2_SOURCES"
cd "$FF2_SOURCE"
FF2_BASELINE_UPSTREAM="$FF2_SOURCES/dag_daw" "$FF2_PY" -m unittest discover -s CES_FL/tests -p 'test_ff2_baseline_suite.py' -v
"$FF2_PY" -m unittest discover -s CES_FL/tests -p 'test_ff2_budget.py' -v
for FF2_METHOD in bce dag daw; do
  "$FF2_PY" CES_FL/src/ff2_fairness_baselines.py preflight \
    --config "$FF2_PROPOSAL/configs/$FF2_METHOD.json" \
    --out "$FF2_WORK/preflight_$FF2_METHOD"
done
"$FF2_PY" "$FF2_PROPOSAL/budget.py" --ledger "$FF2_LEDGER" status
```

Capture command/log/exit status. On repeat handoff, verify existing source/config-bound preflight receipts instead of invoking `fresh_dir` on them again. Require all7 native/loss tests (no skipped upstream tests) and6 budget tests to pass. Existing local ImageNet Xception weights are required; no automatic download. Confirm sufficient disk space for three runs and synthetic checkpoints (allow at least15GB). All real TRAIN/VAL hashes, memberships and demographic mappings must pass. A missing path or changed hash is a blocker, not permission to alter the split.

## 3. First wave: matched BCE / DAG-FDD / DAW-FDD

One GPU per method, up to3 concurrent jobs. Exact configs specify50 epochs, batch128, seed17, full Xception/BN training, float32 SGD and natural once-per-epoch original exposure. Do not silently change these to make the experiment fit. CVaR is computed over the one-GPU batch; do not introduce unverified DDP/gradient accumulation.

Initial session reservation:18 one-GPU hours each. This reserves at most54 of the72 additional GPU-hours; leave18 uncommitted until costs/readiness justify using it. The worker performs a fresh synthetic CUDA save/reload/resume test,20 actual TRAIN backward steps, timing projection and only then training. It has no TEST command. If the estimate plus40% overhead exceeds the session's remaining time, it returns `BUDGET_PENDING`. Reconcile all reservations and choose a feasible continuation within72h; do not shorten the scientific schedule or fabricate a completion.

For each actual allocation, record a **job-specific** JSON outside the repo:

```json
{
  "allocation_id": "ACTUAL_PBS_JOB_ID_OR_VERIFIED_POOL_RESERVATION",
  "gpus": 1,
  "cuda_visible_devices": "ACTUAL_ALLOCATED_BINDING",
  "verified_by": "OPERATOR_NAME",
  "evidence": "PATH_TO_SCHEDULER_RESOURCE_AND_GPU_BINDING_EVIDENCE"
}
```

This is a template, not proof of an allocation. Verify scheduler resources and site binding; never guess device numbers. The worker checks the record against `PBS_JOBID` when present and the actual `CUDA_VISIBLE_DEVICES`. If the site sets binding only inside the job, adapt the wrapper to its documented mechanism and record the exact reviewed wrapper before launch. The provided `one_gpu.pbs` is syntax checked, not locally submitted.

Inside each verified one-GPU allocation, the executable command is:

```bash
"$FF2_PY" -u "$FF2_PROPOSAL/run_one.py" \
  --method "$FF2_METHOD" --work-root "$FF2_WORK" --sources "$FF2_SOURCES" \
  --ledger "$FF2_LEDGER" --allocation-record "$FF2_ALLOCATION_RECORD" \
  --session-id "$FF2_ACTUAL_JOB_ID" --hours 18
```

Or submit the provided PBS wrapper after resolving job-specific ownership/binding. It requests `select=1:ncpus=8:ngpus=1:mem=64gb`, walltime18h. Pass `FF2_SOURCE`, `FF2_METHOD`, `FF2_WORK`, `FF2_SOURCES`, `FF2_LEDGER`, `FF2_ALLOCATION_RECORD`, `FF2_PY` and site-correct `CUDA_VISIBLE_DEVICES` through the scheduler environment. Do not submit blindly with placeholder values. Use separate method jobs; the shared ledger rejects a fourth active GPU or reservations above the remaining cap.

For a confirmed stopped incomplete run, use a new session ID and `--resume` (PBS: `FF2_RESUME=1`). Preserve config/source/checkpoint hashes; do not restart from epoch1. If a worker died before releasing its ledger claim, reconcile its actual scheduler end time and held GPU-hours first. Never erase usage. A complete method is not an instruction to run another seed. Profile-only work also uses a ledgered allocation; `--profile-only` still runs CUDA checks and the20-step timing test.

## 4. Remaining official methods, while GPUs run

Use CPU time to execute `REMAINING_METHODS.md` work orders for PG-FDD and FairAdapter. Their pinned sources are fetched; **their FF2 adapters are not certified ready**. Verify exact paper method, necessary TRAIN labels/views, local weights, losses/gradient paths, BatchNorm/SAM behavior and checkpoint replay. Write new uniquely named requests and exact configs before launch. Do not rename an approximation as an official method. PG-FDD must not reuse hardcoded FF++ counts or invented manipulation labels. FairAdapter retains its actual CLIP backbone; never blindly execute its generation-oriented top-level `main.py`.

Launch any additional ready method only after CPU/CUDA checks and a measured projection fit the shared remaining72h ledger and at most3 allocated GPUs. Keep RSEF-FDD blocked until official source and additional-data requirements are verified. Report specific blockers rather than silently omitting them or claiming all baselines complete. This queue does not include every appendix detector or new ResNet/FF++ experiment.

## 5. Verify, report and push actual evidence

After selected-checkpoint replay completes, release GPU allocations and run on CPU:

```bash
"$FF2_PY" "$FF2_PROPOSAL/report_validation.py" \
  --work-root "$FF2_WORK" --out "$FF2_WORK/validation_report_UNIQUE_TIMESTAMP" --bootstrap 1000
"$FF2_PY" "$FF2_PROPOSAL/budget.py" --ledger "$FF2_LEDGER" status
```

Publish after meaningful milestones: exact launch commit and commands, input/source/config hashes, source pins, scheduler allocation and timestamps, software/input/backward checks, measured/projected GPU-hours, completed epochs, selected epoch and checkpoint SHA, aggregate validation metrics, full supported subgroup counts, paired uncertainty versus native BCE, and the remaining-method blockers. Keep dated evidence and failures. Use a separate publication checkout/branch; do not pull or edit the source under running jobs. Publish to the original repo if your account can, otherwise to the writable fork and give its exact branch/commit.

Keep checkpoints, images, full private row-level scores/inventories and credentials off public GitHub. Published results must distinguish **VAL**, fixed-logit0 versus VAL-calibrated thresholds, and existing Ding's different recipe. Bootstrap does not cover training-seed variability or establish true identity independence. Unknown/unsupported demographic axes remain explicit. The existing Ding FF2 classification/GDA run is separately labeled; these common-recipe rows are not an apples-to-apples claim against Ding's paper TEST tables.

Finish with a clear table per method: READY/RUNNING/COMPLETED_VERIFIED/BUDGET_PENDING/BLOCKED, actual evidence timestamp, epoch/selected checkpoint, validation AUC, gender/race/gender×race fairness, GPU-hours and next dependency. No automatic unattended monitoring is implied. Do not claim beating Ding until comparable completed results support it.
