# Supervisor → GitHub → cluster workflow

## 1. Review the observed state

Read README.md, STATUS.md, the dated handoff, actual source/config and the relevant result JSON/history. Check when STATUS.md was observed. The current completed pilot is seed17 Xception, C-invariance only. Its 98% scores are validation scores, fairness is mixed, test Tables1–3 are incomplete, and the old CESel-MA gate remains FAIL.

A source change is a proposal until it is tested. Historical smoke receipts do not certify newly edited code. Training requests and pushes do not automatically execute jobs. This setup adds no automatic job runner or periodic service.

## 2. Supervisor proposes a new run

Use a branch (recommended) or a deliberate reviewed commit on main. For example, in a clean review clone:

```bash
git clone https://github.com/hardikbha/CFES-BeatDing.git
cd CFES-BeatDing
git switch -c supervisor/REQUEST_ID
mkdir -p CES_FL/proposals/REQUEST_ID
cp CES_FL/proposals/TRAINING_REQUEST_TEMPLATE.json CES_FL/proposals/REQUEST_ID/request.json
```

Replace REQUEST_ID with a unique meaningful name. Fill request.json. Create a real `config.json` compatible with the source's `train-config` command; the request template is metadata, **not** a training config. The archived B/C configs show the actual schema, but copying one does not approve a new experiment. Keep old configs/results immutable.

State the hypothesis, baseline, changed factors, dataset/split hashes, seed, selected encoder/checkpoint hash, preprocessing, update budget, validation selection and threshold policies, intervention-review evidence, required resources, new output directory and dependencies. For B/C comparisons use the same encoder, head initialization, sampling/update budget, selection and calibration. Do not change multiple factors silently or tune against test. A forensic-sensitive/full-selectivity arm needs validated forensic interventions; map consistency requires spatial validation. A ResNet or FairForge-2 study needs its own explicit configuration and software checks.

Review the diff, stage named files, test, commit and push your branch:

```bash
git diff -- CES_FL/src CES_FL/proposals/REQUEST_ID
python -m unittest discover -s CES_FL/tests -v
# Run suitable model smoke checks in the supported environment for actual model changes.
git add -- CES_FL/proposals/REQUEST_ID CES_FL/src/ces_fl_diagram_model_v021.py
git diff --cached --check
git commit -m "Propose REQUEST_ID with exact config and rationale"
git push -u origin supervisor/REQUEST_ID
```

Merge the reviewed change before the operator launches it, or explicitly agree the exact branch commit. Record the actual launch decision; do not fill approvals retroactively. The provided account hardikbha already has admin access. For another person/account, use their own access; [GitHub's collaborator instructions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository) describe invitations.

## 3. Cluster operator fetches without changing active job sources

The historical agents checkout contains unrelated dirty files. Do not blanket-stage, reset or clean it. A separate clone for review/execution is often easier. Check qstat and source dependencies before any pull. Do not pull over an active job's source, configuration or live-read launcher.

From a suitable checkout, resolve the accepted commit and create a new detached worktree. These commands prepare source only; they do not launch training:

```bash
git fetch origin
CES_COMMIT=$(git rev-parse origin/main)
CES_REQUEST_ID=REQUEST_ID
CES_SOURCE_DIR=/home/sachin.chaudhary/ces_fl_sources/${CES_REQUEST_ID}_${CES_COMMIT}
mkdir -p /home/sachin.chaudhary/ces_fl_sources
git worktree add --detach "$CES_SOURCE_DIR" "$CES_COMMIT"
git -C "$CES_SOURCE_DIR" show --stat --oneline HEAD
```

Inspect the request/config at that immutable path, verify code and input hashes, confirm all prerequisites and approvals, inspect existing runs to avoid duplicates, and use a new output directory. Do not run an archived completed config simply because it exists.

## 4. GPU launch is explicit and allocation-dependent

The template `CES_FL/cluster/train_one_gpu.pbs` requests ngpus=1. It requires CES_SOURCE_DIR, CES_TRAIN_CONFIG, CES_OUTPUT_DIR and a verified CUDA_VISIBLE_DEVICES binding. It rejects an edited/branch worktree and existing output path; it records Git commit, code/config hashes and PBS job ID, then writes training outputs under `<CES_OUTPUT_DIR>/training/`.

PBS on this cluster may not populate CUDA_VISIBLE_DEVICES automatically. A low GPU utilization reading or selecting gpu01 is not a reservation. Check the assigned PBS resources and actual device binding; do not borrow another job's GPU or guess an index. Have the site/operator establish the correct job-specific mapping before submission/execution. The template's CUDA check verifies visibility, not ownership. Follow any required site binding mechanism; adjust the resource header only through a deliberate reviewed launcher change.

For an approved request with allocation/binding already resolved, the submission form is:

```text
qsub -v CES_SOURCE_DIR=<immutable_worktree>,CES_TRAIN_CONFIG=CES_FL/proposals/<id>/config.json,CES_OUTPUT_DIR=<new_absolute_output_package>,CUDA_VISIBLE_DEVICES=<verified_job_device> <immutable_worktree>/CES_FL/cluster/train_one_gpu.pbs
```

This is a form to fill, not a runnable command with guessed devices. The template was syntax-checked for this setup; it was not submitted or validated inside a fresh PBS allocation. Only training is covered by this template; evaluation uses the selected checkpoint and its stored validation threshold. No automatic test evaluation or image generation is attached.

GitHub permissions and cluster credentials are separate. The supervisor can propose/push without cluster access; the operator controls actual launch time and resource allocation.

## 5. Record and publish what actually happened

Capture request ID, exact launch commit, command, data/source/config/checkpoint hashes, scheduler job ID, actual GPU mapping, output paths, start/end/error state and software-test evidence. Keep failures and prior checkpoints. A training DONE marker is not by itself proof of correct selection/loading/calibration. Review and generate verification evidence before marking a new run verified.

For experiments using the current audit layout, update status/tracking.json with exact audit/launch roots and arm names. For template output packages, add a run_directories mapping such as {"new_arm": "/absolute/new_output_package/training"} and add new_arm to arms. The observer reads that explicit directory instead of the default full_experiment_runs/<arm> path. It expects config.json, history.jsonl, DONE, best.pt and RUN_VERIFIED.json for a verified completion; missing/contradictory evidence remains UNVERIFIED. History without completion is shown as recorded progress, not proof the process is currently alive. The generic training template does not create RUN_VERIFIED.json; actual post-run verification must be performed and documented before claiming a verified completion. Keep launch evidence in the new dated result directory and do not silently point the dashboard to unrelated or stale paths.

Add actual real benchmark evaluation directories to tracking.json evaluation_records after they exist (dataset, arm, directory). Never use synthetic smoke directories as test results. File presence still needs protocol review; the helper does not certify counterfactual validity, identity independence or statistical significance.

Refresh on the login node and publish only intended files:

```bash
python CES_FL/scripts/refresh_status.py --resources
git diff -- CES_FL/STATUS.md CES_FL/status/
git add -- CES_FL/STATUS.md CES_FL/status/
# Also add only the named new result summaries, request decisions and code/config changes intended for this milestone.
git diff --cached --stat
git diff --cached --check
git commit -m "Record REQUEST_ID actual progress and evidence"
git push origin main
```

Keep aggregate metrics, group counts and textual training/evaluation evidence on GitHub. Large checkpoints, datasets, face images and full private row-level outputs remain on approved cluster storage; publish their paths/hashes, not the files. GitHub is public. Never stage credentials. The initial source/results are a reproducible review snapshot, not a portable dataset release.

## 6. Freshness and failure states

The status file has an explicit UTC observation time and append-only dated snapshot. It does not keep itself current. Run the helper and push after milestones; run it manually whenever a fresh snapshot is needed. If the operator/LLM is inactive, no updates are made. Old RUNNING records are explicitly unverified until the process/job is checked; scheduler query failures are displayed. Historical results remain associated with the exact tested hash even after source edits.

## Authorized benchmark monitor exception

The31October2026 start-now batch explicitly launches a finite observation/publication monitor for its already submitted PBS jobs. It publishes only its dedicated generated status page via GitHub file-content updates; it does not change model/configuration files, submit additional jobs, retry failures or cancel jobs. The process runs on the login node, records its PID/log path in the run root, and stops on batch termination or after24h. Do not start a duplicate monitor for that batch. Its state is separate from the generic manual refresh helper; always check timestamps and monitor liveness.
