# FF++ counterfactual 50+50: allocated preflight submission

Source commit: `fc4e63b8811c5813afa0d298a8172ba9555b7e40` from the user's fork. Actual immutable source checkout: `/home/sachin.chaudhary/ces_fl_sources/cesfl-cf5050-fc4e63b`.

## FF++ actual status

The requested relative directory was absent on this cluster. The exact published commit was fetched, verified and placed in a new detached worktree. Existing experiments, outputs and running sources were preserved. The publication worktree is separate from runtime source.

- Source SHA256: `3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab` — matches the handoff's seven-check CPU receipt.
- Configuration SHA256: `56e77211a966ae12a449c47cdaf8e3fc271f149c6031513747813a1d1abfaef0` — matches the request.
- TRAIN, validation, overlap-exception and ImageNet-weight byte hashes: PASS. Metadata confirms 76,139 TRAIN and 25,386 validation originals with binary labels, contributor fields, no duplicate original paths and all balancing cells. The full source-image byte inventory remains pending in the allocated plan command.
- Target environment: PyTorch 2.7.0+cu126; NumPy 2.2.6; Pillow 11.3.0; sklearn 1.7.2; diffusers 0.35.1; transformers 5.15.1; accelerate 1.10.1. Upstream CPU smoke used PyTorch 2.15.0; those upstream checks do not certify target CUDA or Qwen compatibility.
- PBS **44948.mgmt01**, job name `cf50_check`, requests `select=1:ncpus=8:ngpus=1:mem=48gb`, walltime 02:00:00. Current verified state: **QUEUED**. Scheduler reason: **User has reached queue gpu running job limit**. Two existing GPU-queue jobs are running; they were not stopped or modified.
- No CUDA pass, real-data dry-run pass, real Qwen generation, pilot quality decision, full pretraining or downstream result exists for this run yet.

The submitted job executes milestone 1 only: allocation/CUDA binding guard → exact CUDA synthetic smoke → generation plan with hashes of all TRAIN originals → one real configured backward dry-run with zero optimizer steps. The guard checks real PBS ownership, running state, assigned ngpus and host; it requires a scheduler-provided single-device binding or usable PBS_GPUFILE. It refuses to choose a device from idle-GPU observations. Binding cannot be certified while the job is queued.

The job cannot launch the two-source generator pilot, full training or any later stage. Those commands remain separate milestones from the operator handoff. Pilot, full-bank review and binding gates must not be bypassed. The runner and PBS script were syntax checked; this is not a CUDA test. Preserve any failure receipt before recovery.

## Measured resource information and pending estimates

The pinned cached Qwen snapshot exists: 35 files, 57,720,463,453 logical bytes. This is a file-size inventory, not a verified binary hash inventory. The pilot must record model binary hashes, actual generator/environment behavior, runtime, peak VRAM, image quality and output storage on its own allocated GPU. The home filesystem had approximately 84 TB available at inspection; quota/headroom and per-output storage still need validation before full generation.

The full protocol remains 76,139 sources × 7 diffusion outputs = 532,973 outputs plus 76,139 geometric controls. No silent subset is authorized. Full generation time must be projected from measured pilot seconds/output, and storage from measured per-source output size. No credible measured completion estimate is available before the pilot. Full-bank source-level review is a separate prerequisite; prompt wording and realistic appearance alone do not establish preserved forensic evidence.

After milestone 1 passes, run the separate class-balanced two-source Qwen pilot, create original/edit/no-change/geometry contact sheets, and prepare a pending review template. Do not fabricate approvals. Advance full pretraining/generation only after pilot quality and resources pass. Bind downstream B/N/C only after complete per-source reviewed coverage and a verified 50-epoch P checkpoint. No test evaluation is attached.

## FairForge-2 separate status

**DO NOT RUN.** The exact supervisor review was read from `CES_FL/proposals/ff2_supervisor_review_20261101_r1/REVIEW.md`; its reviewed baseline commit is `f73c98d28f01604e45525b2acff49cd945b1cd33`. Remaining requirements are a new exact versioned FF2 protocol and configurations; tested Ding variants with class-conditioned differentiable fairness; common two-logit score adapter; verified paired controls; per-group and paired uncertainty; group/label mapping and actual launch checks. This FF++ 50+50 configuration does not address or replace that study. No FF2 experiment or configuration change was made.

## Commands and paths

Actual qsub command and scheduler receipt: [checks/SUBMISSION.json](checks/SUBMISSION.json). Exact unchanged operational runner and resource header: [runtime/stage1_allocated.py](runtime/stage1_allocated.py) and [runtime/stage1_preflight.pbs](runtime/stage1_preflight.pbs). Source/config/input hashes: [checks/LOCAL_HASH_PREFLIGHT.json](checks/LOCAL_HASH_PREFLIGHT.json). Metadata counts: [checks/MANIFEST_METADATA.json](checks/MANIFEST_METADATA.json).

The three exact commands are preserved in the runner and upstream operator prompt; outputs will be `$CES_RUN/smoke_cuda`, `$CES_RUN/generation`, `$CES_RUN/dry_pretrain`. Existing directories are never reused silently. Run root: `/home/sachin.chaudhary/ces_fl_runs/cf5050_20261101T065603Z`. PBS log: `/home/sachin.chaudhary/ces_fl_runs/cf5050_20261101T065603Z/logs/stage1.pbs.log`.

A finite read-only observer will publish milestone 1 state/receipts for at most 24h, then stop at a terminal state or observation limit. It cannot submit, retry, cancel or advance jobs. This is newly recorded operational monitoring, not an unattended run-all chain. Always check the observation timestamp. Source remains immutable, and private face images, weights, source inventories and review rows stay outside GitHub.
