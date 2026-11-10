# FF2 queue capacity expansion — 10 November 2026

Observed management UTC: 2026-11-10T05:14:29.761657+00:00.

The user requested dispatch of queued FF2 comparisons to every available GPU,
including PBS when capacity is available. The existing 21 fixed runs continue
from training commit `b7fa561b43a659b24efaf1e928d6bbf41a839b63`.

At this observation: **8 training runs active, 13 queued, 0 complete, 0 failed**;
zero final TEST evaluations. [Training snapshot](TRAINING_STATUS.md).
The completed shared SFD preparation is excluded from the 21-run count.

Eight additional standby consumers are now running, four on each of gpu01 and
gpu02, alongside the original eight. All eight extra PIDs were verified alive
and in `WAITING_FOR_SHARED_GPU` after startup verification. There was no additional
eligible GPU at inspection. They poll the existing shared pool and can start
queued runs if another device becomes eligible; maximum dispatch capacity is
16 physical GPUs, subject to actual availability. One FF2 arm per physical GPU.

Each PBS queue allows two running jobs per user; the user's two slots in each
queue were occupied by existing jobs. No PBS job was submitted. The direct route
uses explicit shared-pool-v2 leases and CUDA UUID binding; it does not claim PBS
allocation or exclusive ownership of an entire GPU.

[Controller](standby_worker.py) reuses the original pinned queue, resource pool,
CUDA checks, execution and verification functions. Startup checks the detached
controller commit, its SHA, all training source hashes, software-check receipt,
common-recipe matrix and dependency hashes. The actual `--verify-only` command
passed before launch; this check does not run a model or claim a GPU.
[Exact binding](CAPACITY_CONTROL_BINDING.json), [gpu01 launch](CAPACITY_LAUNCH_gpu01.json),
[gpu02 launch](CAPACITY_LAUNCH_gpu02.json), [observations](SNAPSHOT.json).
Controller commit is local to a separate cluster repository, with exact script
bytes and SHA included here for review. It has a 24-hour dispatch window and
retains the original bounded per-job timeout and no-automatic-retry policy.

The existing model source and all 21 configurations remain unchanged. Existing
GPU/queue locks prevent duplicate claims. Historical result snapshots are
preserved; this progress packet adds no final result or scientific claim.
GitHub is a dated snapshot, not an unattended live publication service.
