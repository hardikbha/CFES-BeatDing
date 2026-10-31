# Shared research updates and reviewer index

Repository: https://github.com/hardikbha/CFES-BeatDing

This public repository is the shared review record for the cluster experiments. Anyone with the link can read it. Writing changes requires GitHub permission; training execution separately requires cluster access and an approved configuration.

## Where to look

| File or folder | What it provides |
|---|---|
| [Controlled upgrade execution](CES_FL/status/upgrade_invariance_20261031_r1_20261031T135409Z_live.md) | Actual smoke/training job IDs, CUDA gate and validation results for the five explicitly authorized upgrade arms. |
| [Upgrade execution receipt](CES_FL/proposals/upgrade_invariance_20261031_r1/execution_20261031T135409Z/README.md) | Exact fetched commit, detached worktree, input/CPU checks, unchanged configurations and operational execution code/logs. |
| [Current benchmark observations](CES_FL/status/benchmark_s17_20261031T112401_live.md) | Timestamped scheduler states, recorded training progress and emerging test results from the authorized seed17 batch. |
| [FairForge-2 in-domain audit](FAIRFORGE2_INDOMAIN_RESULTS_20261031.md) | Verified historical train-on-FF2/test-on-FF2 results, internal versus public split distinctions, and exact cluster evidence paths. |
| [CES-FL README](CES_FL/README.md) | Current-method overview and links to experiment records. |
| [Source](CES_FL/src/) and [configurations](CES_FL/configs/) | Reviewable implementation and recorded configurations; each result must retain its actual source/configuration identity. |
| [Completed seed17 evidence](CES_FL/results/completed_seed17/) | Historical pilot histories and verification records. These are preserved when new results appear. |
| [Experiment proposals](CES_FL/proposals/) | Explicit requests, dependencies, configurations and launch evidence. A Git push does not launch training. |
| [Supervisor workflow](CES_FL/WORKFLOW.md) | How to review code, propose changes and hand off an exact commit to the cluster operator. |
| [Supervisor/LLM prompt](CES_FL/SUPERVISOR_PROMPT.md) | Starting instructions for another reviewer or research assistant. |

## Publication policy

Publish scoped updates after verified launches, failures, checkpoint selection, evaluation, code/configuration changes and review decisions. Record the experiment ID, observation time, source/configuration hashes, dataset/split identity, seed, scheduler state, checkpoint/result paths, measured metrics and unresolved blockers as applicable. Preserve prior results and label incomplete or unchecked work explicitly.

Keep paper values, local reproductions, historical methods and the current CES-FL method distinct. Separate validation from test and FF2 in-domain results from FF++-to-FF2 transfer. Do not report a fairness improvement merely from detection AUC.

Publish aggregate metrics and relevant textual evidence. Dataset images, private row-level records, credentials and binary weights stay on the cluster; publish permitted paths and hashes rather than those files.

## Freshness and automatic updates

The already authorized benchmark has a finite observation/publication monitor. At **2026-10-31 11:47 UTC**, PID 1762946 was present and its log showed a successful GitHub publication at 11:47:11 UTC. It observes only that submitted batch, stops on completion/failure or after 24 hours, and does not submit new experiments. Its log is `/home/sachin.chaudhary/ces_fl_pipeline/experiments/benchmark_s17_20261031T112401/logs/github_monitor.log`.

The separately authorized controlled-upgrade handoff has its own finite executor, started31October2026 as PID3531362. It may submit only the five exact user-approved configurations after actual CUDA smoke succeeds, stops submissions on failed checks, and publishes that request for at most24h. It cannot authorize additional experiments and does not retry failures, generate images or evaluate test sets. Read its execution receipt and live observations above.

GitHub displays the last successfully published observation, not a direct connection to the scheduler. Read the timestamp; a stale RUNNING entry does not prove a job is still active. Outside these finite request-specific processes, updates are made when the researcher/assistant checks and publishes the evidence; indefinite background monitoring is not configured.

## Update record

- **31 October 2026:** Fetched and verified upgrade commit875bce025084871438ef799ca9086985c4375672 in a new detached worktree. All five real-input preflights and19target-environment CPU tests passed. Submitted initial smoke-only job44880.mgmt01; at submission it was queued and CUDA remained unverified. The guarded executor will publish actual training IDs only after actual smoke success.
- **31 October 2026:** Published the verified FairForge-2 in-domain audit. Historical internal/full FF2 results include Ding classification 99.26% AUC, Ding SFD+GDA 99.36%, older CFES detection-only 97.60% and older CFES 98.10%. Separately, public-only CFES-D reached 98.26%; completed matched public-split Ding/DAW/DAG/PG retrains were not found. See the audit for limitations and exact artifacts.
- **31 October 2026:** Added this shared index and README links; verified public visibility and the existing benchmark publisher. No new training, generation, job cancellation or research-protocol change was made for this documentation update.
