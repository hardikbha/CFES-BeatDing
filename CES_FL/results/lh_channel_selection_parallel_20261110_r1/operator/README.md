# Bounded parallel operator

The user explicitly changed the pinned prompt's one-GPU/sequential scheduling
constraint to parallel execution on available GPUs. At most three physical GPUs
run the three registered configurations; the two aggregate GPU-hour allowance,
three epochs, thirty-minute arm limits and scientific gates remain unchanged.
Scientific code executes from clean detached commit 1a6d63db6baf7f6ad2a8e842a31341d2c128de25.

The controller reserves two hours in the existing original L_H ledger, preserving
historical sessions. It charges the sum of actual physical-device-held time.
One accounting session uses ngpus=1 with elapsed_seconds explicitly defined as
the sum across devices; this is not an assertion of one physical device.
It uses the existing pool-v2 mutex, UUID leases, reservations and utilization
checks. No other process is terminated. A separate detached controller commit
and all helper/science hashes are verified before acquisition.
The compute nodes do not need a Git executable: detached HEAD is read from the
shared Git/worktree metadata and every bound execution file is hashed. A clean
Git status is also checked on the management node before launching.

Only after human qualification: one GPU performs CUDA smoke, frozen TRAIN mask
selection, all three discarded real-input updates and a timing-only calibration/
VAL profile. No VAL score is used to select masks. The profile discards its
logits and computes no validation metric. A conservative cost estimate treats
every production step as the largest joint-event dry-run, adds measured setup,
three TRAIN calibration forwards, four VAL forwards, checkpoint allowance and
a 25% concurrency margin. All three arms must fit both individual and aggregate
limits before the other GPUs are acquired and any arm starts.

External guards cover initialization, data/I/O, checks, training and replay.
They stop only owned process groups before the two-hour aggregate/cohort and
thirty-minute arm boundaries, preserving failures. No automatic retries,
TEST/CAL scoring, new Ding experiment or extension is performed. Completed
arms must pass native selected-model replay before CPU-only comparisons.
Planning requires at most29minutes per arm; an independent watchdog stops
owned jobs by29.5minutes and at1.9aggregate GPU-hours, retaining cleanup margin
inside the documented30minute/2GPU-hour limits. Budget-heartbeat file locks
do not prevent that watchdog from stopping owned children.
