# GPU allocation and binding blocker

The 72-hour FF2 ledger and existing GPU allocations were refreshed before profiling. At launch time the shared ledger had44.141509 GPU-hours uncommitted and no active ledger reservations. The user had three other PBS jobs running; they were left untouched.

Job45394 requested `ngpus=1` on `workq` and was assigned `gpu02/8*8`. The PBS environment lacked `CUDA_VISIBLE_DEVICES`; the operator guard exited32 before model imports, weight loading, training or profiling. Job45396 requested `ngpus=1` as a short diagnostic and was assigned `gpu01/8`. Its captured environment had `CUDA_VISIBLE_DEVICES=UNSET` and `PBS_GPUFILE=UNSET`; PyTorch saw eight H100 devices. It ran no model forward or training.

Using the hostname, node slot, free-memory reading or a guessed GPU index cannot establish which physical device PBS allocated. The site must provide a job-specific mapping before CUDA smoke or timing can run. No direct-SSH/shared-pool substitution was used.

The two allocation windows, measured from PBS start to obituary to include scheduler bookends, consumed0.028611 GPU-hours. The shared ledger now records27.887102/72 used and44.112898 uncommitted. Full evidence is in the linked local job IDs and cluster receipts; this report exposes no row-level data or checkpoints.
