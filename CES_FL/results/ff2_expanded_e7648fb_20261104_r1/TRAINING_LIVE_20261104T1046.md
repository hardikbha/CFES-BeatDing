# F3Net and CORE training — 4 November 2026, 10:46 IST

Both seed17 runs are active under the frozen50-epoch configs. This is a mid-epoch checkpoint observation; no validation epoch has completed and no score is reported yet.

| Method | Node/GPU lease | Progress at last atomic checkpoint | Last checkpoint SHA256 | Selected VAL checkpoint |
|---|---|---:|---|---|
| F3Net, AI-Face FAD-only adaptation | gpu02, GPU7, `GPU-65f63cc5-9366-5c2d-196a-6e7a0a62c1e8` | epoch1, batch500 of1706 | `d0207424b39be33b4122e8dc2a076411323c1c4deedd40e80a552305ab77806b` | not yet selected |
| CORE, paired same-source flip adaptation | gpu01, GPU5, `GPU-789d1ee4-d888-8a11-c94e-3d866900bc4d` | epoch1, batch1000 of1706 | `88e4668d50d507ae557f3bbbaad0eaf4835a87333b4a3383b3990084d43c4d89` | not yet selected |

The shared72-hour ledger at this snapshot has28.010872GPUh charged,42.000GPUh reserved across these two methods,2 direct-SSH UUID leases active and1.989128GPUh uncommitted. These are shared-pool allocations with disclosed UUIDs, not exclusive physical ownership. The configs, source, data split, validation selection and thresholds remain unchanged. No TEST read.

Private run logs/checkpoints stay on cluster under `/home/sachin.chaudhary/ces_fl_runs/ff2_expanded_e7648fb/`; controller and worker PIDs are recorded in each session's `PROCESS.json`. The public handoff shares only aggregate progress and checkpoint hashes.
