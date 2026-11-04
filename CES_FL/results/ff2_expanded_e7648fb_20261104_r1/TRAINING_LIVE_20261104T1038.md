# F3Net and CORE training — live snapshot

Observed 4 November 2026,16:06 IST. Exact source commit `e7648fb25aeb16d49bbef33ad998e1b1cf5a5016`; seed17; FairForge-2 public split; source configs unchanged.

| Method | State | Host / GPU | Config digest | Planned | Training output |
|---|---|---|---|---|---|
| F3Net, AI-Face FAD-only adapter | RUNNING; epoch35 is latest validation record; epoch36 checkpoint is at step60,500 | gpu02 / index7 / `GPU-65f63cc5-9366-5c2d-196a-6e7a0a62c1e8` | `ae529c3af7dc58f55bc1b27064ca866a5eb00013a33e2cb4cd548d74741b2bcf` |50epochs, cap22.75GPUh | `/home/sachin.chaudhary/ces_fl_runs/ff2_expanded_e7648fb/trainings/f3net_seed17_50ep` |
| CORE, same-source flip pair adapter | RUNNING; epoch20 is latest validation record; epoch21 checkpoint is at step34,500 | gpu01 / index5 / `GPU-789d1ee4-d888-8a11-c94e-3d866900bc4d` | `b861f3e0aa608430cbcf53848f446be93364abd342b486aa377ba5946bacf389` |50epochs, cap19.25GPUh | `/home/sachin.chaudhary/ces_fl_runs/ff2_expanded_e7648fb/trainings/core_seed17_50ep` |

Both direct SSH launches use separate shared-pool-v2 UUID leases; these are disclosed as shared reservations, not exclusive physical ownership or PBS allocations. CUDA visibility is restricted to one UUID per worker and checked inside each process. Both passed synthetic CUDA smoke and20step real-TRAIN forward/backward/timing profiles. The profiles used saved input hashes from the exact same TRAIN/VAL manifests. F3Net's latest completed validation AUC is98.800723% at epoch35; CORE's is99.665503% at epoch20. These are intermediate VAL scores, not final checkpoint selections. No TEST data or threshold search is involved.

Shared72hour FF2 ledger:28.010872GPUh charged,42.000GPUh reserved for these two runs,1.989128GPUh uncommitted. These caps include a40% margin and.25h overhead over the measured20step training estimates. Neither run may alter epochs, batch size, optimizer or selection to squeeze into its cap. Incomplete runs must remain marked incomplete.

At this snapshot the shared lease processes are F3 controller1586452/worker1586704 on gpu02 and CORE controller4035250/worker4035508 on gpu01. F3's current `last.pt` SHA256 is `46e6ee6643a9036e4892dd56ba71d61dd5ec2c67a5dbf3c5943ca259feb650cd`; it is a resumable mid-epoch checkpoint, not a selected checkpoint. CORE is also mid-epoch. Training logs are under each `run_ff2_expanded_METHOD_TIMESTAMP/controller.log` and `budgeted_session/command.log`; per-method checkpoints and histories are written only to the private training output paths above.

Other profiled methods: EfficientNet-B4's54.798GPUh margin projection does not fit; UnivFD's38.058h projection fits only by itself; SPSL31.430h fits alone but not in addition to another selected run; SRM CUDA smoke passed but its real-TRAIN profile OOMed under shared card pressure; ViT-B/16 lacks a compatible local weight; UCF and PG-FDD lack verified Specific labels and native paired adapters. No other method was launched.
