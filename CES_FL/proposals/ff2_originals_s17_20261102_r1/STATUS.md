# FF2 original-only pretraining — launched

Observed management UTC: 2026-11-02T05:33:39.225497+00:00.

- Input/source/config/weights hashes PASS; all336,937paths and labels checked; recorded cross-split contributor/path overlaps zero.64TRAIN+64VAL stratified decodes PASS; TESTnotdecodedorscored.
- Native CUDA synthetic suite PASS. Real TRAIN-only two-step encoder/head/gate/fusion backward/update test PASS; losses0.7159984708 and0.7062544227. Full21,594,159parameters trainable; encoder weights and BNbuffers changed.
- Authorized10-epoch P training process launched on `gpu01`, physicalGPU4 (`GPU-9b669a06-9156-a98e-2194-36f2fc11d7ca`), childPID2525928, workerPID2525919. Initial `FF2_TRAIN_START` recorded. Manifest loading precedes first optimizer updates; no completed epoch is claimed in this snapshot.
- Direct SSH with cooperative idle-device leases and20%VRAM cap; noPBSreservationclaimed, no existingjobstopped.
- Full matched Ding comparison, RN/RD and other paired arms remain blocked; no TESTevaluationorautomaticdownstreamrun.

Actual run root: `/home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1`; log: `logs/train.log`; checkpoints/history: `training/`; receipts: `checks/`. Adjacent copied receipts are actual launch/preflight evidence. [Milestone observations](../../status/ff2_originals_s17_20261102_r1_live.md) will be produced by the finite read-only observer; read its timestamp.
