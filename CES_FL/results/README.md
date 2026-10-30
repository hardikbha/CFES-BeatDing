# Results and evidence

completed_seed17 contains copies of the actual completed Xception pilot's verification/provenance, epoch histories and pair-use reports. The corresponding exact configs are under ../configs/completed_seed17/. Group-level metrics are aggregate validation results. The historical software-test receipts are under ../provenance/. These are preserved evidence, not claims that the model was trained again after publication.

A/P/B/C use the original FF++ TRAIN/VAL roles; the current real TEST sets were not evaluated. C is invariance-only. See the report and dated handoff for fairness limitations and controls. Existing checkpoints and image datasets remain on the cluster, with paths/hashes in the evidence.

Add a new dated results directory for future work. Include actual state, command, commit, config/input/checkpoint hashes, group support, metric units and split, selection/calibration, errors and uncertainty. Keep point metrics distinct from confidence intervals and validation distinct from test. Do not upload face images, binary checkpoints, datasets or secrets to this public repository.
