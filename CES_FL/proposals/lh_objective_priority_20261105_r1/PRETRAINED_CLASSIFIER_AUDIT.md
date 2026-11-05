# Complete P50 classifier audit

This is a zero-update attribution check on the pinned FF++ TRAIN manifest and clean FF++ VAL manifest. It compares the full native ResNet-50 P50 model against the selected A0, B0, C0 and H0 checkpoints. It does not read TEST, calibrate on TEST, update weights, or replace any checkpoint.

## Bound artifacts

- CES-FL source commit: `985877b11b5bd77cbcec534592eb6d14509a0af1`
- CES-FL source SHA256: `1997eebf093aa3e37de1cbbe58302a1539ac2e517f966a4607aae89dc695a8e0`
- Native P50 source SHA256: `2d66966b1733feec222fe3a80ff9a6cd7181a6ba28cb9f767ad980968d337098`
- Native P50 checkpoint SHA256: `c28bbb9cfcd06a11894027dc9369c5fd674f640de6c182c28b6d6f0aa8d77f55`
- FF++ TRAIN/VAL hashes: `6ee9326fe8361dc3aee0f803c49b5eada169be5cf3bc3cb44ad3d934ba5ca657` / `52f4844cffe5a03adb08cecde8829244bb50e05eeead36f6d659599f1c24f5b1`
- GPU cap: `0.25 aggregate GPU-hour`; no optimizer step.

The script replays the complete P50 forward path and the four selected-checkpoint predictions on the same VAL rows, checks saved-score caches, reports per-class logit distributions and calibrated/fixed-0.5 metrics, and measures responses to the pre-existing identity/JPEG90/resize90/blur035 bank on a deterministic 128-row TRAIN sample. The response measurements describe sensitivity to processing; they do not validate preserved forensic evidence.

For two prespecified TRAIN microbatches per axis (gender, race, age, skin tone, gender×race and ids8), the script reports weighted-BCE and generic soft-error gradient norms and cosine. The generic axis diagnostic is not identical to every arm's hierarchical training loss. The fixed TRAIN real subset supplies diagnostic thresholds; these do not replace the model's stored validation calibration.

All row-level scores and audit JSON remain private on the cluster. Only aggregate metrics and hashes may be published. A0/B0/C0/H0 use their exact selected checkpoints; the script verifies source/config/checkpoint and no-TEST receipts before inference.
