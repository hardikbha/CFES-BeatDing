# Fixed TRAIN checkpoint audit before the lower-LR run

Executed `diagnose_train.py --device cpu --out /home/sachin.chaudhary/ces_fl_runs/lh_low_lr3_s17_20261103T140700Z/train_drift` with aiface Python, float32 inference, two CPU threads.64fixed TRAIN originals:32real/32fake, with32epoch0 real procedural/sham pairs. Both checkpoints receive identical decoded image tensors and pair settings. Every sampled image hash also matches stored fit-membership in BOTH original checkpoints. No gradients or model changes; TRAIN only.

|Diagnostic|Original selected best, epoch1|Original last, epoch10|
|---|---:|---:|
|Mean real original logit|−7.507145|−61.929424|
|Mean fake original logit|11.225762|105.755539|
|Mean absolute real sham−original logit|0.801508|4.501423|
|Mean absolute asinh(sham)−asinh(original)|0.141542|0.081814|
|Sum transformed difference / sum raw difference|0.176595|0.018175|
|Mean asinh derivative at real original logit|0.136471|0.016778|

On this small fixed TRAIN sample, real/fake score magnitudes grow strongly. Raw sham differences become larger while the current transformed consistency measure becomes smaller. This is consistent with `asinh` having derivative `1/sqrt(1+z²)`; it does not prove the consistency objective caused the drift or establish which gradient dominates. TRAIN observations cannot establish validation or unseen-data performance. No loss, score transform or sampling setting is changed based on the observation.

Original checkpoint hashes: best `6ef2caad3979d101fd94d7350aa3c1a50f95ee8307673cfaae63ac9d89af36f0`; last `1ad77edd4610d5c2dfb29eda35b842a58266140e2fc205ba5e9ff7748432c9d2`. The old retained checkpoint remains epoch1. No threshold is recalibrated by this diagnostic.

The user-authorized next run changes only both learning rates by0.1 and limits duration to3epochs, using the SAME initial P50/fresh seed17 head. Compare with original epochs1–3 and retained best. Large-logit compression remains a known limitation because the loss stays unchanged.

Full aggregate distributions and coverage are in `evidence/TRAIN_DRIFT.json`; source-image paths and row-level scores remain private. The new input preflight is still pending at this documentation snapshot. Held PBS45321.mgmt01 requests ngpus1,ncpus6,32GB and is not released until all CPU/input/qualification gates pass. Source execution is detached29ce92ee8631c9c11ee3649d4f5a1110ecba29fd, scientific SHA08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177.
