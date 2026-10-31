# Authorized seed17 benchmark batch

User instruction on31October2026: start the experiments now. This request completes exploratory benchmark evaluation and a matched ResNet comparison; it does not rerun Ding or relax the historical gate.

Phases:
- X_evaluation: reuse the selected Xception A/B/C checkpoints. Verify B/C validation replication and paired error counts, reuse existing controls, then evaluate fixed checkpoints on FF++TEST,DFDC,Celeb-DF,DFD and FairForge-2.
- P_resnet50: cached ImageNet1K-V1 initialized ResNet, original-TRAIN-only end-to-end BCE pretraining.
- A_resnet50: frozen ImageNet encoder and fresh CES-FL heads; BCE; benchmark selected checkpoint.
- B_resnet50: after successful P, freeze exactly its selected encoder, fresh identical heads, BCE; benchmark.
- C_resnet50: after successful P, same encoder/head initialization, BCE plus existing approved demographic invariance; benchmark.

Saved settings:seed17,10epochs,batch16,47590updates,headLR.001,encoderLR.0001,AdamWweightdecay.0001,balanced(label,IDS8)sampling,224bilinear preprocessing with the existing backbone-specific normalization,earliest best validation-AUC selection,stored validation-real FPR10% calibration plus separate fixed probability0.5 reporting. Forensics and map-consistency weights remain0;36reviewed variants/six TRAIN sources only. No new images are generated.

Manifest checks opened all173665evaluation rows, checked every referenced file and decoded132stratified samples across the five datasets. Full decoding happens during inference. Native DFDC has22857rows but21381uniqueimages:1476exact repeated rows with identical labels/groups. Their native weighting is preserved in the explicit test-only adapter; the strict model/training loader is unchanged. Native groups lacking both classes retain undefined AUC and are excluded by the existing minimum20/class summary rule, with counts reported.

The cached official V1 weights use legacy tar serialization unsupported by weights_only=True in the current PyTorch. normalize_weights.py converted only that format, after checking its official cached hash prefix19c8e357, and verified equality of all267tensor entries and metadata. The old protocol/config copies and conversion receipt are preserved. No model/loss code or numerical initialization values changed.

Bootstrap:1000replicates,seed17,connected components of recorded contributors. No CI is claimed if fewer than2components exist. Proxy identities/hidden donors and a single seed limit inference. These datasets are exploratory test benchmarks; future model development must disclose their inspection. FairForge-2 here is cross-dataset evaluation of FF++-trained models, not FairForge-2 in-domain training.

See protocol.json for exact input/checkpoint/weight hashes and paths, configs/ for actual settings, and the run root for software and runtime evidence. PBS jobs request ngpus=1; B/C use afterok:P dependencies. The queue currently limits each user to2running GPU jobs, and both slots are occupied by other projects. Jobs wait for an actually empty physical device under the allocation; no existing process is evicted. Job IDs and the pinned Git source commit are recorded after submission.
