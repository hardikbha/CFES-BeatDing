# Ding SFD+GDA, connected-gradient 10-epoch controlled adaptation

This is a controlled adaptation row, not a reproduction of Ding's paper-reported 50-epoch GDA run. It uses the existing seed-17 FF++ classification checkpoint and its TRAIN-derived ten-channel SFD mask, then trains only the native `last_linear` classifier for ten epochs.

## Locked settings

- Seed 17, exact Ding/Lin FF++ TRAIN and validation manifests, original rows only, every TRAIN row once per epoch.
- 2,380 updates/epoch, 32 original images/update.
- ResNet is not used for this arm; the architecture is Ding's native Xception with the 512-dimensional adjusted feature head.
- The seed-17 base checkpoint is loaded exactly. The ten SFD output channels from the saved TRAIN-derived `SFD_channels.txt` are masked with the native PyTorch pruning reparameterization.
- Only `last_linear` (1,026 parameters) is trainable. All encoder parameters are frozen and all BatchNorm running statistics are frozen. Frozen BN is a declared controlled-variant difference from the archived GDA run, whose `model.train()` updates running buffers.
- SGD learning rate 0.001, momentum 0.9, weight decay 0.005; Sinkhorn `p=2`, blur/epsilon 0.0005 and fairness coefficient 0.005.
- Input is direct RGB bilinear resize to224×224 and normalization `(x-0.5)/0.5`, matching the current L_H geometry. This differs from the archived Ding 299×299 Xception preprocessing and is disclosed.
- Each main update uses a deterministic shuffled batch of32 distinct originals. The auxiliary reference samples one of eight inherited Ding gender×race groups uniformly, with32 TRAIN reals and32 TRAIN fakes sampled with replacement. The native Sinkhorn loss is applied between main logits and that balanced group reference.
- The reference group logits are detached; the fairness gradient reaches main logits and the classifier. This corrects the archived `total_loss`-inside-`torch.no_grad()` gradient break. The loss normalization and implementation are pinned by source hashes and a numerical/gradient smoke.
- Checkpoint selection is the earliest strict maximum FF++ validation AUC. Each selected checkpoint stores the validation-real empirical10% FPR threshold. Fixed probability0.5 and the stored threshold are both reported. TEST is absent from training/configuration and is evaluated only after the model is frozen.

## Protocol differences from archived native Ding

The archived GDA uses 32 real plus32 fake main examples per update, draws a balanced group reference, updates BN running buffers in train mode, and runs50 epochs. This controlled arm uses32 total main examples with each original once per epoch, freezes BN, uses10 epochs, changes training/evaluation resolution to224, selects by FF++ validation AUC, and repairs the detached fairness-gradient path. Its result must be labeled “Ding SFD+GDA, connected-gradient,10-epoch controlled adaptation.” It must remain separate from the paper-published Ding row and archived-code-as-run row.

## Evidence gate

Before launch, bind the seed-17 base checkpoint hash, SFD channel-list hash, SFD run log, upstream model/GDA source hashes, both manifests and overlap declarations. Require a fresh CUDA smoke and real TRAIN backward check; demonstrate nonzero finite fairness gradients to `last_linear`, a changed SGD momentum/update state when the fairness coefficient changes, fixed mask, frozen BN, checkpoint reload, one-original-once exposure and validation-only selection/calibration. Do not use the archived detached fairness loss as the controlled arm.

Expected training cost is estimated from 49 measured native seed-17 epoch intervals (1,827.35 seconds each at1,976 updates/epoch). Scaling to2,380 updates and96 total forward images/update (32 main +64 reference, versus128 in the native loop) gives about4.58 GPU-hours for ten epochs. Add validation passes and fresh checks; stop before launch if the updated shared ledger cannot cover the measured projection and replay.

## Operator implementation status

The standalone adapter is `CES_FL/src/ding10_connected_control.py`, SHA256 `9d480b3bf018d7fff46c840044011122b953b456df04d154f42f173ceeace324`; its locked configuration is `configs/ding10_connected.json`. CPU synthetic smoke confirms a nonzero connected fairness gradient, a changed SGD momentum/update state when lambda moves from0 to0.005, strict head reload and the ten-channel mask. CPU real-input dry-run confirms finite backward on32 TRAIN originals and zero persistent updates. Exact row matching against the archived Ding real/fake TRAIN CSVs passes for all76,139 paths, labels and eight inherited groups. These checks do not establish GPU throughput or experimental fairness results. The CUDA backward check and ten-epoch run are pending until the ResNet priority arms finish.
