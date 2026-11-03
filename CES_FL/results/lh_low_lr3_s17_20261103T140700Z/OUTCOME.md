# Completed bounded L_H learning-rate experiment

Management observation:3November2026,15:11UTC. Exactly3/3epochs completed; native selected-checkpoint replay PASS and independent saved-score/config/exposure verification PASS. Both PBS45321(training) and45322(CPU comparison) exited0. No fourth epoch, retry, automatic extension or TEST computation.

Decision: **MIXED; does not demonstrate an overall improvement over retained original L_H.** No binary success gate or significance claim was preregistered. The new AUC trajectory improves across its three epochs, unlike the original first-three-epoch decline, but the retained best remains the relevant selected-model reference.

|Selected model, validation|Original L_H epoch1|Lower LR epoch3|
|---|---:|---:|
|AUC %|98.200410|98.185660|
|FNR %|4.306377|4.611628|
|Achieved global FPR %|9.995112|9.995112|
|Gender F_FPR fraction|0.031940|0.043699|
|Race F_FPR fraction|0.056300|0.094555|
|Gender×race F_FPR fraction|0.410837|0.418939|
|Age F_FPR fraction|0.239855|0.192721|
|Skin-bin F_FPR fraction|0.019498|0.036871|
|ids8 F_FPR fraction|0.438923|0.337456|

AUC−0.014750percentage points and FNR+0.305250points. Age/ids8 false-positive disparity improves, while gender/race/joint/skin disparity worsens. The joint worst-group AUC improves93.688088→93.872260%, but age worst-group AUC falls97.423897→97.041648%. All other axes, gaps, F_DP,es_AUC,FPR/FNR and group denominators remain in the full report/CSVs. No favorable epoch or metric is substituted for the highest-validation-AUC model.

The original fixed-TRAIN best-versus-last diagnostic measured mean real logit−7.51→−61.93 and fake11.23→105.76. Raw real sham difference0.801508→4.501423 while the actual transformed difference0.141542→0.081814. This confirms score-scale drift and transformed-score compression on those64TRAIN examples; it does not establish causality or show that the lower-LR run eliminates drift. The loss was not changed.

## Exact provenance

- Execution commit: `29ce92ee8631c9c11ee3649d4f5a1110ecba29fd`.
- Unmodified scientific source SHA256: `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`.
- Config SHA256: `7ac0d405161210e11846ce24af420a77ebdb16074cf39df8d0f80e5742fe6085`; semantic digest `980e6fe1c994f2beaea674889d5566ea5a045a17003136767999ca0154fc117f`.
- Selected checkpoint: epoch3, SHA256 `59143159eea223d97bc436b510f13ca90b750f00851071a860f91ec9231d4a06`.
- Changes only: head_lr0.0001,encoder_lr0.000001,epochs3. Initialization, all losses/coefficients, batch32, preprocessing, frozen BN, sampling and selection/calibration unchanged. Full initial state exactly matches original fresh seed17/P50 state.
- Fresh full preflight101,525TRAIN/VALimages PASS;88byte-identical procedural review cases reused with original named AI scope;11focused tests PASS; CUDA synthetic smoke PASS; real configuration-specific discarded optimizer step PASS, fairness gradient3.3687683553nonzero. No synthetic score is presented as scientific evidence.
- All first-three-epoch exposure fields exactly match original L_H, including every original once/epoch,12,926real pairs/epoch and2,380updates/epoch.
- Native train time1,774.036seconds(29.57minutes). Full scheduler allocation2,060seconds(34minutes20seconds),**0.572222GPUh**, including7.589778seconds of scheduler startup/cleanup beyond the guard. Cap1.5GPUh respected. Original ledger15.936613/24used;8.063387remaining; FF2 ledger untouched.
- GPU02,H100GPU5,UUID `GPU-bff97893-385b-6f3f-831b-113284c2e1f1`; actual ngpus1,ncpus6,32GB PBS allocation, existing explicitly authorized shared UUID lease. Physical exclusivity not claimed.

Private run/log root: `/home/sachin.chaudhary/ces_fl_runs/lh_low_lr3_s17_20261103T140700Z`. Checkpoints, images/reviews and row scores stay on cluster. Aggregate publication: branch `experiment/lh-low-lr3-s17-20261103` in hardikbha/CFES-BeatDing.

Single-seed validation-development evidence after broader prior TEST exposure; not a fresh confirmatory result. Proxy demographic-label and known contributor-overlap limitations remain. No new uncertainty interval was computed, so no statistical superiority, equivalence or Ding-win claim is made. This authorized experiment is finished; additional training/evaluation requires a separate request.
