# L_H early best epoch: evidence and next tasks

Observed upstream result commit `f980e8a667edea6a1accfade28b5044f3215b6e5`, timestamp2026-11-03T12:56:52.363727+00:00. [Pinned aggregate](https://github.com/hardikbha/CFES-BeatDing/blob/f980e8a667edea6a1accfade28b5044f3215b6e5/CES_FL/results/plain_head_s17_20261103T101418Z_cluster/AGGREGATE.json). L_H has9recorded epochs out of10, not yet a verified completion. Epoch1 remains the maximum-validation-AUC checkpoint. No TEST inference. This supersedes earlier4/7epoch observations without changing the scientific config.

All F_FPR below are percentages (stored fractions multiplied by100); lower is better. AUC is also percent.

| Metric | L_A selected epoch3, complete | L_H selected epoch1, interim | L_H epoch9, diagnostic only |
|---|---:|---:|---:|
| AUC |98.232624|98.200410|98.038571|
| Gender F_FPR |5.545779|3.194007|6.917646|
| Race F_FPR |7.005508|5.630005|5.725943|
| Gender×race F_FPR |43.685672|41.083671|43.070685|
| Age F_FPR |21.478224|23.985545|29.370647|
| Skin F_FPR |2.824069|1.949776|5.161312|

Selected H versus selected A: AUC−0.032214percentage points; gender/race/joint/skin F_FPR lower by42.4/19.6/6.0/31.0%relative, while age F_FPR worsens11.7%. These are descriptive single-seed validation differences, not confidence-certified gains. The stored primary bootstrap support is746/1000below the900minimum. Do not change that cutoff.

## What the history actually shows

From epoch1 to9, H total training loss falls0.08830→0.04214, original sensitivity loss0.10115→0.01086, and original control loss0.19040→0.07974. Validation AUC and gender/age/skin disparity deteriorate. Joint F_FPR fluctuates but does not improve beyond epoch1 in these nine observations. This is consistent with overfitting or objective mismatch/competition; it does not prove a bug or establish which loss causes it.

The weighted response-alignment contribution is approximately0.02859 at epoch1 and0.02449 at epoch9; BCE is0.02011 and0.00545 respectively. Scalar loss size is not gradient magnitude. TRAIN-gradient norms/cosines and model responses on held-out procedural probes are not in this aggregate and remain unverified. Both H heads (gated and linear) select epoch1, making an objective/optimization explanation worth testing rather than blaming only the linear head. Shared P50 detection pretraining also means a good downstream model can emerge quickly.

## Next tasks if current L_H is our model candidate

1. Finish the existing ten epochs and independent selected-checkpoint replay. Keep epoch1 if the frozen AUC rule still selects it. Pin code, config, checkpoint hash and its existing validation threshold. Do not reselect the best fairness epoch or force final-epoch reporting for this historical run.
2. Complete the existing four-row head/objective comparison, paired uncertainty and full subgroup counts. Use original validation predictions to plot real-score tails, FPR/FNR versus a common operating-point grid, and joint-cell error contributions. Display all five axes and disclose age regression. No new training is needed for these stored-score diagnostics. Obtain actual permitted images before showing purported failure examples.
3. Before longer training, run ONE prospective component ablation under the current10epoch recipe: L_H with response-alignment weight0, retaining all other settings and objectives. Its purpose is to test whether alignment is useful, not to guarantee a win. Existing strict configs require positive weights, so implementation/tests and an exact new proposal are required before this ablation can run. The full model and L_A already provide controls. This ablation is RECOMMENDED, not implemented or launched here.
4. If the gradient/ablation evidence points to optimization drift, test one explicitly declared learning-rate schedule in a later controlled experiment. Keep losses/sampling/initialization fixed when testing the schedule. Do not simultaneously change batch size, losses, optimizer and checkpoint criterion then attribute an improvement to the scheduler. Later-epoch monotonic gains are not the scientific goal.
5. Once the model/protocol decision is frozen, prepare a single locked evaluation of selected L_H and L_A on FF++TEST(Table1) and DFDC/Celeb-DF/DFD(Table2), with common metrics, declared overlaps, fixed thresholds and bootstrap support. Table3 requires a separately trained ResNet version; Xception cannot fill it. FairForge-2 in-domain training/evaluation is a separate study. Do not use these TEST results to redesign this candidate.
6. Write claims around measured contributions: a plain linear detector plus hierarchical error control and procedural response regularization. The current implementation does not validate demographic counterfactual edits or removal of real manipulation evidence. Novelty requires related-work review and component evidence, not a numerical rating.

The prepared matched50+50 recipe remains useful for a later training-budget comparison, but changes many aspects of current L_H and should not be the immediate fix for epoch1 selection. Its separate baseline/GPU-budget issues remain unresolved. No new Ding computation is included.

## Conversation continuity

The user-provided local session archive is `codex-session-01a1a10b-db89-7fd3-bf63-d575b1c96349.md` in the fairness_cvpr workspace. Its relevant L_H/protocol sections were inspected alongside native code/config/results. Preserve it locally; do not upload the raw conversation, which contains private paths and operational history. This review and PROJECT_MEMORY summarize the evidence needed to continue without treating old snapshots as live status.
