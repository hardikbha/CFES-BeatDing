# CVPR full-effort plan — actual evidence and next gates

**Superseding observation: 9 Nov 2026, 01:43 IST.** Three downstream RF-SJg seeds and the frozen-list TEST exports already completed on the original RFA branch (`30a6b263`). Do not launch duplicate confirmation or TEST jobs. RF-SJg TEST AUC `98.60±0.06%`, gender×race `100F_FPR=47.69±3.91`, versus RF-S `98.98±0.01%` / `60.95±1.92`; gender and age disparity regress. Local corrected GDA is `93.08%` / `9.47`, so an overall superiority claim is unsupported. All repeats share one seed17 pretrained encoder. Current manuscript and public aggregate audit are on `paper/ff2-processing-fairness-20261109`. The 8 Nov plan below is historical; TEST is now exposed. Shared ledger charged `64.6011/72 GPU-h`, with TEST/sanity duration reconciliation pending. No new training or inference in this publication.

**Observed:** 8 Nov 2026, 20:21 IST.
**Acceptance:** no experiment can make CVPR acceptance a sure thing. The goal is to build a clear, reproducible claim with matched baselines, independent seeds and a locked test evaluation.

## Current strongest FF2 candidate

The current best candidate is RF-SJg: a real-only group FPR-risk objective with gentle q75/4:2:0 JPEG augmentation on the classification stream. It is a stretch branch of the real-only RF-S method, not the full counterfactual CES-FL objective.

Seed-17 SELECT result:
- RF-SJg: AUC 99.7549%, FNR 0.4500%, gender×race F_FPR 27.5259%, selected epoch 2.
- Matched RF-S: AUC 99.8032%, FNR 0.2930%, gender×race F_FPR 34.1222%, selected epoch 2.
- Paired seed-17 change: gender×race F_FPR −6.5963 points, AUC −0.0483 points, FNR +0.1570 points.
- It remains far above Ding corrected-GDA's 12.44% FF2 SELECT reference. This is a candidate improvement over RF-S, not a Ding win.

The RF-S no-JPEG control is verified at seeds 7, 17 and 27: mean AUC 99.80±0.01%, FNR 0.29±0.01%, and gender×race F_FPR 34.24±0.21%. Gentle JPEG has only seed 17 so far.

## Work now underway

The fixed-recipe RF-SJg seed-7/27 confirmation is in the pushed branch:
- Branch: experiment/ff2-sjgentle-confirm-20261108
- Launch source commit: 91106b01ab28124c207319707ba841bfd32921d7
- Model code hashes: real-FPR 2d99295cc0b8190a571695b581ef5ba73e7b5fc18994461bdf1a21c1d83089d1; core 8c25ba8cb844b059adbbce75b57785a156333182afa2d5d576985997f2d36733.
- Seed configs are copies of SJg seed17 with only the seed changed.
- Both seed preflight receipts PASS using the identical existing decoded/hash-bound inventory, 218,266 TRAIN / 31,107 SELECT, zero TRAIN/SELECT byte overlap, and no TEST access.
- Seed 7 is running on gpu02 under a direct shared-GPU lease; CUDA is bound to one visible H100, physical UUID GPU-65f63cc5-9366-5c2d-196a-6e7a0a62c1e8. At 20:33 IST it is in epoch 4 around step 4,700/6,821; epoch 2 remains the best feasible checkpoint so far, AUC 99.7603%, FNR 0.4465%, gender×race F_FPR 30.8859%. No TEST was read.
- Four GPU-hours are reserved for seed 7; the 72-hour shared ledger shows 61.8905 charged, 4 reserved and 6.1095 unreserved.
- Seed 27 is pending after seed 7 completes and actual charges are reconciled.

Earlier wrapper/preflight failures are preserved in the run log directory. The first generic preflight rejected RF-SJg's extra config keys; the next attempt to rehash all 249,373 images was stopped as redundant because prior full decoded inventories had identical hashes. The seed-specific inventory binding then passed. The first GPU launch attempt exited before CUDA/training because git is not installed on gpu02; zero GPU-hours were charged, and the wrapper was corrected to verify exact source/config SHA256 on-node. The successful seed-7 launch uses the shared-GPU lease lock; it did not preempt or alter another user's PBS job.

## Frozen confirmation decision

Compare RF-SJg against RF-S at the same seeds, using each checkpoint's stored SELECT-real 10% FPR threshold. Report the individual paired differences and the three-seed mean±standard deviation. The existing selector and guards stay fixed: minimize hard gender×race F_FPR subject to AUC≥99.5% and FNR≤0.5651%; ties go to higher AUC then earlier epoch.

The confirmation requires both new seeds to select feasible checkpoints, mean paired gender×race F_FPR below zero, mean AUC loss no greater than 0.5 percentage points, and each selected run to satisfy the existing absolute AUC/FNR guards. No post hoc seed/model selection and no TEST tuning.

If the confirmation succeeds, freeze the RF-SJg model and final evaluation plan before one held-out TEST evaluation. If it fails, record the failure and stop this branch. A successful FF2 validation confirmation still would not establish superiority over Ding or across datasets.

## CVPR research plan beyond this confirmation

1. **Lock the paper's central claim.** Keep the current real-only RF-SJg result separate from full CES-FL. Do not call a loss-plus-JPEG pilot the validated counterfactual-selectivity method.
2. **Establish the replication.** Complete seeds 7 and 27 under the exact frozen configuration and matched RF-S controls; report per-seed results, group support, paired contributor-cluster uncertainty and actual compute.
3. **Resolve benchmark integrity.** Treat the Xception best-per-cell Ding PDF as a retrospective descriptive oracle; it selects seeds independently by test metric. Do not use those counts as a paper result. The ResNet FF++ audit records 231 fit/test overlaps in recorded IDs; true-person separation is unverified.
4. **Improve the comparison evidence.** Recompute paper tables from locked per-method checkpoints and seed-level means, not best cells. Distinguish published references, historical reproductions and our runs. Include AUC, FPR, FNR, worst-group support, all demographic axes and uncertainty.
5. **Keep TEST sealed.** Freeze model, checkpoint, thresholds, and analysis code before the one authorized final FF2 TEST evaluation. Never choose a method using TEST scores.
6. **Address label validity.** FairForge-2 age/skin groups currently use automated estimates. Any CVPR claim about age/skin fairness requires human-reviewed labels or a clearly narrower proxy-label claim.
7. **Reopen counterfactual CES-FL only with valid interventions.** The matched-edit/forensic-validity gate remains unresolved. No claim of forensic sensitivity or full CES-FL fairness is supported until valid pairs and controls pass independent review.
8. **Do not add ad hoc arms.** Any next method must be a dated, source/config-pinned proposal with one primary hypothesis, matched controls, success/stop criteria and a compute cap.

## Remaining blockers

- The full counterfactual CES-FL objective has not run on FairForge-2 because intervention validity remains unproven.
- RF-SJg is single-seed until the two confirmations finish.
- FairForge-2 TEST remains unrun for in-domain models.
- Some model rows (PG-FDD; verified F3Net/CORE) are still missing from the current FairForge table.
- Current GPU queue limit blocks the submitted duplicate wrappers. The unrelated active jobs identified by the user were left untouched. Only use an idle device if the shared-lease check still passes at launch.
