# RF-SJg seed confirmation — FairForge-2 public

**Scope:** replicate the already completed gentle JPEG real-only-risk run (SJg) at seeds 7 and 27, paired with the completed RF-S controls at the same seeds. Only the random seed changes from the seed-17 SJg config. No new loss, model component, augmentation setting, dataset split, or threshold is introduced.

This is an exploratory confirmation of a validation finding. It is not the full counterfactual CES-FL objective and does not test forensic sensitivity or establish a win over Ding.

## Existing seed-17 evidence

- RF-SJg: completed 10/10 epochs; selected epoch 2; FF2 SELECT AUC **99.7549%**, FNR **0.4500%**, gender×race F_FPR **27.5259%**.
- Matched RF-S seed 17: completed 10/10 epochs; selected epoch 2; SELECT AUC **99.8032%**, FNR **0.2930%**, gender×race F_FPR **34.1222%**.
- Seed-17 paired changes: F_FPR **−6.5963 points**, AUC **−0.0483 points**, FNR **+0.1570 points**.
- The predeclared stretch target against Ding’s corrected GDA (12.44% gender×race F_FPR) was not met.

The RF-S controls for seeds 7 and 27 are also complete and verified. Their outputs are at `/home/sachin.chaudhary/ces_fl_runs/ff2_rfa_20261107/S_s7/out` and `S_s27/out`.

## Frozen hypothesis and decision rule

**Hypothesis:** gentle q75/4:2:0 JPEG augmentation added to RF-S lowers gender×race false-positive disparity on SELECT compared with RF-S at the same seed, without an unacceptable detection loss.

- Primary comparison: paired seed differences `SJg F_FPR − S F_FPR` at the same global threshold calibrated from that checkpoint’s SELECT real images to the fixed 10% empirical target.
- Report each seed pair and the three-seed mean and standard deviation. Seed 17 is the existing pilot; do not rerun it.
- Confirmation requires both new runs to select feasible checkpoints, the mean paired F_FPR difference to be below zero, mean AUC loss to be no more than 0.5 percentage points versus matched RF-S, and each run to satisfy the existing guards: AUC ≥ 0.995 and FNR ≤ 0.005651.
- Use the trainer’s unchanged selector: minimum hard gender×race F_FPR among epochs meeting those AUC/FNR guards; ties go to higher AUC, then earlier epoch.
- If a run has no feasible epoch, retain and report that failure. Do not relax a guard or select a different epoch after seeing outcomes.
- Do not read TEST, tune thresholds on TEST, or attach automatic TEST inference.

## Fixed recipe and provenance

- Base source commit: `9b4b0f42f1fe3823b6abdae826db0f9d63853cfa`; new proposal commit contains only proposal/config/operator files. The model source remains byte-identical.
- Model source SHA256: `ces_fl_realfpr.py` = `2d99295cc0b8190a571695b581ef5ba73e7b5fc18994461bdf1a21c1d83089d1`; head-ablation core = `8c25ba8cb844b059adbbce75b57785a156333182afa2d5d576985997f2d36733`.
- TRAIN SHA256: `4f83e7dc8042533329d7047fefd0f30cbd96f45d42bde2666dd0edc7d58f29cc`; SELECT SHA256: `aeda051259d9e26b07c2810152b601ae115ce7c9e10523f99ea07bbc9f92afa8`.
- Encoder is the same FF2 BCE50 transfer checkpoint, SHA256 `741a8eb5b60ac8efe2ca34be985b423de12facac053acac7f05803df23248a23`.
- Gentle JPEG remains unchanged: 15% of classification-stream images get the exact q75/4:2:0 operation; no random-quality branch, no ramp. The fairness microbatches and validation images remain unmodified.
- Seed-7 config SHA256: `74d487a07134052cae90de832d6755486e4d8d5c08a42024daa3b763a7d35f74`; seed-27 config SHA256: `9dee4b7c468f3fdf2e2b58bf50d09bd60f5df4e158171d09a371a53caf27467f`. These configs are copies of SJg seed17 with only `seed` changed.

## Compute and execution

The shared 72 GPU-hour ledger was last observed at **61.8905 hours charged**, leaving **10.1095 hours**. Reserve at most 4 hours for each of these two runs (8 total); actual RF-SJg seed17 took about 1.169 GPU-hours. Run sequentially, record allocation/binding and charge actual time. If the ledger changes or the requested reserve no longer fits, stop.

A first generic base-head preflight attempt failed because the RF-SJg config has extra RFA keys; that failure is preserved in the run log. A second full image rehash was stopped after 274 seconds and 607 MB because the existing full decode/hash receipts for SJg17, S7 and S27 already share the same inventory digest. The corrected seed rebind verifies the exact source/config and unchanged TRAIN/VAL manifest hashes, then reuses the identical prior decoded inventory digest. The trainer rechecks each image-byte SHA as it loads it. No TEST data is touched. It does not read TEST. Use a fresh output receipt for each seed, then the pinned direct-SSH shared-GPU lease wrapper. It verifies the exact source commit, CUDA visibility, shared locks and a new output directory. No unrelated PBS/SSH jobs are terminated or modified. New outputs and logs belong under `/home/sachin.chaudhary/ces_fl_runs/ff2_sjgentle_confirmation_20261108_r1/`.

## Status

The proposal and exact configs are pushed on branch experiment/ff2-sjgentle-confirm-20261108. See EXECUTION_STATUS_20261108.md for the timestamped preflight, failures, active process, budget and next gate. Seed-7 RF-SJg is running; seed 27 remains pending.
