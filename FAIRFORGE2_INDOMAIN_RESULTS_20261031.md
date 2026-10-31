# FairForge-2 in-domain results audit — 31 October 2026

This answers whether we have models trained on FairForge-2 TRAIN and evaluated on held-out FairForge-2 TEST. **Yes, for several historical runs; a complete matched public-split benchmark is not available.** These are measured saved-checkpoint results, not new training and not Ding's published numbers.

## 1. Internal/full FairForge-2: completed historical runs

Manifest sizes: **502,035 TRAIN; 74,974 SELECT; 182,884 TEST**. TEST contains 10,718 real and 172,166 fake image rows in the Ding score archive.

| Model/checkpoint | Backbone | Run seed | Held-out TEST AUC (%) | Evidence under the agents repository |
|---|---|---:|---:|---|
| Ding classification stage | Xception | 7 initialization | 99.2619 | DINGREPRO_LIN/results/DING_ff2_s7/scores/base_ff2_test.npz |
| Ding local SFD + GDA pipeline | Xception | 7 initialization; GDA code seed 121 | 99.3573 | DINGREPRO_LIN/results/DING_ff2_s7/scores/ff2_test.npz |
| Older CFES detection-only control | CLIP ViT-L/14 | 7 | 97.6043 | CFES/runs_ff2full/clip_A_s7/eval_all/ff2_test.json |
| Older CFES fairness-method run | CLIP ViT-L/14 | 7 | 98.0969 | CFES/runs_ff2full/clip_B_s7/eval_all/ff2_test.json |

**Ding is ahead on AUC in this historical internal-split comparison.** This does not establish a matched causal comparison of fairness objectives: backbones, training schedules, and checkpoint policies differ. Historical clip_A/clip_B labels are unrelated to the present CES-FL A/B/C arm definitions.

The Ding archive has DONE, TRAINED, checkpoints and numeric scores. Its launcher is DINGREPRO_LIN/code/run_task_ff2.sh: five classification epochs, SFD masking of 2% of the targeted channels, then five GDA epochs. The final GDA epoch is evaluated. This is a local, budget-limited adaptation, not a claim that the public repository exactly reproduces the paper's training. The executable GDA gradient behavior was not re-audited during this results retrieval, so scores alone cannot validate the fairness adaptation implementation.

Saved-score AUC was recomputed on CPU with sklearn.metrics.roc_auc_score(label, logit); no model inference or gradient updates were required. The older CFES JSON AUC agrees with its logit-based recomputation. The phase-1-only legacy archive is omitted from the main table because its saturated float32 probabilities yield AUC 97.0854%, whereas its stored logits yield 97.2816%; do not silently mix these scoring conventions.

## 2. Frozen public FairForge-2: completed historical run

Manifest sizes: **218,266 TRAIN; 31,107 SELECT; 87,564 TEST**. Keep this experiment separate from the internal/full split above.

| Model | Training manifest | Held-out TEST rows | AUC (%) | Status |
|---|---|---:|---:|---|
| Older CFES-D, seed 7 | fairforge2_public/train.csv | 87,564 | 98.2598 | Completed, saved checkpoint and evaluation |
| Ding trained on this public TRAIN | — | — | — | No completed matching retrain found |
| DAW-FDD / DAG-FDD / PG-FDD trained on this public TRAIN | — | — | — | No completed matching retrains found |
| Current CES-FL v0.2.1 trained on public FF2 | — | — | — | No completed in-domain run found |

CFES-D evidence:
- CESEL_MA/runs_ff2public/cfesd_s7/config.yaml
- CESEL_MA/runs_ff2public/cfesd_s7/TRAINED
- CESEL_MA/runs_ff2public/cfesd_s7/best.pt
- CESEL_MA/runs_ff2public/cfesd_s7/eval/ff2_test.json
- CESEL_MA/runs_ff2public/cfesd_s7/PUBLIC_LINEAGE
- CESEL_MA/runs_ff2public/clip_s7_p1/provenance.jsonl

The saved configuration initializes CFES-D from the public phase-1 checkpoint; that phase-1 provenance records no previous checkpoint initialization. Its recorded public-TRAIN SHA256 is e93a5f3adabeb728154a909e4431e04a4ab92e3e57d60ba5f477a800d2ebbc36, matching the current public train manifest in this inspection. The historical selection policy is val_oppoint, rather than the current CES-FL AUC-only protocol. This result is the older CFES-D implementation, not the current Qwen-counterfactual CES-FL model.

The directory CFES/runs_ff2id is empty. CFES/results/2026-10-22-ff2-indomain-baselines.md describes prepared official-code Ori/DAW/DAG/PG experiments; prepared commands are not completed results. The separate completed Ding internal/full run above must not be overlooked merely because that baseline directory is empty.

## 3. What does not count as an in-domain FF2 result

- Current CES-FL checkpoints trained on FF++ and evaluated on FF2 are cross-dataset transfer results.
- Previously released detectors evaluated on FF2 without FF2 retraining are zero-shot benchmark results.
- An internally trained model evaluated on a filtered public TEST subset is not a public-only retraining result.
- Ding's published FF++/DFDC/Celeb-DF/DFD tables are not published FF2 experiments.
- The old CFES-D configuration is not evidence that the current demographic-counterfactual architecture has already been trained on FF2.

## 4. Integrity and interpretation limits

Image-path sets for both full and public builds were read from train.csv, realval.csv + fakeval.csv, and test.csv. TRAIN–SELECT, TRAIN–TEST, and SELECT–TEST path intersections are all zero. This is an image-path separation check; it does not independently establish identity/donor separation or rule out copies saved under different paths.

Do not compare fairness numbers under different thresholds or substitute a max-minus-min group gap for Ding's reported F_FPR definition. Existing JSON fairness fields were not relabeled as a common protocol in this report. A fair paper comparison still needs identical public manifests, documented encoder/training budgets, selection/calibration rules and correctly matched fairness metrics, with uncertainty.

These historical results are single-run evidence, not multi-seed conclusions. No new FF2 training, Ding reproduction, generation, cancellation, or protocol change was performed for this audit.

## 5. Exact absolute artifact locations

Repository root:
`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents`

Internal manifests:
`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/data_ding_schema/fairforge2_full/`

Public manifests:
`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/data_ding_schema/fairforge2_public/`

Ding full internal checkpoint:
`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/results/DING_ff2_s7/XceptionDetector_prune2%_only_head/lr_1.0e-03_lambda_5.0e-03/4.pth`

Internal CFES evaluation:
`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CFES/runs_ff2full/clip_B_s7/eval_all/ff2_test.json`

Public CFES-D evaluation:
`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CESEL_MA/runs_ff2public/cfesd_s7/eval/ff2_test.json`

Audit report copy:
`/home/sachin.chaudhary/cesel_ma_audit_logs/fairforge2_indomain_results_20261031.md`

## 6. Recommended next dependency

Prepare a matched public-TRAIN/public-TEST comparison plan using the verified existing runs as historical references. Explicitly identify whether the intended new model is current CES-FL or historical CFES-D; neither the old internal Ding checkpoint nor a cross-dataset evaluation completes that comparison. This report does not authorize additional training.
