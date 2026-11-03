# Current L_H tables, ten-epoch Ding control and convergence request

User explicitly clarified “DannyBox” as **the same ten downstream epochs**. This authorizes a new Ding-on-FF++ comparison and supersedes the previous prohibition for this scope. It does not authorize changing historical Ding results, consuming the separate FF2 baseline72hledger, or silently calling a modified baseline a reproduction of published numbers.

No benchmark run or evaluation was launched here. SSH to the cluster was reset again. Latest available result snapshot remains `f980e8a667edea6a1accfade28b5044f3215b6e5` at2026-11-03T12:56:52Z: L_H9/10recorded, selected1. Require actual ten-epoch completion/replay before binding evaluation.

## Table work

| Paper table | Required L_H work | Current status |
|---|---|---|
| 1 | Selected Xception L_H on fixed FF++TEST25,401rows | Pending; current98.200410%AUC is VAL, not a Table1TEST row |
| 2 | SAME FF++trained Xception checkpoint on DFDC22,857, Celeb-DF28,458, DFD9,385 | Locked evaluation helper prepared; no retraining on these target datasets |
| 3 | Separately trained ResNet-50 L_H, then all four TEST datasets | Requires backbone port, checks, pretraining and downstream training; older ResNet CES-FL A/B/C are not L_H |

The paper's definitions come from [Tables2/3](https://arxiv.org/html/2511.10150v5). Table3 covers both intra-domain FF++ and the three cross-domain datasets. No Xception metric can fill a ResNet row.

The new `scripts/lh_locked_benchmarks.py` binds completed L_A and L_H as controls, verifies the native earliest maximum-validation-AUC checkpoint, its threshold and cache, preserves native DFDC duplicate multiplicity, evaluates each locked manifest once, and separates GPU inference from CPU bootstrap. The exact historical science source remains untouched. Source/config/checkpoint/manifest/overlap evidence is required, not invented. `RUN_VERIFIED.json`, DONE and all ten epochs are mandatory.

Main reporting has two separately labeled views: fixed probability.5 and the original selected FF++VAL-real10%FPR threshold. Cross-domain inference uses that SAME source-domain threshold; target TEST labels never recalibrate it. Report all supports, FPR/FNR, AUC, F_FPR, FDP, es-AUC, gaps and worst-group AUC, plus1000connected-contributor bootstrap draws and actual usable counts. Retain900usable minimum; do not turn insufficient support into a pass. Fixed.5 nonempty-group metrics are also exported to make group-support differences explicit.

## Training work orders, separate from frozen evaluation

**Ding-10 controlled adaptation.** Reuse a verified seed17 native classification checkpoint after50epochs and its TRAIN-only SFD2%inventory if provenance agrees; do not recompute an available stage just to call it new. Implement a separate connected-gradient GDA variant, rather than silently retaining the archived detached fairness term. Use10downstream epochs,32main images/update, each TRAIN original once/epoch and2,380updates/epoch, seed17, original224direct training/evaluation geometry for comparison with current L_H, same validation-AUC selection and dual threshold reporting. This is an explicitly adapted comparison recipe. Keep native head-only optimization/SGD.001/momentum.9/wd.005 and record BN-buffer policy plus additional group forwards; these differ from L_H and prevent claiming complete optimization equivalence. Name it “Ding SFD+GDA, connected-gradient,10epoch adaptation”; keep the published50epoch-method row and archived code-as-run row separate. Before launch, verify nonzero fairness gradients and a fairness-weight perturbation changes the update, SFD provenance and input membership. The existing GDA.py cannot provide a clean fairness baseline by merely changing `--num_epochs`. New baseline training code and smoke evidence are still required.

**ResNet L_H.** Port the current plain classifier/objectives to ResNet-50 features, trainable layer4 plus Linear2048→1 downstream, frozen BN. Use proper declared ImageNet normalization throughout originals/shams/treated views, and no default model download. Prepare ResNet P50 using a documented counterpart of current Xception P50 (original weighted BCE, training exposure/optimizer/BN copied from the verified native pretraining config, not guessed from an older P10run), then L_A/L_H10each sharing the same encoder and identical fresh heads. Preserve current loss values and auxiliary schedule. Require no-regression tests for Xception, exact parameter counts, backbone checkpoint rejection, normalization consistency, gradient/BN tests, resume/replay and actual CUDA checks. Current standalone L_H has Xception only; do not run its config with an ignored `backbone` field or label historical ResNet scores as new results.

**Convergence candidate.** The train objective decreases while validation deteriorates; this is generalization drift, not proof of optimizer non-convergence. A single prospective schedule test can use the same10epochs/init/data/losses/seed/batch/selection, with both AdamW learning rates multiplied at epoch e=0..9 by `0.1+0.9*(1+cos(pi*e/9))/2`. That preserves epoch1rates (head.001, exit1e−5) and ends at.0001/1e−6. Keep full L_H losses unchanged to isolate the schedule. Implement and test schedule/resume logging before launch. Response-alignment-off remains a separate future component ablation; do not combine changes. Freeze this specification before looking at new TEST outcomes. No guarantee later epochs improve; keep normal validation selection and preserve the current candidate.

These work orders are **not implemented training adapters in this package**. They define the next GPU-operator implementation/review work without pretending commands exist. Table1/2 evaluation tooling is implemented and CPU-guard-tested. Training and resource allocation remain pending. Do not launch all jobs just because their work orders are in the same file.

## Thresholding

We already have a selected validation threshold. Additional global thresholds may change fairness/accuracy trade-offs but cannot improve the unchanged-score AUC or es-AUC. A degenerate all-real classifier can have zero FPR disparity while missing every fake; always report FNR/TPR and overall errors with fairness. [Threshold decision versus score estimation](https://scikit-learn.org/stable/modules/classification_threshold.html).

The helper's `threshold-scan` accepts only the selected PRIVATE validation cache, reports all four predeclared targetFPRs1/2/5/10%, and chooses no new winner. These curves are exploratory because VAL already selected the model. They do not change the locked TEST plan. A new deployment threshold requires its own prospectively frozen validation policy; no target-domain TEST tuning. A group-specific postprocessor would require demographic labels at inference and separate identical treatment of baselines, with new claim scope. It is not included in the main Ding comparison. Positive-temperature scaling preserves ordering and therefore does not solve AUC or group ranking disparities.

## Budget and execution

First use already stored predictions for CPU analysis and reuse existing verified checkpoints. Then profile frozen evaluation on a legitimately allocated GPU. Before new Ding/ResNet/schedule training, publish concrete configs, source, smoke tests, dependency/checkpoint hashes and projected aggregate GPU cost against a verified remaining allowance. The user's FF2 baseline budget is separate. No unattended launches or monitoring are created by this package.
