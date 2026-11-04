# Original-image fairness first — 5 November 2026

User requested a pushed training handoff, prioritizing isolation of original-image fairness and aiming to improve the Ding comparison. This package prepares B0/C0 and instructions for diagnostics and source-only generalization. No training or TEST evaluation was launched here.60/80 is an aspiration, not a guarantee or a TEST-tuning objective.

[Exact operator prompt](GPU_OPERATOR_PROMPT.md) · [Staged commands](commands.sh) · [Request and caps](request.json).

| Arm | Native CLI | Objective | Action |
|---|---|---|---|
| A0 | A | original weighted BCE | reuse completed3epoch nuisance-pilot control |
| B0 | B | BCE + hierarchical original-image fairness | NEW, first priority |
| C0 | C | B0 + procedural sensitivity/sham consistency | NEW, after matching review/compute checks |
| H0 | D | C0 + response alignment | reuse completed3epoch nuisance-pilot control |

Exact copied source from84ad983 and unchanged H0 config dictionaries make the CLI objective arm the only new treatment. This does not replace full H or change its masks/coefficients/architecture. B0 forwards only original images and auxiliary original images; C0 requires valid FF++ procedural qualification. Both use the same P50/fresh head/seed/batch/normalization/three-epoch budget as A0/H0. Existing10epoch A/H scores are not matched substitutes.

Source and all config/checkpoint/input hashes are pinned. verify_controls.py independently checks completion, three-epoch selection, checkpoint bytes and saved validation replay before reuse. check_arm_losses.py checks actual native B/C function execution against independent objective equations and verifies B has no edited-view forwards or intervention losses. It is a synthetic software check; real data, private control files, CUDA checks and cost remain operator work.

Prospective analysis: paired B0−A0,C0−B0,H0−C0,B0−H0 at selected validation checkpoints; gender×race F_FPR is primary, with AUC/FNR/worst-group and all other axes reported. No binary success gate is retrospectively invented. Supported intervals must retain at least900/1000usable draws; under-supported primary intervals remain inconclusive. A favorable cell count does not establish a method contribution.

All GPU use remains within verified existing study funds: stage1cohort≤3h,zero-update diagnostic≤0.25h. The source-only generalization follow-up has a5h ceiling but needs actual native P50 config binding, valid TRAIN-only folds, implementation/checks and profile first. Those follow-up trainers are not included in this commit. The prompt gives the operator a bounded implementation/run task; no missing prerequisites are treated as passed. No new budget or FF2 baseline hours are granted.

The complete-classifier diagnostic and source-only held-out-family design follow the attribution study. Existing P50 trained on every family cannot be reused to claim that a family was unseen. External TEST data already observed in Tables1–3 must not select new losses, thresholds, hyperparameters or family folds. Historical tables remain frozen.
