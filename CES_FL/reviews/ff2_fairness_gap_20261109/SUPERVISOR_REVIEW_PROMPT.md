# Copy-and-paste supervisor prompt

Please review a specific unresolved fairness problem in our CES-FL/FairForge-2 project, initially at the theoretical and methodological level.

Repository: https://github.com/hardikbha/CFES-BeatDing
Branch: `review/ff2-fairness-gap-20261109`

Read:

1. `CES_FL/AGENTS.md`.
2. `CES_FL/reviews/ff2_fairness_gap_20261109/PROBLEM.md`.
3. The accompanying `EVIDENCE.json`.
4. The actual `CES_FL/src/ces_fl_realfpr.py`, relevant error-control/model/calibration functions in `ces_fl_head_ablation.py`, and the completed configurations/history under `CES_FL/proposals/ff2_realfpr_stretch_20261107_r1/`.
5. The six-table result packet and `CES_FL/paper/CLAIMS_AND_GAPS.md` cited in the problem report.

We have strong detection but weak fairness relative to a corrected local Ding-style baseline. On TEST, coherent seed 17 CES-FL P has AUC 98.78%, FNR 2.92%, gender×race100F_FPR 42.23; corrected local GDA has 93.08%, 7.80%, 9.47. These are not equal-utility results or an exact reproduction of the paper's original protocol. Gentle JPEG improves the RF loss-only control but does not solve the fairness gap, and gender/age outcomes regress. Per-cell best-seed tables are retrospective oracles, not a single model result.

The current RF model is Xception+GAP+linear, with a two-sided real-image soft exceedance disparity loss at a TRAIN-derived threshold and fixed utility guards. A source-processing diagnostic also shows strong sensitivity, but its JPEG treatment destroys much fake detection. Stronger JPEG, full tuning, ramps and hierarchical/procedural variants have already been attempted. Please distinguish a genuinely new hypothesis from repeating those configurations.

Please return:

- A ranked theoretical diagnosis, with measured facts separated from hypotheses and unverified claims.
- Whether the current surrogate can control the reported hard-FPR metric; how source confounds, group offsets and class imbalance affect that reasoning.
- Whether source/data controls, class-conditional distribution/tail alignment, or a carefully labelled calibration diagnostic is the next justified step, and why.
- At most one focused experiment recommendation: mathematical change, matched control, required data/label validity, unchanged factors, safeguards, uncertainty and stop conditions.
- Closest prior work and a candid assessment of what CES-FL could claim as novel.

Do not implement or launch anything in this review. Do not tune against exposed TEST, lower a historical guard after results, add group-specific thresholds to a global-threshold row, fabricate approvals/labels, or claim that dataset ownership implies model superiority. We need a scientifically justified path to improving fairness while retaining useful detection, not a guaranteed win.
