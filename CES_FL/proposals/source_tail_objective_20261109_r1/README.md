# Experimental source-tail objective

User requested a concrete changed objective on9November2026. Implemented module: `CES_FL/src/source_tail_objective.py`. This is a tested loss primitive and proposal, not an integrated production trainer or a successful new model. No training, inference, TEST/CAL access or old-result modification occurred. Existing unrelated workspace file deletions are not part of this change.

## Proposed objective

Keep the complete selected L_A detector as frozen teacher f0. Student starts with identical full classifier weights and forward computation; freeze the encoder for this diagnostic. Source IDs are documented TRAIN-real acquisition labels, currently FFHQ/FairFace, not demographic guesses or fake pixel-origin assumptions. All original classification examples and existing fixed inverse-class×ids8 BCE weights remain in the classification term.

Fit reference constants on teacher scores for real TRAIN images only: thresholds t_k at fixed quantiles85/90/95%, scale a=TRAIN-real teacher IQR and temperature T=1. Reject a degenerate scale. Freeze all constants. Let

    r_s,k(theta) = mean_real_aux_from_source_s sigmoid((z_theta(x)-t_k)/(T*a))
    b_k = mean_all_TRAIN_real sigmoid((z_teacher(x)-t_k)/(T*a))

Then

    L = L_weighted_BCE
        + lambda_real * mean_s,k relu(r_s,k(theta)-b_k)^2
        + lambda_fake * mean_main_fake relu((z_teacher(x)-z_theta(x))/a)^2.

Teacher scores are detached. The matched control uses lambda_real0 and lambda_fake1; treatment uses lambda_real0.1 and lambda_fake1. These are unoptimized engineering defaults, not values selected from TEST or claimed optimal. Both arms must use the same auxiliary source batches and main original exposure. Require at least16auxiliary reals per declared source; absence/unknown sources fail explicitly. Fake-free minibatches contribute zero fake-guard loss and must be counted. Actual source sidecar, manifests, reference hashes and learner/teacher checkpoints still need binding before a training proposal is executable.

## Meaning and limits

- BCE preserves the ordinary real/fake discrimination objective.
- The source-tail term penalizes sources above the fixed pooled teacher reference. It has no direct incentive to raise the soft error of a source already below that reference. Shared parameters, other objectives and recalibrated thresholds can still raise some errors.
- The fake guard discourages using lower fake scores to obtain lower false alarms. A zero penalty implies nondecreasing logits only on the observed fake examples, at the same fixed threshold. A finite penalty is not a hard constraint; it offers no guarantee for unseen images or a different calibrated threshold.
- Frozen thresholds/scale avoid chasing the student's moving score scale. Multiple thresholds test a local score band rather than one moving soft risk. The target is still a surrogate and cannot guarantee hard FPR parity.
- This targets acquisition-source behavior; demographic fairness is an evaluation outcome and may fail to improve. It does not impute the absent FFHQ×Others cells, establish source causality, or repair the unsupported standardized estimand.
- No demographic input is required at inference. No JPEG augmentation, generated images or forensic-removal claim is introduced.

## Before training

Preserve historical SELECT utility guards AUC≥99.5%,FNR≤0.5651%; define all marginal non-regression tolerances and supported uncertainty before fitting. Compare hard errors at the same declared source-validation calibration rule, report both native thresholds and all group supports, and retain insufficient-support outcomes. Current SELECT is exposed development and the encoder was previously selected on it; neither TRAIN fit nor another SELECT comparison provides fresh confirmation. TEST/CAL remain out of scope. Any independent confirmation role requires verified unused data and independence.

Only one control/treatment comparison is proposed. Source metadata, reference calibration, source-balanced auxiliary sampling, frozen-teacher preservation, head-only optimizer, cost/ledger and actual trainer integration have not been certified here. Do not launch by importing this module alone or substitute it into a live RF run. Diagnose whether the source effect/utility constraint is feasible before unfreezing more layers or tuning coefficients.

The module's synthetic checks verify source-gradient direction, zero direct tail gradient below target, fake-score preservation penalty, detached teacher/reference, exact control reduction and rejection of missing source support. Evidence is in `evidence/checks.log`; these are software properties, not benchmark results.

## Prior work and claim scope

This is an objective hypothesis, not established novelty. Existing RF already regularizes real-image risk; earlier project variants already tested one-sided penalties. The differences here are acquisition-source indexing, fixed teacher-derived threshold band/scale, and a matched fake-score preservation guard. Distribution alignment, class-conditioned representation alignment and fairness post-processing already exist in [Ding](https://arxiv.org/abs/2511.10150),[FairReL](https://arxiv.org/abs/2608.28777),[FYC](https://arxiv.org/abs/2610.37882) and [Hardt et al.](https://arxiv.org/abs/1610.02413). No guarantee of beating these methods or reaching60/80 follows from this formulation.
