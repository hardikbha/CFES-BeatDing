# FairForge-2 status after supervisor review

**Verdict: DO NOT RUN the full comparison.** This is separate from the FF++ counterfactual 50+50 study. No FF2 training, generation, objective/configuration repair or experiment launch occurred in this continuation.

Read source: `/home/sachin.chaudhary/ces_fl_sources/cesfl-cf5050-fc4e63b/CES_FL/proposals/ff2_supervisor_review_20261101_r1/REVIEW.md`, commit `fc4e63b8811c5813afa0d298a8172ba9555b7e40`. The review itself audits upstream `f73c98d28f01604e45525b2acff49cd945b1cd33`.

1. Exact public-split Ding and downstream configurations, exposure budgets, seed and BN policy are missing/unbound. The earlier 10-epoch P file is not an approved 50+50 comparison.
2. Define archive-faithful, gradient-only diagnostic and class-conditioned differentiable Ding variants separately, with fairness-only and lambda-dependent update tests; retain TRAIN-only SFD mask provenance.
3. A common tested score adapter must use fake minus real logit for Ding, with class orientation/order/preprocessing and softmax-equivalence checks.
4. The sixteen reviewed age edits cover eight sources. Matched no-change and geometric controls remain unverified; RN/RD are not a broad demographic study.
5. Individual-group CIs and paired method differences need shared contributor-component draws and fixed checkpoints/thresholds. Existing aggregate bootstrap is incomplete for those promises.
6. ids8 and gender_race are different groupings; document mapping, label provenance, support/missingness and source/donor limitations. Age/skin labels are proxies.

A new versioned FF2 proposal with actual passing software/data/allocated-CUDA evidence is required before the full comparison. The FF++ source/configuration must not be substituted. Prior public-split metadata/file checks remain historical evidence and do not close these methodological blockers. The earlier CESel-MA gate remains FAIL.
