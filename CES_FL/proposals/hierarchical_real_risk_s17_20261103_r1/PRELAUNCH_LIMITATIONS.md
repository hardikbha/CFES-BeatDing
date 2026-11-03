# Additional falsification checks, before any candidate training

The response-offset fixture confirms that a group additive score offset can leave procedural response alignment zero while the new original-real risk penalty is positive. The new term addresses that specific blind spot.

A second explicit fixture keeps every group soft risk at0.5, but hard FPR differs0.5 versus1.0. The new hierarchical risk penalty is zero there. This is a limitation of the prescribed soft-mean risk objective, not a numerical implementation error. No unrequested quantile/tail loss, temperature change or sampler is substituted. Validation FPR/FNR safeguards decide the empirical result; this objective is not a parity guarantee.

The first two added fixtures incorrectly paired uniform priors with nonuniform within-parent weights; the consistency validator rejected them as designed. The final fixtures use the declared consistent priors and pass. Earlier failure output is preserved. No scientific source, coefficients or criteria changed for these corrections. All13hierarchical,13legacy,9support and7resource tests pass; full CPU/CUDA software smoke passes with unchanged sourceSHA2549cb1345d1e2ee617e8a77ca288a0acbbfd05cf480f7b91a46b8627fcb74b2.

Initialization state tensors, trainable flags and all five first-cycle auxiliary sample indices/original/control/treated tensors exactly match D. Hash serialization is explicitly sorted state keys plus contiguous tensor bytes; these hashes need not match earlier reports that used a different serialization. Contributor diversity is still logged for transparency but does NOT weight H's loss.

The unchanged paired-bootstrap eligibility/seed generated746/1000usable primary replicates in every E/F comparison. This is a support-driven limitation, declared before launch. It is not relaxed to obtain PASS. An improved numerical result can remain INCONCLUSIVE. No TEST confirmation or seed-level uncertainty is claimed.
