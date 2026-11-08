# Claims, limits and publication gates

Observed 9 Nov 2026. This document does not authorize a new run or change any preregistration.

| Proposed claim | Current evidence | Wording permitted now | Missing evidence |
|---|---|---|---|
| Processing history affects detector decisions | Existing frozen L_A SELECT probe, 242 real / 360 fake images, 24 fake methods | Large score/decision changes under specified processing treatments | Replication across models, source/contributor-aware uncertainty, controlled isolation of all processing factors |
| Gentle augmentation improves primary disparity over RF-S | Three matched downstream seeds on existing frozen TEST exports; lower gender×race F_FPR for every seed | Descriptive reduction in primary disparity at a utility cost | Contributor-aware intervals; independently repeated encoder pretraining |
| Improvement in all demographic axes | Contradicted: gender and age worsen | Report regressions explicitly | A new protected experiment would be needed to support a broader claim |
| We beat Ding | Not supported: corrected local GDA has much lower F_FPR and much lower AUC | Utility/fairness trade-off against a disclosed local adaptation | Exact matched budgets/protocols, original-framework validity, defensible joint decision rule |
| Full CES-FL counterfactual method works | Not tested by RF-S/RF-SJg | The current pilot uses original-image risk control and JPEG augmentation | Validated demographic/forensic interventions, processing/no-change controls, matched baselines and independent review |
| Fairness labels are human ground truth | Not established; age/skin estimates are present | Estimated-label subgroup analysis | Human review, missingness/uncertainty/adjudication, documented label provenance |
| Three independent complete training seeds | False: all RF runs share one seed17 BCE50 encoder | Three downstream seeds conditional on a shared encoder | Independent pretraining repetitions if claiming full-training variability |
| Statistical significance | Not established | Mean ± downstream sample SD; descriptive paired differences | Paired source/contributor-cluster bootstrap with declared unit and multiplicity policy |
| Strict pre-TEST freeze order independently audited | Incomplete | A freeze record and fixed checkpoint list exist | Resolve login/GPU clock skew, remote push timing and original checkpoint-hash attestation |
| Complete GPU compute accounting | Incomplete | Shared ledger charged64.6011/72 GPU-h | TEST and sanity inference start/end durations; distinguish shared ledger from per-method total |
| Novelty sufficient for CVPR | Not established | Candidate contribution is processing-confound audit plus a carefully bounded intervention study | Broader primary-literature comparison, stronger controls, generalization and uncertainty |

Do not combine per-cell best seeds/models into a row called one method. Do not fill missing TEST entries from validation. Do not present corrected GDA as identical to the published or archived detached-loss version. Do not treat fewer false positives at a different achieved FPR as a guaranteed fairness improvement.

The historical source-diagnosis report uses stronger causal language than this audit supports. The draft intentionally says **processing sensitivity consistent with source confounding**, rather than claiming compression is the sole cause of demographic unfairness. It also discloses that the diagnostic transform severely reduces fake detection.

TEST has been opened by the prior upstream study. Nothing in this paper-writing step authorizes post-TEST tuning, a new split, new training, or another inference pass.
