# FF2 response-strength screen, seed17

The user requested tuning of ramp and low-response training. Existing seed17
SELECT curves show response activation at epoch3 coinciding with an AUC fall
from99.8182% to99.5539%. The epoch3 weighted response scalar averages1.34times
the classification scalar. Scalar magnitudes do not establish gradient conflict
or causality; this motivates a bounded strength/schedule experiment.

Four fixed configurations: fairness-only control (response ceiling0), and
smooth ceilings0.05/0.10/0.20. All use two response-free epochs, then six epochs
of smoothstep t²(3−2t) to their ceiling, then hold it. Ceiling is the only
scientific difference among these four arms. The old schedules reached1/3at
their first active epoch; the new maximum starts at0.014815. "Low" means lower
response strength, not an explicit variance penalty.

All start from the same P50 encoder and fresh seed17 linear head, never a
selected TEST-exposed downstream checkpoint. Ten full epochs, batch32,
AdamW head1e−3/encoder1e−5, clip5, frozen BN, joint auxiliary groups, unchanged
unweighted BCE/fairness0.1, PS-v2 probes and identical forward exposure remain.
No LR schedule, mask or other loss change is introduced in this screen. The
unused inherited GDA diagnostic is computed as in the common recipe; no Ding
model is trained or evaluated. All four initial parameter and actual input
fingerprints must match in real CUDA checks, including a second discarded
update at the plateau with active response gradients where appropriate.

Fresh full TRAIN218266/SELECT31107 byte/decode audit and exact source/teacher/
manifest/dependency bindings precede GPU work. The original88-case recipe
panel/reviewer decisions are reused verbatim, with explicit provenance; the
operator, strengths, image bytes and sheet bytes are unchanged. No new human
review or approval of every training image is claimed.

Select the earliest strict maximum SELECT AUC across all ten epochs. Keep
pre-activation selections: they cannot establish a response benefit. Report
best post-activation and final-epoch values as clearly separate diagnostics,
never replacement primary checkpoints. Calibrate a global threshold on SELECT
reals at10% empirical FPR. TRAIN-real thresholds for the soft fairness loss
remain unchanged. No TEST/CAL manifest or evaluation command is present.

Primary exploratory comparison is joint100F_FPR against the new fairness-only
control, retaining all axes, AUC, FPR/FNR, support and regressions. Retain the
absolute SELECT utility screens AUC≥99.5% and FNR≤0.5651%; also report AUC/FNR
changes against the control, without tuning away a regression. A benefit needs
a selected post-activation checkpoint and lower primary disparity than the
control. Paired contributor bootstrap uses2000draws and≥95%valid support;
otherwise uncertainty is INCONCLUSIVE. All intervals remain conditional on
already selected checkpoints; one retrospectively chosen training seed and
prior TEST exposure prevent confirmatory superiority claims.

This finite screen allows up to four compatible shared H100 UUID leases
(two workers/node),16GB reservations plus4GB admission headroom, allocator18%.
Each native run has4hours to finish ten epochs; each held-GPU stage is capped
at4.25hours including checks and replay. Maximum17aggregate GPU-hours is
reserved inside the reconciled remaining original L_H allowance; no new grant
or ledger transfer is made. Actual held-card time, including failed checks and
idle/loading, is charged. Preserve other jobs. No automatic retries, extra
candidates, altered epochs, new seeds, TEST/CAL scoring or next-stage extension.
If the bounded screen does not help, publish the failure rather than silently
changing the selector, guards or evidence.
