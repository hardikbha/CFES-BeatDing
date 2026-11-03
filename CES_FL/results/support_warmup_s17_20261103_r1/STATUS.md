# CES-FL support-aware seed17 pilots

Observed management UTC: 2026-11-03T07:23:58.390346+00:00

Scientific commit8d5f0996a3f1cc02fbeaff9d25c1d95f714e8e77; source SHA2560eeafd838da3ca158336dcfe3b8195842dd15bc9fecd67ebbaf2ea03ba3c2680. Both candidates seed17,tenepochs. Existing CES-FL controls reused; no new Ding model computation, no new FF2 experiments, no TEST.

Explicit user-authorized pool-v2 shared UUID leases; no PBS or exclusive-GPU allocation claimed. Existing Ding jobs untouched.

Shared GPU budget used12.2822/24TOTAL hours; unspent conservative reservations3.1786h. Both GPUs counted by full task-held walltime even when shared.

|Candidate|Actual state|Complete epochs|Selected epoch|VAL AUC (%)|Intersection F_FPR (fraction)|Gender F_FPR|Age F_FPR|
|---|---|---:|---:|---:|---:|---:|---:|
|E_support|RUNNING|9/10|2|98.228795|0.468281|0.054478|0.312441|
|F_support_ramp3|COMPLETED_VERIFIED|10/10|1|98.319331|0.495578|0.048598|0.296048|

Values are selected-so-far VAL until native ten-epoch completion/replay. All supported axes, FPR/FNR, group support, achieved calibration, pair exposure and hashes are in LIVE_STATE.json. Gates are learned weights; maps are signed local contributions. AI visual review qualifies representative procedural recipes, not biological identity or real-forgery validity.

Prospective decision rules and comparison limits: ../../proposals/support_warmup_s17_20261103_r1/README.md and DING_COMPARISON_AUDIT.md. Female-Black validation has only30real images/two recorded components. No statistical superiority or paper TEST win follows from a favorable pilot metric.

E changes the error-control package; F adds a3epoch alignment ramp. A final schedule is separately recorded/compute-gated; no automatic promotion or TEST inference. Observer runs every10minutes for at most24hours and cannot train, submit jobs or generate reviews.
