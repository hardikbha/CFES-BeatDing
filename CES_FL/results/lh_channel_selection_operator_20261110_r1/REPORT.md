# Full L_H channel-selection operator audit: BLOCKED before GPU work

Observed management UTC: 2026-11-10T09:49:29.647376+00:00.

**Software checks passed. The scientific execution gate is BLOCKED: the new intervention panel has 344 cases and zero cases reviewed by a named human. No qualification receipt exists. No production mask or training result exists; benefit is unassessed.**

## Pinned source and isolation

Requested source: [1a6d63db6baf7f6ad2a8e842a31341d2c128de25](https://github.com/hardiksharma6555/CFES-BeatDing/commit/1a6d63db6baf7f6ad2a8e842a31341d2c128de25).
Execution uses a new clean detached worktree at `/home/sachin.chaudhary/ces_fl_sources/lh-channel-selection-1a6d63db6baf-20261110`; its source SHA is `95574295b76f16f5e5d8f70f01502848b4616fc7437030a5dfdf2ff1219f62f0`.
The run directory is `/home/sachin.chaudhary/ces_fl_runs/lh_channel_selection_20261110_r1_operator_1a6d63d`. The source worktree remains unchanged. A separate publication worktree holds this new results branch. The requested fork is readable but not writable by this session; this report is published to the existing writable `hardikbha/CFES-BeatDing` mirror.

This is a justified exploratory fixed-mask comparison with three prospective controls: none, random and response-informed. Random and informed remove 41 channels from the same utility-eligible half. Full L_H hierarchy, weighted original BCE and all response losses are retained; both encoder and trained head warm-start the exact completed L_A teacher. This is a different initialization/recipe from the currently running common-recipe suite, not a substitute control for that suite or a native-paper reproduction. A fixed mask adds no learned parameters and does not establish disentanglement or valid forgery interventions.

## Fresh CPU verification

| Check | Actual outcome |
|---|---|
| Environment | Python 3.12.11, torch 2.7.0+cu126, NumPy 2.2.6, Pillow 11.3.0, scikit-learn 1.7.2; CUDA hidden/unavailable |
| Targeted CPU tests | 14/14 PASS |
| Selector numerical/gradient/reload and AI-review rejection checks | PASS |
| Full-width original/full-L_H synthetic lifecycle, resume and replay | PASS |
| Actual 41-channel masked full-L_H synthetic lifecycle | PASS |
| Thirteen original objective/preprocessing/evaluation function ASTs | Exact parity with archived core 8c25ba8 |
| Teacher checkpoint, DONE, completed arm-A receipt | PASS; checkpoint SHA `cce4476491829702d60a74ed7b2044aa8436768d6e08e0129da5a1fd84b0cb6d` |
| TRAIN manifest SHA | `4f83e7dc8042533329d7047fefd0f30cbd96f45d42bde2666dd0edc7d58f29cc` |
| VAL manifest SHA | `aeda051259d9e26b07c2810152b601ae115ce7c9e10523f99ea07bbc9f92afa8` |
| Exact `commands.sh prepare` on CPU | PASS; 876.55seconds; no GPU |
| Fresh image preflight | All 249,373 TRAIN/VAL originals hashed and decoded; zero cross-split byte overlap |
| Teacher-versus-fresh image inventory | Identical SHA `9f5d4b587d5dd87fe04a7db9b353c3aabc043b892c931dfd7f2b059112276621` |
| New panel replay | All 344 saved sheets match the source runtime pixel-for-pixel |

CPU fixture metrics are software-test output only, never experimental results.
[CPU commands/log hashes](CPU_CHECKS.json), [unit test output](unit_tests.txt), [selector receipt](SELECTOR_SMOKE.json), [native lifecycle receipt](SMOKE_RESULT.json), [masked lifecycle receipt](MASKED_LIFECYCLE_PASS.json), [objective parity](FRESH_OBJECTIVE_AST_PARITY.json), [checkpoint/manifest bytes](INPUT_HASH_CHECKS.json), [full data verification](DATA_PREFLIGHT_VERIFIED.json), [exact bound config](BOUND_AUDIT_CONFIG.json).

## Exact blocking gate: substantive new named-human review

The [pinned operator prompt](https://github.com/hardiksharma6555/CFES-BeatDing/blob/1a6d63db6baf7f6ad2a8e842a31341d2c128de25/CES_FL/proposals/lh_channel_selection_20261110_r1/GPU_OPERATOR_PROMPT.md) requires: “Review EVERY generated case with a named human”; prior approvals and AI decisions do not qualify changed panels.

The new bank has 86 unique TRAIN-real sources and 344 cases, covering both operators at both strengths. All 22 original stratified sources are retained; each joint group has at least 8 sources (observed 9–15). Mechanical replay found zero empty masks and zero exactly identical control/probe tensors. That verifies implementation consistency, not face suitability, demographic preservation, perceptual comparability or real-forgery relevance.

The generated review template contains **0/344 approved named-human reviews**, and `QUALIFIED.json` is absent. No human identity or approval has been invented. No case has been discarded or auto-approved. Rejected, occluded, padded, hat or low-texture cases must stay in the record and block qualification if unresolved; any recipe revision needs a separate version.

Private review location: `/home/sachin.chaudhary/ces_fl_runs/lh_channel_selection_20261110_r1_operator_1a6d63d/preview/`.
Template: `review_PENDING_PRIVATE.json`. Preview manifest SHA: `441dcf264c38a73c3e5929a0869e2f36a757ea7d8c0a227a685c9fbbb0ef4a65`.
[Aggregate panel checks](PANEL_VERIFICATION.json). Face images, full row inventories and per-case review metadata remain private.

## Allocation and remaining existing allowance

The historical original L_H ledger was read without modification. Its 59 recorded sessions sum exactly to 24.6467914626 GPU-h; the recorded cap is 48 GPU-h and recorded reservations are 0. The nominal recorded remainder is **23.3532085374GPU-h**, which arithmetically covers the requested 2 h. This is a verified recorded balance, not a fresh cohort reservation or permission to transfer another ledger. The original scope/extension restrictions are preserved in [BUDGET_CHECK.json](BUDGET_CHECK.json). Current scope, unrecorded charges and holds must be reconciled immediately before any future reservation; no historical budget or new grant was changed here.

Resource snapshots found candidate shared-pool capacity, but an idle-looking GPU is not an allocation. The user's PBS running slots were occupied. **No GPU lease, CUDA allocation receipt, cohort start/deadline or budget hold was created while human review is missing.** Thus actual CUDA UUID binding, CUDA smoke, TRAIN-only mask selection and the three real-input dry runs/cost-fit gates are **NOT REACHED**, not passed. [gpu01 observation](RESOURCE_gpu01.json), [gpu02 observation](RESOURCE_gpu02.json).

Only after a real named human reviews every case and qualification passes may a later operator acquire exactly one real allocation and establish one fixed cohort of at most 2 hours deadline within the freshly reconciled existing allowance. Every prior gate, including all-three-arm runtime fit and at most 30 minutes per arm, still must pass before any training. No partial preferential arm, shortened epoch budget, retry or extension is authorized by this report.

## Results and preservation

| Configuration | Production training | Scientific result |
|---|---|---|
| Full L_H, no mask | NOT STARTED | None |
| Full L_H, random mask | NOT STARTED | None |
| Full L_H, response-informed mask | NOT STARTED | None |

New GPU usage: **0GPU-h**. No benchmark TEST/CAL data were read by this task, no new Ding experiment was launched, and no active job was stopped/restarted/edited. All 31 source/config hashes in the active common-recipe suite match their pinned record; active process groups were observed alive. [Preservation checks](ACTIVE_SOURCE_PRESERVATION.json).

No new validation scores, bootstrap comparisons, utility-guard outcomes, fairness improvement, speedup or superiority claim exists. The ≥95% valid-replicate requirement, AUC ≥99.5% and FNR ≤0.5651% guards remain unchanged. [Machine-readable gate state](GATE_STATUS.json). This is a dated operator audit, not unattended publication or training automation.
