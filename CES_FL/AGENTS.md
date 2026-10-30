# Instructions for work under CES_FL

Read README.md, STATUS.md, WORKFLOW.md and the actual source/configuration before changing or running anything. STATUS.md is a timestamped observation, not continuous monitoring. Inspect real artifacts and scheduler state before execution.

- Keep completed_seed17 configurations/results and historical smoke receipts immutable. Add new dated experiments instead of rewriting historical evidence.
- Current verified study is seed17 Xception A/P/B/C; C is demographic invariance only. B/C is a comparison, not another arm. Validation scores are not Ding test scores.
- The earlier CESel-MA gate remains FAIL. Do not revise it after results or represent this exploratory branch as a passed gate.
- Preserve train/validation/test roles, selection/calibration policy, input membership and counterfactual approvals. Never tune on test or fabricate labels/approvals.
- Use proposals/ for new training requests with exact configuration, code commit, data hashes, purpose, dependencies and approval/launch records. A Git push does not submit a training job.
- No automatic training, image generation, model downloads, job termination or scheduler changes from status/publish helpers. Follow the user's explicit scope for execution.
- GPU jobs must request ngpus. Verify the actual allocation and CUDA binding; a host name or low GPU utilization is not a reservation.
- Launch new work from an immutable source checkout; do not pull over source or launchers a running job reads. Never reset/clean shared outputs or include unrelated dirty files in a commit.
- Publish aggregate metrics, relevant textual evidence and code/config hashes after meaningful work. Preserve dated snapshots. Keep datasets, face images, binary weights/checkpoints and credentials outside this public repository.
- A changed source hash invalidates any implication that old smoke receipts tested the new code. Run appropriate software checks and record their real command/output.
- Describe gates as learned weights and maps as signed local contributions. Forensic sensitivity requires validated forensic interventions; map consistency stays disabled without spatial validation.
