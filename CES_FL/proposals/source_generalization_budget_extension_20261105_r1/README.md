# Source-generalization compute-cap extension — 5 November 2026

This dated addendum records the user's instruction to increase GPU budget after the original five-hour source-generalization gate blocked both runs. It supersedes **only the GPU-hour cap** in `source_generalization_20261105_r1/GPU_OPERATOR_PROMPT.md`; it does not change the split, model, losses, transforms, initialization, training recipe, epoch count, selection, or calibration.

## New cap and estimate

- Verified global ledger at authorization: 48.0 total GPU-hours; 23.786942 committed/consumed; **24.213058 GPU-hours unreserved**.
- New source-generalization pair cap: **23.8 aggregate GPU-hours**, including both CUDA smokes already run (0.025042 GPU-hours), both complete training schedules, replay and held-idle time. This leaves approximately 0.413 GPU-hours unreserved in the global ledger.
- Exact configs remain S0 `d8fec9f...` and S1 `a809f20...`; source, fold, manifests, weights and checks remain the pinned hashes in `SOURCEGEN_STATUS.json`.
- Measured one-step profile projects 10.584836 GPU-hours for S0 and 10.496465 for S1 before safety margin, 21.081301 total. A 1.10 multiplier gives 23.189431 GPU-hours, below the new cap. These are measured-profile extrapolations, not completed-run times.
- Each arm may reserve up to **11.88 GPU-hours**. Together with the 0.025042 hours already used by the two smokes, this reserves at most 23.785 GPU-hours and remains below the 23.8 cap. Run both arms simultaneously on distinct, pool-eligible H100s to reduce wall time. The stage runner stops its own arm at its reserve; any incomplete result remains incomplete and must not be silently resumed or extended.

## Conditions

Use the matching seed-17 smoke receipts, which passed on separate H100 UUIDs and recorded identical fresh-initialization hashes. Recheck GPU pool leases and PBS allocations immediately before launch. No preemption, no FairForge-2 budget, no changed epoch count, no TEST, and no outcome-based checkpoint selection. If the global ledger has less than 23.8 unreserved at launch, or a second safe H100 is unavailable, stop and record the blocker. The original five-hour proposal and its blocked status remain preserved as history.
