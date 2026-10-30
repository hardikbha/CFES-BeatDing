# New training proposals

Completed configs are immutable under ../configs/completed_seed17/. Create a new `<request_id>/request.json` and `<request_id>/config.json` here for each proposed experiment. TRAINING_REQUEST_TEMPLATE.json is request metadata, not executable trainer configuration.

Use state DRAFT_DO_NOT_RUN until the actual launch decision is recorded. Describe the controlled comparison, exact changed factors, source/config/data/checkpoint hashes, seed, selection/calibration rules, validity dependencies, expected output path and real software checks. Use a unique output directory. Append actual launch commit/job ID and final result references only after they exist.

No proposal created by this setup has been approved or launched. See ../WORKFLOW.md. A supervisor push makes a proposal/code change available for review and pulling; it does not schedule a job.
