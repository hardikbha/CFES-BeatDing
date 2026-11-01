# FF++ 50+50: direct SSH checks and two-source pilot

Observed UTC: **2026-11-01T08:16:47.179231+00:00**.

The user explicitly superseded the queued PBS route with direct SSH. Job44948 was cancelled before execution. This route uses bounded shared VRAM and claims no PBS allocation or exclusive GPU ownership.

Source commit: `fc4e63b8811c5813afa0d298a8172ba9555b7e40`. Scientific source/configuration/input hashes remain pinned.

| Milestone | Actual status |
|---|---|
| preflight | PASS |
| pilot | SUBMITTED_UNVERIFIED |
| CUDA software smoke | PASS, 7 actual checks on cuda |
| Real configured batch backward | PASS, zero optimizer steps |
| Full P / generation / downstream B,N,C | NOT STARTED |
| Human/forensic-validity approval | NOT GRANTED |
| FairForge-2 comparison | DO NOT RUN |

Completed pilot sources: 0/2. Generated condition counts: `{}`.

The initial SSH wrapper stopped before model execution because PyTorch reports bare UUIDs while NVIDIA prefixes them withGPU-. The failed attempt is preserved. The operational recovery compares normalized full UUIDs and passed three guards: actual bare/prefixed equality, rejection of another device, rejection of malformed IDs. No scientific source, loss, architecture or configuration was changed.

The previous environment-recovered pilot hit physical GPU OOM when another reserved task loaded40.3GiB. This retry additionally holds the existing shared dispatcher legacy GPU lock, preventing new pool work on the selected card while existing processes continue. No pool jobs or configuration were altered. The real pilot is limited to two originals, one per class, with seven independent generated conditions each and geometric controls. No generated image is automatically approved. Full pretraining/generation requires quality/resource checks; downstream binding also requires actual full per-source reviews and verified P.

Training preflight receipts are reused from the verified unchanged aiface run; source/config/input/inventory bytes match. The failed aiface generator lacked QwenImageEditPlusPipeline. This recovery uses the existing Qwen-compatible llada-image environment plus an isolated CPU-import overlay, leaving training in aiface and all science settings unchanged. Runtime, storage and model-binary provenance will be reported from actual pilot receipts. The full plan remains 76,139 sources and532,973diffusion outputs; coverage is not silently reduced. The observer cannot advance stages.

**FairForge-2 remains DO NOT RUN.** [Independent unresolved supervisor review](../proposals/ff2_supervisor_review_20261101_r1/REVIEW.md). The FF++ configuration must not be substituted.

[All actual receipts, bindings, failures and measured pilot profile](../results/cf5050_registry_recovery_20261101T081523Z/status.json).

Cluster root: `/home/sachin.chaudhary/ces_fl_runs/cf5050_registry_recovery_20261101T081523Z`. Logs: `logs/preflight_worker.log`, `logs/smoke_cuda.log`, `logs/generation_plan.log`, `logs/dry_pretrain.log`, `logs/qwen_two_source_pilot.log`.

A finite24h observer publishes these already explicitly launched milestones. It cannot launch, retry, cancel, approve edits, change settings or evaluate test data. Source inventories, face images, weights and per-image reviews remain private. Read the observation timestamp.
