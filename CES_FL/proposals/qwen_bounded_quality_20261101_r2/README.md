# Authorized 32-source Qwen quality pilot — no training

The user explicitly approved proceeding after the bounded diagnostic and 4–6 GPU-hour generation estimate were explained. This r2 records executable selection, software checks and the actual generation-only scope; r1 remains a historical draft. The user's report of reviewing the earlier 14 outputs is preserved. Their “go ahead” is **not** recorded as individual validity decisions or evidence that forgery artifacts survived editing.

## Scope and fixed settings

Exactly **32 TRAIN originals**, 16 real and 16 fake; two per 16 label × ids8 strata. All 32 original bytes were rechecked and decoded, with no repeat of the previous two rendered sources. Unique contributor proxy sets are disjoint within the sample: 48 video-subject proxies, not verified person identities. Eight proxies/source IDs overlap VAL under the unchanged benchmark split. It is not an identity-disjoint study or a population fairness evaluation.

The 224 diffusion outputs comprise no-change, younger, older, feminine, masculine, lighter and darker for each original, plus 32 geometry controls. Source/config remains pinned to `fc4e63b8811c5813afa0d298a8172ba9555b7e40`; **no scientific source/config/architecture/loss changes**. Qwen2511 revision `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`, BF16, 40 steps, true CFG4 and 1MP geometry are unchanged. Generation is offline from each original independently, not chained. No download, training, test evaluation or automatic later milestone is attached.

## Exact selection and training isolation

The pinned generator already consumes an explicit plan source list. A new plan lists exactly these 32 full-inventory records, preserving the original full TRAIN config and hashes and adding `quality_only=true`, `full_train_binding_prohibited=true`, parent inventory provenance and the private selection hash. It does not change or replace the 76,139-original production plan. The production `audit_bank` path was actually exercised: it rejects this subset with `Plan must cover exact full TRAIN inventory` before any review or binding. No synthetic fixture can enter the production loader. The generic `--max-sources 32` is not used because that would select the first class-alternating images, not the stratified sample.

## Actual software checks

Command, exit **0**:

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/ces_fl_runs/qwen_quality32_20261101T095157Z/runtime/test_quality_selection.py
```

All **six CPU checks passed**: real exact selection/hashes/stratification; real production full-coverage binding rejection; synthetic fixture production rejection; disjoint shards, all independent originals/conditions/controls, no approvals and byte-identical resume; changed original rejection; changed inventory rejection. Synthetic render timings are not Qwen throughput. Original CUDA smoke/real-batch receipts are inherited explicitly for the unchanged core; the generator environment and prior real Qwen success are unchanged. Actual CUDA UUID binding is verified again at launch. Runtime helper hashes and exact commands are included.

## Resource policy and tentative time

Use the authorized direct-SSH route on gpu01, holding the existing shared dispatcher exclusion plus the project GPU lease. Require **53,248 MiB free**, recheck after five seconds and use the existing **55% Torch allocator cap**. Do not join/replace others' reservations or terminate their jobs. This is advisory exclusion for new cooperative dispatch; existing processes continue. No PBS allocation or exclusive ownership is claimed. The worker waits at most two hours, then records failure if there is no eligible device. It does not automatically retry.

**Estimated generation: 4–6 hours after binding**, based on 224 ×69.52 seconds =4.33 GPU-hours measured in the old two-source pilot. Waiting, loading variability, retries and human review are additional. This is not a training ETA or a guaranteed finish. No eight-GPU reservation is assumed.

## Outputs and status

Root: `/home/sachin.chaudhary/ces_fl_runs/qwen_quality32_20261101T095157Z`. Generation candidates: `generation/items/`; actual logs: `logs/pilot_worker.log` and `logs/qwen_32_source_quality.log`; bindings/failures/profiles: `checks/`; after successful generation: `quality_review_pending.csv` with **zero automatic approvals**. Images, per-image metadata, reviews and model weights remain private.

[Timestamped running status](../../status/qwen_quality32_20261101T095157Z_live.md) is updated by a finite read-only observer, normally every ten minutes and on stage changes/termination, for at most eight hours. The observer cannot launch, retry, train, approve, cancel, calibrate or evaluate test. Read timestamps; the page is not a scheduler connection.

**Training and the original full bank remain stopped.** New outputs require independent per-pair review and substantive forensic-validity assessment. Realistic appearance, target appearance or favorable detector scores alone do not establish valid counterfactuals. Any subsequent training study needs an explicit justified configuration, valid source coverage and existing validation-only selection/calibration. No fairness gain or win over Ding is claimed. **FairForge-2 remains DO NOT RUN.**

## Actual launch observed

Management UTC: 2026-11-01T09:59:39.587000+00:00. Worker **3806205** is live on **gpu01**, physical **GPU 3**, H10080GB, UUID `GPU-6a002609-de85-4949-c707-ee9317b284f0`. CUDA UUID binding passed and **73,140 MiB free** was rechecked before loading. Phase: RUNNING. The shared dispatcher exclusion and project lease are held. Operational code was published in `c761b0d` before launch. No training or full-bank generation was started. Expected generation completion approximately **19:30–21:30 IST, 1 November**, subject to actual runtime/failure; human review follows.
