# Qwen review completed automatically; human validity decisions still required

Observed management UTC: 2026-11-01T09:43:09.620570+00:00. This is a new observation after `e150105`, not an experiment launch.

## Work actually executed

The command below completed successfully with exit code 0:

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/cesel_ma_audit_logs/prepare_bounded_qwen_review_20261101.py
```

Actual output: `AUTOMATIC_RECHECK_PASS_HUMAN_VALIDITY_PENDING`; **14** candidate hashes, original-input associations, original hashes, recorded geometry and geometric-control hashes reverified; **0** human-approved rows. The source and configuration hashes still match the previously CUDA-tested `fc4e63b8811c5813afa0d298a8172ba9555b7e40` implementation. The complete source inventory file and all four configured external inputs match their pinned hashes. This recheck did not rehash every unselected TRAIN image or rerun old CUDA smoke checks.

Both existing contact sheets were visually inspected again. The no-change reconstruction changes image texture/sharpness; some edits add facial hair or alter facial structure, expression, and texture. These observations are review flags, not formal human rejections. They neither establish identity equivalence nor establish preservation of authenticity or manipulation evidence. No demographic label is inferred from appearance by this audit.

No matching CES-FL Python training/generation worker was observed on gpu01 or gpu02. `qstat` and NVIDIA queries were executed; raw command/output receipts are included. Other users' and projects' jobs were not modified. All 14 pending decisions remain unchanged. No new training, generation, model download or test evaluation occurred. This is a timestamped snapshot, not a live monitor. Eight GPUs are not reserved for this work.

## Private review artifacts

- Portable page containing **14** four-image comparisons (original / resize-only / editor no-change / candidate): `/home/sachin.chaudhary/ces_fl_runs/qwen_review_revision_20261101T094302Z/PRIVATE_REVIEW_14.html`.
- Existing decisions file, preserved unchanged: `/home/sachin.chaudhary/ces_fl_runs/cf5050_registry_recovery_20261101T081523Z/pilot_review_pending.csv`.
- Existing full pilot report: `/home/sachin.chaudhary/ces_fl_runs/cf5050_registry_recovery_20261101T081523Z/REPORT_QWEN_PILOT.md`.
- New command receipts, automatic checks and proposed sample manifest: `/home/sachin.chaudhary/ces_fl_runs/qwen_review_revision_20261101T094302Z`.

The page does not assign approvals. Human reviewers must enter reviewer identity, target validity, unintended changes, identity uncertainty and forensic-label/evidence assessment. Preserve uncertainty; do not check forensic preservation solely because an edit looks plausible or its detector score is favorable. Earlier six/eight-source approvals do not apply to these files.

## New bounded proposal: quality feasibility only

The old 76,139-source full-bank protocol is unchanged. **The proposal below is not a reduced-coverage execution of that protocol.** It is a separate proposed editor-validity diagnostic, with no executable training configuration or approved generation job.

A deterministic seed-17 selection has been prepared and its **32** original images actually decoded and hashed against the existing inventory: **16 real + 16 fake**, two per each of the **16 label × ids8 cells**. Selection is independent of detector/test scores. Contributor-proxy sets are disjoint within these 32 originals, covering **48 video-subject proxies** and **48 source-video IDs**. This does not establish 48 independently identified people. Full counts and family support are in `evidence/BOUNDED_SOURCE_SUMMARY.json`.

This selected candidate sample has **0** image-path overlap with VAL, but **8** shared identity proxies and **8** shared source-video IDs with VAL under the preserved benchmark split. It is not an identity-disjoint training/validation study. These overlaps are disclosed rather than removed by changing the existing split. Group labels and age/skin estimates retain their existing proxy provenance; the sample is too small for population fairness conclusions.

Proposed generation preserves the pinned Qwen model, 40 steps, true CFG 4, BF16, 1MP geometry, original-only inputs, seven independent conditions and geometric controls. **32 × 7 = 224** diffusion outputs, plus 32 geometric controls. No precision, resolution, steps or prompt was silently changed. Existing approvals are not copied and invalid candidates remain excluded. A failed group cannot be silently dropped.

This proposal is currently **DRAFT / DO NOT LAUNCH**: first finish the 14-pair review and decide whether the current editor's confounds justify additional same-recipe generation. A selected-source generation interface and its software checks are also required; the existing `--max-sources 32` alternates classes from source order and does **not** implement the proposed stratified selection. Do not bypass the full-bank coverage guard or pass this private sample manifest as a replacement TRAIN dataset. If review identifies a concrete prompt/editor correction, record it in a subsequent version before generating.

## Tentative time and cost

| Work | Estimate and status |
|---|---|
| Automatic integrity review, 32-source selection and private review page | Completed in this session; not a GPU training run |
| Human review of current 14 candidates | Allow roughly 20–40 minutes for an initial appearance review; expert forensic assessment/adjudication has no measured duration and may remain inconclusive |
| Proposed 32-source, 224-output generation | **4.33 GPU-hours** from the previous measured mean; plan **4–6 hours on one equivalently fast continuously available GPU**, before retries/review/queue |
| Two equally fast dedicated GPUs | Arithmetic base estimate **2.16 hours**, before load, imbalance, retries and review; two suitable GPUs are not currently reserved |
| Full original bank | Approximately 10,292 GPU-hours remains the previous extrapolation; not launched |
| 50+50 training and full B/N/C comparison | No defensible end-to-end ETA yet: no accepted/reviewed bank or revised training configuration, and no epoch timing for this exact run |

The ranges are planning allowances, not confidence intervals or promised completion times. The measured per-output pilot range was 56.74–90.50 seconds, based on two correlated sources on a shared GPU. The human-review allowance is a planning guess, not measured reviewer throughput. Current free VRAM fluctuates and does not establish ownership; all reservations and CUDA binding must be rechecked at any future launch.

## Launch decision and next dependency

**No new training or generation was launched.** The user's request to run the next work was applied to the automatic recheck, actual source selection, private review preparation and versioned revision. It is not recorded as per-pair human validity approval.

The immediate dependency is the completed, evidence-bearing 14-pair human review. The existing handoff explicitly requires stopping if pilot validity/resources do not pass, and per-source reviewed edits plus no-change controls before binding training. After review, choose whether to revise the editor or approve the bounded quality-only scope. No fairness gain, improvement over Ding or successful full selectivity experiment is claimed. **FairForge-2 remains DO NOT RUN** under its independent review. Historical results and the failed preregistered gate are unchanged.

## Later human-review acknowledgement

Observed management UTC: 2026-11-01T09:48:03.630000+00:00. The user reports: “Reviewed; decisions still need recording.” This applies to the current two-source pilot, including controls. No individual pass/fail/uncertain decisions or assertion of forensic preservation were supplied in that response. A follow-up requested those decisions. The original pending review CSV was not modified, no approvals were inferred, and no training or generation was launched. Human review being reported complete is distinct from the required per-pair decisions and substantive forensic-validity evidence being recorded.
