# Verified results and original-only launch readiness

Observed management UTC: **2026-11-02T10:05:56.376051+00:00**. Scientific code commit: `1ffdce6d746482f5de8766dff5622c08d6b3fded`; latest fetched main: `67d2308225e422a91c715c40427f40c3df6812cf` (fetched2November09:57UTC). No source/configuration, training loss, architecture, selection, calibration or old result was changed. No new full training, TEST evaluation, image generation or job termination occurred.

## Existing completed FF++ results — seed17

These are the existing original-only50+50 models, trained under science commit `58773bbef9d9fe79dd313610f694ddeea12c6037`. They are **validation**, not test results. Their old batch16/label×ids8 weighted sampling differs from the newly proposed complete-quota64 matrix; these values cannot fill the new matrix's pending result cells.

| Existing arm | Epochs completed | Selected epoch | VAL AUC (%) | gender×race F_FPR↓ (fraction) | FPR gap↓ (pp) | Worst-group AUC↑ (%) | Overall FNR↓ (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| BCE | 50 | 22 | 98.448762 | 0.657819251 | 47.935606 | 95.032007 | 3.841458 |
| Existing fairness | 50 | 37 | 98.487114 | 0.684208602 | 50.619890 | 94.049637 | 3.587865 |

The existing fairness arm changes AUC by **+0.038352percentage points**, F_FPR by **+0.026389352**, FPRgap by **+2.684284pp**, and worst-group AUC by **−0.982370pp**. Its FNR is lower, but its primary fairness outcome is worse as a point estimate. No population significance or overall win is established. Female-Black validation support remains30real frames across only2recorded components; uncertainty is substantial.

Both models use their selected checkpoint's validation-real global threshold and achieve409/4092validationfalse positives (9.995112%). Thresholds/checkpoints were not changed. No TEST calibration or test-score reading.

## What was freshly checked

- All55protected code/config/report hashes still match the published proposal manifest. Existing CPU/CUDA smoke receipts remain source-matched; the suite was not unnecessarily rerun.
- Actual TRAIN/VAL/configuration-side input files, cached backbone weights and bound seed17 P50 checkpoint were rehashed. See INPUT_HASH_CHECK.json. This is a byte-hash check, not a repeat of full image decoding.
- Rehashed and CPU-loaded actual BCE/fair best.pt files with weights_only. Both SHA256s match native replay receipts; all saved model tensors are finite. Checkpoint metadata, seed17, selected epoch and validation metrics agree with the complete50epoch histories and AUC-only selection. No paired draws were used. This check did not perform another full validation forward pass; the prior native independent replay remains the inference evidence.
- Latest scheduler and gpu01/gpu02 process queries found no running process matching the proposed CDF trainer. The registered output inventory has no new CDF-study runs. A queued/running job from another research project is not counted as this study.

| Existing checkpoint | Fresh SHA256 |
|---|---|
| BCE | `71d9b50377d3e0812c2721d9271ca3f4d59aaeedfa798a69853d9bff769bfb0b` |
| fair | `21a9fedc42be28b261e59c8bcb6f969bb422744453a9e0217fd5a32f6ab6d974` |

Actual verification command, exit0:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/sachin.chaudhary/.conda/envs/qwen38/bin/python -u \
  /home/sachin.chaudhary/cesel_ma_audit_logs/cdf_followup_check_20261102T095742Z/VERIFY_RESULTS.py \
  > /home/sachin.chaudhary/cesel_ma_audit_logs/cdf_followup_check_20261102T095742Z/verify_results.log 2>&1
```

The initial ad-hoc inspection assumed a flat seed field; the actual archived configuration stores it under pretrain. The corrected reader follows the executable schema and leaves the model/checkpoints untouched.

## Proposed new four-arm matrix — no trained results yet

| Arm | Seeds7/17/27 training results | Readiness |
|---|---|---|
| BCE, new matched protocol | Not produced | Source/software checks PASS; each arm's preflight/backward remains required |
| Complete group-DRO | Not produced | Same; historical ten-epoch R2 is a different study |
| Mean-score fairness | Not produced | Same; old fair metrics above are not new quota64 results |
| CDF fairness | Not produced | Prior seed17 real TRAIN backward PASS with0optimizersteps; no model training/performance result |

Seed17's verified P50 checkpoint is bound. Seeds7and27 have **null pretraining bindings** in the committed configs; do not substitute Ding's classification checkpoints or invent hashes. A full three-seed matrix cannot launch as currently bound. The documented production review/release is still not recorded. Counterfactual nochange/score/response arms remain blocked by valid reviewed quartet data and are outside this check. FairForge-2 remains held.

## Fresh GPU attempt — blocked before model execution

A bounded attempt to add real TRAIN backward checks for BCE/DRO/mean-fair stopped before creating a CUDA model or running any arm. Exact failure:

```text
AssertionError: No idle cooperatively available GPU; no other job modified
```

The guard failed at REAL_CHECK_GUARD.py:27 because apparently free-memory devices had existing cooperative pool reservations. Do not equate free VRAM with an unreserved device. No reservation was removed or bypassed, no existing job was killed, and no backward pass or optimizer update occurred in this attempt. This is an allocation failure, not evidence of a numerical CUDA failure. The prior source-matched15/15CUDA suite remains historical evidence.

Actual command, exit1:

```bash
CES_CHECK_ROOT=$(cat /home/sachin.chaudhary/cesel_ma_audit_logs/CDF_FOLLOWUP_LATEST_PATH.txt)
ssh -o BatchMode=yes -o ConnectTimeout=10 gpu01 \
  /home/sachin.chaudhary/.conda/envs/aiface/bin/python -u "$CES_CHECK_ROOT/REAL_CHECK_GUARD.py" \
  > "$CES_CHECK_ROOT/real_checks.log" 2>&1
```

Next dependency: obtain legitimate GPU binding and complete each arm's check, bind actual P50 outputs for7/27, and record the reviewed original-only release. Do not report future scores or leave a claim that training is running. Full counterfactual training and a paper comparison retain their separate data/comparator/support blockers.

## Existing FairForge-2 completion

The previously authorized original-only pretraining finished10epochs, selected epoch9. Its native replay receipt reports **99.810536% validation AUC**, on public VAL31,107; TEST was not evaluated. The native receipt's checkpoint SHA is `43152de1b98dce48801114efb2454cb0bc6aa7400445dd5f139759022229895a`; this FF2 status check did not rehash that checkpoint or rerun inference. This is an original-only pretraining result, not a CDF/counterfactual result or a matched Ding comparison. All further FF2 work remains HOLD per user instruction.

## Outputs

- Private report/evidence: `/home/sachin.chaudhary/cesel_ma_audit_logs/cdf_followup_check_20261102T095742Z/`.
- Public textual snapshot: `CES_FL/results/cdf_originals_readiness_20261102T095742Z/` on proposal branch `proposal/cesfl-tail-response-20261102-r1`.
- Frozen proposal and exact configurations: `CES_FL/proposals/cdf_response_20261102_r1/`.

No new training score exists for the proposed CDF/DRO matrix. Integrity checks passed; new per-arm GPU checks are operationally blocked, and full-matrix pretraining bindings/release remain incomplete.
