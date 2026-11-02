# Executed checks and launch readiness

Final tested scientific source is pinned in [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json). Older v1–v5 debug receipts are preserved privately and do not certify a later source hash. Final public CPU/CUDA receipts are v6. Environment: `/home/sachin.chaudhary/.conda/envs/qwen38/bin/python`, torch2.10.0+cu128, torchvision0.25.0+cu128, GeomLoss0.3.1. No package/model download or existing job termination occurred.

## Actual commands and outputs

Working directory: `/home/sachin.chaudhary/ces_fl_sources/cesfl-tail-response-20261102-r1`.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/sachin.chaudhary/.conda/envs/qwen38/bin/python -u \
  CES_FL/tests/smoke_cdf_proposal.py --device cpu \
  --out /home/sachin.chaudhary/cesel_ma_audit_logs/cdf_response_20261102_r1_cpu_v6 \
  > /home/sachin.chaudhary/cesel_ma_audit_logs/cdf_response_20261102_r1_cpu_v6.log 2>&1
```

Actual output: `Ran 15 tests in 21.805s`, `OK (skipped=1)`, exit0; 14passed, one explicit CUDA-only skip. [Full stdout](evidence/CPU_STDOUT.log), [receipt](evidence/CPU_RESULT.json). CPU observed UTC `2026-11-02T06:43:18.186895+00:00`.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 gpu01 \
  /home/sachin.chaudhary/.conda/envs/aiface/bin/python -u \
  /home/sachin.chaudhary/cesel_ma_audit_logs/cdf_response_20261102_r1_gpu_smoke_guard_v6.py \
  > /home/sachin.chaudhary/cesel_ma_audit_logs/cdf_response_20261102_r1_cuda_v6.log 2>&1
```

The Python launcher is copied verbatim as [CUDA_SSH_GUARD.py](evidence/CUDA_SSH_GUARD.py); it starts qwen38 Python, checks one visible device and caps its allocator. Actual output: `Ran 15 tests in 9.952s`, `OK`, exit0; all15passed. [Full stdout](evidence/CUDA_STDOUT.log), [receipt](evidence/CUDA_RESULT.json), [exact child command/exit](evidence/CUDA_COMMAND_EXIT.json), [binding](evidence/CUDA_BINDING.json). This was **user-authorized direct SSH**, with cooperative idle-device leases and no existing process stopped; it is **not a PBS allocation**. Node clocks differ from management by roughly eleven minutes; embedded node UTC is recorded as-is, not silently treated as synchronized. Use management observations and monotonic elapsed seconds for ordering.

Tests actually exercise: all seven objective arms; nonzero fairness-only gradients and lambda-dependent updates; equal means/unequal FPR tails; response/anchor score-offset cancellation; fake-score exclusion from real-CDF gradients; missing groups/nonfinite rejection; complete group-DRO state; Ding softmax/logit orientation and separate label conditioning; constant scores; both backbones' trainable gradients and frozen encoder/BN invariance; signed-map logit decomposition; reload; identical head initialization; identical quota schedules; paired seed/component bootstrap, one-component refusal and one-eligible-group behavior; nonfinite threshold rejection; tiny synthetic P+paired run with selected-checkpoint VAL replay; zero-update dry-run; unapproved/synthetic bank release refusal. Tiny synthetic optimizer steps are software fixtures, not research experiment results.

The synthetic full-resolution224px GPU check uses64originals plus24paired views and **zero optimizer steps**. Xception peak allocated11,565.369MiB and forward/backward0.2871s; ResNet-50 peak7,450.070MiB and0.3486s. These are fixture timings, not end-to-end training throughput or a scheduler reservation. Parameter counts were independently instantiated on CPU in [PARAMETER_COUNTS.json](evidence/PARAMETER_COUNTS.json):

| Component | Xception | ResNet-50 |
|---|---:|---:|
| Encoder | 20,806,952 | 23,508,032 |
| Global head | 262,401 | 262,401 |
| Patch head | 262,401 | 262,401 |
| Image gate | 131,201 | 131,201 |
| Patch gate | 131,201 | 131,201 |
| Fusion | 3 | 3 |
| Total / trainable under proposed downstream policy | 21,594,159 | 24,295,239 |
| Trainable with frozen encoder | 787,207 | 787,207 |

### Real inputs and reusable P50

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/sachin.chaudhary/.conda/envs/qwen38/bin/python -u \
  CES_FL/src/ces_fl_cdf_study.py preflight \
  --config CES_FL/proposals/cdf_response_20261102_r1/configs/xception_s17.json \
  --arm cdf_fair --device cpu \
  --out /home/sachin.chaudhary/cesel_ma_audit_logs/cdf_response_20261102_r1_real_preflight_v2 \
  > /home/sachin.chaudhary/cesel_ma_audit_logs/cdf_response_20261102_r1_real_preflight_v2.log 2>&1
```

Exit0; [REAL_PREFLIGHT.json](evidence/REAL_PREFLIGHT.json) PASS. Checked exact TRAIN76,139/VAL25,386 path existence/labels/groups and split declaration; no image-path overlap. Recorded TRAIN/VAL identity/source overlap163, preserved and disclosed, not waived as person independence. Rehashed/loaded the actual native P50 best checkpoint `3174e5b01a84c91cc70afae538505663ca96454d897e6b793c0672826a8f08b5`; verified source/input/seed/backbone/settings, native50epoch history, DONE and validation-replay receipt. Paired bank is null; this preflight certifies **only the original-only arm**, not counterfactual data or all images decoding.

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 gpu01 \
  /home/sachin.chaudhary/.conda/envs/aiface/bin/python -u \
  /home/sachin.chaudhary/cesel_ma_audit_logs/cdf_response_20261102_r1_real_gpu_dry_guard_r3.py \
  > /home/sachin.chaudhary/cesel_ma_audit_logs/cdf_response_20261102_r1_real_cuda_dry_r3.log 2>&1
```

Actual exit0, elapsed73.6086s including input checks/import/load; physical gpu01 GPU6, UUID `GPU-61f54ca7-3cee-1c93-def3-118523e95241`,45%allocator cap,81,087MiB free before binding. [Exact child command](evidence/REAL_DRY_COMMAND_EXIT.json) runs `ces_fl_cdf_study.py dry-run --arm cdf_fair --device cuda` with the committed seed17 config; [guard](evidence/REAL_DRY_SSH_GUARD.py), [binding](evidence/REAL_DRY_BINDING.json), [provenance](evidence/REAL_DRY_PROVENANCE.json), [result](evidence/REAL_DRY_RUN.json). Real64image TRAIN batch decoded; finite gradients for all model parameters, BCE0.6815300584, CDF0.0134027824, total0.6828703284; **optimizer_steps=0**, paired_views0. Validation was metadata-only in this check, with no inference or gradient contribution. No TEST CSV/images/scores opened. This does not establish full-dataset decoding or scientific performance.

Calibration fixtures and the independent parameter counts were executed through [this exact qwen38 Python heredoc](evidence/CALIBRATION_COUNTS_COMMAND.md). The calibration asserts achieved FP counts for `(scores,target,expected_FP)` = `([0,1,2,3,4],.4,2)`, `([0,0,0,1,1],.4,2)`, `([0,1,1,1,1],.4,0)`, `([5],.1,0)`. [Actual thresholds/FP/FPR](evidence/CALIBRATION_FIXTURES.json). Achieved FPRs are0.4,0.4,0,0; tied scores can undershoot the requested target. No TEST scores/labels were supplied. The receipt records actual assertions; it does not claim an unexecuted file-based test command.

## Failures found and preserved

1. Corrected OT constant-score debug fixture failed with a zero auto-diameter schedule (`divide by zero`, `ValueError: Maximum allowed size exceeded`). Fixed the new kernel's diameter at1.0, justified by probabilities in[0,1]. Final CPU/CUDA constant-score regressions pass. The archive is unchanged.
2. First **actual** P50 import preflight failed `KeyError: 'completed_epochs'`: native RUN_VERIFIED stores that budget in summary/history. [Failure stdout](evidence/INITIAL_PREFLIGHT_FAILURE.log). Added a strict native-storage adapter checking50contiguous history epochs, verified summary/replay and checkpoint SHA. The native-storage regression and actual preflight subsequently passed. No completed P50 was rerun.
3. First private real-GPU dry-run harness used runpy without the source directory on sys.path and failed `ModuleNotFoundError: ces_fl_counterfactual_5050`. [Failure stdout](evidence/INITIAL_DRY_HARNESS_FAILURE.log). An intermediate private-harness quoting edit also failed compilation before a successful launch. The r3 harness explicitly sets sys.path and passed. These were launcher errors, not a demonstrated defect in direct `python CES_FL/src/ces_fl_cdf_study.py` invocation. No failed attempt updated parameters or overwrote an old output.
4. Final review added nonfinite-threshold rejection and defined worst-group AUC when only one eligible group exists while leaving disparity undefined. Final v6 tests cover both. Earlier smoke hashes are not used to certify this change.

## Compute estimate and limits

Actual prior50epoch wall times: P9,839.41s(2.73h), BCE10,201.23s(2.83h), fair10,277.92s(2.85h). [Historical P receipt](evidence/HISTORICAL_P_PASS.json), [BCE](evidence/HISTORICAL_bce_PASS.json), [fair](evidence/HISTORICAL_fair_PASS.json). Those used the old batch16/native exposure and a different GPU-sharing/runtime configuration; they are **not benchmarks for new batch64/CDF quotas**.

The first three-seed Xception four-arm stage needs12downstream runs plus2new P50 runs, reusing seed17 P:14new stages. At the old roughly2.8h/stage reference, budget about**39GPU-hours**, with an indicative planning range**28–70GPU-hours** before queues/I/O/retries. Both backbones' four-original-arm matrix adds15stages:29newstages total, about81GPU-hours at that reference. All seven arms/three seeds/two backbones would be47newstages after reuse, about132GPU-hours before extra paired forwards and bank preparation. These estimates are uncertain and are **not launch commitments**. The new full-resolution synthetic response step alone scales to about4.75GPU-hours over59,500steps for Xception before I/O/optimizer/validation; benchmark real throughput only after release. Do not equate maximal GPU occupancy with shorter wall time on shared resources. No bank-generation cost/ETA is known because a qualified operation and bank do not exist.

## Remaining blockers and verdict

| Item | Status |
|---|---|
| Final-source CPU/CUDA synthetic software | PASS |
| Xception seed17 original-only input hash/P50 import + real TRAIN backward | PASS, no optimizer update |
| Complete real image decode / other arms' own real-input checks | Not executed; required before their launch |
| Other seed/backbone P50 bindings | BLOCKED_INPUTS, no invented hashes |
| Independently reviewed production release | Missing; template approved=false |
| Matched edit/operation bank and independent exclusions | BLOCKED_INPUTS; no generation authorized |
| Corrected Ding production stage wrapper / SFD binding | Not implemented; loss kernel PASS only |
| Public FairForge-2 full comparison | HOLD; existing original-only pretraining may finish |
| Strong subgroup confirmation support on current VAL | Below fixed floor: Female-Black2recorded components/class; other deficient cells retained and reported |
| New TEST evaluation / confirmatory superiority evidence | Not authorized, not executed |

**DO NOT RUN the full study.** Review may release the staged non-counterfactual controls once their own prerequisites are met; it cannot release response arms without the bank. Existing jobs, native artifacts and failed historical gates remain preserved. No experiment result establishing better fairness, detection, generalization or novelty was generated by these software checks.

The allowed existing FairForge-2 original-only job reached10/10and produced DONE/RUN_VERIFIED during this task; final status is in FINAL_JOB_OBSERVATION.json. No follow-on was launched. This completion is a separate pre-existing experiment, not evidence for the new CDF/response hypothesis.
