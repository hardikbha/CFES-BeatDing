# Direct SSH continuation: CUDA/input checks passed, P50 running

Observed management UTC: 2026-11-01T12:30:37.435123+00:00.

Exact research source remains **58773bbef9d9fe79dd313610f694ddeea12c6037**, detached and clean at `/home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb`. All four scientific source/config/test hashes match the original manifest. No loss, architecture, optimizer, split membership, seed, epoch budget, validation selection or calibration policy changed. Existing experiments and the Qwen pilot are preserved.

## Route and actual checks

The user explicitly requested direct SSH to gpu01/gpu02. PBS smoke44963 was rechecked Q with no execution vnode/host and then cancelled before execution to avoid duplicates. Its old logs/receipts remain intact. Its obsolete read-only observer was retired only after verifying PID/cmdline. No running GPU experiment was terminated. The new route has **no PBS allocation** and makes no claim of exclusive GPU ownership.

Each single-milestone worker acquires the shared-dispatcher exclusion and the existing project lease, explicitly protects the active Qwen UUID, requires20GiB free headroom and applies a15% Torch allocator cap. Existing processes continue. Both nodes were inspected; actual stages ran on gpu02 GPU5, UUID `GPU-bff97893-385b-6f3f-831b-113284c2e1f1`. Normalized PyTorch/NVIDIA UUID equality passed. The operational wrapper changed for this route; the research scripts did not. Six direct-route safety tests and four truthful-observer tests passed; outputs/commands are preserved.

The **new six-group CUDA suite passed** under aiface/Torch2.7.0+cu126 in87.33seconds (worker binding/start overhead included). Actual command:

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb/CES_FL/tests/smoke_manipulation_response.py --device cuda --out /home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/smoke_cuda
```

This exercised numerical/gradient references, collapse/offset cases, bank/review guards, identical initialization/frozen BN, all five synthetic training arms with checkpoint/validation replay, and bitwise resume/tied calibration. It remains synthetic software evidence: it does not establish valid manipulation quartets, fairness gains or novelty. The log's tiny fixture training is not a scientific arm result.

Full original-data readiness then passed: **76,139 TRAIN +25,386 VAL** images all decoded and SHA256-hashed; all were380×380. Every TRAIN hash matched the existing full inventory. Exact TRAIN/VAL image-byte overlap was **0**. The preserved benchmark declares **163** overlapping video-subject/source proxies between TRAIN/VAL; these are documented benchmark exceptions, not evidence of a person-disjoint split. No TEST images were read. Class counts: TRAIN12,926real/63,213fake; VAL4,092real/21,294fake. Full group support is in the aggregate receipt.

Actual allocated-node helper command (zero optimizer updates):

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/runtime/real_inputs.py
```

It executed the exact base backward command:

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb/CES_FL/src/ces_fl_counterfactual_5050.py dry-run --config /home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb/CES_FL/proposals/counterfactual_5050_20261101_r1/protocol_config.json --stage pretrain --device cuda --out /home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/P_real_dry
```

The16-image CUDA backward pass **passed**: total/classification loss0.5129827857, all21,594,159parameters trainable with finite gradients, peak allocated2,367,416,320bytes, **0 optimizer steps**. BN updates belonged only to the discarded dry-run model. The full decode+backward milestone took340.95seconds. Source/input/threshold policies were unchanged.

## P50 actually launched

Duplicate checks found no active manipulation-response pretrainer on either node, no exact-config completed P50 in known experiment roots, and no existing output at the new P path. No legacy ten-epoch P was substituted.

Launch command (exit0):

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/runtime/launch_stage.py --stage P --host gpu02
```

Actual training command:

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python /home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb/CES_FL/src/ces_fl_manipulation_response.py pretrain --config /home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb/CES_FL/proposals/manipulation_response_20261101_r1/config.json --out /home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/P --device cuda
```

Worker PID **1390084**, actual training PID **1393109**, **gpu02 physicalGPU5 H10080GB**. CUDA binding and **26,369MiB free** were checked before launch; this is shared VRAM, not an exclusive card. P trains50epochs on original TRAIN only with the fixed class×ids8 loss weights, full fine-tuning and updating BN. Selection remains validation AUC only. A completed epoch or final result is not claimed before its history/replay receipts exist.

| Arm | Actual status at this observation | Dependency |
|---|---|---|
| P | RUNNING | 50epoch training and independent validation replay |
| bce | BLOCKED_INPUTS | Verified shared P50, exact binding and BCE backward check |
| fair | BLOCKED_INPUTS | Same shared P50, exact binding and fair backward check |
| nochange | BLOCKED_INPUTS | P50 plus valid reviewed matched quartet bank, audit/bind and own backward check |
| score | BLOCKED_INPUTS | Same bank/P50 and score backward check |
| response | BLOCKED_INPUTS | Same bank/P50 and response backward check |

No automatic chain or downstream launch is attached. Next operator milestone after verified P50 is the documented original-only binding and configured BCE/fair backward checks. No unattended run-all executor was installed. The read-only observer cannot perform any of these actions.

## Outputs, monitoring and remaining blockers

Run root: `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z`. P checkpoint/history outputs: `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/P`. Actual logs: `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/logs/P.log`; process/binding/failure evidence: checks/. Private full original image inventory remains on the cluster and is not published. Original quartet preparation remains `/home/sachin.chaudhary/ces_fl_runs/manip_response_20261101T104925Z/quartet_preparation`; it contains prospective input paths, not a complete reviewed bank.

The existing Qwen32 generation is separate on gpu01 and was not stopped, repurposed, or extended. Candidate age/no-change images alone are not full manipulation quartets. The operator adapter, t0/ta outputs, matched processing controls and qualified independent review still block the paired arms. No approval was fabricated and no forensic-validity claim follows from software success. FairForge-2 remains DO NOT RUN; no test evaluation or new seed/architecture was launched.

[Timestamped per-arm status](../../../status/manip_response_ssh_20261101T115720Z_live.md) and its machine record are published by a finite **read-only** observer (at most72hours, normally every ten minutes/on state changes). It cannot launch, retry, approve, cancel or evaluate TEST. Actual completed epochs, group metrics/support, calibration, hashes and failures appear as they are recorded. No completion time is promised before this exact P run has measured epoch timing. Existing test results were previously viewed; this validation development experiment does not erase that history or establish a win over Ding.
