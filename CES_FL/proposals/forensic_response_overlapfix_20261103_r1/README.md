# FF++ shared-audit compatibility fix — 3 November 2026

Parent experiment: [forensic_response_20261103_r1](../forensic_response_20261103_r1/README.md), pinned commit `9030dd77d1d93593a4a3808be20b3b903aaeb0da`. The parent checkout, proposal, test receipts, image and results are preserved. This follow-up is on a separate branch, `fix/cesfl-overlap-audit-20261103`.

## Reported failure and verified cause

The operator reports that FF++ preflight stopped at the parser's `Unexpected overlap exception keys` rejection despite the overlap audit matching its configured SHA256. That audit inventories six keys: identity/source contributors for TRAIN/VAL, TRAIN/TEST and VAL/TEST. The old parser allowed only the two TRAIN/VAL keys in training mode.

The operator also reports zero GPU time spent on this new study and no change to the existing Ding run. These are **operator-reported facts**; this patch review did not access the cluster, inspect its job state, or obtain the real audit bytes. The original failure log should remain in its original run directory. No local synthetic test is described as a passed real-data preflight.

The failure was independently reproduced against the pinned function using a six-key synthetic inventory. The replacement accepts that inventory without altering its bytes. See [parser regression evidence](evidence/parser_regression.json). The old smoke tests covered narrower exception files and missed this compatibility case.

## Minimal change

Only the `split_audit` function changes in the production Python source:

- TRAIN/VAL mode recognizes the six known historical keys, but compares actual contributor overlaps only against `identity_ids:train:val` and `source_ids:train:val`.
- The audit receipt records `checked_overlap_keys` and the four `deferred_overlap_keys`. Deferred declarations have not been checked against TEST data.
- Unexpected keys, changed/undeclared TRAIN/VAL overlap and shared image paths still fail. Malformed declaration containers are rejected explicitly.
- Evaluation still requires the separate, explicit `fit:test` inventory for TRAIN+VAL versus TEST. A six-key historical audit is not silently converted into evaluation authorization or reused as proof of fit:TEST separation.

**Do not trim or rewrite the real audit file.** Its original hash and all declarations remain intact. The two configuration templates here are byte-identical copies of the parent templates. Dataset membership, preprocessing, architecture, loss functions, optimizer, seeds, epoch count, sampling, selection and thresholds are unchanged. The original experiment hypothesis and A/B/C/D baselines remain in force.

New source SHA256: `89530025a00b9b0405fba7026d3eed5f2749d86620174ad3ffde93a44f8b770b`.

## Checks and remaining dependencies

**Verified locally: all13 unit tests and the fresh full CPU smoke suite pass for the new source hash.** No real-data or CUDA pass is claimed.

The regression suite now includes full six-key inventories, altered TRAIN/VAL overlaps, malformed/unknown keys, wrong-stage exception reuse and path leakage. CPU smoke evidence must match the new source hash; the parent receipt does not certify this patch. See the new `evidence/` receipts for actual outcomes.

The first full local rerun encountered a PyTorch checkpoint-write failure while the filesystem was nearly full. Its traceback is retained in `evidence/smoke_cpu_attempt1_write_failure.log`. Only temporary checkpoint binaries from two superseded synthetic smoke attempts were removed to free space; their text receipts/logs, the pinned final smoke, real research runs and the new failed attempt were preserved. Cleanup is recorded separately. The full suite was retried in a fresh output directory. Check available disk space as well as GPU memory before executing the cluster smoke suite (its synthetic checkpoint package is about1.4GiB, plus temporary write space).

Use [the updated operator handoff](GPU_OPERATOR_PROMPT.txt) from a new immutable checkout and fresh output directory. Bind a new configuration, rerun **CPU preflight first**, and inspect the checked/deferred keys before reserving GPU time for this study. Fresh previews/qualification and CUDA/software/real-input receipts are required because they are source-bound. Do not transplant old receipts or relabel the failed run successful.

Real FF++ preflight, demographic support, operator qualification, CUDA checks and scientific training remain pending until the operator supplies matching evidence. No cluster training, TEST evaluation, editor generation or Ding-job operation is performed by this patch.

## Novelty assessment

This compatibility fix adds no methodological novelty. The parent model's candidate contribution remains alignment of processing-control-relative forensic responses across groups, conditioned on the intervention operator and strength. D versus C isolates that additional objective. The current two procedural operators and fairness surrogate make a research hypothesis testable; they do not establish that the method is new relative to all prior literature, valid on real manipulations, or better than Ding. The bounded literature context and limitations in the parent proposal still apply. No novelty score or performance claim should be increased because a parser test passes.
