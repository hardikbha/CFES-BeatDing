# Software verification and limits

Model SHA256: `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`. Local environment: Python3.9, PyTorch2.8.0, CPU, no CUDA. These are software checks with synthetic data, not scientific benchmark results.

| Check | Outcome | Evidence |
|---|---|---|
| New architecture/config/helper tests |11 passed|`evidence/unit_tests_final.log`|
| Existing forensic-response tests |13 passed|`evidence/forensic_regression.log`|
| Existing hierarchical-real tests |12 passed,1 skipped because pinned older source unavailable|`evidence/hier_regression.log`|
| Existing support/warmup tests |9 passed|`evidence/support_regression.log`|
| Standalone linear CPU smoke |PASS|`evidence/SMOKE_CPU_RESULT.json`, `evidence/smoke_cpu_v3.log`|
| Source/helpers compile; CLI help; shell syntax |PASS|`evidence/cli_help.txt`; commands below|
| CUDA smoke / actual production input verification / actual selected-reference replay |PENDING GPU node|No local CUDA or cluster datasets/checkpoints|

Smoke tests cover full-width224px parameter counts, direct linear formula and CAM decomposition, exit/head gradients and frozen encoder BN; synthetic two-epoch A and H training; native independent selected-checkpoint validation replay; original/pair coverage; bitwise uninterrupted-versus-resumed H weights; fixed calibration under synthetic TEST-label flip; paired self-comparison and linear map export without patch gates. See the receipt for source-bound checks. Tiny fixture scores have no scientific meaning.

Focused tests verify exact reference gated initialization/loss/gradient parity, AST equality of unchanged science functions, strict head/config/checkpoint separation, exact two-config differences, review byte/membership tamper rejection, and completed-reference/earliest-AUC selection guards. They do not prove fairness, causal validity, real-data accuracy or historical artifacts absent from this machine.

## Preserved failures

`smoke_cpu_attempt1.log`: strict zero-tolerance comparison of two repeat forwards differed by9.313e−10. Only that numerical test tolerance was changed to rtol1e−5/atol1e−7; no scientific loss changed.

`smoke_cpu.log`: second attempt encountered PyTorch checkpoint writer failure while broader tests ran concurrently under severe local disk pressure. Full-suite `all_tests.log` reports64tests,3errors,1skip: two cluster-only operator imports require `/home/sachin.chaudhary/ces_fl_runs/forensic_response_c332fd34_20261103/allocated_stage.py`; an unrelated existing upgrade integration hit another writer failure. Therefore **full discovery did not pass locally**. Do not hide these errors or modify unrelated archived tests to make the report green.

Only failed synthetic checkpoint binaries created by this task were deleted to recover space; their paths/sizes/hashes are in `evidence/failed_fixture_cleanup.json`. Logs/text receipts were kept. Standalone smoke then passed in a fresh `smoke_cpu_v3` directory. Successful binary fixtures remain outside the public repo.

## Reproducible commands

From the repo root, with a suitable Python environment:

```bash
python -m unittest discover -s CES_FL/tests -p 'test_head_ablation.py' -v
python -m unittest discover -s CES_FL/tests -p 'test_forensic_response.py' -v
python -m unittest discover -s CES_FL/tests -p 'test_hierarchical_real.py' -v
python -m unittest discover -s CES_FL/tests -p 'test_support_warmup.py' -v
python CES_FL/src/ces_fl_head_ablation.py smoke --head-type linear --device cpu --out /new/private/synthetic_output
python CES_FL/src/ces_fl_head_ablation.py parameters --head-type linear --encoder-policy exit
python -m py_compile CES_FL/src/ces_fl_head_ablation.py CES_FL/proposals/plain_head_s17_20261103T101418Z/reuse_review.py CES_FL/proposals/plain_head_s17_20261103T101418Z/replay_reference.py
bash -n CES_FL/proposals/plain_head_s17_20261103T101418Z/commands.sh
```

Do not run disk-heavy smoke/integration suites concurrently on low-space storage. Use fresh output paths. A modified source requires fresh receipts. Cluster-level review reuse and legacy checkpoint VAL replay remain unverified until executed against the actual private artifacts.
