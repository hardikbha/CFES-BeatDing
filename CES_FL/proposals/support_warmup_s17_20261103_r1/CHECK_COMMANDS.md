# Actual check commands and concise evidence

These document completed checks. Existing output directories and receipts are preserved: do not copy these commands to rerun into those paths. GPU commands below are the native argv dispatched inside the actual pool-v2 UUID-bound wrapper; a bare hostname or idle utilization is not allocation. The wrapper checks one visible H100, CUDA and bf16, sets allocator fraction0.16, and records actual UUID / torch2.7.0+cu126 before dispatch.

Scientific source: `/home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-execution-20261103/CES_FL/src/ces_fl_forensic_response.py`; commit8d5f0996a3f1cc02fbeaff9d25c1d95f714e8e77 / SHA0eeafd838da3ca158336dcfe3b8195842dd15bc9fecd67ebbaf2ea03ba3c2680.

CPU unit commands, from the preparation worktree (same scientific source bytes):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/sachin.chaudhary/.conda/envs/aiface/bin/python -u -m unittest discover -s CES_FL/tests -p test_forensic_response.py -v
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/sachin.chaudhary/.conda/envs/aiface/bin/python -u -m unittest discover -s CES_FL/tests -p test_support_warmup.py -v
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/sachin.chaudhary/.conda/envs/aiface/bin/python -u -m unittest discover -s CES_FL/tests -p test_support_operator.py -v
```

Actual outputs: legacy `Ran13tests in2.004s / OK`; support `Ran9tests in1.791s / OK`; final resource guards `Ran5tests in0.002s / OK`. Full exact outputs, including normal spacing, are in evidence/cpu_legacy_units_final.log, cpu_support_units_final.log and operator_guards_v2.log.

Full software smoke:

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u /home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-execution-20261103/CES_FL/src/ces_fl_forensic_response.py smoke --device cpu --out /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/software_cpu
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u /home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-execution-20261103/CES_FL/src/ces_fl_forensic_response.py smoke --device cuda --out /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/software_cuda
```

Actual native results: status PASS for both, nine check groups each. CUDA receipt has cuda_tested=true, fixture_only=true, production total21,594,159/trainable7,575,591. CUDA task-held56.712220seconds. JSON receipts are in evidence/CPU_SMOKE_PASS.json and CUDA_SMOKE_PASS.json; full private logs remain at the run root. Synthetic TEST-label-flip isolation is a software fixture, not benchmark TEST evaluation.

Fresh real-data gates for each candidate:

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u /home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-execution-20261103/CES_FL/src/ces_fl_forensic_response.py preflight --config /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/E_support/config.json --out /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/E_support/preflight
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u /home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-execution-20261103/CES_FL/src/ces_fl_forensic_response.py preview --config /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/E_support/config.json --preflight /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/E_support/preflight --out /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/E_support/operator_preview
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u /home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-execution-20261103/CES_FL/src/ces_fl_forensic_response.py qualify --preview /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/E_support/operator_preview --review /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/E_support/operator_preview/review_COMPLETED_PRIVATE.json --out /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/E_support/qualification.json
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u /home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-execution-20261103/CES_FL/src/ces_fl_forensic_response.py dry-run --config /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/E_support/config.json --preflight /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/E_support/preflight --arm D --device cuda --out /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/E_support/dry
```

E_support: native preflight PASS, decoded_images101525, train_rows76139, val_rows25386, cross_split_content_overlap0; six-key audit retains163recorded TRAIN/VALexceptions and defers four TESTdeclarations. Preview88cases, all genuinely named AI-reviewed, fresh native qualification recorded. CUDA dry-run PASS, one discarded optimizer step,32real pairs, peak780161024bytes. E5.839014s/F0.777747s cold/warm timings are not scientific throughput comparisons. No human or causal-forgery validation claimed. Exact source/config hashes and numeric loss components are in the corresponding E/F evidence JSONs.

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u /home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-execution-20261103/CES_FL/src/ces_fl_forensic_response.py preflight --config /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/F_support_ramp3/config.json --out /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/F_support_ramp3/preflight
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u /home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-execution-20261103/CES_FL/src/ces_fl_forensic_response.py preview --config /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/F_support_ramp3/config.json --preflight /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/F_support_ramp3/preflight --out /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/F_support_ramp3/operator_preview
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u /home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-execution-20261103/CES_FL/src/ces_fl_forensic_response.py qualify --preview /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/F_support_ramp3/operator_preview --review /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/F_support_ramp3/operator_preview/review_COMPLETED_PRIVATE.json --out /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/F_support_ramp3/qualification.json
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u /home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-execution-20261103/CES_FL/src/ces_fl_forensic_response.py dry-run --config /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/F_support_ramp3/config.json --preflight /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/F_support_ramp3/preflight --arm D --device cuda --out /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/F_support_ramp3/dry
```

F_support_ramp3: native preflight PASS, decoded_images101525, train_rows76139, val_rows25386, cross_split_content_overlap0; six-key audit retains163recorded TRAIN/VALexceptions and defers four TESTdeclarations. Preview88cases, all genuinely named AI-reviewed, fresh native qualification recorded. CUDA dry-run PASS, one discarded optimizer step,32real pairs, peak780161024bytes. E5.839014s/F0.777747s cold/warm timings are not scientific throughput comparisons. No human or causal-forgery validation claimed. Exact source/config hashes and numeric loss components are in the corresponding E/F evidence JSONs.

TRAIN-only diagnostics and applicable frozen CES-FL validation replay:

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/runtime/diagnose_support_warmup.py --source /home/sachin.chaudhary/ces_fl_sources/cesfl-support-warmup-execution-20261103 --run /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1 --out /home/sachin.chaudhary/ces_fl_runs/support_warmup_s17_20261103_r1/diagnostics
```

Actual output: status PASS; gradient_snapshots10; controls_replayed3; test_read=false; optimizer_steps0. Controls are our completed CES-FL A/C/D, not Ding. Exact old selected validation reports reproduced; checkpoint hashes unchanged. Combined CUDA checks plus diagnostics/replay held244.098667seconds, including the two disposable backward checks.

Optional CPU throughput probe: `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 /home/sachin.chaudhary/.conda/envs/aiface/bin/python`, importing the pinned source and calling `bootstrap(existing_D_VAL, other=existing_C_VAL, n=2, seed=17)`. Actual4.842806seconds. This is timing only, not a confidence interval or decision. First attempt used unrelated default internvl and failed on NumPy C-extension import before bootstrap; both outcome/fix records are preserved in evidence/. The full1000replicate analysis has not yet run; it waits native completions.

Operator launch commands and actual PIDs/devices: GPU_OPERATOR_PROMPT.txt. Pending full analysis argv is recorded there and in the frozen CPU controller; it must not be described as passed before its actual outputs exist. No benchmark TEST/Ding/FairForge-2 command belongs to this proposal.
