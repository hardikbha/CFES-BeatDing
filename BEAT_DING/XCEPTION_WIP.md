# Xception arm + Ding s17/s27 fast loader: work in progress, STOPPED by the user 26 Oct ~18:45 ("no more required")

Workflow wf_7aa4a185-577 (xception-arm-and-ding-fast) was stopped during its Implement stage. It had not been verified
or fixed, and nothing it built was ever run on a GPU.

**Restored.** The live runner-facing files are back to their pre-workflow state, so the queue cannot launch
half-built tasks. The runner had already listed dingfast_s17/s27 as ready.

| File | Restored from |
|---|---|
| CESEL_MA/run/tasks.yaml | .bak_xc_oct26 |
| CESEL_MA/run/ma_run.sh | .bak_xc_oct26 |
| BEAT_DING/run/bd_run.sh | .bak_xc_oct26 |
| BEAT_DING/cfes/engine.py | .bak_xc_oct26 |
| BEAT_DING/scripts/train.py | .bak_xc_oct26 |
| BEAT_DING/scripts/bd_report.py | .bak_xc_oct26 |
| BEAT_DING/tests/bd_checks.py | .bak_xc_oct26 |

**Kept for a later resume.** The workflow's edited versions of those files are saved as `<file>.xc_wip_oct26`, and
the unswapped `run/bd_tasks.json.new` as `run/bd_tasks.json.xc_wip_oct26`. The new files it added are unused
by anything live:

- **BEAT_DING code:** cfes/ding_xception.py, cfes/model_xception_bd.py, scripts/xception_checks.py,
  scripts/bd_e2.py.
- **BEAT_DING configs:** configs/ffpp_ding_xception_bd.yaml, configs/xception_bd_p2.yaml, configs/bd_xception/{base_E,BD0-BD4}.yaml.
- **BEAT_DING tests:** tests/{identity_bd_step.py, validate_bd_tasks.py, x_ding_ref.py, bd_fixture.py}.
- **DINGREPRO_LIN:** code/Ding_mod/GDA_fastloader.py and .diff, code/run_task_full_fast.sh,
  code/{fastloader_compare.py, fastloader_equiv_harness.py, fastloader_speed_summary.py, check_audit_npz.py}.

**Design (complete):** XCEPTION_DESIGN.md. **Decisions recorded:** Amendment 4a in the plan (D1 X0, D2 b, D3 yes).
Amendment 4b was not written.

**To resume:** re-run the workflow script with resumeFromRunId wf_7aa4a185-577, or restore the .xc_wip files after
review.
