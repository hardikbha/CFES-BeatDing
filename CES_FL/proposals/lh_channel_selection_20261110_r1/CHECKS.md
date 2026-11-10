# Actual verification of the delivered source

Source SHA256: `95574295b76f16f5e5d8f70f01502848b4616fc7437030a5dfdf2ff1219f62f0`.

-14targeted CPU unit tests passed (`evidence/unit_tests_final.log`).
-Selector numerical reference, masked gradient/CAM/frozen-buffer checks, checkpoint reload and AI-review rejection passed (`evidence/SELECTOR_SMOKE.json`).
-Full-width224px backward, synthetic2epoch A and full-L_H training, exact interrupted resume, global-threshold calibration and native replay passed (`evidence/SMOKE_RESULT.json`).
-An additional2epoch full-L_H run with41channels actually masked passed. Mask/early layers/BN stayed fixed; exit/head changed; all original L_H loss keys remained; selected checkpoint replay passed (`evidence/MASKED_LIFECYCLE_PASS.json`).
-13original objective/preprocessing/evaluation function ASTs are unchanged (`evidence/LH_OBJECTIVES_UNCHANGED.json`).
-`python -m py_compile`, `bash -n commands.sh`, standalone CLI parameter count and Git diff whitespace checks passed.

Commands used:

```bash
python -m unittest discover -s CES_FL/tests -p test_lh_channel_selection.py -v
python CES_FL/src/ces_fl_lh_channel_selection.py selector-smoke --out NEW_SELECTOR_DIR
python CES_FL/src/ces_fl_lh_channel_selection.py smoke --device cpu --out NEW_SYNTHETIC_DIR
python CES_FL/proposals/lh_channel_selection_20261110_r1/tests/masked_lifecycle.py --smoke-root NEW_SYNTHETIC_DIR --out NEW_MASKED_DIR
```

All images/labels in software tests were synthetic fixtures. No real benchmark TEST was accessed. No CUDA hardware, real teacher bytes, production audit bank, human approvals, production mask, real training, speedup, scientific novelty or fairness benefit has been verified here. The GitHub teacher receipt was inspected; private cluster bytes still need independent verification. The script deliberately blocks missing/unverified production dependencies.

Earlier logs under `evidence/superseded/` retain their original source hashes and do not certify the delivered file.
