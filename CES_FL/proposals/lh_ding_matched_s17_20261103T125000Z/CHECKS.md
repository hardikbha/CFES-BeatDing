# Software checks and execution limits

CPU environment: Python3.9, PyTorch2.8.0. No CUDA, pretrained model download or benchmark data used. Source hashes are in `evidence/source_manifest.json`.

- Six focused checks PASS: exact pair/image units and final partial batches; deterministic fake-once/real-replacement draws; torchvision-equivalent rectangular validation preprocessing; strict protocol rejection; production-width Xception gradient/BN/parameter checks with75px synthetic inputs; synthetic tiny P→A/H transfer, two-epoch training, final-epoch selection despite tied earlier AUC, replay, exposure and bitwise resumed weights.
- Eleven existing head-ablation regression checks PASS; the pinned science source is unchanged.
- Adapter/test Python compile and CLI help PASS.
- Actual ImageNet checkpoint load, full224px production batch on CUDA, real input preflight, target-device profiling, actual H review binding,50epoch runs, uncertainty and TEST remain PENDING. The end-to-end loop fixture uses a tiny synthetic network; the separate gradient test uses real full-width Xception. Neither is a benchmark performance result.

Commands (run from repo root):

```bash
python -m unittest discover -s CES_FL/tests -p test_matched_recipe.py -v
python -m unittest discover -s CES_FL/tests -p test_head_ablation.py -v
python -m py_compile CES_FL/src/ces_fl_matched_recipe.py CES_FL/tests/test_matched_recipe.py
python CES_FL/src/ces_fl_matched_recipe.py --help
```

Evidence: `evidence/tests_cpu.log`, `evidence/head_regression.log`. Full repository discovery was not rerun and is not claimed passing. No scheduler or GPU-node validation has occurred. The user's later request prioritizes inspecting L_H's early best epoch; this prepared50+50 recipe is deferred, not the next automatically authorized launch.
