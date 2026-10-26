# Amendment 7 report

Changed/new files:
- `scripts/train_clipfair.py`: line 16 (copy import), lines 46–82 (pooled checkpoint and final fallback), 126–132 (persistent eligibility state/resume guard), 205–207 and 215 (loop integration). Backup: `scripts/train_clipfair.py.bak_codex_oct26`.
- `tests/test_amend7_fallback.py`: lines 1–76, four CPU mocked-loop tests using production selection and checkpoint I/O.
- `codex_ops/ACTIONS_01.md`: lines 1–14, action log (initial version backed up as `.bak_codex_oct26`). Test outputs: `codex_ops/test_amend7_fallback.log`, `codex_ops/clipfair_checks.log`.

Selection remains `scripts/train.py:126–140`: AUC minus 0.5 × FPR gap. Pooled candidates require TPR ≥ 80 and FPR ≤ 20 and epoch ≥ 0. Any eligible trained epoch prevents fallback, even if it does not improve the start score. Otherwise best_pooled.pt is copied to best.pt. TRAINED records selection_fallback (false / true / "none_eligible") and best_epoch. Evaluation still reads best.pt. Shared train/engine modules were inspected and unchanged. New-code resumes retain history; pre-amendment checkpoints must restart because they lack pooled history. Already-running imported code is unaffected.

Tests (both exit 0; no skips):
```text
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 /home/sachin.chaudhary/.conda/envs/pulid/bin/python tests/test_amend7_fallback.py
Ran 4 tests; OK; ALL_PASS
CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/sachin.chaudhary/.conda/envs/pulid/bin/python tests/clipfair_checks.py
18/18 checks passed; ALL_PASS (including real CLIP on CPU)
```

Moved only:
- `runs/ffpp_daw_s7` → `runs/ffpp_daw_s7_DEGENERATE_oct26`
- `runs/ffpp_daw_s7.RUNLOCK.gone.gpu01_44112.mgmt01_1969594_1792981318236049630_1101328702` → same name plus `_DEGENERATE_oct26`.

Other runs/ entries were unchanged during the moves. No network, SSH, PBS, installs or GPU work.
