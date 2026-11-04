# Actual local checks

These are CPU software and synthetic mechanism checks. No real-data training, private model diagnosis, CUDA check, benchmark TEST inference or performance improvement was executed locally.

```bash
python -m unittest discover -s CES_FL/tests -p test_lh_nuisance.py -v
python CES_FL/scripts/diagnose_lh_nuisance.py --mode synthetic --out NEW_COUNTEREXAMPLE_OUTPUT
python CES_FL/src/ces_fl_nuisance_resnet50.py smoke --device cpu --nuisance-mode matched --out NEW_SMOKE_OUTPUT
```

Executed evidence:

- `evidence/unit_tests_final.log`:6focused checks, PASS. They cover AST identity of model/loss/selection functions, original-data control parity, matched nuisance cases across all groups/classes, unaugmented evaluation, metadata/RNG independence of case assignment, zero-strength pair cancellation and the mathematical scale/offset counterexamples.
- `evidence/SMOKE_CPU_MATCHED.json`:PASS, final scientific source hash bound. Full-width224px ResNet/layer4 backward, correct23,510,081total/14,966,785trainable counts, finite gradients and frozen BN; synthetic A/H lifecycle, identical initialization, wrong-backbone rejection, checkpoint reload, bitwise resume, calibration, source coverage and synthetic label-flip isolation.
- The lifecycle fixture freezes the encoder to bound checkpoint/optimizer storage. Production configurations use layer4training. Full layer4gradients were checked separately in the same smoke. This limitation is explicit, not a claim of an actual production GPU run.
- `evidence/counterexamples_final/SYNTHETIC_COUNTEREXAMPLES.json`:actual values from four random texture tensors and score-space examples. A fixed texture-only score can satisfy the procedural sensitivity hinge; logit scaling can also satisfy it. These counterexamples do not show that real L_H checkpoints use the shortcuts.
- Four configs differ only by nuisance_mode/error_control, share P50/input/source hashes, and preserve original rates/losses. All have three epochs and native0.75hourboundary allowance.
- `bash -n commands.sh`, Python compilation and diagnostic/report CLI help checks passed.
- `evidence/report_guards.log`: missing cohort arms and synthetic fixture receipts cannot be published as a complete production comparison; current source/config/smoke bindings agree.

The first full lifecycle attempt failed with exit120 while saving large synthetic checkpoints on a nearly full local disk. `evidence/smoke_cpu_attempt1_failure.json` records the source, failure and sizes of removed generated binaries. Only this attempt's synthetic checkpoint binaries were removed; its metadata remains local. The isolated worktree was narrowed to CES_FL, and the successful retry used the explicitly frozen-encoder lifecycle fixture described above. Historical models, real results and user data were not deleted or overwritten.

The checkpoint diagnostic and four-arm report are prepared but have not been run against private production checkpoints/caches here. Source-specific CUDA smoke, native real-input backward, all four TRAIN/VAL preflights, genuine procedural qualification, full cohort resource reservations and four actual training runs remain operator work. Publication is not launch.
