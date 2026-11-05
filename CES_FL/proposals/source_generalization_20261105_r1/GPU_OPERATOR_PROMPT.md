# GPU operator handoff — source-only generalization

Use only the private TRAIN-derived fold produced by `make_train_fold.py` and the two configs bound by `prepare_configs.py`. Do not read the public or private TEST manifest, TEST scores, or external benchmark scores.

## Prerequisites

1. Confirm the proposal branch/commit is pushed and clean.
2. Verify the adapter, native P50 source, transform source, ImageNet weight, fold, manifests, and config hashes from the private receipts.
3. Run `train_source_generalization.py preflight` for S0. It must decode every fit/development image, confirm no recorded identity/source component overlap, and preserve the fixed ids8 cells. Bind the same decoded-input receipt to S1 only if its native config, manifests, weights, source and fold hashes are identical; the augmentation flag is the sole difference.
4. Run the CUDA smoke on each arm on bound H100 devices. Both must report the same fresh seed-17 initial-state SHA256, finite full-model gradients, a discarded one-step update, no TEST read, and a measured projected 50-epoch cost. Do not train if the combined conservative estimate exceeds 5 aggregate GPU-hours.
5. Check the global GPU ledger and shared-pool leases. Reserve no more than 5 aggregate GPU-hours for checks, two full schedules and replay. Never preempt another task. Prefer two distinct eligible H100s for the two independent arms; if only one is eligible, run the arms sequentially on that GPU. Record process-to-UUID receipts.

## Training

Launch exactly one full 50-epoch schedule per arm, in separate fresh output directories:

```bash
python train_source_generalization.py train \
  --config PRIVATE_CONFIGS/S0_original.json \
  --fold-audit PRIVATE_FOLD/FOLD_AUDIT_PRIVATE.json \
  --p50-source PINNED_P50_SOURCE/ces_fl_resnet50_p50.py \
  --transform-source PINNED_985877_SOURCE/ces_fl_nuisance_resnet50.py \
  --preflight PRIVATE_CHECKS/S0_PREFLIGHT.json \
  --smoke-receipt PRIVATE_CHECKS/S0_SMOKE.json \
  --device cuda --out PRIVATE_RUNS/S0_original

python train_source_generalization.py train \
  --config PRIVATE_CONFIGS/S1_nuisance_bank.json \
  --fold-audit PRIVATE_FOLD/FOLD_AUDIT_PRIVATE.json \
  --p50-source PINNED_P50_SOURCE/ces_fl_resnet50_p50.py \
  --transform-source PINNED_985877_SOURCE/ces_fl_nuisance_resnet50.py \
  --preflight PRIVATE_CHECKS/S1_PREFLIGHT.json \
  --smoke-receipt PRIVATE_CHECKS/S1_SMOKE.json \
  --device cuda --out PRIVATE_RUNS/S1_nuisance_bank
```

Use the verified `torch_guarded_python.sh` wrapper inside the existing shared-pool runner so `CUDA_VISIBLE_DEVICES`, H100 device UUID and allocator cap are checked before model imports. Do not invoke those two commands unbound on a login node.

## Stop conditions and reporting

Stop if either config/source/input hash differs, the fresh initial-state fingerprints differ, any ids8 class cell is missing, CUDA smoke/backward is nonfinite, a GPU binding fails, or the full paired schedule no longer fits the five-hour cap. Do not reduce epochs, alter batch size, alter transforms, change the held-out family, or use a different checkpoint to recover a failed gate.

Report fit/development/unused counts, the sparse development group supports, exact runtime and GPU-hours, selected epoch, clean development AUC, validation-real calibration and FPR/FNR, and transform exposure. The 269 development real rows represent only 41 recorded contributor components; do not make unsupported subgroup fairness claims or call the component graph true-person identity ground truth. Keep manifests, row-level results and checkpoints private. No TEST evaluation is attached to training.
