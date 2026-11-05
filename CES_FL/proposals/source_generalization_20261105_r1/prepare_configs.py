#!/usr/bin/env python3
"""Bind two matched source-generalization configs to a private TRAIN fold."""
import argparse
import hashlib
import json
from pathlib import Path

P50_SOURCE_SHA256 = "2d66966b1733feec222fe3a80ff9a6cd7181a6ba28cb9f767ad980968d337098"
ADAPTER_SOURCE = Path(__file__).with_name("train_source_generalization.py")
TRANSFORM_SOURCE_SHA256 = "1997eebf093aa3e37de1cbbe58302a1539ac2e517f966a4607aae89dc695a8e0"
TRANSFORMS = ["identity", "jpeg90", "resize90", "blur035"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--p50-config", required=True, type=Path)
    p.add_argument("--fold-audit", required=True, type=Path)
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--development", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Output exists; do not overwrite experiment configs")
    base = json.loads(a.p50_config.read_text())
    fold = json.loads(a.fold_audit.read_text())
    if fold.get("status") != "PASS_WITH_SUPPORT_LIMITATIONS":
        raise SystemExit("TRAIN fold audit did not pass its explicit status")
    if sha(a.train) != fold["hashes"]["fit_train.csv"] or sha(a.development) != fold["hashes"]["development.csv"]:
        raise SystemExit("Fold manifest hashes do not match private audit")
    if sha(a.weights) != base["input_hashes"]["weights"]:
        raise SystemExit("Pinned local ImageNet weights changed or are absent")
    native = dict(base)
    native.update(train=str(a.train.resolve()), val=str(a.development.resolve()),
                  protocol_overlap_file=None, pair_bank=None, pair_bank_sha256=None,
                  pretrain_checkpoint=None, pretrain_checkpoint_sha256=None)
    native["input_hashes"] = {"train": sha(a.train), "val": sha(a.development), "weights": sha(a.weights)}
    # Keep the exact public P50 recipe. The stage adapter runs the native
    # 50-epoch original-only pretraining stage; it does not use the existing
    # full-split P50 checkpoint.
    if (native["pretrain_epochs"], native["batch_size"], native["lr"],
        native["pretrain_encoder_lr"], native["weight_decay"], native["balance_axis"],
        native["selection"]) != (50, 16, .001, .0001, .0001, "ids8", "validation_auc_only"):
        raise SystemExit("Unexpected native P50 recipe; do not guess replacement settings")
    common = {"protocol": "TRAIN-only contributor-component leave-one-family-out robustness",
              "seed": 17, "fold_audit_sha256": sha(a.fold_audit),
              "p50_native_source_sha256": P50_SOURCE_SHA256,
              "adapter_source_sha256": sha(ADAPTER_SOURCE),
              "transform_source_sha256": TRANSFORM_SOURCE_SHA256,
              "native_config": native,
              "native_config_digest": digest(native),
              "selection": "earliest maximum development AUC; clean development images only",
              "calibration": "existing native 10% empirical validation-real rule; global threshold",
              "test_manifest_read": False, "test_scores_read": False}
    a.out.mkdir(parents=True)
    for arm, enabled in (("S0_original", False), ("S1_nuisance_bank", True)):
        config = {**common, "arm": arm, "augmentation_enabled": enabled,
                  "transform_bank": TRANSFORMS if enabled else ["identity"],
                  "transform_schedule": "same SHA256(seed17,image_id)+epoch modulo four; independent of label, group, and score"}
        path = a.out / f"{arm}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
    summary = {"status": "CONFIGS_BOUND", "fold_audit_sha256": sha(a.fold_audit),
               "fit_manifest_sha256": sha(a.train), "development_manifest_sha256": sha(a.development),
               "weights_sha256": sha(a.weights), "config_sha256": {
                   "S0_original": sha(a.out / "S0_original.json"),
                   "S1_nuisance_bank": sha(a.out / "S1_nuisance_bank.json")},
               "test_read": False}
    (a.out / "CONFIG_BINDING_PRIVATE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
