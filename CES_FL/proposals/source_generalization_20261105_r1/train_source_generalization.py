#!/usr/bin/env python3
"""Matched P50-50 source-generalization experiment on a private TRAIN fold.

S0 is native original-only weighted BCE. S1 uses the same P50 recipe and one
TRAIN-image view per source/epoch, chosen from identity/JPEG90/resize90/blur035
for both labels. Validation always uses untransformed development images.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

P50_SHA = "2d66966b1733feec222fe3a80ff9a6cd7181a6ba28cb9f767ad980968d337098"
CES_SHA = "1997eebf093aa3e37de1cbbe58302a1539ac2e517f966a4607aae89dc695a8e0"
TRAIN_SHA = "6ee9326fe8361dc3aee0f803c49b5eada169be5cf3bc3cb44ad3d934ba5ca657"
WEIGHT_SHA = "d9ad71c77abfb482d7d9fc338f4536beda71d6ef4b495b7f60aad8f689832d11"
TRANSFORMS = ("identity", "jpeg90", "resize90", "blur035")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_module(name, path, expected):
    if sha(path) != expected:
        raise RuntimeError(f"Pinned {name} source hash mismatch")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def setup(config_path, fold_audit_path, p50_source, transform_source):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    if cfg.get("adapter_source_sha256") != sha(__file__):
        raise RuntimeError("Adapter source/config hash mismatch")
    if cfg.get("p50_native_source_sha256") != P50_SHA or cfg.get("transform_source_sha256") != CES_SHA:
        raise RuntimeError("Pinned architecture/transform source mismatch")
    if cfg.get("arm") not in ("S0_original", "S1_nuisance_bank"):
        raise ValueError("Unknown source-generalization arm")
    expected_bank = ["identity"] if cfg["arm"] == "S0_original" else list(TRANSFORMS)
    if cfg.get("transform_bank") != expected_bank or cfg.get("augmentation_enabled") != (cfg["arm"] == "S1_nuisance_bank"):
        raise RuntimeError("Arm/transform binding mismatch")
    fold = json.loads(Path(fold_audit_path).read_text())
    if sha(fold_audit_path) != cfg["fold_audit_sha256"] or fold.get("status") != "PASS_WITH_SUPPORT_LIMITATIONS":
        raise RuntimeError("Private TRAIN-fold audit is missing, changed, or invalid")
    native_cfg = cfg["native_config"]
    if digest(native_cfg) != cfg["native_config_digest"]:
        raise RuntimeError("Native P50 config digest mismatch")
    if native_cfg["input_hashes"].get("train") != sha(native_cfg["train"]):
        raise RuntimeError("Fold TRAIN bytes changed")
    if native_cfg["input_hashes"].get("val") != sha(native_cfg["val"]):
        raise RuntimeError("Development bytes changed")
    if native_cfg["input_hashes"].get("weights") != WEIGHT_SHA or sha(native_cfg["weights"]) != WEIGHT_SHA:
        raise RuntimeError("Local ImageNet weights missing or changed")
    p50 = load_module("cesfl_p50_native", p50_source, P50_SHA)
    nuisance = load_module("cesfl_nuisance_transforms", transform_source, CES_SHA)
    p50.validate_config(native_cfg)
    if native_cfg["pretrain_epochs"] != 50 or native_cfg["downstream_epochs"] != 50:
        raise RuntimeError("Native P50 full-epoch recipe required")
    rows, val, weights, split = p50.data_preflight(native_cfg)
    if not rows or not val:
        raise RuntimeError("Empty source-generalization fit/development fold")
    return cfg, fold, p50, nuisance, rows, val, weights, split


class FoldDataset(Dataset):
    def __init__(self, rows, weights, p50, nuisance, seed, epoch, size, augmentation):
        self.rows = rows
        self.weights = weights
        self.reader = p50.PairImages([], size=size, backbone="resnet50")
        self.nuisance = nuisance
        self.seed, self.epoch, self.augmentation = seed, epoch, augmentation

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        if self.augmentation:
            pixels = self.nuisance.load_pixels(row, self.reader.size)
            offset = int(hashlib.sha256(f"{self.seed}:source-generalization:{row['image']}".encode()).hexdigest()[:16], 16)
            case = TRANSFORMS[(offset + self.epoch) % len(TRANSFORMS)]
            x = self.nuisance.normalize(self.nuisance.nuisance_view(pixels, case))
        else:
            case = "identity"
            x = self.reader.load(row["image"])
        return {"x": x, "y": float(row["label"]), "weight": float(self.weights[i]),
                "index": i, "case": case, "label": int(row["label"]),
                "axes": {a: row[a] for a in self.nuisance.AXES}}


def model_init(p50, c, device):
    p50.seed_all(c["seed"])
    model = p50.DiagramCES(c["backbone"], c["weights"], c["hidden"], c["gate_hidden"],
                           pretrained=True, encoder_policy="trainable", bn_policy="update",
                           head_seed=c["head_seed"])
    return model.to(device)


def decode_all(rows):
    for row in rows:
        with Image.open(row["image"]) as image:
            image.convert("RGB").load()


def preflight(config_path, fold_audit, p50_source, transform_source, out):
    cfg, fold, p50, nuisance, rows, val, weights, split = setup(config_path, fold_audit, p50_source, transform_source)
    out = Path(out)
    if out.exists():
        raise SystemExit("Preflight output exists; preserve it")
    decode_all(rows)
    decode_all(val)
    c = cfg["native_config"]
    receipt = {"status": "PASS", "arm": cfg["arm"], "config_sha256": sha(config_path),
               "config_digest": digest(cfg), "native_config_digest": digest(c),
               "adapter_source_sha256": sha(__file__), "p50_source_sha256": P50_SHA,
               "transform_source_sha256": CES_SHA, "fold_audit_sha256": sha(fold_audit),
               "train_manifest_sha256": sha(c["train"]), "development_manifest_sha256": sha(c["val"]),
               "weights_sha256": sha(c["weights"]), "fit_rows": len(rows), "development_rows": len(val),
               "fit_class_counts": {str(k): int(v) for k, v in Counter(r["label"] for r in rows).items()},
               "development_class_counts": {str(k): int(v) for k, v in Counter(r["label"] for r in val).items()},
               "recorded_split_audit": split, "all_fit_and_dev_images_decoded": True,
               "training_epochs": 50, "batch_size": c["batch_size"], "selection": "earliest max development AUC",
               "test_manifest_read": False, "test_scores_read": False}
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, receipt)
    print(json.dumps(receipt, indent=2))


def smoke(config_path, fold_audit, p50_source, transform_source, preflight_path, device, out):
    cfg, fold, p50, nuisance, rows, val, weights, split = setup(config_path, fold_audit, p50_source, transform_source)
    pre = json.loads(Path(preflight_path).read_text())
    if pre.get("status") != "PASS" or pre.get("config_digest") != digest(cfg):
        raise RuntimeError("Source/config-bound real-input preflight missing")
    if device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Expected one verified CUDA device")
    dev = torch.device(device)
    c = cfg["native_config"]
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = model_init(p50, c, dev)
    model.train()
    ds = FoldDataset(rows, weights, p50, nuisance, c["seed"], 0, c["size"], cfg["augmentation_enabled"])
    data_start = time.perf_counter()
    ids = list(range(min(c["batch_size"], len(ds))))
    sample = [ds[i] for i in ids]
    batch = torch.utils.data.default_collate(sample)
    data_seconds = time.perf_counter() - data_start
    before = state_hash(model)
    optimizer = p50.build_optimizer(model, {**c, "encoder_lr": c["pretrain_encoder_lr"]})
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    step_start = time.perf_counter()
    z = model(batch["x"].to(dev))["logit"].float()
    y = batch["y"].to(dev).float()
    loss = p50.objective(z, y, batch["weight"].to(dev))
    if not torch.isfinite(loss["total"]):
        raise RuntimeError("Nonfinite source-generalization loss")
    optimizer.zero_grad(set_to_none=True)
    loss["total"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):
        raise RuntimeError("Missing or nonfinite real-input gradients")
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.))
    if not math.isfinite(grad_norm):
        raise RuntimeError("Nonfinite gradient norm")
    optimizer.step()  # discarded one-step smoke; no checkpoint or selection use
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    step_seconds = time.perf_counter() - step_start
    after = state_hash(model)
    if before == after:
        raise RuntimeError("Smoke optimizer step did not update the model")
    if not bool(torch.isfinite(z).all()):
        raise RuntimeError("Nonfinite logits")
    # A short clean-development forward profile; it has no gradient and no
    # optimizer state is changed. Full-dev prediction is repeated each epoch.
    model.eval()
    val_probe = val[:min(c["batch_size"], len(val))]
    val_reader = p50.PairImages([], size=c["size"], backbone="resnet50")
    val_x = torch.stack([val_reader.load(r["image"]) for r in val_probe]).to(dev)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    val_start = time.perf_counter()
    with torch.inference_mode():
        val_z = model(val_x)["logit"].float()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    val_seconds = time.perf_counter() - val_start
    if not torch.isfinite(val_z).all():
        raise RuntimeError("Nonfinite development probe logits")
    epoch_batches = len(p50.EpochBatches(len(rows), c["batch_size"], c["seed"], 0))
    dev_batches = math.ceil(len(val) / c["batch_size"])
    projected_arm_gpu_hours = 50 * (epoch_batches * step_seconds + dev_batches * val_seconds) / 3600
    cases = Counter(batch["case"])
    result = {"status": "PASS", "arm": cfg["arm"], "device": str(dev), "torch": str(torch.__version__),
              "config_digest": digest(cfg), "adapter_source_sha256": sha(__file__),
              "batch_size": len(ids), "loss": float(loss["total"].detach()), "gradient_norm": grad_norm,
              "fit_batch_decode_seconds": data_seconds, "optimizer_step_seconds": step_seconds,
              "development_probe_rows": len(val_probe), "development_probe_seconds": val_seconds,
              "fit_steps_per_epoch": epoch_batches, "development_batches_per_epoch": dev_batches,
              "projected_full_50_epoch_gpu_hours_before_safety_margin": projected_arm_gpu_hours,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
              "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "initial_state_sha256": before, "post_discarded_step_state_sha256": after,
              "discarded_step_changed_model": True, "transform_counts": dict(cases),
              "val_gradient_updates": False, "test_manifest_read": False, "test_scores_read": False}
    out = Path(out)
    if out.exists():
        raise SystemExit("Smoke output exists; preserve it")
    out.parent.mkdir(parents=True, exist_ok=True)
    write(out, result)
    print(json.dumps(result, indent=2))


def verify_run(out, cfg, p50, device):
    out = Path(out)
    ck = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    if ck.get("adapter_source_sha256") != sha(__file__) or ck.get("config_digest") != digest(cfg):
        raise RuntimeError("Selected checkpoint source/config mismatch")
    c = cfg["native_config"]
    model = p50.DiagramCES(c["backbone"], hidden=c["hidden"], gate_hidden=c["gate_hidden"],
                           pretrained=False, encoder_policy="trainable", bn_policy="frozen",
                           head_seed=c["head_seed"])
    model.load_state_dict(ck["model"], strict=True)
    model.to(device).eval()
    val = p50.read_rows(c["val"])
    z = p50.predict(model, val, device, c["batch_size"], c["workers"], c["size"])
    replay = p50.metrics(val, z, ck["calibration"]["threshold_logit"], c["min_support"])
    saved = ck["validation"]
    max_diff = abs(replay["overall"]["auc"] - saved["overall"]["auc"])
    max_diff = max(max_diff, abs(replay["overall"]["fpr"] - saved["overall"]["fpr"]))
    max_diff = max(max_diff, abs(replay["overall"]["fnr"] - saved["overall"]["fnr"]))
    if max_diff > 1e-5:
        raise RuntimeError("Selected development checkpoint replay mismatch")
    receipt = {"status": "PASS", "selected_checkpoint_sha256": sha(out / "best.pt"),
               "adapter_source_sha256": sha(__file__), "p50_native_source_sha256": P50_SHA,
               "config_digest": digest(cfg), "selected_epoch": ck["epoch"], "validation_replayed": True,
               "max_metric_difference": max_diff, "test_evaluated": False, "test_manifest_read": False}
    write(out / "RUN_VERIFIED.json", receipt)
    return receipt


def train(config_path, fold_audit, p50_source, transform_source, preflight_path, smoke_path, device, out):
    cfg, fold, p50, nuisance, rows, val, weights, split = setup(config_path, fold_audit, p50_source, transform_source)
    pre = json.loads(Path(preflight_path).read_text())
    smoke = json.loads(Path(smoke_path).read_text())
    if pre.get("status") != "PASS" or pre.get("config_digest") != digest(cfg):
        raise RuntimeError(f"Missing or stale stage3 preflight: {preflight_path}")
    if smoke.get("status") != "PASS" or smoke.get("config_digest") != digest(cfg):
        raise RuntimeError(f"Missing or stale stage3 smoke: {smoke_path}")
    if device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Expected exactly one verified CUDA device")
    out = Path(out).resolve()
    if out.exists():
        raise SystemExit("Training output exists; do not duplicate/overwrite")
    out.mkdir(parents=True)
    write(out / "config.json", cfg)
    c = cfg["native_config"]
    p50.seed_all(c["seed"])
    model = model_init(p50, c, device).to(device)
    init_hash = state_hash(model)
    if init_hash != smoke.get("initial_state_sha256"):
        raise RuntimeError("Fresh training initialization differs from source-bound smoke")
    opt = p50.build_optimizer(model, {**c, "encoder_lr": c["pretrain_encoder_lr"]})
    train_weights = torch.tensor(weights, dtype=torch.float32)
    write(out / "RUN.json", {"status": "RUNNING", "arm": cfg["arm"], "adapter_source_sha256": sha(__file__),
                              "p50_native_source_sha256": P50_SHA, "transform_source_sha256": CES_SHA,
                              "config_digest": digest(cfg), "initial_state_sha256": init_hash,
                              "model_counts": model.counts(), "fit_rows": len(rows), "development_rows": len(val),
                              "epochs_planned": 50, "test_manifest_read": False, "test_scores_read": False})
    history = []
    best_auc, best_epoch = -math.inf, None
    start = time.perf_counter()
    for epoch in range(50):
        epoch_start = time.perf_counter()
        p50.seed_all(c["seed"] + epoch)
        model.train()
        ds = FoldDataset(rows, weights, p50, nuisance, c["seed"], epoch, c["size"], cfg["augmentation_enabled"])
        loader = DataLoader(ds, batch_sampler=p50.EpochBatches(len(rows), c["batch_size"], c["seed"], epoch),
                            num_workers=c["workers"], generator=torch.Generator().manual_seed(c["seed"] + epoch),
                            pin_memory=False)
        seen, case_counts, cell_counts = set(), Counter(), Counter()
        loss_sum = 0.0
        for step, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=False)
            y = batch["y"].to(device).float()
            w = batch["weight"].to(device).float()
            z = model(x)["logit"].float()
            losses = p50.objective(z, y, w)
            if not torch.isfinite(losses["total"]):
                raise RuntimeError("Nonfinite training loss")
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad):
                raise RuntimeError("Missing/nonfinite trainable gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            opt.step()
            batch_ix = batch["index"].tolist()
            if seen.intersection(batch_ix):
                raise RuntimeError("Duplicate original exposure in an epoch")
            seen.update(batch_ix)
            loss_sum += float(losses["total"].detach()) * len(batch_ix)
            for local_i, (label, case) in enumerate(zip(batch["label"].tolist(), batch["case"])):
                case_counts[f"label={label}:{case}"] += 1
                for axis, groups in batch["axes"].items():
                    cell_counts[f"{axis}:{groups[local_i]}:label={label}:{case}"] += 1
            if (step + 1) % 100 == 0:
                print(json.dumps({"arm": cfg["arm"], "epoch": epoch + 1, "step": step + 1,
                                  "steps": len(loader)}), flush=True)
        if seen != set(range(len(rows))):
            raise RuntimeError("The epoch did not expose each fitting original exactly once")
        zval = p50.predict(model, val, device, c["batch_size"], c["workers"], c["size"])
        labels = np.asarray([r["label"] for r in val])
        calibration = p50.calibrate(zval[labels == 0], c["target_fpr"])
        report = p50.metrics(val, zval, calibration["threshold_logit"], c["min_support"])
        auc = report["overall"]["auc"]
        selected = auc > best_auc  # strict inequality preserves earliest maximum
        if selected:
            best_auc, best_epoch = auc, epoch + 1
        rec = {"epoch": epoch + 1, "steps": len(loader), "fit_rows": len(seen), "originals_once_each": True,
               "training_loss_mean": loss_sum / len(rows), "development_validation": report,
               "calibration": calibration, "transform_counts": dict(case_counts),
               "transform_group_class_counts": dict(cell_counts), "selection": "validation_auc_only_earliest_tie",
               "selected_this_epoch": selected, "epoch_seconds": time.perf_counter() - epoch_start,
               "test_evaluated": False}
        history.append(rec)
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(rec, allow_nan=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        checkpoint = {"format": "CESFL_SOURCE_GENERALIZATION_P50_v1", "adapter_source_sha256": sha(__file__),
                      "p50_native_source_sha256": P50_SHA, "transform_source_sha256": CES_SHA,
                      "config_digest": digest(cfg), "config": cfg, "epoch": epoch + 1,
                      "model": model.state_dict(), "optimizer": opt.state_dict(),
                      "calibration": calibration, "validation": report, "best_auc": best_auc,
                      "best_epoch": best_epoch, "test_evaluated": False}
        if selected:
            tmp = out / "best.pt.tmp"
            torch.save(checkpoint, tmp); os.replace(tmp, out / "best.pt")
        tmp = out / "last.pt.tmp"
        torch.save(checkpoint, tmp); os.replace(tmp, out / "last.pt")
        print(json.dumps({"arm": cfg["arm"], "epoch": epoch + 1, "development_auc": auc,
                          "selected": selected, "seconds": time.perf_counter() - epoch_start}), flush=True)
    duration = time.perf_counter() - start
    write(out / "summary.json", {"status": "TRAINED_PENDING_REPLAY", "arm": cfg["arm"],
                                  "epochs_completed": 50, "selected_epoch": best_epoch,
                                  "selected_validation_auc": best_auc, "elapsed_seconds": duration,
                                  "fit_rows": len(rows), "development_rows": len(val), "test_evaluated": False})
    verified = verify_run(out, cfg, p50, device)
    summary = json.loads((out / "summary.json").read_text())
    summary["status"] = "COMPLETED_VERIFIED"
    write(out / "summary.json", summary)
    (out / "DONE").write_text("Completed full native 50-epoch schedule and development replay; TEST not read.\n")
    return {"status": "COMPLETED_VERIFIED", "selected_epoch": best_epoch,
            "selected_validation_auc": best_auc, "elapsed_seconds": duration,
            "checkpoint_sha256": verified["selected_checkpoint_sha256"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("preflight", "smoke", "train"))
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--fold-audit", required=True, type=Path)
    p.add_argument("--p50-source", required=True, type=Path)
    p.add_argument("--transform-source", required=True, type=Path)
    p.add_argument("--preflight", type=Path)
    p.add_argument("--smoke-receipt", type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if a.command == "preflight":
        preflight(a.config, a.fold_audit, a.p50_source, a.transform_source, a.out)
    elif a.command == "smoke":
        if not a.preflight:
            raise SystemExit("Smoke requires matching real-input preflight")
        smoke(a.config, a.fold_audit, a.p50_source, a.transform_source, a.preflight, a.device, a.out)
    else:
        if not a.preflight or not a.smoke_receipt:
            raise SystemExit("Training requires matching preflight and smoke receipts")
        print(json.dumps(train(a.config, a.fold_audit, a.p50_source, a.transform_source,
                               a.preflight, a.smoke_receipt,
                               a.device, a.out), indent=2))


if __name__ == "__main__":
    main()
