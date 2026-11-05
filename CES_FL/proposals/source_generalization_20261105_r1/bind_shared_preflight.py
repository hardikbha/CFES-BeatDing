#!/usr/bin/env python3
"""Bind one decoded-input preflight to a second arm with identical data/model recipe."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-preflight", required=True, type=Path)
    p.add_argument("--source-config", required=True, type=Path)
    p.add_argument("--target-config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Output exists; preserve previous preflight receipts")
    receipt = json.loads(a.source_preflight.read_text())
    source, target = json.loads(a.source_config.read_text()), json.loads(a.target_config.read_text())
    if receipt.get("status") != "PASS" or receipt.get("config_digest") != digest(source):
        raise SystemExit("Source preflight does not match its bound config")
    if digest(source["native_config"]) != digest(target["native_config"]):
        raise SystemExit("Native model/data recipe differs; decoded input validation cannot be reused")
    for key in ("p50_native_source_sha256", "transform_source_sha256", "fold_audit_sha256"):
        if source.get(key) != target.get(key):
            raise SystemExit(f"Source/fold binding differs: {key}")
    inputs = target["native_config"]["input_hashes"]
    if (receipt.get("train_manifest_sha256") != inputs.get("train") or
            receipt.get("development_manifest_sha256") != inputs.get("val") or
            receipt.get("weights_sha256") != inputs.get("weights")):
        raise SystemExit("Decoded input receipt hashes differ from target arm")
    bound = dict(receipt)
    bound.update({"status": "PASS", "arm": target["arm"],
                  "config_sha256": sha(a.target_config), "config_digest": digest(target),
                  "preflight_executor_source_sha256": receipt.get("adapter_source_sha256"),
                  "adapter_source_sha256": target.get("adapter_source_sha256"),
                  "shared_input_preflight": True,
                  "shared_input_preflight_sha256": sha(a.source_preflight),
                  "reuse_reason": "same exact fit/development bytes, ImageNet weights, native config, source and fold; only deterministic TRAIN-view policy differs"})
    bound.pop("config_sha256_original", None)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(bound, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "arm": target["arm"],
                      "config_digest": bound["config_digest"],
                      "shared_input_preflight_sha256": bound["shared_input_preflight_sha256"],
                      "test_manifest_read": False}, indent=2))


if __name__ == "__main__":
    main()
