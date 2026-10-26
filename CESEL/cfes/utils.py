import hashlib
import json
import os
import random
import time

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def code_hash(root):
    """sha256 over every .py/.yaml under root, recorded in each run so results map to exact code."""
    h = hashlib.sha256()
    for d, _, files in sorted(os.walk(root)):
        if "runs" in d.split(os.sep) or "__pycache__" in d:
            continue
        for f in sorted(files):
            if f.endswith((".py", ".yaml")):
                with open(os.path.join(d, f), "rb") as fh:
                    h.update(f.encode() + fh.read())
    return h.hexdigest()[:16]


class JsonlLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, **rec):
        rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")


def save_checkpoint(path, model, optimizer, scheduler, scaler, state, full=True):
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict() if full else model.trainable_state_dict(), "full": full,
                "optimizer": optimizer.state_dict() if optimizer else None,
                "scheduler": scheduler.state_dict() if scheduler else None,
                "scaler": scaler.state_dict() if scaler else None,
                "state": state,
                "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, restore_rng=False):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ck["model"], strict=ck.get("full", True))
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected[:5]}")
    if optimizer and ck.get("optimizer"):
        optimizer.load_state_dict(ck["optimizer"])
    if scheduler and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    if scaler and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    if restore_rng and ck.get("rng"):
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        if ck["rng"]["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
    return ck["state"]
