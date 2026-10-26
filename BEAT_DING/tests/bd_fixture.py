"""Synthetic Lin/Ding-schema dataset for the Beat-Ding CPU checks (tests/bd_checks.py): small RGB PNGs with Lin-style frame
names, every (label, intersectional group) cell present in every split, a (fake, own source real) pair table, and the
--set override files that point a config at it (CLIP stub backbone; real ResNet-50 architecture at 64 px)."""
import os
import zlib

import numpy as np
import pandas as pd
from PIL import Image

SPLITS = {"train": ("eight", 3), "val": ("eight", 3), "ffpp_test": ("eight", 3), "dfdc": ("eight", 2),
          "celebdf": ("six", 3), "dfd": ("six", 2)}
EIGHT = [("male", "asian"), ("male", "white"), ("male", "black"), ("male", "others"),
         ("nonmale", "asian"), ("nonmale", "white"), ("nonmale", "black"), ("nonmale", "others")]
SIX = [g for g in EIGHT if g[1] != "asian"]


def _img(seed, size):
    r = np.random.default_rng(seed)
    base = Image.fromarray(r.integers(30, 220, size=(10, 10, 3)).astype(np.uint8)).resize((size, size), Image.BICUBIC)
    a = np.asarray(base).astype(np.float32)
    yy, xx = np.mgrid[:size, :size]
    blob = np.exp(-(((xx - size / 2) / (size * 0.25)) ** 2 + ((yy - size * 0.48) / (size * 0.32)) ** 2))[..., None]
    a = a * (1 - 0.6 * blob) + 0.6 * blob * r.integers(90, 255, 3) + r.normal(0, 6, a.shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def make(root, size=96):
    """Writes <root>/{split}.csv, <root>/pairs.csv, images under <root>/img/<split>/. Returns {name: path}."""
    out, vid = {}, 0
    for split, (scheme, k) in SPLITS.items():
        d = os.path.join(root, "img", split)
        os.makedirs(d, exist_ok=True)
        names = EIGHT if scheme == "eight" else SIX
        rows = []
        for gi, (sex, race) in enumerate(names):
            for lab in (0, 1):
                vid += 1
                for j in range(k):
                    n = f"0_youtube_{vid:03d}_{j}.png" if lab == 0 else f"1_Deepfakes_{vid:03d}_{vid + 500:03d}_{j}.png"
                    p = os.path.join(d, n)
                    if not os.path.exists(p):
                        _img(zlib.crc32(f"{split}|{gi}|{lab}|{j}".encode()), size).save(p)
                    rows.append({"img_path": p, "label": lab, "ismale": 1 if sex == "male" else -1,
                                 "isasian": 1 if race == "asian" else -1, "iswhite": 1 if race == "white" else -1,
                                 "isblack": 1 if race == "black" else -1, "intersec_label": gi})
        out[split] = os.path.join(root, f"{split}.csv")
        pd.DataFrame(rows).to_csv(out[split], index=False)
    tr = pd.read_csv(out["train"])
    fakes, reals = tr[tr.label == 1].reset_index(drop=True), tr[tr.label == 0].reset_index(drop=True)
    pr = pd.DataFrame({"fake_path": fakes.img_path, "real_path": reals.img_path[np.arange(len(fakes)) % len(reals)].values,
                       "method": "Deepfakes", "bg_diff": [0.01, 0.05] * (len(fakes) // 2) + [0.01] * (len(fakes) % 2),
                       "face_diff": 0.03, "n_candidates": 1})
    pr["aligned"] = pr.bg_diff <= 0.015
    out["pairs"] = os.path.join(root, "pairs.csv")
    pr.to_csv(out["pairs"], index=False)
    return out


def test_sets(fx):
    return ("eval.test_sets={" + ", ".join(f"{k}: {{csv: {fx[k]}, scheme: {SPLITS[k][0]}}}"
                                            for k in ("ffpp_test", "dfdc", "celebdf", "dfd")) + "}")


def clip_overrides(fx, epochs=3):
    """CLIP family on the stub backbone (same classes / layout as the FairReL wrapper, 24 blocks), 64 px, 1 step per epoch."""
    return ["model.backbone=stub", "model.stub={image: 64, patch: 8, dim: 32, layers: 24, heads: 2, seed: 0}",
            "data.input_size=64", "data.batch_size=16", f"data.train.csv={fx['train']}", f"data.val.csv={fx['val']}",
            f"pairs.csv={fx['pairs']}", "pairs.per_step=4", test_sets(fx), "eval.batch_size=32",
            "debug.max_steps_per_epoch=1", f"optim.epochs={epochs}", "optim.warmup_steps=1", "log_every=1"]


def resnet_overrides(fx, epochs=3):
    """ResNet-50 family: the real Ding 'adjust_channel' architecture (ImageNet weights) at 64 px (layer4 grid 2)."""
    return ["data.input_size=64", "data.eval_size=64", "model.resnet.grid=2", "data.batch_size=8",
            f"data.train.csv={fx['train']}", f"data.val.csv={fx['val']}", f"pairs.csv={fx['pairs']}", "pairs.per_step=2",
            test_sets(fx), "eval.batch_size=32", "debug.max_steps_per_epoch=1", f"optim.epochs={epochs}", "log_every=1"]


def write_setfile(path, items):
    with open(path, "w") as f:
        f.write("\n".join(items) + "\n")
    return path


def xception_overrides(fx, epochs=3, recal_batches=2):
    """Beat-Ding Xception family (backend xception_bd): Ding's real Xception 'adjust_channel' architecture (Ding's phase-1
    checkpoint via model.xception.init_from, set by the caller) at 64 px (feature grid 2); step-0 BN re-estimation on
    `recal_batches` batches of 4 real + 4 fake."""
    return ["data.input_size=64", "data.eval_size=64", "model.xception.grid=2", "data.batch_size=8",
            f"data.train.csv={fx['train']}", f"data.val.csv={fx['val']}", f"pairs.csv={fx['pairs']}", "pairs.per_step=2",
            test_sets(fx), "eval.batch_size=32", "debug.max_steps_per_epoch=1", f"optim.epochs={epochs}", "log_every=1",
            f"model.xception.bn_recal.batches={recal_batches}", "model.xception.bn_recal.per_class=4"]
