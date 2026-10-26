"""Datasets in Ding et al.'s CSV schema (img_path,label,ismale,isasian,iswhite,isblack,intersec_label).

Both Lin et al.'s FF++/DFDC/Celeb-DF/DFD manifests and our FairForge-2 export use this schema, so one reader
serves both. Images are returned in [0,1] at `input_size`; CLIP normalisation happens inside the model so that
counterfactual interventions operate in pixel space.
"""
import os
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

# Ding/Lin intersec_label order.
SCHEMES = {
    "eight": ["male,asian", "male,white", "male,black", "male,others",
              "nonmale,asian", "nonmale,white", "nonmale,black", "nonmale,others"],
    "six": ["male,white", "male,black", "male,others", "nonmale,white", "nonmale,black", "nonmale,others"],
}

_FFPP_NAME = re.compile(r"^(\d+)_([^_]+)_(.+)_(\d+)\.png$")   # label_Method_TTT_SSS_frame.png (FF++ fakes)
_FRAME_SUFFIX = re.compile(r"^(.*)_(\d+)$")


def ffpp_method(path):
    m = _FFPP_NAME.match(os.path.basename(path))
    return m.group(2) if m else "unknown"


_FFPP_FAKE_IDS = re.compile(r"^\d+_[^_]+_(\d+)_(\d+)_\d+\.png$")
_FFPP_REAL_ID = re.compile(r"^\d+_youtube_(\d+)_\d+\.png$")


def ffpp_identities(path):
    """FF++ identities in a Lin-style frame: target and source ids for fakes, the video id for reals."""
    b = os.path.basename(path)
    m = _FFPP_FAKE_IDS.match(b)
    if m:
        return {m.group(1), m.group(2)}
    m = _FFPP_REAL_ID.match(b)
    return {m.group(1)} if m else set()


# Lin/Ding-protocol extracted video frames (FF++, DFDC, Celeb-DF, DFD): <label 0|1>_<video name>_<frame index>.
_LIN_VIDEO_FRAME = re.compile(r"^([01]_.+)_(\d+)$")


def cluster_key(path):
    """Leak-free clustering key (Codex review F8, 25 Oct; copied from agents/CESEL/cfes/data.py), dataset-aware:
      * Lin/Ding video-frame names (<label>_<video>_<frame>, FF++ / DFDC / Celeb-DF / DFD): the video = stem minus the
        trailing frame index (identical to the previous key on these names);
      * anything else (e.g. FairForge-2 generated images such as pixart_Asian_Female_105076007.jpg): the FULL image
        identity (normalised path) -- a trailing number is an image id there, not a frame index.
    read_split() overrides the key with "person:<person_id>" wherever the manifest's meta provides a person id."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _LIN_VIDEO_FRAME.match(stem)
    return m.group(1) if m else "img:" + os.path.normpath(path)


def read_split(spec):
    """spec: {csv: path or [paths], scheme: eight|six, meta: optional csv keyed by img_path, max_items: int}.
    Returns a DataFrame with img_path, label, group, method, cluster."""
    paths = spec["csv"] if isinstance(spec["csv"], list) else [spec["csv"]]
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    need = {"img_path", "label", "intersec_label"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{paths}: missing columns {missing}")
    df = df.rename(columns={"intersec_label": "group"})
    df["label"] = df["label"].astype(int)
    df["group"] = df["group"].astype(int)
    n_groups = len(SCHEMES[spec.get("scheme", "eight")])
    bad = ~df["group"].between(0, n_groups - 1)
    if bad.any():
        raise ValueError(f"{paths}: {int(bad.sum())} rows with group outside 0..{n_groups - 1}")
    if spec.get("meta"):
        meta = pd.read_csv(spec["meta"])
        keep = [c for c in ("img_path", "method", "person_id") if c in meta.columns]
        df = df.merge(meta[keep].drop_duplicates("img_path"), on="img_path", how="left")
    if "method" not in df.columns:
        df["method"] = df["img_path"].map(ffpp_method)
    df.loc[df["label"] == 0, "method"] = "real"
    df["method"] = df["method"].fillna("unknown")
    df["cluster"] = df["img_path"].map(cluster_key)
    if "person_id" in df.columns:
        has = df["person_id"].notna()
        df.loc[has, "cluster"] = "person:" + df.loc[has, "person_id"].astype(str)
    if spec.get("max_items"):
        df = pd.concat([g.sample(min(len(g), spec["max_items"] // 2), random_state=0)
                        for _, g in df.groupby("label")]).reset_index(drop=True)
    return df.reset_index(drop=True)


RESAMPLE = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR}


def load_image_u8(path, size, resample="bicubic"):
    """(size, size, 3) uint8 RGB, the exact pixels load_image() returns before the float conversion."""
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), RESAMPLE[resample])
    return np.asarray(img, dtype=np.uint8).copy()


def to_tensor(u8):
    """uint8 HWC -> float32 CHW in [0, 1] (the pipeline's image range; CLIP normalisation is inside the model)."""
    return torch.from_numpy(np.ascontiguousarray(u8)).permute(2, 0, 1).float() / 255.0


def load_image(path, size, resample="bicubic"):
    return to_tensor(load_image_u8(path, size, resample))


class ImageTable(Dataset):
    def __init__(self, df, size, resample="bicubic"):
        self.df, self.size, self.resample = df, size, resample
        self.paths = df["img_path"].tolist()
        self.labels = torch.tensor(df["label"].values)
        self.groups = torch.tensor(df["group"].values)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return dict(image=load_image(self.paths[i], self.size, self.resample), label=self.labels[i], group=self.groups[i],
                    index=i)


def _split_key(key, default_epoch):
    """Dataset key: an int index, or (epoch, index) from an EpochTagged sampler (works with persistent workers)."""
    if isinstance(key, (tuple, list)):
        return int(key[0]), int(key[1])
    return int(default_epoch), int(key)


class AugImageTable(ImageTable):
    """ImageTable + Beat-Ding augmentation A (cfes/bd_aug.py) on both classes; label and group are never changed.
    Parameters from bd_aug.item_rng(seed, epoch, stream, index). aug_op = index into bd_aug.OPS (-1 = none)."""

    def __init__(self, df, size, resample, aug_cfg, seed, stream):
        super().__init__(df, size, resample)
        from . import bd_aug
        self._aug, self.aug_cfg, self.seed, self.stream, self.epoch = bd_aug, bd_aug.config(aug_cfg), int(seed), stream, 0

    def __getitem__(self, key):
        e, i = _split_key(key, self.epoch)
        prm = self._aug.draw(self._aug.item_rng(self.seed, e, self.stream, i), self.aug_cfg)
        img = self._aug.apply(load_image_u8(self.paths[i], self.size, self.resample), prm)
        op = -1 if prm["op"] is None else self._aug.OPS.index(prm["op"])
        return dict(image=to_tensor(img), label=self.labels[i], group=self.groups[i], index=i, aug_op=op,
                    aug_flip=int(prm["flip"]))


class EpochTagged(Sampler):
    """Wraps an index sampler (or, batch=True, a batch sampler) and yields (epoch, index) keys, so a dataset in a
    persistent DataLoader worker knows the epoch of the item it loads. The wrapped order is unchanged. set_epoch()
    forwards to the wrapped sampler when it has one; other attributes (describe(), ...) are forwarded too."""

    def __init__(self, base, batch=False):
        self.base, self.batch, self.epoch = base, bool(batch), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        if hasattr(self.base, "set_epoch"):
            self.base.set_epoch(epoch)

    def __iter__(self):
        e = self.epoch
        for item in self.base:
            yield [(e, int(i)) for i in item] if self.batch else (e, int(item))

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        if name in ("base", "batch", "epoch"):
            raise AttributeError(name)
        return getattr(self.base, name)


class BalancedRealFakeSampler(Sampler):
    """Each batch: half real, half fake, drawn without replacement within an epoch (the smaller class is
    recycled). Epoch length = ceil(n_items / batch), i.e. one dataset's worth of images (the same exposure per
    epoch as FairReL's recipe), unless `steps` is given."""

    def __init__(self, labels, batch_size, steps=None, seed=0):
        labels = np.asarray(labels)
        self.real = np.flatnonzero(labels == 0)
        self.fake = np.flatnonzero(labels == 1)
        if len(self.real) == 0 or len(self.fake) == 0:
            raise ValueError("balanced sampler needs both classes")
        self.half = batch_size // 2
        self.steps = steps or int(np.ceil(len(labels) / (2 * self.half)))
        self.seed, self.epoch = seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _stream(self, pool, n, rng):
        out = []
        while len(out) < n:
            out.extend(rng.permutation(pool).tolist())
        return out[:n]

    def __iter__(self):
        rng = np.random.default_rng(self.seed + 1000 * self.epoch)
        r = self._stream(self.real, self.steps * self.half, rng)
        f = self._stream(self.fake, self.steps * self.half, rng)
        for s in range(self.steps):
            batch = r[s * self.half:(s + 1) * self.half] + f[s * self.half:(s + 1) * self.half]
            yield from rng.permutation(batch).tolist()

    def __len__(self):
        return self.steps * self.half * 2


class StratifiedGroupBatchSampler(Sampler):
    """Batch sampler: every batch holds k images from EACH (label, group) cell present in the table, so every
    subgroup is represented in every step (used by the alignment term, and given to every compared arm with
    detection labels so data exposure stays matched). Each cell is shuffled per epoch and cycled; deterministic
    given (seed, epoch)."""

    def __init__(self, labels, groups, k, steps, seed=0):
        labels, groups = np.asarray(labels), np.asarray(groups)
        self.cells = {}
        for lab in np.unique(labels):
            for g in np.unique(groups):
                idx = np.flatnonzero((labels == lab) & (groups == g))
                if len(idx):
                    self.cells[(int(lab), int(g))] = idx
        self.k, self.steps, self.seed, self.epoch = int(k), int(steps), seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed * 7 + 1000 * self.epoch + 12345)
        streams = {}
        for c, idx in self.cells.items():
            out = []
            while len(out) < self.steps * self.k:
                out.extend(rng.permutation(idx).tolist())
            streams[c] = out
        for s in range(self.steps):
            yield [i for c in sorted(streams) for i in streams[c][s * self.k:(s + 1) * self.k]]

    def __len__(self):
        return self.steps

    def describe(self):
        return {f"{lab}:{g}": int(len(v)) for (lab, g), v in sorted(self.cells.items())}


class PairTable(Dataset):
    """(fake, real source) pairs from scripts/build_pairs.py. Returns both images and, for pixel-aligned pairs,
    the per-patch fraction of face pixels changed by the manipulation (the evidence target)."""

    def __init__(self, csv, size, grid, face_mask, diff_tau, align_tau=None, max_items=None, seed=0,
                 resample="bicubic", group_of=None, aug_cfg=None, aug_seed=0):
        df = pd.read_csv(csv)
        if align_tau is not None:
            df["aligned"] = df["bg_diff"] <= align_tau
        if max_items:
            df = df.sample(min(len(df), max_items), random_state=seed)
        self.df = df.reset_index(drop=True)
        self.size, self.grid, self.tau, self.resample = size, grid, diff_tau, resample
        self.face = face_mask  # (size,size) bool, core face region
        # Beat-Ding: each pair image's OWN group label (img_path -> intersectional group of the training manifest), so
        # the pair images enter the group-conditional terms (ESA / group-AUC floor). Absent (None): items unchanged.
        self.groups = None
        if group_of is not None:
            gf, gr = self.df.fake_path.map(group_of), self.df.real_path.map(group_of)
            if gf.isna().any() or gr.isna().any():
                raise ValueError(f"{csv}: {int(gf.isna().sum() + gr.isna().sum())} pair images without a group label")
            self.groups = (torch.tensor(gf.astype(int).values), torch.tensor(gr.astype(int).values))
        # Beat-Ding augmentation A: ONE draw per pair, the same op + parameters + flip for the fake and its source real
        self.aug_cfg, self.aug_seed, self.epoch = None, int(aug_seed), 0
        if aug_cfg is not None:
            from . import bd_aug
            self._aug, self.aug_cfg = bd_aug, bd_aug.config(aug_cfg)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, key):
        e, i = _split_key(key, self.epoch)
        r = self.df.iloc[i]
        fu, su = load_image_u8(r.fake_path, self.size, self.resample), load_image_u8(r.real_path, self.size, self.resample)
        f, s = to_tensor(fu), to_tensor(su)
        aligned = bool(r.aligned)
        frac = torch.zeros(self.grid, self.grid)
        if aligned:                                    # evidence target from the UNPROCESSED pair (flipped below)
            changed = ((f - s).abs().mean(0) > self.tau) & self.face
            cell = self.size // self.grid
            frac = changed.float().reshape(self.grid, cell, self.grid, cell).mean((1, 3))
        out = dict(fake=f, real=s, aligned=aligned, frac=frac, index=i)
        if self.aug_cfg is not None:
            prm = self._aug.draw(self._aug.item_rng(self.aug_seed, e, "pairs", i), self.aug_cfg)
            out["fake"], out["real"] = to_tensor(self._aug.apply(fu, prm)), to_tensor(self._aug.apply(su, prm))
            if prm["flip"]:
                out["frac"] = torch.flip(frac, dims=[1])
            out["aug_op"] = -1 if prm["op"] is None else self._aug.OPS.index(prm["op"])
            out["aug_flip"] = int(prm["flip"])
        if self.groups is not None:
            out["fake_group"], out["real_group"] = self.groups[0][i], self.groups[1][i]
        return out


def cycle(loader):
    while True:
        yield from loader
