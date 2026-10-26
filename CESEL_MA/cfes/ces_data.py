"""CES data additions on top of cfes/data.py (Ding schema). Each image also carries a manipulation CONDITION code:

    cond = 0 for reals; 1 + index of the fake's family in data.families; len(families) + 1 = "other" (unmapped)
    family = data.family_map[method]  (FF++ method from the file name, FF2 method from provenance.csv)

The condition defines the "comparable manipulation condition" of L_floor cells (group x class x condition).
With data.families empty every fake has condition 1 (class-only cells).
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data import ImageTable, StratifiedGroupBatchSampler, load_image


def condition_codes(df, cfg):
    d = cfg["data"]
    fams = list(d.get("families") or [])
    fmap = d.get("family_map") or {}
    code = {f: i + 1 for i, f in enumerate(fams)}
    other = len(fams) + 1 if fams else 1

    def one(row):
        if int(row.label) == 0:
            return 0
        if not fams:
            return 1
        return code.get(fmap.get(row.method, row.method), other)
    return np.array([one(r) for r in df[["label", "method"]].itertuples()], dtype=np.int64)


def condition_names(cfg):
    fams = list(cfg["data"].get("families") or [])
    return ["real"] + (fams + ["other"] if fams else ["fake"])


class CondImageTable(ImageTable):
    """ma (CESel-MA): optional (N,5) float matrix of demographic axes (cfes/ma_labels.COLS); returned as d["ma"]."""

    def __init__(self, df, size, cond, resample="bicubic", ma=None):
        super().__init__(df, size, resample)
        self.cond = torch.as_tensor(cond)
        self.ma = None if ma is None else torch.as_tensor(ma, dtype=torch.float32)

    def __getitem__(self, i):
        d = super().__getitem__(i)
        d["cond"] = self.cond[i]
        if self.ma is not None:
            d["ma"] = self.ma[i]
        return d


class CESPairTable(Dataset):
    """(fake, real source) pairs; both images must be TRAINING images (their group and condition are looked up in the
    training table, rows outside it are dropped and counted in self.report). exclude_families: drop pairs whose
    csv `family` is listed (FF2: synth / gan fakes have an assigned cell real in FAKE_MANIFEST.real_path, not a pixel
    source, so they use the EMA baseline instead)."""

    def __init__(self, csv, train_df, cond, size, max_items=None, seed=0, resample="bicubic", exclude_families=()):
        df = pd.read_csv(csv)
        info = {p: (int(g), int(c)) for p, g, c in zip(train_df.img_path, train_df.group, cond)}
        self.ma_info = None                                   # CESel-MA: per-image axes of the training table
        if "ma_group" in train_df.columns:
            from .ma_labels import ma_matrix
            self.ma_info = dict(zip(train_df.img_path, torch.as_tensor(ma_matrix(train_df))))
        keep = df.fake_path.isin(info) & df.real_path.isin(info)
        self.report = {"pairs_in": int(len(df)), "dropped_not_in_train": int((~keep).sum())}
        if exclude_families:
            if "family" not in df.columns:
                raise ValueError(f"pairs.exclude_families set but {csv} has no family column")
            ex = df.family.isin(list(exclude_families))
            self.report["dropped_excluded_family"] = int((ex & keep).sum())
            keep &= ~ex
        df = df[keep]
        if max_items:
            df = df.sample(min(len(df), int(max_items)), random_state=seed)
        self.df = df.reset_index(drop=True)
        self.report["pairs"] = int(len(self.df))
        self.info, self.size, self.resample = info, size, resample

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        gf, cf = self.info[r.fake_path]
        gr, cr = self.info[r.real_path]
        d = dict(fake=load_image(r.fake_path, self.size, self.resample),
                 real=load_image(r.real_path, self.size, self.resample),
                 fake_group=gf, real_group=gr, fake_cond=cf, real_cond=cr,
                 frac=torch.zeros(1), aligned=False, index=i)
        if self.ma_info is not None:
            d["fake_ma"], d["real_ma"] = self.ma_info[r.fake_path], self.ma_info[r.real_path]
        return d


class StratifiedCondBatchSampler(StratifiedGroupBatchSampler):
    """k images from each (label, group, condition) cell per step (by_cond), so the L_floor cells (group x class x
    condition) have >= k members in the stratified batch alone. Same RNG scheme as StratifiedGroupBatchSampler."""

    def __init__(self, labels, groups, conds, k, steps, seed=0):
        labels, groups, conds = np.asarray(labels), np.asarray(groups), np.asarray(conds)
        super().__init__(labels, groups, k, steps, seed)
        self.cells = {}
        for key in sorted(set(zip(labels.tolist(), groups.tolist(), conds.tolist()))):
            idx = np.flatnonzero((labels == key[0]) & (groups == key[1]) & (conds == key[2]))
            self.cells[tuple(int(v) for v in key)] = idx

    def describe(self):
        return {":".join(str(v) for v in key): int(len(v)) for key, v in sorted(self.cells.items())}
