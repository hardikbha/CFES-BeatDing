"""MFER: Matched Forensic Evidence Response.

A matched set i holds, for each demographic group g present, one manipulated image and its comparable real
control: (fake[i,g], real[i,g]). With s(.) the detector's image score (logit, fake > 0):

    delta[i,g] = s(fake[i,g]) - s(real[i,g])

Losses (each has its own weight in cfg.loss; weight 0 switches it off):
    mfer_margin : mean over valid (i,g) of softplus(m - delta[i,g])          fake above its own matched real
    mfer_cons   : mean over sets with >= 2 valid groups of
                  (1/|G_i|) sum_g (delta[i,g] - mean_g delta[i,g])^2        equal evidence response across groups
    mfer_det    : balanced BCE on every valid matched image (fakes -> 1, reals -> 0)
Averaging: mfer_margin and mfer_det average over valid images/pairs, so a set with more groups contributes more;
mfer_cons averages over sets. Only VALID slots are forwarded through the model (padding never enters the encoder,
so it cannot affect batch statistics).
Only the training loss uses matched sets. Inference takes one image and returns s(x); it needs no demographic
label and no matched image.

Matched sets come from a CSV (set_id, group, fake_path, real_path) that must be produced by a validated
procedure: the SAME manipulation shown on faces of different groups, each with a comparable real control.
read_matched_sets() never creates matches; it only validates and drops incomplete ones.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .data import load_image
from .losses import detection_loss as _bce

TERMS = ("mfer_margin", "mfer_cons", "mfer_det")

REQUIRED = ("set_id", "group", "fake_path", "real_path")


def read_matched_sets(df, min_groups=2, max_groups=None):
    """Validate a matched-set table. Drops elements without a fake or real path, keeps the first element when a
    group appears twice in a set, and drops sets with fewer than `min_groups` distinct groups. Sets with more than
    `max_groups` groups are counted in the report (MatchedSetTable then applies its oversize policy).
    Returns (clean_df, report)."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"matched-set table lacks columns {missing}")
    rep = {"rows_in": int(len(df)), "sets_in": int(df.set_id.nunique())}
    d = df.dropna(subset=["fake_path", "real_path", "group"]).copy()
    rep["dropped_missing_path_or_group"] = int(len(df) - len(d))
    d["group"] = d["group"].astype(int)
    dup = d.duplicated(["set_id", "group"], keep="first")
    rep["dropped_duplicate_group_in_set"] = int(dup.sum())
    d = d[~dup]
    ng = d.groupby("set_id").group.nunique()
    keep = ng[ng >= min_groups].index
    rep["dropped_sets_below_min_groups"] = int((ng < min_groups).sum())
    d = d[d.set_id.isin(keep)].reset_index(drop=True)
    rep.update(sets=int(d.set_id.nunique()), elements=int(len(d)),
               elements_per_group={int(k): int(v) for k, v in d.group.value_counts().sort_index().items()},
               groups_per_set={int(k): int(v) for k, v in d.groupby("set_id").group.nunique().value_counts().items()})
    if max_groups is not None:
        rep["sets_over_max_groups"] = int((d.groupby("set_id").size() > max_groups).sum())
    return d, rep


class MatchedSetTable(Dataset):
    """One item = one matched set, padded to `max_groups` elements with a validity mask.

    oversize (a set with more than max_groups elements):
      "error"  (default) refuse at construction, listing the sets
      "random" each time the set is drawn, use a uniformly random subset of max_groups elements (torch RNG, so it
               follows the DataLoader worker seeding)"""

    def __init__(self, df, size, max_groups, resample="bicubic", oversize="error"):
        self.sets = [g for _, g in df.groupby("set_id", sort=True)]
        self.size, self.K, self.resample = size, max_groups, resample
        if oversize not in ("error", "random"):
            raise ValueError(f"mfer.oversize must be error or random, got {oversize}")
        self.oversize = oversize
        big = [str(g.set_id.iloc[0]) for g in self.sets if len(g) > max_groups]
        if big and oversize == "error":
            raise ValueError(f"{len(big)} matched sets have more than max_groups={max_groups} elements "
                             f"(e.g. {big[:5]}); raise mfer.max_groups or set mfer.oversize: random")

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, i):
        g = self.sets[i]
        if len(g) > self.K:
            g = g.iloc[torch.randperm(len(g))[:self.K].sort().values.numpy()]
        fake = torch.zeros(self.K, 3, self.size, self.size)
        real = torch.zeros_like(fake)
        group = torch.full((self.K,), -1, dtype=torch.long)
        mask = torch.zeros(self.K, dtype=torch.bool)
        for k, r in enumerate(g.itertuples()):
            fake[k] = load_image(r.fake_path, self.size, self.resample)
            real[k] = load_image(r.real_path, self.size, self.resample)
            group[k], mask[k] = int(r.group), True
        return dict(fake=fake, real=real, group=group, mask=mask)


def deltas(score_fake, score_real):
    """(S,K) fake scores minus (S,K) matched real scores."""
    return score_fake - score_real


def margin_loss(delta, mask, margin):
    v = delta[mask]
    if v.numel() == 0:
        return delta.sum() * 0.0, {"mfer_margin_items": 0}
    return F.softplus(margin - v).mean(), {"mfer_margin_items": int(v.numel()), "mfer_delta_mean": float(v.detach().mean()),
                                            "mfer_delta_absmean": float(v.detach().abs().mean()),
                                            "mfer_margin_violation_frac": float((v.detach() < margin).float().mean())}


def consistency_loss(delta, mask, group):
    """Within-set variance of delta over valid elements; sets with < 2 valid distinct groups are excluded (they
    contribute nothing, never NaN)."""
    m = mask.float()
    n = m.sum(1)
    distinct = torch.tensor([len(set(group[i][mask[i]].tolist())) for i in range(len(group))], device=delta.device)
    use = (n >= 2) & (distinct >= 2)
    if not use.any():
        return delta.sum() * 0.0, {"mfer_cons_sets": 0}
    mean = (delta * m).sum(1) / n.clamp(min=1)
    var = (((delta - mean[:, None]) ** 2) * m).sum(1) / n.clamp(min=1)
    loss = var[use].mean()
    return loss, {"mfer_cons_sets": int(use.sum()), "mfer_delta_spread": float(var[use].detach().sqrt().mean()),
                  "mfer_response_var": float(var[use].detach().mean())}


def flatten_for_forward(batch):
    """(S,K,3,H,W) fakes/reals -> one (2V,3,H,W) tensor holding only the V valid slots: valid fakes first, then
    the valid reals in the same order. Padded slots are never forwarded."""
    m = batch["mask"]
    return torch.cat([batch["fake"][m], batch["real"][m]])


def unflatten_scores(scores, mask):
    """(2V,) scores from flatten_for_forward -> (S,K) fake and real score tensors; padded slots hold 0 and are
    excluded by the mask in every loss."""
    V = int(mask.sum())
    sf = scores.new_zeros(mask.shape).masked_scatter(mask, scores[:V])
    sr = scores.new_zeros(mask.shape).masked_scatter(mask, scores[V:2 * V])
    return sf, sr


def detection_loss(scores, mode="balanced"):
    """BCE on the forwarded valid matched images: first half fakes (label 1), second half reals (label 0)."""
    V = scores.shape[0] // 2
    y = torch.cat([torch.ones(V, dtype=torch.long), torch.zeros(V, dtype=torch.long)]).to(scores.device)
    loss, _ = _bce(scores, y, mode)
    return loss, {"mfer_det_items": int(2 * V)}


def describe(df):
    return {"sets": int(df.set_id.nunique()), "elements": int(len(df)),
            "per_group": {int(k): int(v) for k, v in df.group.value_counts().sort_index().items()},
            "mean_groups_per_set": float(np.mean(df.groupby("set_id").group.nunique())) if len(df) else 0.0}
