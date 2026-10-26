"""CESel-MA demographic axes (decisions/2026-10-26-final-model-spec.md section 2): per-image labels used in TRAINING
(L_floor-MA) and in VALIDATION checkpoint selection only, never at test.

Axis       values                                  column (after attach())       missing
gender     0 = male, 1 = non-male                  ma_gender (int)               -1
age bin    0: <30, 1: 30-44, 2: 45-59, 3: 60+      ma_age_bin (int)              -1   (from a numeric age, ma.age_edges)
skin tone  continuous ITA (degrees)                ma_ita (float)                NaN
           4 bins by the TRAIN quartile edges      ma_ita_bin (int)              -1   (edges declared by DEMO_AXES)
gender x race  Ding/Lin intersec order 0..7         ma_group (int)                falls back to the Lin/Xu `group`

Label source (cfg ma.labels): one CSV per split keyed by img_path; the column names are mapped by ma.labels.columns
(the contract with agents/DEMO_AXES is in README.md "Label contract"). A row of the split without a label row keeps the
missing codes: such an image takes part in detection and in every other loss, but not in the floor / gap of that axis.
Gender may come as a 0/1 non-male column (columns.gender) or as DEMO_AXES' cl_male (columns.male, 1 = male); gender x
race may come as a group column or be derived from (gender, columns.race name); the ITA bin may come as a column
(columns.ita_bin, DEMO_AXES ita4) or be cut from ITA at the edges in ma.labels.ita_edges_json (dotted key
ma.labels.ita_edges_key; 3 increasing values, TRAIN quartiles) or ma.ita_edges; the kernel anchors of the skin-tone floor span the TRAIN ITA range (ma.ita_range, else min / max of the
TRAIN labels).
"""
import json
import os

import numpy as np
import pandas as pd

AXES = ("gender", "age_bin", "ita_bin", "group")
RACE_CODE = {"asian": 0, "white": 1, "black": 2, "others": 3}        # Ding/Lin intersec order within a gender
COLS = ("ma_gender", "ma_age_bin", "ma_ita", "ma_ita_bin", "ma_group")


def age_bin(age, edges=(30, 45, 60)):
    """numeric age -> 0..len(edges) (<30, 30-44, 45-59, 60+); NaN -> -1."""
    a = np.asarray(age, dtype=np.float64)
    out = np.searchsorted(np.asarray(edges, dtype=np.float64), a, side="right").astype(np.int64)
    out[~np.isfinite(a)] = -1
    return out


def ita_bin(ita, edges):
    t = np.asarray(ita, dtype=np.float64)
    out = np.searchsorted(np.asarray(edges, dtype=np.float64), t, side="right").astype(np.int64)
    out[~np.isfinite(t)] = -1
    return out


def ita_edges(macfg):
    """3 edges from ma.ita_edges, else ma.labels.ita_edges_json at the dotted key ma.labels.ita_edges_key (default
    "edges"; DEMO_AXES: "ffpp_protocol.ita_gw.edges" / "ff2.ita_gw.edges")."""
    if macfg.get("ita_edges") is not None:
        e = [float(x) for x in macfg["ita_edges"]]
    else:
        lab = macfg.get("labels") or {}
        p = lab.get("ita_edges_json")
        if not p or not os.path.exists(p):
            raise FileNotFoundError(f"ma: ITA bin edges not found (ma.ita_edges or ma.labels.ita_edges_json = {p})")
        node = json.load(open(p))
        for k in str(lab.get("ita_edges_key") or "edges").split("."):
            node = node[k]
        e = [float(x) for x in node]
    if len(e) != 3 or not all(a < b for a, b in zip(e, e[1:])):
        raise ValueError(f"ma: ITA edges must be 3 increasing values (TRAIN quartiles), got {e}")
    return e


def _read(path):
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"ma: label file not found: {path}")
    return pd.read_csv(path, low_memory=False)


def attach(df, cfg, split):
    """Add the COLS columns to a read_split() frame (row order unchanged). split: train | val | <test set name>.
    Returns (df, report)."""
    m = cfg["ma"]
    lab = m["labels"]
    col = dict({"img_path": "img_path", "gender": "gender", "male": None, "race": None, "age": "age", "age_bin": None,
                "ita": "ita", "ita_bin": None, "group": "group_gr"}, **(lab.get("columns") or {}))
    L = _read(lab[split] if split in lab else (lab.get("test") or {}).get(split))
    L = L.drop_duplicates(col["img_path"]).set_index(col["img_path"])
    key = df["img_path"]
    hit = key.isin(L.index)
    out = df.copy()

    def get(c, fill):
        if not c or c not in L.columns:
            return pd.Series(fill, index=df.index)
        return key.map(L[c]).where(hit, fill)
    if col.get("male"):                      # DEMO_AXES convention: cl_male 1 = male -> gender 0 = male, 1 = non-male
        g = 1 - pd.to_numeric(get(col["male"], np.nan), errors="coerce")
    else:
        g = pd.to_numeric(get(col["gender"], np.nan), errors="coerce")
    out["ma_gender"] = np.where(np.isfinite(g), g, -1).astype(np.int64)
    if col.get("age_bin"):
        ab = pd.to_numeric(get(col["age_bin"], np.nan), errors="coerce")
        out["ma_age_bin"] = np.where(np.isfinite(ab), ab, -1).astype(np.int64)
    else:
        out["ma_age_bin"] = age_bin(pd.to_numeric(get(col["age"], np.nan), errors="coerce").values,
                                    m.get("age_edges", (30, 45, 60)))
    t = pd.to_numeric(get(col["ita"], np.nan), errors="coerce").values.astype(np.float64)
    out["ma_ita"] = t
    if col.get("ita_bin"):                   # declared bins shipped with the labels (DEMO_AXES ita4)
        tb = pd.to_numeric(get(col["ita_bin"], np.nan), errors="coerce")
        out["ma_ita_bin"] = np.where(np.isfinite(tb), tb, -1).astype(np.int64)
    else:
        out["ma_ita_bin"] = ita_bin(t, ita_edges(m))
    if col.get("race"):                      # gender x race in the Ding/Lin intersec order from (gender, race name)
        rc = get(col["race"], None).map(RACE_CODE)
        gr = pd.Series(np.where((out["ma_gender"].values >= 0) & rc.notna().values,
                                4 * out["ma_gender"].values + rc.fillna(0).values, np.nan), index=df.index)
    else:
        gr = pd.to_numeric(get(col["group"], np.nan), errors="coerce")
    fb = df["group"].values.astype(np.int64)
    out["ma_group"] = np.where(np.isfinite(gr), gr, fb).astype(np.int64)
    bad = (out["ma_gender"] > 1) | (out["ma_gender"] < -1) | (out["ma_age_bin"] > 3) | (out["ma_group"] < 0) | \
          (out["ma_group"] > 7) | (out["ma_ita_bin"] > 3)
    if bad.any():
        raise ValueError(f"ma: {int(bad.sum())} label rows out of range in split {split}")
    rep = {"split": split, "rows": int(len(df)), "labelled": int(hit.sum()),
           "gender": int((out.ma_gender >= 0).sum()), "age_bin": int((out.ma_age_bin >= 0).sum()),
           "ita": int(np.isfinite(out.ma_ita).sum()),
           "group_from_labels": int(np.isfinite(gr).sum())}
    return out, rep


def ita_range(cfg, train_df):
    r = cfg["ma"].get("ita_range")
    if r is not None:
        return float(r[0]), float(r[1])
    t = train_df["ma_ita"].values
    t = t[np.isfinite(t)]
    if t.size < 2:
        raise ValueError("ma: fewer than 2 finite TRAIN ITA values; cannot define the kernel anchors")
    return float(t.min()), float(t.max())


def ma_matrix(df):
    """(N,5) float32 matrix in COLS order (gender, age_bin, ita, ita_bin, group), for the datasets / batches."""
    return np.stack([df[c].values.astype(np.float32) for c in COLS], 1)
