"""Ding et al. (CVPR'26) Tables 2 and 3 with every published row, followed by our rows (mean +- sd over seeds).

  python scripts/ding_tables.py --table 2 --rows "CFES-CLIP (ours)|ViT-L/14|runs/clip_s7_p2:eval_snap_ep39,runs/clip_s17_p2:eval_snap_ep36"
  python scripts/ding_tables.py --table 3 --rows "Ori (ours, phase 1 best)|ResNet-50|runs/resnet_s7_p1:eval_snap_best,..." ...
Published rows are parsed from refs/Ding_CVPR2026.txt (pdftotext -layout of the CVF open-access PDF).
Our numbers come from evaluate.py summary.csv (Ding Eq. 8-10 at threshold 0.5, full test sets).
"""
import argparse
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLS = ["gender_F_FPR", "gender_F_DP", "gender_es_AUC", "race_F_FPR", "race_F_DP", "race_es_AUC",
        "intersection_F_FPR", "intersection_F_DP", "intersection_es_AUC", "AUC"]
DS = {"FF++": "ffpp_test", "DFDC": "dfdc", "Celeb-DF": "celebdf", "DFD": "dfd"}
NUM = r"(\d+\.\d+|\d+\.\d|\d+)"


ORDER = {2: (["DFDC", "Celeb-DF", "DFD"], 7), 3: (["FF++", "DFDC", "Celeb-DF", "DFD"], 5)}


def published(table):
    """Rows in the PDF come in fixed blocks per dataset (Table 2: 7 methods; Table 3: 5); the dataset name is printed
    on a middle row of its block, so rows are assigned by block order, and the name is checked against the block."""
    txt = open(os.path.join(ROOT, "refs/Ding_CVPR2026.txt")).read().splitlines()
    start = next(i for i, l in enumerate(txt) if l.strip().startswith(f"Table {table}. Eval"))
    raw = []
    for l in txt[start + 1:]:
        if l.strip().startswith("Table ") or l.strip().startswith("which is designed"):
            break
        nums = re.findall(r"\d+\.\d+", l)
        if len(nums) >= 10:
            raw.append(l)
    names, per = ORDER[table]
    if len(raw) != len(names) * per:
        raise ValueError(f"Table {table}: expected {len(names) * per} rows, parsed {len(raw)}")
    rows = []
    for i, l in enumerate(raw):
        ds = names[i // per]
        nums = re.findall(r"\d+\.\d+", l)
        head = l[:l.find(nums[-10])]
        bb = "ViT-L/14" if "ViT-L/14" in head else ("ResNet-50" if "ResNet-50" in head else "Xception")
        method = re.sub(r"\s+", " ", head.replace(bb, ""))
        for n in DS:
            method = re.sub(rf"(^|\s){re.escape(n)}(\s|$)", " ", method)
        rows.append((ds, method.strip(), bb, [float(x) for x in nums[-10:]]))
    for blk in range(len(names)):                       # the block's own label must appear inside it
        block = "\n".join(raw[blk * per:(blk + 1) * per])
        if not re.search(rf"(^|\s){re.escape(names[blk])}(\s|$)", block):
            raise ValueError(f"Table {table}: block {blk} lacks label {names[blk]}")
    return rows


def ours(spec):
    label, bb, runs = spec.split("|")
    per = []
    for r in runs.split(","):
        run, ev = r.split(":")
        per.append(pd.read_csv(os.path.join(ROOT, run, ev, "summary.csv")).set_index("test_set"))
    return label, bb, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=int, choices=[2, 3], required=True)
    ap.add_argument("--rows", action="append", default=[])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pub = published(a.table)
    datasets = list(dict.fromkeys(d for d, *_ in pub))
    mine = [ours(s) for s in a.rows]
    hdr = ("| Dataset | Method | Backbone | g F_FPR↓ | g F_DP↓ | g es-AUC↑ | r F_FPR↓ | r F_DP↓ | r es-AUC↑ | "
           "i F_FPR↓ | i F_DP↓ | i es-AUC↑ | AUC↑ |")
    out = [hdr, "|" + "---|" * 13]
    for d in datasets:
        for _, m, bb, v in [p for p in pub if p[0] == d]:
            m = {"Ours": "Ding et al. CVPR'26 (published)", "Ori": "Ori (published, Ding)"}.get(m, m)
            out.append(f"| {d} | {m} | {bb} | " + " | ".join(f"{x:.2f}" for x in v) + " |")
        for label, bb, per in mine:
            vals = np.array([[p.loc[DS[d], c] for c in COLS] for p in per])
            if len(per) > 1:
                cells = [f"{m:.2f} ± {s:.2f}" for m, s in zip(vals.mean(0), vals.std(0, ddof=1))]
            else:
                cells = [f"{x:.2f}" for x in vals[0]]
            out.append(f"| {d} | **{label}** (n={len(per)}) | {bb} | " + " | ".join(cells) + " |")
    s = "\n".join(out)
    print(s)
    if a.out:
        open(a.out, "w").write(s + "\n")


if __name__ == "__main__":
    main()
