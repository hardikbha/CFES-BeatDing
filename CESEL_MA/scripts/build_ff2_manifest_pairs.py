"""FairForge-2 (fake, real source) pairs from the generation provenance: FAKE_MANIFEST.csv real_path, i.e. the real
image the fake was generated from (swap target / edit input / restoration input / identity reference), NOT a
nearest-background search (CFES scripts/build_pairs.py --pairing ff2 matched by person and background instead).

  python scripts/build_ff2_manifest_pairs.py \
     --manifest ../T_scaleup/generation/plan/FAKE_MANIFEST.csv \
     --train ../../data_ding_schema/fairforge2_full/train.csv --out data/pairs_ff2full_manifest_train.csv

A pair is kept only when BOTH the fake and its real_path are rows of the FF2 TRAIN table (no SELECT/TEST image can
enter training). Output columns: fake_path, real_path, method, family (CESPairTable needs fake_path and real_path).
"""
import argparse
import json

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    m = pd.read_csv(a.manifest, usecols=["fake_path", "real_path", "method", "family", "split"])
    tr = pd.read_csv(a.train, usecols=["img_path", "label"])
    fakes, reals = set(tr.img_path[tr.label == 1]), set(tr.img_path[tr.label == 0])
    rep = {"manifest_rows": int(len(m)), "train_fakes": len(fakes), "train_reals": len(reals)}
    m = m[m.fake_path.isin(fakes)]
    rep["manifest_fakes_in_train"] = int(len(m))
    m = m[m.real_path.notna()]
    rep["with_real_path"] = int(len(m))
    keep = m.real_path.isin(reals)
    rep["real_path_not_in_train_dropped"] = int((~keep).sum())
    m = m[keep].drop_duplicates("fake_path")
    rep["pairs"] = int(len(m))
    rep["train_fakes_without_pair"] = len(fakes) - int(len(m))
    rep["pairs_per_family"] = {k: int(v) for k, v in m.family.value_counts().items()}
    m[["fake_path", "real_path", "method", "family"]].to_csv(a.out, index=False)
    json.dump(rep, open(a.out + ".report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
