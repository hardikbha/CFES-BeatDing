"""Unique image-path lists for face_attrs.py (one per set). Keys are the paths the score npz files use.
FF++/DFDC/CDF/DFD: DINGREPRO_LIN/data CSVs (identical rows/order to every model's *_scores.npz, verified in
rescore.py). FF2: rows of data_ding_schema/fairforge2_full/test.csv that belong to the frozen PUBLIC TEST, plus the
public TRAIN reals (manifests/ding/realtrain.csv, for the ITA bin edges)."""
import os

import pandas as pd

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
D = f"{R}/agents/DINGREPRO_LIN/data"
F = f"{R}/release_ff2_public"
OUT = f"{R}/agents/DEMO_AXES/labels/raw/lists"
IMG_ROOT = "/home/sachin.chaudhary/ICLR_2027/"
SETS = {"ffpp_train": ["ff++_ding/train.csv"], "ffpp_val": ["ff++_ding/realval.csv", "ff++_ding/fakeval.csv"],
        "ffpp_test": ["ff++/test.csv"], "dfdc": ["dfdc/test.csv"], "celebdf": ["celebdf/test.csv"], "dfd": ["dfd/test.csv"]}


def rel(p):  # identical to release_ff2_public/build_public.rel
    q = p[len(IMG_ROOT):]
    for a, b in (("fairforge_scaleup/real/", "images/real/"), ("fairforge_scaleup/fakes/", "images/fake/"),
                 ("FairForge2_v1.0/real/", "images/real/ff2_v1/"), ("FairForge2_v1.0/fake/", "images/fake/ff2_v1/")):
        if q.startswith(a):
            return b + q[len(a):]
    raise ValueError(p)


def unrel(r):
    for b, a in (("images/real/ff2_v1/", "FairForge2_v1.0/real/"), ("images/fake/ff2_v1/", "FairForge2_v1.0/fake/"),
                 ("images/real/", "fairforge_scaleup/real/"), ("images/fake/", "fairforge_scaleup/fakes/")):
        if r.startswith(b):
            return IMG_ROOT + a + r[len(b):]
    raise ValueError(r)


def main():
    os.makedirs(OUT, exist_ok=True)
    for s, fs in SETS.items():
        p = pd.concat([pd.read_csv(f"{D}/{f}") for f in fs]).img_path.drop_duplicates()
        p.to_csv(f"{OUT}/{s}.txt", index=False, header=False)
        print(s, len(p))
    man = pd.read_csv(f"{F}/manifests/public_manifest.csv", low_memory=False, usecols=["rel_path", "split", "label"])
    T = pd.read_csv(f"{R}/data_ding_schema/fairforge2_full/test.csv", usecols=["img_path"])
    T["rel"] = T.img_path.map(rel)
    pub = set(man[man.split == "TEST"].rel_path)
    t = T[T.rel.isin(pub)].img_path
    t.to_csv(f"{OUT}/ff2_test.txt", index=False, header=False)
    print("ff2_test", len(t))
    tr = pd.read_csv(f"{F}/manifests/ding/realtrain.csv").img_path.map(unrel)
    assert all(os.path.exists(x) for x in tr.iloc[:50])
    tr.to_csv(f"{OUT}/ff2_trainreal.txt", index=False, header=False)
    print("ff2_trainreal", len(tr))


if __name__ == "__main__":
    main()
