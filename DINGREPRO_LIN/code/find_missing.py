import pandas as pd, os

EM = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan"
FR = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/FAIRREL_REPRO"

extracted_dirs = {
    "ffpp_train": f"{FR}/data/extracted/ffpp_train",
    "ffpp_val":   f"{EM}/work/extracted/ffpp_val",
    "ffpp_test":  f"{EM}/work/extracted/ffpp",
    "dfd":        f"{EM}/work/extracted/dfd",
    "dfdc":       f"{EM}/work/extracted/dfdc",
    "celebdf":    f"{EM}/work/extracted/celebdf",
}
have = {k: set(os.listdir(v)) for k, v in extracted_dirs.items()}

def missing_for(csv, key):
    df = pd.read_csv(csv)
    base = df['img_path'].apply(lambda p: os.path.basename(str(p).replace("\\","/")))
    miss = sorted(set(base) - have[key])
    return miss

jobs = [
    (f"{EM}/work/ff++/realtrain.csv", "ffpp_train"),
    (f"{EM}/work/ff++/faketrain.csv", "ffpp_train"),
    (f"{EM}/work/ff++/realval.csv",   "ffpp_val"),
    (f"{EM}/work/ff++/fakeval.csv",   "ffpp_val"),
    (f"{EM}/work/ff++/test.csv",      "ffpp_test"),
    (f"{EM}/work/dfd/test.csv",       "dfd"),
    (f"{EM}/work/dfdc/test.csv",      "dfdc"),
    (f"{EM}/work/celebdf/test.csv",   "celebdf"),
]
allmiss = {}
for csv, key in jobs:
    m = missing_for(csv, key)
    allmiss.setdefault(key, set()).update(m)
    print(csv, key, "missing", len(m))

for k, v in allmiss.items():
    outp = f"/tmp/missing_{k}.txt"
    with open(outp, "w") as f:
        f.write("\n".join(sorted(v)))
    print(k, "TOTAL UNIQUE MISSING", len(v), "->", outp)
