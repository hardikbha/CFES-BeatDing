import pandas as pd, os, sys

EM = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan"
FR = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/FAIRREL_REPRO"
OUT = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/data"

def index_dir(d):
    return {f: os.path.join(d, f) for f in os.listdir(d)}

idx = {
    "ffpp_train": index_dir(f"{FR}/data/extracted/ffpp_train"),
    "ffpp_val":   index_dir(f"{EM}/work/extracted/ffpp_val"),
    "ffpp_test":  index_dir(f"{EM}/work/extracted/ffpp"),
    "dfd":        index_dir(f"{EM}/work/extracted/dfd"),
    "dfdc":       index_dir(f"{EM}/work/extracted/dfdc"),
    "celebdf":    index_dir(f"{EM}/work/extracted/celebdf"),
}
for k, v in idx.items():
    print(k, "indexed", len(v), "files")

def remap(src_csv, key, out_csv):
    df = pd.read_csv(src_csv)
    base = df['img_path'].apply(lambda p: os.path.basename(str(p).replace("\\", "/")))
    resolved = base.map(idx[key])
    n_missing = resolved.isna().sum()
    df2 = df.copy()
    df2['img_path'] = resolved
    before = len(df2)
    df2 = df2.dropna(subset=['img_path'])
    print(f"{src_csv} -> {out_csv}: raw={before} resolved={len(df2)} missing={n_missing}")
    df2.to_csv(out_csv, index=False)
    return before, len(df2), n_missing

results = {}
results['ffpp_train_real'] = remap(f"{EM}/work/ff++/realtrain.csv", "ffpp_train", f"{OUT}/ff++_ding/realtrain.csv")
results['ffpp_train_fake'] = remap(f"{EM}/work/ff++/faketrain.csv", "ffpp_train", f"{OUT}/ff++_ding/faketrain.csv")
results['ffpp_val_real']   = remap(f"{EM}/work/ff++/realval.csv",   "ffpp_val",   f"{OUT}/ff++_ding/realval.csv")
results['ffpp_val_fake']   = remap(f"{EM}/work/ff++/fakeval.csv",   "ffpp_val",   f"{OUT}/ff++_ding/fakeval.csv")
results['ffpp_test']       = remap(f"{EM}/work/ff++/test.csv",      "ffpp_test",  f"{OUT}/ff++/test.csv")
results['dfd_test']        = remap(f"{EM}/work/dfd/test.csv",       "dfd",        f"{OUT}/dfd/test.csv")
results['dfdc_test']       = remap(f"{EM}/work/dfdc/test.csv",      "dfdc",       f"{OUT}/dfdc/test.csv")
results['celebdf_test']    = remap(f"{EM}/work/celebdf/test.csv",   "celebdf",    f"{OUT}/celebdf/test.csv")

print()
print("ffpp train total:", results['ffpp_train_real'][1] + results['ffpp_train_fake'][1], "(Ding reports 76139)")
print("ffpp val total:", results['ffpp_val_real'][1] + results['ffpp_val_fake'][1], "(Ding reports 25386)")
print("ffpp test:", results['ffpp_test'][1], "(Ding reports 25401)")
print("dfd test:", results['dfd_test'][1], "(Ding reports 9385)")
print("dfdc test:", results['dfdc_test'][1], "(Ding reports 22857)")
print("celebdf test:", results['celebdf_test'][1], "(Ding reports 28458)")

# combined ff++ train.csv for SFD.py
real = pd.read_csv(f"{OUT}/ff++_ding/realtrain.csv")
fake = pd.read_csv(f"{OUT}/ff++_ding/faketrain.csv")
pd.concat([real, fake]).to_csv(f"{OUT}/ff++_ding/train.csv", index=False)
