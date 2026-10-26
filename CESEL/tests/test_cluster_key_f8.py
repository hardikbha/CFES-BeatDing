"""Codex review F8 check (CPU, reads manifests only): the dataset-aware cluster_key leaves every FF++-protocol key
(train / val / FF++ / DFDC / Celeb-DF / DFD) and every leak-free-video drop count UNCHANGED, and FF2-public no longer
drops the 1,274 PixArt test rows (image ids, not frame indices).
  CUDA_VISIBLE_DEVICES="" ~/.conda/envs/pulid/bin/python tests/test_cluster_key_f8.py"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cfes import data as D  # noqa: E402
from cfes.config import load_config  # noqa: E402

_OLD = re.compile(r"^(.*)_(\d+)$")


def old_key(p):
    st = os.path.splitext(os.path.basename(p))[0]
    m = _OLD.match(st)
    return m.group(1) if m else st


def old_clusters(df):
    o = df["img_path"].map(old_key)
    if "person_id" in df:
        h = df["person_id"].notna()
        o[h] = "person:" + df.loc[h, "person_id"].astype(str)
    return o


def run(cfgp):
    cfg = load_config(os.path.join(ROOT, cfgp))
    tr, va = D.read_split(cfg["data"]["train"]), D.read_split(cfg["data"]["val"])
    out = {"train_changed": int((old_clusters(tr) != tr["cluster"]).sum()),
           "val_changed": int((old_clusters(va) != va["cluster"]).sum())}
    tk, otk = set(tr["cluster"]), set(old_clusters(tr))
    for name, spec in cfg["eval"]["test_sets"].items():
        df = D.read_split(spec)
        o = old_clusters(df)
        out[name] = dict(n=len(df), old=int(o.isin(otk).sum()), new=int(df["cluster"].isin(tk).sum()),
                         changed=int((o != df["cluster"]).sum()),
                         old_methods=df[o.isin(otk)]["method"].value_counts().to_dict(),
                         new_methods=df[df["cluster"].isin(tk)]["method"].value_counts().to_dict())
        print(f"  {name}: n={len(df)} leak-free-video drops old={out[name]['old']} new={out[name]['new']} "
              f"keys changed={out[name]['changed']}", flush=True)
    return out


# unit cases
assert D.cluster_key("/x/ffpp_train/0_youtube_025_111.png") == "0_youtube_025"
assert D.cluster_key("/x/1_Deepfakes_289_228_91.png") == "1_Deepfakes_289_228"
assert D.cluster_key("/x/0_22__kitchen_still_241.png") == "0_22__kitchen_still"
k1 = D.cluster_key("/d/fake/pixart/Asian_Female/pixart_Asian_Female_105076007.jpg")
k2 = D.cluster_key("/d/fake/pixart/Asian_Female/pixart_Asian_Female_105883745.jpg")
assert k1 != k2 and k1.startswith("img:"), (k1, k2)
print("PASS unit cases")
print("FF++ (configs/ces_ffpp.yaml)")
f = run("configs/ces_ffpp.yaml")
assert f["train_changed"] == 0 and f["val_changed"] == 0
for k in ("ffpp_test", "dfdc", "celebdf", "dfd"):
    assert f[k]["changed"] == 0 and f[k]["old"] == f[k]["new"], (k, f[k])
print("PASS FF++ protocol: every cluster key and every leak-free-video drop count unchanged")
print("FF2-public (configs/ces_ff2_public.yaml)")
p = run("configs/ces_ff2_public.yaml")
t = p["ff2_test"]
print(f"  ff2_test dropped methods old {t['old_methods']} -> new {t['new_methods']}")
assert t["old_methods"].get("pixart", 0) == 1274 and t["new_methods"].get("pixart", 0) == 0
print("PASS FF2-public: the 1,274 PixArt test rows are no longer dropped")
print("ALL_PASS test_cluster_key_f8")
