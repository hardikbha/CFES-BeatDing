"""FairFace (Karkkainen & Joo, WACV'21) demographic annotation, FairForensics/GenFace protocol (secondary axis).

Model: official res34_fair_align_multi_7_20190809.pt (dchen236/FairFace; ResNet-34, 18 outputs =
7 race [White, Black, Latino_Hispanic, East Asian, Southeast Asian, Indian, Middle Eastern] + 2 gender [Male, Female]
+ 9 age [0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+]); softmax per head, as in predict.py.

Preprocessing = predict.py's (dlib mmod CNN detector -> 5-point landmarks -> dlib.get_face_chip(size=300,
padding=0.25) -> Resize(224,224) -> ImageNet normalisation), with these documented deviations (README):
  D1 detector run on the image resized to max side 300 px with 0 upsampling (predict.py: 800 px, 1 upsample;
     7.4 s/img on CPU, infeasible for 0.6 M images). Our inputs are already face crops, so the face is large.
  D2 one face per image: the largest detection (predict.py writes every detected face).
  D3 fallbacks when mmod finds no face: dlib HOG detector (max side 400, 1 upsample), then the whole-image rectangle
     (shrunk 10%) as landmark-predictor box. The mode is recorded per image (det_mode).
  D4 landmarks/chip are computed on the original-resolution image (predict.py: on the 800-px resized copy).
Runs on CPU (env ffaa: dlib 19.24.2 + numpy 1.26 + torch 2.1.2; the FA env's dlib is broken with numpy 2).

  OMP_NUM_THREADS=1 ~/.conda/envs/ffaa/bin/python fairface_annotate.py --workers 16
Resumable: per-chunk part files under labels_fairface/_parts/<set>/; merged CSV written when a set completes.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
DA = f"{R}/agents/DEMO_AXES"
OUT = os.environ.get("FF_OUT", f"{DA}/labels_fairface")
PARTS = f"{OUT}/_parts"
FFROOT = "/home/sachin.chaudhary/models_external/fairface"
WEIGHTS = f"{FFROOT}/fair_face_models/res34_fair_align_multi_7_20190809.pt"
DLIBM = f"{FFROOT}/repo/dlib_models"
CHUNK = 400

RACE7 = ["White", "Black", "Latino_Hispanic", "East Asian", "Southeast Asian", "Indian", "Middle Eastern"]
AGE9 = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
# FairForensics Table I IDS order
IDS8 = ["MYW", "FYW", "MAW", "FAW", "MAN", "FAN", "MYN", "FYN"]
_IDS_IDX = {(s, a, r): i for i, (s, a, r) in enumerate(
    [("M", "Y", "W"), ("F", "Y", "W"), ("M", "A", "W"), ("F", "A", "W"),
     ("M", "A", "N"), ("F", "A", "N"), ("M", "Y", "N"), ("F", "Y", "N")])}

DL = f"{R}/agents/DINGREPRO_LIN/data"
FF2P = f"{R}/data_ding_schema/fairforge2_public"
# processing order: test sets first
SETS = [
    ("ffpp_test", "ding", f"{DL}/ff++/test.csv"),
    ("dfdc", "ding", f"{DL}/dfdc/test.csv"),
    ("celebdf", "ding", f"{DL}/celebdf/test.csv"),
    ("dfd", "ding", f"{DL}/dfd/test.csv"),
    ("ff2pub_test", "ff2", "TEST"),
    ("ff2pub_select", "ff2", "SELECT"),
    ("ffpp_val", "dingval", None),
    ("ffpp_train", "ding", f"{DL}/ff++_ding/train.csv"),
    ("ff2pub_train", "ff2", "TRAIN"),
    ("ff2pub_cal", "ff2", "CAL"),
]


def ff2_abs(q):
    """= release_ff2_public/build_training_csvs.absp (release-relative -> on-disk path used by our loaders)"""
    for b, a in (("images/real/ff2_v1/", "FairForge2_v1.0/real/"), ("images/fake/ff2_v1/", "FairForge2_v1.0/fake/"),
                 ("images/real/", "fairforge_scaleup/real/"), ("images/fake/", "fairforge_scaleup/fakes/")):
        if q.startswith(b):
            return "/home/sachin.chaudhary/ICLR_2027/" + a + q[len(b):]
    raise ValueError(q)


_MAN = None


def load_set(name, kind, src):
    global _MAN
    if kind == "ding":
        d = pd.read_csv(src)
        return pd.DataFrame({"img_path": d.img_path.drop_duplicates().values})
    if kind == "dingval":
        d = pd.concat([pd.read_csv(f"{DL}/ff++_ding/realval.csv"), pd.read_csv(f"{DL}/ff++_ding/fakeval.csv")])
        return pd.DataFrame({"img_path": d.img_path.drop_duplicates().values})
    if _MAN is None:
        _MAN = pd.read_csv(f"{R}/release_ff2_public/manifests/public_manifest.csv", low_memory=False,
                           usecols=["rel_path", "split"])
    m = _MAN[_MAN.split == src]
    return pd.DataFrame({"img_path": m.rel_path.map(ff2_abs).values, "rel_path": m.rel_path.values})


# ---------------------------------------------------------------- worker
_W = {}


def _init():
    import torch
    import torchvision
    import dlib
    torch.set_num_threads(1)
    m = torchvision.models.resnet34()
    m.fc = torch.nn.Linear(m.fc.in_features, 18)
    m.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    m.eval()
    _W.update(torch=torch, dlib=dlib, model=m,
              cnn=dlib.cnn_face_detection_model_v1(f"{DLIBM}/mmod_human_face_detector.dat"),
              hog=dlib.get_frontal_face_detector(),
              sp=dlib.shape_predictor(f"{DLIBM}/shape_predictor_5_face_landmarks.dat"))


def _resize_max(img, m):
    from PIL import Image
    h, w = img.shape[:2]
    s = m / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return np.ascontiguousarray(np.asarray(Image.fromarray(img).resize((nw, nh), Image.BILINEAR)), dtype=np.uint8), s


def _largest(rects):
    return max(rects, key=lambda r: r.width() * r.height())


def chip(path):
    from PIL import Image
    dlib = _W["dlib"]
    img = np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB")), dtype=np.uint8)
    small, s = _resize_max(img, 300)
    dets = _W["cnn"](small, 0)
    mode, conf, nf = "mmod300", np.nan, len(dets)
    if nf:
        d = max(dets, key=lambda d: d.rect.width() * d.rect.height())
        r, conf = d.rect, float(d.confidence)
    else:
        sm4, s = _resize_max(img, 400)
        h = _W["hog"](sm4, 1)
        nf = len(h)
        if nf:
            r, mode = _largest(h), "hog400"
        else:
            H, W = sm4.shape[:2]
            r, mode = dlib.rectangle(int(.05 * W), int(.05 * H), int(.95 * W), int(.95 * H)), "fullrect"
    R_ = dlib.rectangle(int(r.left() / s), int(r.top() / s), int(r.right() / s), int(r.bottom() / s))
    shape = _W["sp"](img, R_)
    c = dlib.get_face_chip(img, shape, size=300, padding=0.25)
    return c, mode, conf, nf


def work(task):
    name, ci, paths = task
    out = f"{PARTS}/{name}/{ci:05d}.csv"
    if os.path.exists(out):
        return name, ci, len(paths), 0.0
    t0 = time.time()
    torch = _W["torch"]
    from PIL import Image
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    rows, X = [], []
    for p in paths:
        try:
            c, mode, conf, nf = chip(p)
            x = np.asarray(Image.fromarray(c).resize((224, 224), Image.BILINEAR), np.float32) / 255.0
            X.append(((x - mean) / std).transpose(2, 0, 1))
            rows.append([p, mode, conf, nf, 1])
        except Exception as e:  # unreadable image: keep the row, mark it
            X.append(np.zeros((3, 224, 224), np.float32))
            rows.append([p, "error:" + type(e).__name__, np.nan, 0, 0])
    P = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            o = _W["model"](torch.from_numpy(np.stack(X[i:i + 32]))).numpy().astype(np.float64)
            P.append(np.concatenate([_sm(o[:, :7]), _sm(o[:, 7:9]), _sm(o[:, 9:18])], 1))
    P = np.concatenate(P)
    df = pd.DataFrame(rows, columns=["img_path", "det_mode", "det_conf", "n_faces", "ok"])
    for j, n in enumerate([f"p_race_{r}" for r in RACE7] + ["p_male", "p_female"] + [f"p_age_{a}" for a in AGE9]):
        df[n] = P[:, j]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out + ".tmp", index=False, float_format="%.6g")
    os.replace(out + ".tmp", out)
    return name, ci, len(paths), time.time() - t0


def _sm(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


# ---------------------------------------------------------------- derived labels
def video_id(p):
    """frame file '<k>_<video>_<frame>.png' -> '<video>' (same key as AI-Face 'vid' in outputs/aiface_*.csv)"""
    b = os.path.basename(p).rsplit(".", 1)[0]
    b = b.split("_", 1)[1] if b.split("_", 1)[0].isdigit() else b
    return b.rsplit("_", 1)[0]


def derive(df, video):
    pr = df[[f"p_race_{r}" for r in RACE7]].values
    pa = df[[f"p_age_{a}" for a in AGE9]].values
    df["race7"] = np.array(RACE7)[pr.argmax(1)]
    df["race2"] = np.where(pr.argmax(1) == 0, "White", "NonWhite")
    df["sex"] = np.where(df.p_male.values >= 0.5, "Male", "Female")
    a9 = pa.argmax(1)
    df["age9"] = np.array(AGE9)[a9]
    df["age2"] = np.where(a9 <= 3, "0-29", "30+")
    df["age4"] = np.array(["<30", "<30", "<30", "<30", "30-49", "30-49", "50-69", "50-69", "70+"])[a9]
    df["p_age_lt30"] = pa[:, :4].sum(1)
    df["ids8"] = [_IDS_IDX[(s[0], "Y" if a == "0-29" else "A", "W" if r == "White" else "N")]
                  for s, a, r in zip(df.sex, df.age2, df.race2)]
    df["ids8_name"] = np.array(IDS8)[df.ids8.values]
    if video:
        df["vid"] = df.img_path.map(video_id)
        g = df.groupby("vid")
        # tie-break (equal frame votes) = the same rule applied to the video's mean probabilities
        mp = g[[f"p_race_{r}" for r in RACE7] + ["p_male"] + [f"p_age_{a}" for a in AGE9]].mean()
        a9m = mp[[f"p_age_{a}" for a in AGE9]].values.argmax(1)
        tieb = {"sex": pd.Series(np.where(mp.p_male.values >= 0.5, "Male", "Female"), index=mp.index),
                "age2": pd.Series(np.where(a9m <= 3, "0-29", "30+"), index=mp.index),
                "race2": pd.Series(np.where(mp[[f"p_race_{r}" for r in RACE7]].values.argmax(1) == 0, "White", "NonWhite"),
                                   index=mp.index),
                "age4": pd.Series(np.array(["<30"] * 4 + ["30-49"] * 2 + ["50-69"] * 2 + ["70+"])[a9m], index=mp.index)}

        for col in ("sex", "age2", "race2", "age4"):
            vc = {}
            for v, s_ in g[col]:
                c = s_.value_counts()
                vc[v] = tieb[col][v] if (len(c) > 1 and c.iloc[0] == c.iloc[1]) else c.index[0]
            fr = g[col].agg(lambda s: s.value_counts().iloc[0] / len(s))
            df[f"vid_{col}"] = df.vid.map(vc)
            df[f"vid_{col}_agree"] = df.vid.map(fr)
        df["vid_ids8"] = [_IDS_IDX[(s[0], "Y" if a == "0-29" else "A", "W" if r == "White" else "N")]
                          for s, a, r in zip(df.vid_sex, df.vid_age2, df.vid_race2)]
        df["vid_ids8_name"] = np.array(IDS8)[df.vid_ids8.values]
        df["vid_n_frames"] = df.vid.map(g.size())
    return df


def merge_set(name, base, video):
    parts = sorted(glob.glob(f"{PARTS}/{name}/*.csv"))
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df = base.merge(df, on="img_path", how="left", validate="one_to_one")
    assert df.ok.notna().all() and len(df) == len(base), name
    df = derive(df, video)
    df.to_csv(f"{OUT}/{name}.csv", index=False, float_format="%.6g")
    return df


DERIVED = ["race7", "race2", "sex", "age9", "age2", "age4", "p_age_lt30", "ids8", "ids8_name", "vid", "vid_sex",
           "vid_sex_agree", "vid_age2", "vid_age2_agree", "vid_race2", "vid_race2_agree", "vid_age4", "vid_age4_agree",
           "vid_ids8", "vid_ids8_name", "vid_n_frames"]


def rederive(names):
    """Recompute the derived columns of merged label files from their stored raw probabilities (no inference)."""
    for name, kind, _ in SETS:
        f = f"{OUT}/{name}.csv"
        if name in names and os.path.exists(f):
            df = pd.read_csv(f, low_memory=False)
            df = derive(df.drop(columns=[c for c in DERIVED if c in df.columns]), kind != "ff2")
            df.to_csv(f, index=False, float_format="%.6g")
            print("re-derived", name, len(df), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rederive", action="store_true", help="only recompute derived columns of finished sets")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sets", default=",".join(s[0] for s in SETS))
    ap.add_argument("--limit", type=int, default=0, help="debug: first N images per set")
    a = ap.parse_args()
    want = a.sets.split(",")
    if a.rederive:
        return rederive(want)
    os.makedirs(PARTS, exist_ok=True)
    from multiprocessing import Pool
    bases, tasks = {}, []
    for name, kind, src in SETS:
        if name not in want:
            continue
        b = load_set(name, kind, src)
        if a.limit:
            b = b.iloc[:a.limit]
        bases[name] = (b, kind != "ff2")
        P = b.img_path.tolist()
        tasks += [(name, i // CHUNK, P[i:i + CHUNK]) for i in range(0, len(P), CHUNK)]
    left = {n: len(range(0, len(b), CHUNK)) for n, (b, _) in bases.items()}
    total = sum(len(b) for b, _ in bases.values())
    done, t0, prog = 0, time.time(), f"{OUT}/PROGRESS.json"
    state = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "workers": a.workers, "total_images": total, "sets": {}}
    with Pool(a.workers, initializer=_init) as pool:
        for name, ci, n, dt in pool.imap(work, tasks):
            done += n
            left[name] -= 1
            st = state["sets"].setdefault(name, {"images": len(bases[name][0]), "chunks_done": 0, "cpu_s": 0.0})
            st["chunks_done"] += 1
            st["cpu_s"] += dt
            if left[name] == 0:
                df = merge_set(name, *bases[name])
                st["merged"] = time.strftime("%Y-%m-%d %H:%M:%S")
                st["det_mode"] = df.det_mode.value_counts().to_dict()
                print(f"[{time.strftime('%H:%M:%S')}] merged {name}: {len(df)} rows", flush=True)
            el = time.time() - t0
            state.update(done_images=done, elapsed_s=round(el), img_per_s=round(done / max(el, 1), 2),
                         eta_min=round((total - done) / max(done / max(el, 1), 1e-9) / 60, 1),
                         updated=time.strftime("%Y-%m-%d %H:%M:%S"))
            json.dump(state, open(prog + ".tmp", "w"), indent=1)
            os.replace(prog + ".tmp", prog)
    print("ALL DONE", json.dumps(state)[:2000], flush=True)


if __name__ == "__main__":
    main()
