"""Fidelity check of preprocessing deviation D1-D4 (fairface_annotate.py) against FairFace's own predict.py pipeline.

For a fixed random sample of images per set (seed 0), runs predict.py's preprocessing verbatim (dlib.load_rgb_image,
resize to max side 800 with dlib.resize_image, mmod CNN detector with 1 upsample, 5-point landmarks on the resized
image, dlib.get_face_chips(size=300, padding=0.25); the largest face is kept when several are found) and compares the
predictions with ours (fairface_annotate.chip) on the same images and the same weights.

  OMP_NUM_THREADS=1 ~/.conda/envs/ffaa/bin/python fairface_fidelity.py --n 60 --workers 16
Writes results/fairforensics_ids/fidelity_official_vs_ours.csv (+ _summary.json).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fairface_annotate as FA  # noqa: E402

OUT = f"{FA.DA}/results/fairforensics_ids"


def official_chip(path):
    dlib = FA._W["dlib"]
    img = dlib.load_rgb_image(path)
    h, w, _ = img.shape
    if w > h:
        nw, nh = 800, int(800 * h / w)
    else:
        nw, nh = int(800 * w / h), 800
    img = dlib.resize_image(img, rows=nh, cols=nw)
    dets = FA._W["cnn"](img, 1)
    if len(dets) == 0:
        return None, 0
    faces = dlib.full_object_detections()
    for d in dets:
        faces.append(FA._W["sp"](img, d.rect))
    chips = dlib.get_face_chips(img, faces, size=300, padding=0.25)
    areas = [d.rect.width() * d.rect.height() for d in dets]
    return chips[int(np.argmax(areas))], len(dets)


def predict(chip):
    from PIL import Image
    torch = FA._W["torch"]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    x = np.asarray(Image.fromarray(chip).resize((224, 224), Image.BILINEAR), np.float32) / 255.0
    x = ((x - mean) / std).transpose(2, 0, 1)[None]
    with torch.no_grad():
        o = FA._W["model"](torch.from_numpy(x)).numpy().astype(np.float64)
    return np.concatenate([FA._sm(o[:, :7]), FA._sm(o[:, 7:9]), FA._sm(o[:, 9:18])], 1)[0]


def labels(p):
    a9 = int(np.argmax(p[9:18]))
    return {"sex": "Male" if p[7] >= 0.5 else "Female", "age2": "0-29" if a9 <= 3 else "30+",
            "race2": "White" if int(np.argmax(p[:7])) == 0 else "NonWhite", "race7": FA.RACE7[int(np.argmax(p[:7]))],
            "age9": FA.AGE9[a9]}


def one(task):
    name, p = task
    t = time.time()
    oc, nf = official_chip(p)
    t_off = time.time() - t
    mc, mode, conf, _ = FA.chip(p)
    pm = predict(mc)
    r = {"set": name, "img_path": p, "ours_det_mode": mode, "official_n_faces": nf, "official_s": t_off}
    r.update({f"ours_{k}": v for k, v in labels(pm).items()})
    if oc is not None:
        po = predict(oc)
        r.update({f"off_{k}": v for k, v in labels(po).items()})
        r["max_abs_prob_diff"] = float(np.abs(po - pm).max())
        r["p_male_diff"] = float(po[7] - pm[7])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    from multiprocessing import Pool
    t0 = time.time()
    tasks = []
    for name, kind, src in FA.SETS[:5]:
        b = FA.load_set(name, kind, src).sample(a.n, random_state=0)
        tasks += [(name, p) for p in b.img_path]
    with Pool(a.workers, initializer=FA._init) as pool:
        rows = pool.map(one, tasks, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/fidelity_official_vs_ours.csv", index=False)
    ok = df.off_sex.notna()
    S = {"n_images": int(len(df)), "official_no_face": int((~ok).sum()), "per_set": {}}
    for s, g in [("all", df[ok])] + [(s, g[g.off_sex.notna()]) for s, g in df.groupby("set")]:
        S["per_set"][s] = {"n": int(len(g)), **{f"agree_{k}": round(100 * float((g[f"ours_{k}"] == g[f"off_{k}"]).mean()), 2)
                                              for k in ("sex", "age2", "race2", "race7", "age9")},
                           "agree_ids8": round(100 * float(((g.ours_sex == g.off_sex) & (g.ours_age2 == g.off_age2) &
                                                            (g.ours_race2 == g.off_race2)).mean()), 2),
                           "median_max_abs_prob_diff": round(float(g.max_abs_prob_diff.median()), 4)}
    S["runtime_s"] = round(time.time() - t0)
    S["official_s_per_image_median"] = round(float(df.official_s.median()), 2)
    json.dump(S, open(f"{OUT}/fidelity_official_vs_ours_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
