"""Follow-up check (run after analyze_dfd_failures.py): DFD multi-person scenes.
(a) exact-duplicate frames (md5 of the PNG pixels) shared between different videos;
(b) 'wrong-person crop' proxy: the frame's AI-Face cleaner gender (cl_male, frame level) disagrees with the gender of
    BOTH actors named in the video (actor gender = majority cl_male over that actor's REAL videos, 0_<id>__*).
    For a real video only the one named actor is checked. Reports FPR/FNR (matched threshold) inside vs outside.
Read-only; writes crop_identity_check.json + duplicate_frames.csv here."""
import hashlib, json, sys
import numpy as np, pandas as pd
from PIL import Image
BD = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING"
AG = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents"
OUT = f"{BD}/results/failure_cases/dfd"
d = np.load(f"{BD}/runs/BD0_s7/eval/dfd_scores.npz", allow_pickle=True)
e = np.load(f"{AG}/DINGREPRO_LIN/results/audit_scores/full_s7_ep49_dfd.npz", allow_pickle=True)
L = pd.read_csv(f"{AG}/DEMO_AXES/labels/labels_dfd.csv")
t = json.load(open(f"{BD}/runs/BD0_s7/eval/bd_eval.json"))["thresholds"]["matched"]
df = pd.DataFrame({"img_path": d["img_path"], "video": d["cluster"], "label": d["label"], "group": d["group"],
                   "logit": d["logit"].astype(float), "ding": e["logit"].astype(float), "cl_male": L.cl_male.values})
df["err"] = np.where(df.label == 0, df.logit >= t, df.logit < t)
df["ding_err"] = np.where(df.label == 0, df.ding >= 0, df.ding < 0)
body = df.video.str[2:]
df["ids"] = body.str.split("__").str[0].str.split("_")
df["scene"] = body.str.split("__").str[1]

def md5(p):
    return hashlib.md5(np.asarray(Image.open(p).convert("RGB")).tobytes()).hexdigest()
df["md5"] = [md5(p) for p in df.img_path]
dup = df.groupby("md5").filter(lambda g: g.video.nunique() > 1)
dup_vids = dup.groupby("md5").video.apply(lambda s: " | ".join(sorted(set(s))))
dup_pairs = dup_vids.value_counts().reset_index()
dup_pairs.columns = ["videos_sharing_identical_frames", "n_identical_frame_hashes"]
dup_pairs.to_csv(f"{OUT}/duplicate_frames.csv", index=False)

real = df[df.label == 0]
actor_male = real.assign(a=real.ids.str[0]).groupby("a").cl_male.mean().round(3).to_dict()
def mismatch(r):
    sexes = [actor_male.get(i) for i in r.ids]
    if any(s is None for s in sexes):
        return np.nan
    exp = {int(s >= 0.5) for s in sexes}
    return float(r.cl_male not in exp)
df["mismatch"] = df.apply(mismatch, axis=1)
res = {"threshold_matched": t, "actor_male_share_over_real_frames": actor_male,
       "frames_in_duplicate_hash_across_videos": int(len(dup)), "videos_involved_in_duplicates": int(dup.video.nunique()),
       "duplicate_real_frames": int((dup.label == 0).sum()), "duplicate_fake_frames": int((dup.label == 1).sum())}
for lab, nm in ((0, "real"), (1, "fake")):
    s = df[(df.label == lab) & df.mismatch.notna()]
    for mm in (0.0, 1.0):
        q = s[s.mismatch == mm]
        res[f"{nm}_gender_mismatch_{int(mm)}"] = {"n": int(len(q)), "videos": int(q.video.nunique()),
                                                  "err_rate_ours": 100 * float(q.err.mean()) if len(q) else None,
                                                  "err_rate_ding": 100 * float(q.ding_err.mean()) if len(q) else None}
    res[f"{nm}_frames_unscored_actor_unknown"] = int(((df.label == lab) & df.mismatch.isna()).sum())
# videos whose frames are byte-identical to frames of a REAL video (fake 'contains' an unswapped crop)
rh = set(df[df.label == 0].md5)
fk = df[df.label == 1]
res["fake_frames_identical_to_a_real_frame"] = int(fk.md5.isin(rh).sum())
json.dump(res, open(f"{OUT}/crop_identity_check.json", "w"), indent=1)
print(json.dumps(res, indent=1)); print(dup_pairs.head(30).to_string())
