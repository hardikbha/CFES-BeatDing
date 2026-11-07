# Pilot QC: decode/size, landmark error vs v2 template (FF2-H vs the FF2 v1.0 crop of the SAME item as reference),
# JPEG quantisation tables / sampling vs FF2, master-sha agreement with stored hashes, private contact sheet.
import os, sys, io, json, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ff2h_common as H
H.single_thread_env()
import numpy as np, pandas as pd
from PIL import Image, JpegImagePlugin
from multiprocessing import Pool

APP = None


def _init():
    global APP, V, C, cv2
    APP = H.make_app()
    import canon_v2 as V, canonicalize_ff2 as C, cv2 as _cv2
    cv2 = _cv2


def qinfo(p):
    im = Image.open(p)
    return (hashlib.md5(repr(sorted(im.quantization.items())).encode()).hexdigest()[:8],
            JpegImagePlugin.get_sampling(im), im.size)


def ref_one(p):
    """FF2 v1.0 crop of the same item: quant table, sampling, size, landmark error vs template(256)."""
    q, s, sz = qinfo(p)
    bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    f, _, _ = V.detect_v2(APP, bgr)
    T = C.template(256)
    e = np.linalg.norm(f[0]["kps"] - T, axis=1) if f else np.full(5, np.nan)
    return dict(ref_q=q, ref_samp=s, ref_size=sz[0], ref_lm_err_mean=float(np.mean(e)), ref_lm_err_max=float(np.max(e)))


if __name__ == "__main__":
    inv = pd.read_pickle(H.WORK + "/_inventory_PRIVATE.pkl")
    rec = pd.read_csv(H.WORK + "/_pilot_records_PRIVATE.csv").drop_duplicates("idx", keep="last").set_index("idx")
    ids = np.load(H.WORK + "/pilot_idx.npy")
    d = inv.loc[ids, ["rel_path", "abs_path", "split", "src_dataset", "label", "method", "origin", "stored_master_sha256", "stored_iod"]].join(rec)
    S = {"n": int(len(d)), "n_err": int(d.err.notna().sum()) if "err" in d else 0,
         "errors": d.err.value_counts().to_dict() if "err" in d else {}}
    ok = d[d.out_sha256.notna()].copy()
    ok["new_path"] = H.OUT_ROOT + ok.rel_path
    NQ = [qinfo(p) for p in ok.new_path]
    ok["new_q"] = [a for a, _, _ in NQ]; ok["new_samp"] = [b for _, b, _ in NQ]; ok["new_size"] = [c[0] for _, _, c in NQ]
    ok["sha_on_disk_ok"] = [hashlib.sha256(open(p, "rb").read()).hexdigest() == s for p, s in zip(ok.new_path, ok.out_sha256)]
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 16, initializer=_init) as p:
        R = p.map(ref_one, ok.abs_path.tolist(), chunksize=8)
    ok = pd.concat([ok.reset_index(), pd.DataFrame(R)], axis=1)
    ok["grp"] = ok.src_dataset + "|" + ok.label.map({0: "real", 1: "fake"})
    S["chk_hw"] = ok.chk_hw.value_counts().to_dict()
    S["new_size"] = ok.new_size.value_counts().to_dict()
    S["new_qtable_hash"] = ok.new_q.value_counts().to_dict(); S["ref_qtable_hash"] = ok.ref_q.value_counts().to_dict()
    S["new_sampling"] = {str(k): int(v) for k, v in ok.new_samp.value_counts().items()}
    S["ref_sampling"] = {str(k): int(v) for k, v in ok.ref_samp.value_counts().items()}
    S["qtable_match_frac"] = float((ok.new_q == ok.ref_q).mean())
    S["sha_on_disk_ok_frac"] = float(ok.sha_on_disk_ok.mean())
    sm = ok.stored_master_sha256.notna()
    S["master_sha_vs_stored"] = dict(n_with_stored=int(sm.sum()), n_match=int((ok.master_sha256[sm] == ok.stored_master_sha256[sm]).sum()))
    S["lm_src_master"] = ok.lm_src_master.value_counts().to_dict(); S["lm_src_align"] = ok.lm_src_align.value_counts().to_dict()
    S["largest_face_is_match_frac"] = float(ok.lm_largest_is_match.mean())
    S["interp"] = ok.interp.value_counts().to_dict()
    S["no_face_on_output"] = int((ok.chk_n_faces == 0).sum())
    # landmark error vs template: FF2-H output vs FF2 v1.0 crop of the same item (reference distribution)
    q = lambda s: {k: round(float(s.quantile(v)), 3) for k, v in [("p50", .5), ("p95", .95), ("p99", .99), ("max", 1.)]}
    S["lm_err_mean_px"] = {g: dict(ff2h=q(x.chk_lm_err_mean.dropna()), ff2_v10=q(x.ref_lm_err_mean.dropna())) for g, x in ok.groupby("grp")}
    S["lm_err_mean_px_all"] = dict(ff2h=q(ok.chk_lm_err_mean.dropna()), ff2_v10=q(ok.ref_lm_err_mean.dropna()))
    tol = float(ok.ref_lm_err_mean.quantile(.99))
    S["centring_tolerance_px(ref_p99)"] = round(tol, 3)
    S["frac_ff2h_within_tol"] = float((ok.chk_lm_err_mean <= tol).mean())
    S["scale_by_grp"] = {g: q(x.scale) for g, x in ok.groupby("grp")}
    S["align_scale_by_grp"] = {g: q(x.align_scale) for g, x in ok.groupby("grp")}
    S["iod_master_by_grp"] = {g: q(x.iod_master) for g, x in ok.groupby("grp")}
    si = ok.stored_iod.notna()
    S["stored_iod_vs_measured_master"] = dict(n=int(si.sum()), median_rel_diff=float((ok.iod_master[si] / ok.stored_iod[si] - 1).abs().median()),
                                             frac_rel_diff_gt_5pct=float(((ok.iod_master[si] / ok.stored_iod[si] - 1).abs() > .05).mean()))
    S["upsampled_frac_by_method"] = ok[ok.label == 1].groupby("method").scale.apply(lambda s: float((s > 1).mean())).round(3).to_dict()
    S["out_bytes_median_by_grp"] = ok.groupby("grp").out_bytes.median().to_dict()
    S["ref_ff2_v10_bytes_median_by_grp"] = ok.groupby("grp").abs_path.apply(lambda s: float(np.median([os.path.getsize(p) for p in s]))).to_dict()
    json.dump(S, open(H.WORK + "/pilot_qc_summary.json", "w"), indent=1, default=str)
    print(json.dumps(S, indent=1, default=str))
    # private contact sheet: 20 items (5 per source x label), FF2 v1.0 crop | FF2-H crop
    import cv2
    sys.path.insert(0, H.VENDOR); import canonicalize_ff2 as C
    pick = pd.concat([ok[ok.grp == g].sample(min(5, int((ok.grp == g).sum())), random_state=1) for g in sorted(ok.grp.unique())])
    tiles = []
    for _, r in pick.iterrows():
        a = cv2.imread(r.abs_path); b = cv2.imread(r.new_path)
        T = C.template(256).astype(int)
        b2 = b.copy()
        for x, y in T:
            cv2.circle(b2, (int(x), int(y)), 2, (0, 255, 0), -1)
        t = np.concatenate([a, b, b2], 1)
        cv2.putText(t, "%s %s" % (r.grp, r.method), (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 255), 1)
        tiles.append(t)
    rows = [np.concatenate(tiles[i:i + 2], 1) for i in range(0, len(tiles), 2)]
    w = max(x.shape[1] for x in rows)
    rows = [np.pad(x, ((0, 0), (0, w - x.shape[1]), (0, 0))) for x in rows]
    cv2.imwrite(H.WORK + "/_pilot_contact_sheet_PRIVATE.jpg", np.concatenate(rows, 0), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
