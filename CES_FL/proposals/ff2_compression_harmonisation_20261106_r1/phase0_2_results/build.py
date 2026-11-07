# FF2-H build, recipe R1 (no label / source exceptions):
#   lossless master -> resize to FairFace median IOD (INTER_AREA shrink / INTER_CUBIC enlarge)
#   -> JPEG q75 4:2:0 round-trip (cv2, standard IJG tables) -> canonical profile v2 (SCRFD canvas detection,
#   ArcFace 5-pt similarity to 256, frozen encoder JPEG q95 4:4:4).
# Usage: build.py <index_file.npy> <records_out.csv> <n_workers> [--check]
#   index_file: row indices into _inventory_PRIVATE.pkl (public_manifest order)
#   --check: also re-detect landmarks on the 256 output (pilot QC only)
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ff2h_common as H
H.single_thread_env()
import numpy as np, pandas as pd
from multiprocessing import Pool

IOD_TARGET = float(json.load(open(H.WORK + "/phase0_summary.json"))["IOD_TARGET"])
APP = None
CHECK = "--check" in sys.argv


def _init():
    global APP, V, C, cv2
    APP = H.make_app()
    import canon_v2 as V, canonicalize_ff2 as C, cv2 as _cv2
    cv2 = _cv2
    V.assert_env(); C.assert_frozen_env()


def jpeg_rt(bgr, q=75):  # identical to ff2_source_diag_20261106/recrop_score.py::jpeg_rt
    ok, b = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q, int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                                       int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420)])
    if not ok:
        raise RuntimeError("jpeg_rt encode failed")
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def master_kps(bgr):
    f, _, _ = V.detect_v2(APP, bgr)
    if f:
        return f[0]["kps"], "scrfd_v2canvas_master", f[0]["det_score"], len(f)
    f = C.detect(APP, bgr)  # v1 shared detector (0.25 replicate pad) as the declared fallback
    if f:
        return f[0]["kps"], "scrfd_v1pad_master", f[0]["det_score"], len(f)
    return None, "none", np.nan, 0


def process(task):
    idx, mp, rp = task
    r = dict(idx=idx)
    try:
        data = open(mp, "rb").read()
        r["master_sha256"] = H.sha256_bytes(data)
        bgr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("master_decode")
        r["master_h"], r["master_w"] = bgr.shape[:2]
        kps, src, ds, nf = master_kps(bgr)
        r.update(lm_src_master=src, det_score_master=ds, n_faces_master=nf)
        if kps is None:
            raise ValueError("noface_master")
        iod = C.face_scale(kps); r["iod_master"] = iod
        k = IOD_TARGET / iod; r["scale"] = k
        if abs(k - 1.0) < 1e-9:
            small, r["interp"] = bgr, "none"
        elif k < 1:
            small, r["interp"] = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA), "AREA"
        else:
            small, r["interp"] = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC), "CUBIC"
        r["rs_h"], r["rs_w"] = small.shape[:2]
        rt = jpeg_rt(small)
        sx, sy = small.shape[1] / bgr.shape[1], small.shape[0] / bgr.shape[0]
        exp = np.stack([(kps[:, 0] + .5) * sx - .5, (kps[:, 1] + .5) * sy - .5], 1).astype(np.float32)
        f2, _, _ = V.detect_v2(APP, rt)
        if f2:
            d = [float(np.linalg.norm(f["kps"] - exp, axis=1).mean()) for f in f2]
            j = int(np.argmin(d))
            r["lm_largest_is_match"] = int(j == 0); r["lm_dev_px"] = d[j]
            if d[j] < 0.25 * IOD_TARGET:
                kps2, r["lm_src_align"], r["det_score_align"] = f2[j]["kps"], "scrfd_v2canvas_rt", f2[j]["det_score"]
            else:
                kps2, r["lm_src_align"] = exp, "scaled_master_kps(mismatch)"
        else:
            kps2, r["lm_src_align"] = exp, "scaled_master_kps(noface)"
        r["iod_rt"] = C.face_scale(kps2)
        out, a, info = V.canonicalize_v2(rt, kps2)
        r.update(align_scale=info["align_scale"], align_rot_deg=info["align_rot_deg"], out_bytes=info["out_bytes"],
                 out_sha256=info["out_sha256"])
        dst = H.OUT_ROOT + rp
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + ".tmp%d" % os.getpid()
        with open(tmp, "wb") as fh:
            fh.write(out); fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp, dst)
        if os.path.getsize(dst) != len(out):
            raise IOError("short_write")
        if CHECK:
            dec = cv2.imdecode(np.frombuffer(open(dst, "rb").read(), np.uint8), cv2.IMREAD_COLOR)
            r["chk_hw"] = "%dx%d" % dec.shape[:2]
            f3, _, _ = V.detect_v2(APP, dec)
            T = C.template(256)
            if f3:
                e = np.linalg.norm(f3[0]["kps"] - T, axis=1)
                r["chk_lm_err_mean"], r["chk_lm_err_max"] = float(e.mean()), float(e.max())
            r["chk_n_faces"] = len(f3)
    except Exception as e:
        r["err"] = f"{type(e).__name__}:{e}"[:160]
    return r


if __name__ == "__main__":
    idx_file, rec_out, nw = sys.argv[1], sys.argv[2], int(sys.argv[3])
    inv = pd.read_pickle(H.WORK + "/_inventory_PRIVATE.pkl")
    ids = np.load(idx_file)
    done = set()
    if os.path.exists(rec_out):
        try:
            done = set(pd.read_csv(rec_out, usecols=["idx", "out_sha256"]).dropna().idx.astype(int))
        except Exception:
            done = set()
    tasks = [(int(i), inv.master_path.iat[i], inv.rel_path.iat[i]) for i in ids if int(i) not in done]
    prog = rec_out + ".progress"
    t0 = time.time(); n = 0; nerr = 0
    header = not os.path.exists(rec_out)
    COLS = ["idx", "master_sha256", "master_h", "master_w", "lm_src_master", "det_score_master", "n_faces_master",
            "iod_master", "scale", "interp", "rs_h", "rs_w", "lm_largest_is_match", "lm_dev_px", "lm_src_align",
            "det_score_align", "iod_rt", "align_scale", "align_rot_deg", "out_bytes", "out_sha256",
            "chk_hw", "chk_lm_err_mean", "chk_lm_err_max", "chk_n_faces", "err"]
    buf = []
    with Pool(nw, initializer=_init) as p, open(rec_out, "a") as fo:
        for r in p.imap_unordered(process, tasks, chunksize=8):
            buf.append(r); n += 1; nerr += int("err" in r)
            if len(buf) >= 500 or n == len(tasks):
                pd.DataFrame(buf).reindex(columns=COLS).to_csv(fo, header=header, index=False); fo.flush()
                header = False; buf = []
                el = time.time() - t0
                open(prog, "w").write(json.dumps(dict(done=n, total=len(tasks), already=len(done), errors=nerr,
                                                      elapsed_s=round(el), rate_per_s=round(n / el, 2),
                                                      eta_s=round((len(tasks) - n) / max(n / el, 1e-9)))) + "\n")
        if buf:
            pd.DataFrame(buf).reindex(columns=COLS).to_csv(fo, header=header, index=False)
    print("finished", n, "errors", nerr, "sec", round(time.time() - t0))
