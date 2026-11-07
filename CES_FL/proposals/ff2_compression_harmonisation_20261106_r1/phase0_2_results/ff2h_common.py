# FF2-H common helpers: path mapping (public rel_path <-> on-disk v1.0 path), master resolution, recipe R1.
# Read-only with respect to every existing file. CPU only.
import os, glob, hashlib
import numpy as np, pandas as pd

ICLR = "/home/sachin.chaudhary/ICLR_2027/"
R = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009"
PUB = f"{R}/release_ff2_public/manifests/public_manifest.csv"
RM = f"{R}/agents/T_scaleup/real_tiers/data/REAL_MANIFEST.csv"
FIN = "/home/sachin.chaudhary/ICLR_2027/fairforge_scaleup/fin_records/*/*.csv"
ACQ = "/home/sachin.chaudhary/ICLR_2027/fairforge2/metadata/acquisition_manifest_*.csv"
FF2ROOT = "/home/sachin.chaudhary/ICLR_2027/fairforge2/"
FFHQ_M = "/home/sachin.chaudhary/ICLR_2027/fairforge2/masters/real/ffhq1024/%s.webp"
FF_SRC = "/home/sachin.chaudhary/ICLR_2027/fairforge_scaleup/sources/fairface_1.25/"
OUT_ROOT = "/home/sachin.chaudhary/ICLR_2027/fairforge2_h/"
WORK = "/home/sachin.chaudhary/ces_fl_runs/ff2h_build_20261107"
VENDOR = f"{R}/release_ff2_public/tools/inswapper_regen/vendor"

_MAP = (("images/real/ff2_v1/", "FairForge2_v1.0/real/"), ("images/fake/ff2_v1/", "FairForge2_v1.0/fake/"),
        ("images/real/", "fairforge_scaleup/real/"), ("images/fake/", "fairforge_scaleup/fakes/"))


def abs_of_rel(rp):
    """inverse of build_public.rel(): public rel_path -> on-disk FF2 v1.0 path"""
    for a, b in _MAP:
        if rp.startswith(a):
            return ICLR + b + rp[len(a):]
    raise ValueError(rp)


def rel_of_abs(p):
    assert p.startswith(ICLR), p
    q = p[len(ICLR):]
    for a, b in _MAP:
        if q.startswith(b):
            return a + q[len(b):]
    raise ValueError(p)


def new_abs(p_old):
    return OUT_ROOT + rel_of_abs(p_old)


def load_fin():
    cols = ["item_id", "fake_path", "fake_sha256", "master_path", "master_sha256", "iod", "profile", "align_scale"]
    L = []
    for f in sorted(glob.glob(FIN)):
        d = pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False)
        L.append(d[d.fake_path.notna()])
    return pd.concat(L, ignore_index=True)


def load_acq():
    cols = ["image_id", "arm", "master_path", "master_sha256", "iod", "align_scale", "admitted", "canonical_path", "out_sha256"]
    L = []
    for f in sorted(glob.glob(ACQ)):
        if os.path.getsize(f) == 0:
            continue
        d = pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False).assign(_file=os.path.basename(f))
        L.append(d)
    return pd.concat(L, ignore_index=True)


def resolve_masters():
    """One row per public-manifest row (same order) with master_path + kind + reference iod (stored)."""
    m = pd.read_csv(PUB, low_memory=False)
    m["abs_path"] = m.rel_path.map(abs_of_rel)
    m["master_path"] = None; m["master_kind"] = None; m["ref_sha_ok"] = np.nan; m["stored_iod"] = np.nan
    m["stored_master_sha256"] = None
    # reals
    a = (m.label == 0) & (m.src_dataset == "FFHQ")
    assert m.loc[a, "item_id"].str.startswith("ffhqA/").all()
    m.loc[a, "master_path"] = m.loc[a, "item_id"].str[6:].map(lambda i: FFHQ_M % i)
    m.loc[a, "master_kind"] = "ffhq1024_webp"
    c = (m.label == 0) & (m.src_dataset == "FairFace")
    assert m.loc[c, "item_id"].str.startswith("ff/images/").all()
    m.loc[c, "master_path"] = FF_SRC + m.loc[c, "item_id"].str[3:]
    m.loc[c, "master_kind"] = "fairface125_jpeg"
    ffs = pd.read_csv(FF_SRC + "manifest_1.25.csv")
    shamap = dict(zip(FF_SRC + ffs.path, ffs.sha256))
    m.loc[c, "stored_master_sha256"] = m.loc[c, "master_path"].map(shamap)
    rm = pd.read_csv(RM, usecols=["path", "sha256", "face_iod_src_px", "tier"], low_memory=False).set_index("path")
    r = m.label == 0
    m.loc[r, "ref_sha_ok"] = (m.loc[r, "abs_path"].map(rm.sha256) == m.loc[r, "sha256"]).astype(float)
    # scale-up fakes
    s = (m.label == 1) & (m.origin == "scaleup")
    fin = load_fin().drop_duplicates("fake_path").set_index("fake_path")
    m.loc[s, "master_path"] = m.loc[s, "abs_path"].map(fin.master_path)
    m.loc[s, "stored_master_sha256"] = m.loc[s, "abs_path"].map(fin.master_sha256)
    m.loc[s, "stored_iod"] = m.loc[s, "abs_path"].map(fin.iod)
    m.loc[s, "ref_sha_ok"] = (m.loc[s, "abs_path"].map(fin.fake_sha256) == m.loc[s, "sha256"]).astype(float)
    m.loc[s & m.master_path.notna(), "master_kind"] = "scaleup_fin_records"
    # FF2 v1.0 fakes: acquisition manifests (admitted rows, keyed by image_id; verified by canonical out_sha256)
    v = (m.label == 1) & (m.origin == "ff2_v1.0")
    acq = load_acq()
    acq = acq[acq.admitted == 1]
    ids = m.loc[v, "item_id"].str.replace("^ff2/", "", regex=True)
    # prefer the acquisition row whose canonical out_sha256 equals the public sha256
    key = pd.DataFrame({"image_id": ids.values, "sha256": m.loc[v, "sha256"].values, "idx": m.index[v]})
    j = key.merge(acq, left_on=["image_id", "sha256"], right_on=["image_id", "out_sha256"], how="left")
    j = j.drop_duplicates("idx").set_index("idx")
    m.loc[v, "master_path"] = j.master_path.reindex(m.index[v]).map(lambda p: FF2ROOT + p if isinstance(p, str) else None).values
    m.loc[v, "stored_master_sha256"] = j.master_sha256.reindex(m.index[v]).values
    m.loc[v, "stored_iod"] = j.iod.reindex(m.index[v]).values
    m.loc[v, "ref_sha_ok"] = j.out_sha256.reindex(m.index[v]).notna().astype(float).values
    m.loc[v & m.master_path.notna(), "master_kind"] = "ff2v1_acquisition"
    # fallback for v1.0 fakes missing from the acquisition manifests: generation manifests (admitted rows),
    # master = fairforge2/masters/<arm>/<out_filename>; verified later by master_sha256 recorded at generation
    u = v & m.master_path.isna()
    if u.any():
        L = []
        for f in sorted(glob.glob(FF2ROOT + "metadata/generation_manifest_*.csv")):
            if os.path.getsize(f) == 0:
                continue
            d = pd.read_csv(f, low_memory=False)
            if {"out_filename", "admitted", "arm", "master_sha256"} <= set(d.columns):
                L.append(d[d.admitted == 1][["out_filename", "arm", "master_sha256"]])
        g = pd.concat(L).drop_duplicates("out_filename")
        g["image_id"] = g.out_filename.str.rsplit(".", n=1).str[0]
        g = g.set_index("image_id")
        iid = m.loc[u, "item_id"].str.replace("^ff2/", "", regex=True)
        fn = iid.map(g.out_filename); arm = iid.map(g.arm)
        mp = [FF2ROOT + "masters/%s/%s" % (a, f) if isinstance(f, str) else None for a, f in zip(arm, fn)]
        m.loc[u, "master_path"] = mp
        m.loc[u, "stored_master_sha256"] = iid.map(g.master_sha256).values
        m.loc[u & m.master_path.notna(), "master_kind"] = "ff2v1_generation_manifest"
    return m


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def single_thread_env():
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def make_app():
    """SCRFD (buffalo_l detection only) on CPU with onnxruntime capped to 1 intra + 1 inter thread.
    insightface's model_zoo does not pass sess_options through, so InferenceSession is wrapped before import."""
    single_thread_env()
    import onnxruntime as ort
    if not getattr(ort.InferenceSession, "_ff2h_capped", False):
        _orig = ort.InferenceSession

        class _Capped(_orig):
            _ff2h_capped = True

            def __init__(self, *a, **kw):
                so = kw.get("sess_options") or ort.SessionOptions()
                so.intra_op_num_threads = 1
                so.inter_op_num_threads = 1
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                kw["sess_options"] = so
                kw["providers"] = ["CPUExecutionProvider"]
                super().__init__(*a, **kw)
        ort.InferenceSession = _Capped
    import cv2
    cv2.setNumThreads(1)
    import sys
    if VENDOR not in sys.path:
        sys.path.insert(0, VENDOR)
    import canonicalize_ff2 as C
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"], allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=tuple(C.CONFIG["det_size"]), det_thresh=C.CONFIG["det_thresh"])
    return app
