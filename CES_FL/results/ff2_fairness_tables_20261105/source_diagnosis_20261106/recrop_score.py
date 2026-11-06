# Counterfactual re-crops of FFHQ-sourced VAL images from their lossless masters, scored with ONE FF2 checkpoint (L_A).
# VAL only. CPU only. Nothing is written except aggregate-capable private tables in this directory.
import os, sys, io, json, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '8'); os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np, pandas as pd, cv2
from PIL import Image
cv2.setNumThreads(1)
OUT = '/home/sachin.chaudhary/ces_fl_runs/ff2_source_diag_20261106'
sys.path.insert(0, '/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/release_ff2_public/tools/inswapper_regen/vendor')
import canonicalize_ff2 as C, canon_v2 as V
sys.path.insert(0, OUT); from feat_stats import feats
import torch; torch.set_num_threads(8)
sys.path.insert(0, '/home/sachin.chaudhary/ces_fl_sources/ff2-lh-ps-v2-20261106/CES_FL/src')
import ces_fl_head_ablation as H
from insightface.app import FaceAnalysis

V.assert_env(); C.assert_frozen_env()
CK = '/home/sachin.chaudhary/ces_fl_runs/ff2_lh_ps_20261106_r1/L_A'
model, ck = H.load_trained(CK + '/best.pt', torch.device('cpu'))
SZ = ck['config']['size']
VS = json.load(open(CK + '/validation_scores_PRIVATE.json')); THR = float(VS['threshold'])
assert VS['manifest_sha256'] == 'aeda051259d9e26b07c2810152b601ae115ce7c9e10523f99ea07bbc9f92afa8'
STORED = dict(zip([r['image'] for r in VS['rows']], VS['logits']))
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'], allowed_modules=['detection'])
app.prepare(ctx_id=-1, det_size=tuple(C.CONFIG['det_size']), det_thresh=C.CONFIG['det_thresh'])
TARGET_SCALE = float(sys.argv[1])  # FairFace VAL-real median v2 align_scale (from REAL_MANIFEST)


def score(jpeg_bytes):
    with Image.open(io.BytesIO(jpeg_bytes)) as im:
        a = np.asarray(im.convert('RGB').resize((SZ, SZ), Image.Resampling.BILINEAR), np.float32).copy() / 255.
    x = H.normalize(torch.from_numpy(a).permute(2, 0, 1))[None]
    with torch.inference_mode():
        return float(model(x)['logit'][0])


def jpeg_rt(bgr, q=75):  # standard IJG tables, 4:2:0, like the FairFace 1.25 source files
    ok, b = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q, int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                                       int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420)])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)


def v2(src):
    f, _, _ = V.detect_v2(app, src)
    if not f: raise ValueError('no face')
    d, _, info = V.canonicalize_v2(src, f[0]['kps']); return d, info['align_scale']


def arms(bgr):
    out = {}
    f1 = C.detect(app, bgr)
    if not f1: raise ValueError('no face v1')
    d, rec = C.canonicalize(bgr, f1[0]['kps']); out['A1_v1_recrop'] = (d, rec['align_scale'])
    out['A2_v2_profile_from_master'] = v2(bgr)
    iod = C.face_scale(f1[0]['kps'])
    k = V.TEMPLATE_IOD_256 / (TARGET_SCALE * iod)  # source scale so that the v2 warp upsamples like FairFace
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    out['A3_fairface_geometry'] = v2(small)
    out['A4_fairface_geometry_jpeg75_420'] = v2(jpeg_rt(small))
    h, w = small.shape[:2]
    lo = cv2.resize(bgr, (max(8, w // 2), max(8, h // 2)), interpolation=cv2.INTER_AREA)
    up = cv2.resize(lo, (w, h), interpolation=cv2.INTER_CUBIC)
    out['A6_fairface_geometry_2x_upsampled_jpeg75_420'] = v2(jpeg_rt(up))
    d, rec = C.canonicalize(jpeg_rt(bgr), f1[0]['kps']); out['A5_v1_after_master_jpeg75_420'] = (d, rec['align_scale'])
    return out


def run(items, tag):
    rows = []
    for i, (img, master, lab, method) in enumerate(items):
        r = dict(kind=tag, label=lab, method=method)
        try:
            stored = open(img, 'rb').read()
            r['A0_stored_logit_file'] = STORED.get(img, np.nan)
            r['A0_stored'] = score(stored)
            bgr = cv2.imread(master, cv2.IMREAD_COLOR)
            for name, (d, sc) in arms(bgr).items():
                r[name] = score(d); r[name + '_scale'] = sc
                if name == 'A1_v1_recrop': r['A1_bytes_identical'] = int(d == stored)
                fs = feats_bytes(d); r[name + '_log_lap'] = fs['log_lap']; r[name + '_hf_y'] = fs['hf_y']; r[name + '_noise'] = fs['noise']
            fs = feats_bytes(stored); r['A0_stored_log_lap'] = fs['log_lap']; r['A0_stored_hf_y'] = fs['hf_y']; r['A0_stored_noise'] = fs['noise']
        except Exception as e:
            r['err'] = f'{type(e).__name__}:{e}'[:120]
        rows.append(r)
        if (i + 1) % 50 == 0: print(tag, i + 1, flush=True)
    return rows


def feats_bytes(d):
    p = OUT + '/_tmp_arm_%s.jpg' % os.environ.get('TAG', 'all')
    open(p, 'wb').write(d); return feats(p)


if __name__ == '__main__':
    v = pd.read_csv('/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/val.csv', usecols=['image', 'label', 'source_ids', 'method'])
    reals = v[(v.label == 0) & v.source_ids.str.startswith('ffhq:')]
    items = [(r.image, '/home/sachin.chaudhary/ICLR_2027/fairforge2/masters/real/ffhq1024/%s.webp' % r.source_ids.split(':')[1], 0, 'real')
             for r in reals.itertuples()]
    items = items[:int(os.environ.get('LIMIT', 10**9))]
    rows = [] if os.environ.get('SKIP_REALS') else run(items, 'ffhq_real')
    if len(sys.argv) > 2:  # control: FFHQ-sourced VAL fakes (tier A, profile v1) from their generator masters
        fin = pd.read_pickle(OUT + '/_fin_trval_PRIVATE.pkl')
        fk = fin[(fin.sp == 'val') & (fin.src == 'ffhq') & (fin.profile == 'v1') & fin.master_path.notna()]
        fk = fk.groupby('method').sample(n=int(sys.argv[2]), random_state=0, replace=False) if fk.groupby('method').size().min() >= int(sys.argv[2]) else fk.groupby('method').head(int(sys.argv[2]))
        rows += run([(r.fake_path, r.master_path, 1, r.method) for r in fk.itertuples()], 'ffhq_fake')
    df = pd.DataFrame(rows); df.to_pickle(OUT + '/_recrop_scores_%s_PRIVATE.pkl' % os.environ.get('TAG', 'all'))
    os.remove(OUT + '/_tmp_arm_%s.jpg' % os.environ.get('TAG', 'all')) if os.path.exists(OUT + '/_tmp_arm_%s.jpg' % os.environ.get('TAG', 'all')) else None
    arms_ = ['A0_stored', 'A1_v1_recrop', 'A2_v2_profile_from_master', 'A3_fairface_geometry', 'A4_fairface_geometry_jpeg75_420',
             'A5_v1_after_master_jpeg75_420', 'A6_fairface_geometry_2x_upsampled_jpeg75_420']
    ok = df[df.get('err', pd.Series(index=df.index, dtype=object)).isna()]
    print('threshold', THR, 'n ok', ok.groupby('kind').size().to_dict(), 'errors', len(df) - len(ok))
    print('stored-vs-file max |dlogit|', float((ok.A0_stored - ok.A0_stored_logit_file).abs().max()))
    print('A1 byte-identical', ok.groupby('kind').A1_bytes_identical.mean().to_dict())
    for kind, g in ok.groupby('kind'):
        for a in arms_:
            print(kind, a.ljust(48), 'flag-fake rate %.3f' % (g[a] >= THR).mean(), 'median logit %.2f' % g[a].median(),
                  'scale %.3f' % g[a + '_scale'].median() if a + '_scale' in g else '',
                  'log_lap %.2f hf_y %.2f noise %.2f' % (g[a + '_log_lap'].median(), g[a + '_hf_y'].median(), g[a + '_noise'].median()) if a + '_log_lap' in g else '')
