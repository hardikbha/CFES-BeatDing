# Follow-up: separate JPEG quality from chroma subsampling in the FairFace-geometry arm (VAL, L_A, CPU).
import os, sys, json
os.environ['TAG'] = 'split'
import numpy as np, pandas as pd, cv2
sys.argv = [sys.argv[0], '1.162']
import recrop_score as R
def rt(bgr, q, sub):
    f = {'420': cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420, '444': cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444}[sub]
    ok, b = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q, int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), int(f)])
    return cv2.imdecode(b, cv2.IMREAD_COLOR)
CFG = [(75, '444'), (95, '420'), (90, '444'), (85, '444'), (50, '444')]
def arms(bgr):
    f1 = R.C.detect(R.app, bgr); k = R.V.TEMPLATE_IOD_256 / (R.TARGET_SCALE * R.C.face_scale(f1[0]['kps']))
    small = cv2.resize(bgr, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return {f'G_q{q}_{s}': R.v2(rt(small, q, s)) for q, s in CFG}
R.arms = arms
v = pd.read_csv('/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/val.csv', usecols=['image', 'label', 'source_ids'])
reals = v[(v.label == 0) & v.source_ids.str.startswith('ffhq:')]
items = [(r.image, '/home/sachin.chaudhary/ICLR_2027/fairforge2/masters/real/ffhq1024/%s.webp' % r.source_ids.split(':')[1], 0, 'real') for r in reals.itertuples()]
fin = pd.read_pickle(R.OUT + '/_fin_trval_PRIVATE.pkl')
fk = fin[(fin.sp == 'val') & (fin.src == 'ffhq') & (fin.profile == 'v1') & fin.master_path.notna()].groupby('method').sample(n=6, random_state=1)
rows = R.run(items, 'ffhq_real') + R.run([(r.fake_path, r.master_path, 1, r.method) for r in fk.itertuples()], 'ffhq_fake')
d = pd.DataFrame(rows); d = d[d.err.isna()] if 'err' in d else d
d.to_pickle(R.OUT + '/_recrop_scores_jpegsplit_PRIVATE.pkl')
S = {}
for kind, g in d.groupby('kind'):
    S[kind] = {'n': int(len(g))}
    for q, s in CFG:
        a = f'G_q{q}_{s}'; S[kind][a] = dict(flag_fake_rate=float((g[a] >= R.THR).mean()), median_logit=float(g[a].median()))
json.dump(S, open(R.OUT + '/recrop_jpegsplit_summary.json', 'w'), indent=1); print(json.dumps(S, indent=1))
