# Feasibility only (CPU): face-parsing class fractions inside the L_H intervention support, for the 22 preview
# sources and a group-stratified TRAIN-real sample. No training, no VAL/TEST access.
import json, sys, random, csv, numpy as np, torch
from PIL import Image
from collections import deque
from facexlib.parsing import init_parsing_model
torch.set_num_threads(8)
net = init_parsing_model(model_name='parsenet', device='cpu', model_rootpath='/home/sachin.chaudhary/gfpgan/weights')
FACE = {1, 2, 3, 4, 5, 6, 10, 11, 12, 13}   # skin, brows, eyes, eyeglasses, nose, mouth, lips
OCCL = {0, 7, 8, 9, 14, 15, 16, 17, 18}     # background, ears, earring, neck, necklace, cloth, hair, hat
def support(h, w, cx=0.0, cy=-0.035, rx=0.475, ry=0.575):
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing='ij')
    d = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    return d <= 1.0
def union_support(h, w):  # union over the operator's random centre range at max radii, plus max warp shift
    m = np.zeros((h, w), bool)
    for cx in (-.10, 0, .10):
        for cy in (-.12, -.035, .05):
            m |= support(h, w, cx, cy, .55 + .045, .65 + .035)
    return m
def padding_mask(a, thr=6):
    dark = a.max(2) <= thr; h, w = dark.shape; seen = np.zeros_like(dark); q = deque()
    for y in range(h):
        for x in (0, w - 1):
            if dark[y, x] and not seen[y, x]: seen[y, x] = 1; q.append((y, x))
    for x in range(w):
        for y in (0, h - 1):
            if dark[y, x] and not seen[y, x]: seen[y, x] = 1; q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy < h and 0 <= xx < w and dark[yy, xx] and not seen[yy, xx]: seen[yy, xx] = 1; q.append((yy, xx))
    return seen
def measure(path):
    im = Image.open(path).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR); a = np.asarray(im)
    x = torch.from_numpy(np.asarray(im.resize((512, 512), Image.Resampling.BILINEAR), np.float32) / 255.).permute(2, 0, 1)[None]
    x = (x - 0.5) / 0.5
    with torch.no_grad(): lab = net(x)[0].argmax(1)[0].numpy()
    lab = np.asarray(Image.fromarray(lab.astype(np.uint8)).resize((224, 224), Image.Resampling.NEAREST))
    core = support(224, 224); uni = union_support(224, 224)
    f = lambda s: float(np.isin(lab[core], list(s)).mean())
    return {'face_frac': f(FACE), 'occl_frac': f(OCCL), 'hat_frac': float((lab[core] == 18).mean()),
            'cloth_frac': float((lab[core] == 16).mean()), 'pad_in_union': float(padding_mask(a)[uni].mean())}
if __name__ == '__main__':
    out = {}
    pv = json.load(open('/home/sachin.chaudhary/ces_fl_runs/ff2_lh_priority_20261104_r1/preview/preview_PRIVATE.json'))
    rv = {c['id']: c for c in json.load(open('/home/sachin.chaudhary/ces_fl_runs/ff2_lh_priority_20261104_r1/review_reused_exact_preview.json'))['cases']}
    seen = {}
    for c in pv['cases']:
        src = c['id'].rsplit('_', 2)[0]
        seen.setdefault(src, {'image': c['image'], 'groups': c['groups'], 'approved_all': True})
        seen[src]['approved_all'] &= bool(rv[c['id']]['approved'])
    for s, v in seen.items(): v.update(measure(v['image']))
    out['preview_sources'] = seen
    rows = [r for r in csv.DictReader(open('/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/train.csv')) if r['label'] == '0']
    by = {}
    for r in rows: by.setdefault(r['gender_race'], []).append(r)
    rng = random.Random(17); samp = []
    for g, rs in sorted(by.items()):
        if g.lower() == 'unknown': continue
        for r in rng.sample(rs, min(60, len(rs))): samp.append((g, r['image'], r.get('source_ids', '')))
    out['sample'] = [{'group': g, 'src': s.split(':')[0], **measure(p)} for g, p, s in samp]
    json.dump(out, open('feasibility_PRIVATE.json', 'w'))
    for s, v in seen.items(): print(s, v['approved_all'], {k: round(v[k], 3) for k in ('face_frac', 'occl_frac', 'hat_frac', 'cloth_frac', 'pad_in_union')})
