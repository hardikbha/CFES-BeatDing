# CPU census (TRAIN reals only): fraction of the L_H probe support that falls on artificial black padding, per group/source.
import csv, json, numpy as np
from multiprocessing import Pool
from PIL import Image
from scipy import ndimage
def support(h, w, cx, cy, rx, ry):
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing='ij')
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
U = np.zeros((224, 224), bool)
for cx in (-.10, 0, .10):
    for cy in (-.12, -.035, .05):
        U |= support(224, 224, cx, cy, .595, .685)
def padding(a, thr=6):
    dark = a.max(2) <= thr; lab, n = ndimage.label(dark)
    edge = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    return np.isin(lab, list(edge))
def one(r):
    a = np.asarray(Image.open(r[0]).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR))
    p = padding(a); pd = ndimage.binary_dilation(p, iterations=6)
    return {'group': r[1], 'src': r[2], 'pad_union': float(p[U].mean()), 'clip_loss': float(pd[U].mean())}
if __name__ == '__main__':
    rows = [(r['image'], r['gender_race'], r['source_ids'].split(':')[0]) for r in csv.DictReader(open('/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/train.csv')) if r['label'] == '0']
    with Pool(24) as pool: res = pool.map(one, rows, chunksize=64)
    json.dump(res, open('padding_census_PRIVATE.json', 'w'))
    import pandas as pd
    d = pd.DataFrame(res); d['any'] = d.pad_union > 0; d['gt10'] = d.pad_union > 0.10
    print(len(d)); print(d.groupby('src')[['any', 'gt10', 'clip_loss']].mean().round(4))
    print(d.groupby('group')[['any', 'gt10', 'clip_loss']].mean().round(4))
