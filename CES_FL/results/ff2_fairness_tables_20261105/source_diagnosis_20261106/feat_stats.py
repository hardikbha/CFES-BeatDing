# Low-level image statistics on FF2 TRAIN/VAL crops (read-only). Writes per-image feature table (private, not for report).
import sys, os, io, hashlib
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np, pandas as pd, cv2
from PIL import Image, JpegImagePlugin
from multiprocessing import Pool
cv2.setNumThreads(1)
OUT = '/home/sachin.chaudhary/ces_fl_runs/ff2_source_diag_20261106'
MAN = '/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/%s.csv'
N = 256
yy, xx = np.mgrid[:N, :N]
R = np.hypot(np.minimum(yy, N - yy), np.minimum(xx, N - xx)) / (N / 2)  # 0..~1.41, 1 = Nyquist on axis
BINS = np.linspace(0, 1.0, 17)
HAN = np.outer(np.hanning(N), np.hanning(N))


def radial(ch):
    f = np.abs(np.fft.fft2((ch - ch.mean()) * HAN)) ** 2
    out = []
    for a, b in zip(BINS[:-1], BINS[1:]):
        m = (R >= a) & (R < b)
        out.append(f[m].mean())
    out = np.asarray(out); return out / out.sum()


def noise_sigma(g):  # Immerkaer 1996
    K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float64)
    c = cv2.filter2D(g, cv2.CV_64F, K)[1:-1, 1:-1]
    return np.sqrt(np.pi / 2) * np.abs(c).mean() / 6.0


def blockiness(g, off=0):
    d = np.abs(np.diff(g, axis=1)); e = np.abs(np.diff(g, axis=0))
    cols = np.arange(d.shape[1]); rows = np.arange(e.shape[0])
    b = (d[:, (cols % 8) == (7 + off) % 8].mean() + e[(rows % 8) == (7 + off) % 8, :].mean())
    nb = (d[:, (cols % 8) != (7 + off) % 8].mean() + e[(rows % 8) != (7 + off) % 8, :].mean())
    return b / max(nb, 1e-9)


def feats(path):
    try:
        data = open(path, 'rb').read()
        im = Image.open(io.BytesIO(data))
        qt = im.quantization; samp = JpegImagePlugin.get_sampling(im)
        qh = hashlib.md5(repr(sorted(qt.items())).encode()).hexdigest()[:8]
        rgb = np.asarray(im.convert('RGB'), np.float64)
        if rgb.shape[:2] != (N, N):
            rgb = cv2.resize(rgb, (N, N), interpolation=cv2.INTER_AREA)
        ycc = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb).astype(np.float64)
        Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
        ry = radial(Y); rc = 0.5 * (radial(Cr) + radial(Cb))
        hsv = cv2.cvtColor(rgb.astype(np.float32) / 255., cv2.COLOR_RGB2HSV)
        lap = cv2.Laplacian(Y, cv2.CV_64F).var()
        # chroma vs luma high-frequency (4:2:0 sources lose chroma HF)
        def hf(ch):
            f = np.abs(np.fft.fft2((ch - ch.mean()) * HAN)) ** 2
            return f[R >= 0.5].sum() / max(f[R > 0.02].sum(), 1e-9)
        hfy, hfc = hf(Y), 0.5 * (hf(Cr) + hf(Cb))
        # border/padding: near-black fraction, constant rows/cols at the edges (replicate streaks)
        dark = float((rgb.max(-1) < 8).mean())
        edge_flat = float(np.mean([np.std(Y[:, :4], 1).mean() < 0.5, np.std(Y[:, -4:], 1).mean() < 0.5,
                                   np.std(Y[:4, :], 0).mean() < 0.5, np.std(Y[-4:, :], 0).mean() < 0.5]))
        # local-variance based: fraction of 8x8 blocks that are nearly flat (upsampled / smooth sources)
        blk = Y.reshape(32, 8, 32, 8).std((1, 3))
        flat = float((blk < 1.0).mean())
        r = dict(bytes=len(data), q_hash=qh, sampling=samp, w=im.size[0],
                 log_lap=np.log(lap + 1e-6), noise=noise_sigma(Y), hf_y=np.log(hfy + 1e-12), hf_c=np.log(hfc + 1e-12),
                 chroma_ratio=np.log(hfc + 1e-12) - np.log(hfy + 1e-12), block8=blockiness(Y),
                 sat=float(hsv[..., 1].mean()), val=float(hsv[..., 2].mean()), y_std=float(Y.std()),
                 cr_mean=float(Cr.mean()), cb_mean=float(Cb.mean()), dark=dark, edge_flat=edge_flat, flat8=flat)
        for i, v in enumerate(ry): r[f'ry{i:02d}'] = np.log(v + 1e-12)
        for i, v in enumerate(rc): r[f'rc{i:02d}'] = np.log(v + 1e-12)
        return r
    except Exception as e:
        return dict(err=str(e)[:100])


def sample(split, seed):
    d = pd.read_csv(MAN % split, usecols=['image', 'label', 'source_ids', 'method', 'family', 'race'])
    d['src'] = d.source_ids.astype(str).str.split(':').str[0]
    rng = np.random.default_rng(seed)
    if split == 'val':
        parts = [d[d.label == 0], d[(d.label == 1) & (d.src == 'ffhq')],
                 d[(d.label == 1) & (d.src == 'img')].sample(6000, random_state=seed)]
    else:
        parts = [d[(d.label == 0) & (d.src == 'ffhq')], d[(d.label == 0) & (d.src == 'img')].sample(6000, random_state=seed),
                 d[(d.label == 1) & (d.src == 'ffhq')].sample(8000, random_state=seed),
                 d[(d.label == 1) & (d.src == 'img')].sample(12000, random_state=seed)]
    s = pd.concat(parts).copy(); s['split'] = split; return s


if __name__ == '__main__':
    s = pd.concat([sample('val', 1), sample('train', 2)]).reset_index(drop=True)
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as p:
        F = p.map(feats, s.image.tolist(), chunksize=64)
    f = pd.DataFrame(F)
    out = pd.concat([s.reset_index(drop=True), f], axis=1)
    out.to_pickle(OUT + '/_features_PRIVATE.pkl')
    print(out.groupby(['split', 'label', 'src']).size()); print('errors', out.get('err', pd.Series()).notna().sum())
