# Independent recomputation of per-image luminance mean/std, Laplacian variance, HSV saturation (DFDC).
import numpy as np, pandas as pd
from PIL import Image
from multiprocessing import Pool
S="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/runs/BD0_s7/eval/dfdc_scores.npz"
def f(p):
    im=np.array(Image.open(p).convert("RGB")).astype(np.float64)
    L=im@np.array([0.299,0.587,0.114])
    lap=-4*L[1:-1,1:-1]+L[:-2,1:-1]+L[2:,1:-1]+L[1:-1,:-2]+L[1:-1,2:]
    mx=im.max(2); mn=im.min(2)
    s=np.divide(mx-mn,mx,out=np.zeros_like(mx),where=mx>0)
    return p,L.mean(),L.std(),lap.var(),s.mean(),im.shape[0],im.shape[1]
if __name__=="__main__":
    paths=sorted(set(np.load(S,allow_pickle=True)["img_path"].astype(str)))
    with Pool(8) as pool: r=pool.map(f,paths,chunksize=64)
    pd.DataFrame(r,columns=["img_path","lum","lstd","lap","sat","h","w"]).to_csv(__file__.replace("imgstats.py","imgstats_mine.csv"),index=False)
    print("done",len(r))
