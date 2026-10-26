import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
import pandas as pd
from unittest.mock import patch
from cfes.metrics import rank_prob, threshold_at_tpr, fairness_block
import fair_a1 as A
import train as T

# Validation threshold must denote one fixed logit, even if test extrema differ.
y = np.array([0,0,1,1]); zv = np.array([-2.,-1.,1.,2.]); zt = np.array([-100.,-1.,1.,2.])
t = threshold_at_tpr(y, rank_prob(zv), .9)
wrong = rank_prob(zt) >= t
correct = zt >= threshold_at_tpr(y, zv, .9)
assert not np.array_equal(wrong, correct)
print(f'REPRO rank threshold: val threshold={t:.6f}, test decisions={wrong.tolist()}, fixed-logit decisions={correct.tolist()}')
# Production validate() ignores rank_metrics and ranks sigmoid-saturated scores.
y = np.tile([0,0,1,1], 2); g = np.repeat([0,1], 4)
z = np.tile([-50.,40.,41.,42.],2)
p = 1/(1+np.exp(-z))
cfg = {'optim':{'amp':False}, 'eval':{'prob_float64':True,'rank_metrics':True,'val_tpr':.9}, 'data':{'val':{'scheme':'eight'}}}
with patch.object(T.engine, 'predict', return_value=(p,z)):
    res = T.validate(None,cfg,None,pd.DataFrame({'label':y,'group':g}),None)
exact = 100*fairness_block(y,rank_prob(z),g,'eight',.5)['overall']['auc']
assert res['auc'] != exact
print(f"REPRO validation saturation: reported AUC={res['auc']}, logit/rank AUC={exact}")
# Independent direct definitions and exhaustive A1 grid + tie-break comparison.
rng = np.random.default_rng(5)
y = np.tile([0,1],80); g = np.repeat(np.arange(8),20); z = rng.normal(size=160)+3*y
f = fairness_block(y,rank_prob(z),g,'eight',.5)
pred = z>=0; pooled=pred[y==0].mean()
expected=100*sum(abs(pred[(g==k)&(y==0)].mean()-pooled) for k in range(8))
assert np.isclose(f['intersection']['F_FPR'],expected)
assert np.isclose(f['intersection']['F_DP'],100*np.ptp([pred[g==k].mean() for k in range(8)]))
from sklearn.metrics import roc_auc_score
auc=roc_auc_score(y,z)
es=100*auc/(1+sum(abs(roc_auc_score(y[g==k],z[g==k])-auc) for k in range(8)))
assert np.isclose(f['intersection']['es_AUC'],es)
u,fr,tr,ff=A.val_curves(z,y,g,'eight'); j=A.pick_a1(u,fr,tr,ff)
manual=[]
for t0 in np.unique(z):
    d=z>=t0; fp=d[y==0].mean(); tp=d[y==1].mean()
    if tp>=.8 and fp<=.2:
        gap=100*sum(abs(d[(y==0)&(g==k)].mean()-fp) for k in range(8))
        manual.append((round(gap,9),-tp,-t0,t0))
assert np.isclose(u[j],min(manual)[-1])
print('PASS F_FPR, F_DP, es-AUC formulas and A1 exhaustive threshold/tie-break oracle')
rec=A.a1_for((z,y,g),{'ff2_test':(z,y,g)})
assert rec['t'] is not None and 'ff2_test' not in rec['test']
print('REPRO FF2 A1 omission: input ff2_test is silently absent from A1 test output')
# All four removed cues should leave a constant; saved float32 summation errors instead rank images.
sk=np.array([[.1,.2,.3,.4],[.1,.2,.3,.4],[1.,2.,3.,4.],[1.,2.,3.,4.]],dtype=np.float32)
s=sk.sum(1).astype(np.float64); residual=s-sk.astype(np.float64).sum(1)
assert np.ptp(residual)>0
print(f'REPRO full deletion numerical residual: {residual.tolist()}, AUC={roc_auc_score([0,0,1,1],residual)} (exact constant: 0.5)')
