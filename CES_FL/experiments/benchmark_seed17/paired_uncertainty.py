"""Frozen-score paired seed/component bootstrap, including individual groups.

No calibration, checkpoint selection or data splitting. Intervals remain
conditional on observed seeds and recorded contributors, not true-person proof.
"""
import itertools
import numpy as np
from uncertainty import components


class Metric:
    def __init__(self, y, z, threshold, mask):
        ids = np.flatnonzero(mask)
        ids = ids[np.argsort(z[ids], kind='stable')]
        self.ids, self.y, self.pred = ids, y[ids], z[ids] >= threshold
        self.starts = np.r_[0, 1+np.flatnonzero(np.diff(z[ids]))] if len(ids) else np.array([], dtype=int)

    def calculate(self, weights):
        w = weights[self.ids]; p, n = w*self.y, w*(1-self.y)
        np_, nn = p.sum(), n.sum(); total = np_+nn
        auc = None
        if np_ > 0 and nn > 0:
            pp, nr = np.add.reduceat(p, self.starts), np.add.reduceat(n, self.starts)
            auc = float(np.dot(pp, np.cumsum(nr)-nr+.5*nr)/(np_*nn))
        return {'auc': auc, 'fpr': float(np.dot(n,self.pred)/nn) if nn>0 else None,
                'fnr': float(np.dot(p,~self.pred)/np_) if np_>0 else None,
                'positive_rate': float(np.dot(w,self.pred)/total) if total>0 else None}


def signflip_p(differences):
    d = np.asarray(differences,dtype=float)
    if not len(d) or not np.isfinite(d).all(): raise ValueError('Invalid paired seed differences')
    reference = abs(d.mean())
    return sum(abs(np.mean(d*np.asarray(s)))>=reference-1e-14 for s in itertools.product([-1,1],repeat=len(d))) / (2**len(d))


def paired_bootstrap(rows, a, b, threshold_a, threshold_b, axes, *, reps=2000, seed=1729, min_support=20):
    seeds = sorted(a)
    if seeds!=sorted(b) or seeds!=sorted(threshold_a) or seeds!=sorted(threshold_b) or len(seeds)<3:
        raise ValueError('At least three matched seeds/thresholds required')
    if any(not np.isfinite(v) for v in [*threshold_a.values(),*threshold_b.values()]):
        raise ValueError('Frozen thresholds must be finite')
    if reps<1 or not rows: raise ValueError('Empty data/bootstrap')
    y = np.asarray([r['label'] for r in rows]); ids=components(rows); nc=int(ids.max()+1)
    if set(y)!={0,1}: raise ValueError('Both original classes required')
    if any(np.asarray(s).shape!=(len(rows),) or not np.isfinite(s).all() for s in [*a.values(),*b.values()]):
        raise ValueError('Score alignment/finite check failed')
    groups, supports = {}, {}
    for axis in axes:
        labels=np.asarray([r.get(axis,'unknown') for r in rows]);eligible=[]
        for name in sorted(set(labels)):
            mask=labels==name; nr=int(sum(mask&(y==0)));nf=int(sum(mask&(y==1)))
            supports[axis+'.'+name]={'real':nr,'fake':nf,'real_components':len(set(ids[mask&(y==0)])),
                                    'fake_components':len(set(ids[mask&(y==1)])),
                                    'eligible_both_classes':name!='unknown' and min(nr,nf)>=min_support}
            if supports[axis+'.'+name]['eligible_both_classes']:eligible.append(name)
        groups[axis]=(labels,eligible)
    models={}
    for method,scores,thresholds in [('a',a,threshold_a),('b',b,threshold_b)]:
        for s in seeds:
            metrics={'overall':Metric(y,np.asarray(scores[s]),thresholds[s],np.ones(len(y),dtype=bool))}
            for axis,(labels,_) in groups.items():
                for name in sorted(set(labels)):
                    metrics[axis+'.'+name]=Metric(y,np.asarray(scores[s]),thresholds[s],labels==name)
            models[(method,s)]=metrics
    def report(method,s,weights):
        raw={k:m.calculate(weights) for k,m in models[(method,s)].items()}
        result={f'{k}.{metric}':v for k,m in raw.items() for metric,v in m.items() if metric!='positive_rate'}
        for axis,(_,eligible) in groups.items():
            values=[raw[axis+'.'+g] for g in eligible];overall=raw['overall']
            for metric in ['auc','fpr','fnr']:
                valid=len(values)>1 and all(v[metric] is not None for v in values)
                result[axis+'.'+metric+'_gap']=max(v[metric] for v in values)-min(v[metric] for v in values) if valid else None
            valid_auc=bool(values) and all(v['auc'] is not None for v in values)
            result[axis+'.worst_group_auc']=min(v['auc'] for v in values) if valid_auc else None
            result[axis+'.es_AUC']=overall['auc']/(1+sum(abs(v['auc']-overall['auc']) for v in values)) if valid_auc and len(values)>1 and overall['auc'] is not None else None
            result[axis+'.F_FPR']=sum(abs(v['fpr']-overall['fpr']) for v in values) if len(values)>1 and overall['fpr'] is not None and all(v['fpr'] is not None for v in values) else None
            result[axis+'.F_DP']=max(v['positive_rate'] for v in values)-min(v['positive_rate'] for v in values) if len(values)>1 and all(v['positive_rate'] is not None for v in values) else None
        return result
    fixed={s:{m:report(m,s,np.ones(len(rows))) for m in ['a','b']} for s in seeds}
    differences={key:[fixed[s]['b'][key]-fixed[s]['a'][key] for s in seeds]
                 for key in fixed[seeds[0]]['a'] if all(fixed[s][m][key] is not None for s in seeds for m in ['a','b'])}
    if nc<2:
        return {'intervals':None,'reason':'Fewer than two recorded contributor components','components':nc,
                'supports':supports,'paired_seed_differences':differences}
    rng=np.random.default_rng(seed); samples={}
    for _ in range(reps):
        seed_draw=rng.choice(seeds,size=len(seeds),replace=True)
        weights=np.bincount(rng.integers(nc,size=nc),minlength=nc)[ids]
        # Identical seed/component draws for both methods; fixed thresholds.
        reports={s:{m:report(m,int(s),weights) for m in ['a','b']} for s in set(seed_draw)}
        for key in fixed[seeds[0]]['a']:
            if all(reports[s][m][key] is not None for s in seed_draw for m in ['a','b']):
                samples.setdefault('delta.'+key,[]).append(np.mean([reports[s]['b'][key]-reports[s]['a'][key] for s in seed_draw]))
            for method in ['a','b']:
                if all(reports[s][method][key] is not None for s in seed_draw):
                    samples.setdefault(method+'.'+key,[]).append(np.mean([reports[s][method][key] for s in seed_draw]))
    return {'method':'paired hierarchical seed and connected-contributor percentile bootstrap',
            'components':nc,'few_components':nc<20,'seeds':seeds,'replicates':reps,'supports':supports,
            'thresholds_frozen':True,'paired_seed_differences':differences,
            'exact_two_sided_signflip_p':{k:signflip_p(v) for k,v in differences.items()},
            'intervals':{k:{'low':float(np.quantile(v,.025)),'high':float(np.quantile(v,.975)),
                            'valid_replicates':len(v),'invalid_replicates':reps-len(v)} for k,v in samples.items()},
            'limitations':'Conditional on observed seeds and recorded source graph; hidden donors, label uncertainty and prospective holdout validity are not established.'}
