"""Fixed-score, contributor-component bootstrap; never calibrates or selects."""
import numpy as np


def components(rows):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    anchors = []
    for row in rows:
        tokens = [field+':'+v for field in ('identity_ids', 'source_ids') for v in row[field].split(';') if v]
        anchor = find(tokens[0])
        for token in tokens[1:]:
            parent[find(token)] = find(anchor)
        anchors.append(tokens[0])
    names = [find(x) for x in anchors]
    mapping = {name: i for i, name in enumerate(sorted(set(names)))}
    return np.asarray([mapping[x] for x in names], dtype=int)


class WeightedMetric:
    def __init__(self, y, z, threshold, mask):
        self.indices = np.flatnonzero(mask)
        order = np.argsort(z[self.indices], kind='stable')
        self.indices = self.indices[order]
        self.y = y[self.indices]
        self.pred = z[self.indices] >= threshold
        values = z[self.indices]
        self.starts = np.r_[0, 1+np.flatnonzero(np.diff(values))] if len(values) else np.array([], dtype=int)

    def calculate(self, weights):
        w = weights[self.indices]
        positive, negative = w*self.y, w*(1-self.y)
        np_, nn = positive.sum(), negative.sum()
        if np_ == 0 or nn == 0:
            return None
        p = np.add.reduceat(positive, self.starts)
        n = np.add.reduceat(negative, self.starts)
        auc = np.dot(p, np.cumsum(n)-n+0.5*n)/(np_*nn)
        return {'auc': float(auc), 'fpr': float(np.dot(negative, self.pred)/nn),
                'fnr': float(np.dot(positive, ~self.pred)/np_),
                'positive_rate': float(np.dot(w, self.pred)/(np_+nn))}


def bootstrap(rows, logits, threshold, axes, reps=1000, seed=17, min_support=20):
    ids = components(rows)
    count = int(ids.max()+1)
    base = {'method': 'percentile bootstrap of connected recorded-contributor components',
            'replicates': reps, 'seed': seed, 'components': count,
            'limits': 'Single-seed dataset uncertainty; recorded identity/video proxies and hidden donor connections do not establish true person independence.'}
    if count < 2:
        return {**base, 'intervals': None, 'reason': 'Recorded shared contributors connect all rows; fewer than two independent recorded components.'}
    y = np.array([r['label'] for r in rows])
    z = np.asarray(logits)
    overall = WeightedMetric(y, z, threshold, np.ones(len(y), dtype=bool))
    groups = {}
    for axis in axes:
        labels = np.array([r.get(axis, 'unknown') for r in rows])
        good = []
        for name in np.unique(labels):
            mask = labels == name
            if name != 'unknown' and min(np.sum(mask & (y == 0)), np.sum(mask & (y == 1))) >= min_support:
                good.append(WeightedMetric(y, z, threshold, mask))
        if len(good) > 1:
            groups[axis] = good
    values = {}
    def add(key, value):
        values.setdefault(key, []).append(float(value))
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        multiplicity = np.bincount(rng.integers(count, size=count), minlength=count)
        weights = multiplicity[ids]
        pooled = overall.calculate(weights)
        if pooled is None:
            continue
        for key in ('auc', 'fpr', 'fnr'):
            add('overall.'+key, pooled[key])
        for axis, members in groups.items():
            result = [m.calculate(weights) for m in members]
            if any(m is None for m in result):
                continue
            for key in ('auc', 'fpr', 'fnr'):
                a = [m[key] for m in result]
                add(axis+'.'+key+'_gap', max(a)-min(a))
            add(axis+'.worst_group_auc', min(m['auc'] for m in result))
            add(axis+'.F_FPR', sum(abs(m['fpr']-pooled['fpr']) for m in result))
            add(axis+'.F_DP', max(m['positive_rate'] for m in result)-min(m['positive_rate'] for m in result))
            add(axis+'.es_AUC', pooled['auc']/(1+sum(abs(m['auc']-pooled['auc']) for m in result)))
    return {**base, 'few_components_warning': count < 20,
            'eligibility': 'Original-sample min_support fixed; replicates with undefined class metrics are omitted explicitly',
            'intervals': {k: {'low': float(np.quantile(v, .025)), 'high': float(np.quantile(v, .975)),
                              'valid_replicates': len(v)} for k, v in values.items()}}
