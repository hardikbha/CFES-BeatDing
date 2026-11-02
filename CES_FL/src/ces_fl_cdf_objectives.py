"""New proposal only: real-score CDF parity and controlled response objectives.

No changes to historical CES-FL losses. TRAIN-only score knots are detached;
validation/test scores must never be arguments to this training objective.
Smooth empirical CDF alignment is a surrogate, not an FPR/fairness guarantee.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ('bce', 'dro', 'mean_fair', 'cdf_fair', 'nochange', 'score', 'response')


def validate(z, y, groups, n_groups):
    if z.ndim != 1 or z.shape != y.shape or z.shape != groups.shape or not len(z):
        raise ValueError('Scores/labels/groups must be aligned nonempty vectors')
    if z.device != y.device or z.device != groups.device or groups.dtype != torch.long:
        raise ValueError('All tensors must share device; group IDs must be long')
    if not torch.isfinite(z).all() or not ((y == 0) | (y == 1)).all():
        raise ValueError('Nonfinite score or nonbinary label')
    if n_groups < 2 or groups.min() < 0 or groups.max() >= n_groups:
        raise ValueError('Invalid declared group inventory')


def real_cdf_parity(z, y, groups, priors, temperature=.25, min_count=2):
    """Max over pooled TRAIN-real knots of mean absolute group survival gaps.

    At any fixed t, FPR_g(t)=P(z>=t|real,g). Population equality of the entire
    real-score distributions implies equality at any single global t, including
    a later validation-calibrated t. Finite minibatches and smoothing do NOT
    establish population equality. This function never consumes that VAL t.
    priors are fixed TRAIN-real group proportions, not minibatch frequencies.
    """
    validate(z, y, groups, len(priors))
    if not math.isfinite(temperature) or temperature <= 0 or min_count < 1:
        raise ValueError('Invalid CDF temperature/support')
    if (priors.device != z.device or priors.ndim != 1 or priors.requires_grad
            or not torch.isfinite(priors).all() or (priors <= 0).any()
            or not torch.isclose(priors.sum(), priors.new_tensor(1.))):
        raise ValueError('Priors must be fixed positive TRAIN proportions summing to one')
    real = y == 0
    counts = torch.stack([(real & (groups == g)).sum() for g in range(len(priors))])
    if (counts < min_count).any():
        raise ValueError('Every declared real group needs minimum batch support; do not drop groups')
    knots = torch.unique(z[real].detach(), sorted=True)
    rates = torch.stack([torch.sigmoid((z[real & (groups == g), None] - knots[None]) / temperature).mean(0)
                         for g in range(len(priors))])
    pooled = (priors[:, None] * rates).sum(0)
    per_knot = (rates - pooled[None]).abs().mean(0)
    loss = per_knot.max()
    return loss, {'knots': knots, 'soft_group_survival': rates, 'soft_pooled_survival': pooled,
                  'real_counts': counts, 'per_knot_mean_abs_gap': per_knot}


def mean_score_parity(z, y, groups, n_groups, temperature=1.):
    validate(z, y, groups, n_groups)
    real = y == 0
    if any(not (real & (groups == g)).any() for g in range(n_groups)):
        raise ValueError('Missing declared real group')
    rates = torch.stack([torch.sigmoid(z[real & (groups == g)] / temperature).mean() for g in range(n_groups)])
    return (rates - torch.sigmoid(z[real] / temperature).mean()).abs().mean()


def manipulation_terms(q, margin=.5):
    """Original r; matched preprocessing xr; x0,xa,T(x0),T(xa).

    The response-only term is invariant to additive offsets shared by the two
    scores in each operation response. It is never claimed to anchor scores.
    """
    if set(q) != {'original', 'resize', 'x0', 'xa', 't0', 'ta'}:
        raise ValueError('Six complete, aligned views required in every paired arm')
    reference = q['original']
    if reference.ndim != 1 or not reference.numel():
        raise ValueError('Empty/malformed paired pool')
    if any(v.shape != reference.shape or v.device != reference.device or not torch.isfinite(v).all() for v in q.values()):
        raise ValueError('Paired tensor alignment/device/finite check failed')
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError('Invalid response margin')
    r0, ra = torch.asinh(q['t0'] - q['x0']), torch.asinh(q['ta'] - q['xa'])
    return {'nochange': (torch.asinh(reference) - torch.asinh(q['x0'])).abs().mean(),
            'score': (torch.asinh(q['x0']) - torch.asinh(q['xa'])).abs().mean(),
            'alignment': (r0 - ra).abs().mean(),
            'sensitivity': (F.relu(margin-r0) + F.relu(margin-ra)).mean()/2,
            'resize_diagnostic': (reference - q['resize']).abs().mean(),
            'response0': r0, 'responsea': ra}


class CompleteGroupDRO(nn.Module):
    """Standard exponentiated q on label×group BCE; every cell must be present."""
    def __init__(self, n_cells, eta=.01):
        super().__init__()
        if n_cells < 2 or not math.isfinite(eta) or eta <= 0:
            raise ValueError('Invalid DRO settings')
        self.eta = eta
        self.register_buffer('log_q', torch.full((n_cells,), -math.log(n_cells)))

    def forward(self, losses, cells):
        if (losses.ndim != 1 or not losses.numel() or losses.shape != cells.shape or cells.dtype != torch.long
                or losses.device != cells.device or cells.device != self.log_q.device
                or not torch.isfinite(losses).all() or cells.min() < 0 or cells.max() >= len(self.log_q)):
            raise ValueError('Malformed DRO cells/losses')
        counts = torch.bincount(cells, minlength=len(self.log_q))
        if (counts == 0).any():
            raise ValueError('Missing label×group cell; quota sampler required')
        means = losses.new_zeros(len(counts)).index_add(0, cells, losses) / counts
        with torch.no_grad():
            self.log_q.add_(self.eta * means.detach())
            self.log_q.sub_(torch.logsumexp(self.log_q, dim=0))
        return (self.log_q.exp() * means).sum()


def objective(z, y, groups, priors, arm, *, lambda_cdf=.1, lambda_mean=.1,
              lambda_pair=.1, lambda_response=.1, temperature=.25, margin=.5,
              q=None, dro=None):
    validate(z, y, groups, len(priors))
    if arm not in ARMS:
        raise ValueError('Unknown arm')
    if any(not math.isfinite(x) or x < 0 for x in [lambda_cdf, lambda_mean, lambda_pair, lambda_response]):
        raise ValueError('Invalid loss weight')
    zero = z.sum()*0
    sample = F.binary_cross_entropy_with_logits(z, y.float(), reduction='none')
    cls = sample.mean()
    if arm == 'dro':
        if dro is None:
            raise ValueError('Missing persistent DRO state')
        cls = dro(sample, y.long()*len(priors)+groups)
    cdf, mean, pair, response = zero, zero, zero, zero
    if arm == 'mean_fair':
        mean = mean_score_parity(z, y, groups, len(priors))
    if arm in ('cdf_fair', 'nochange', 'score', 'response'):
        cdf, _ = real_cdf_parity(z, y, groups, priors, temperature)
    if arm in ('nochange', 'score', 'response'):
        terms = manipulation_terms(q, margin)
        pair = terms['nochange'] if arm == 'nochange' else terms['score']
        if arm == 'response':
            response = terms['alignment'] + terms['sensitivity']
    return {'total': cls + lambda_cdf*cdf + lambda_mean*mean + lambda_pair*pair + lambda_response*response,
            'classification': cls, 'cdf': cdf, 'mean': mean, 'pair': pair, 'response': response}


def binary_logit(two_logits):
    if two_logits.ndim != 2 or two_logits.shape[1] != 2 or not torch.isfinite(two_logits).all():
        raise ValueError('Ding score adapter requires finite [N,2] real/fake logits')
    return two_logits[:, 1] - two_logits[:, 0]


def corrected_ding_cc_ot(two_logits, y, groups, n_groups, *, epsilon=5e-4, min_count=2):
    """Ding-CC-OT-r1, explicitly distinct from archived GDA and paper-run claims.

    Class-conditioned fake-probability distributions; differentiable entropic OT,
    GeomLoss0.3.1, p=2, blur=sqrt(epsilon), debias=False, tensorized backend.
    Raw OT (not debiased divergence) matches the declared regularized cost choice.
    No KDE is added; this remains a disclosed empirical-distribution adaptation.
    """
    if not torch.is_grad_enabled() or not two_logits.requires_grad:
        raise ValueError('Corrected GDA must run with gradients enabled on detector logits')
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('Invalid entropic regularization')
    z = binary_logit(two_logits)
    validate(z, y, groups, n_groups)
    try:
        from geomloss import SamplesLoss
    except ImportError as error:
        raise RuntimeError('Corrected Ding baseline requires separately tested GeomLoss0.3.1') from error
    # Fake probabilities live in [0,1]. A fixed nonzero diameter avoids the
    # legacy GeomLoss log(0) schedule failure on a constant-score classifier.
    criterion = SamplesLoss('sinkhorn', p=2, blur=math.sqrt(epsilon), diameter=1., debias=False, backend='tensorized')
    terms = {}
    with torch.enable_grad():
        probabilities = torch.sigmoid(z)
        for label in [0, 1]:
            pool = probabilities[y == label, None]
            distances = []
            for group in range(n_groups):
                mask = (y == label) & (groups == group)
                if mask.sum() < min_count:
                    raise ValueError('Missing class-conditioned group support')
                distances.append(criterion(probabilities[mask, None], pool))
            terms[label] = torch.stack(distances).mean()
        # Eq6 sums the two class contributions and averages over groups.
        total = terms[0] + terms[1]
    return total, terms
