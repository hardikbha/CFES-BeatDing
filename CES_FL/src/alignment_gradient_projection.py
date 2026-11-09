"""One-sided raw-gradient projection for a bounded L_H optimization experiment.

This is PCGrad-style gradient handling, not a new loss or a utility guarantee.
Only the weighted alignment gradient is projected relative to original BCE.
AdamW momentum/preconditioning, clipping, other losses and validation shift mean
nonnegative raw-gradient dot products do NOT guarantee AUC/FNR preservation.
"""
from contextlib import contextmanager
import math
import time

import torch

VERSION = 'LH_ALIGNMENT_PROJECTION_v1'


def project_alignment(total, bce, alignment, enabled=True):
    """Return combined gradients plus scalar diagnostics; never modify inputs.

    total is the native total gradient, INCLUDING the weighted alignment term.
    For dot(a,b)<0, a' = a - dot(a,b)/dot(b,b) * b; return total+(a'-a).
    Computing the correction directly avoids subtract/re-add roundoff when the
    projection is inactive. A tiny/zero BCE gradient is left unchanged.
    """
    if not total or len(total) != len(bce) or len(total) != len(alignment):
        raise ValueError('Gradient lists must be nonempty and aligned')
    for t, b, a in zip(total, bce, alignment):
        if t.shape != b.shape or t.shape != a.shape or t.device != b.device or t.device != a.device:
            raise ValueError('Gradient shape/device mismatch')
        if not all(torch.isfinite(v).all() for v in (t, b, a)):
            raise ValueError('Nonfinite gradient')
    dot = float(sum((a.double() * b.double()).sum() for a, b in zip(alignment, bce)))
    norm_b2 = float(sum(b.double().square().sum() for b in bce))
    norm_a2 = float(sum(a.double().square().sum() for a in alignment))
    cosine = dot / math.sqrt(norm_b2 * norm_a2) if norm_b2 and norm_a2 else None
    coefficient = -dot / norm_b2 if enabled and dot < 0 and norm_b2 > 1e-30 else 0.
    if not math.isfinite(coefficient):
        raise ValueError('Nonfinite projection coefficient')
    result = [t + coefficient * b if coefficient else t.clone() for t, b in zip(total, bce)]
    if not all(torch.isfinite(v).all() for v in result):
        raise ValueError('Projection produced a nonfinite gradient')
    return result, {
        'bce_gradient_norm': math.sqrt(norm_b2),
        'alignment_gradient_norm': math.sqrt(norm_a2),
        'alignment_to_bce_ratio': math.sqrt(norm_a2 / norm_b2) if norm_b2 else None,
        'raw_alignment_bce_cosine': max(-1., min(1., cosine)) if cosine is not None else None,
        'raw_alignment_bce_dot': dot,
        'projected_alignment_bce_dot': dot + coefficient * norm_b2,
        'projection_coefficient': coefficient,
        'projection_applied': coefficient > 0,
    }


class ProjectedStep:
    """Drop-in finite_step hook; no model or scientific source file is edited."""
    def __init__(self, config, mode='projected', deadline_epoch=None):
        if mode not in ('native', 'projected'):
            raise ValueError('Unknown update mode')
        if config['alignment_warmup_epochs'] != 0:
            raise ValueError('This fixed pilot requires no alignment ramp')
        self.coefficient = config['aux_every'] * config['lambda_alignment']
        self.mode, self.deadline_epoch = mode, deadline_epoch
        self.steps = self.alignment_steps = self.conflicts = self.projections = 0
        self.dot_sum = self.correction_sum = 0.

    def __call__(self, losses, model, optimizer):
        if self.deadline_epoch is not None and time.time() >= self.deadline_epoch:
            raise TimeoutError('Cohort deadline reached; preserve partial run')
        if not all(torch.isfinite(v).all() for v in losses.values()):
            raise RuntimeError('Nonfinite native loss')
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer.zero_grad(set_to_none=True)
        gb = torch.autograd.grad(losses['classification'], params, retain_graph=True, allow_unused=True)
        ga = torch.autograd.grad(self.coefficient * losses['response_alignment'], params,
                                 retain_graph=True, allow_unused=True)
        # Preserve the native backward path for the baseline total and all other
        # components. Only the alignment conflict correction changes the update.
        losses['total'].backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise RuntimeError('Missing/nonfinite native total gradient')
        gb = [torch.zeros_like(p) if g is None else g for p, g in zip(params, gb)]
        ga = [torch.zeros_like(p) if g is None else g for p, g in zip(params, ga)]
        combined, stats = project_alignment([p.grad for p in params], gb, ga, self.mode == 'projected')
        for p, gradient in zip(params, combined):
            p.grad = gradient
        torch.nn.utils.clip_grad_norm_(params, 5.)
        optimizer.step()
        self.steps += 1
        self.alignment_steps += int(stats['alignment_gradient_norm'] > 0)
        self.conflicts += int(stats['raw_alignment_bce_dot'] < 0)
        self.projections += int(stats['projection_applied'])
        self.dot_sum += stats['raw_alignment_bce_dot']
        self.correction_sum += stats['projection_coefficient']
        return stats

    def summary(self):
        return {'format': VERSION, 'mode': self.mode, 'steps': self.steps,
                'nonzero_alignment_steps': self.alignment_steps,
                'negative_dot_steps': self.conflicts, 'projected_steps': self.projections,
                'sum_raw_alignment_bce_dot': self.dot_sum,
                'sum_projection_coefficients': self.correction_sum,
                'alignment_weight': self.coefficient,
                'gradient_space': 'raw trainable-parameter Euclidean space before clipping/AdamW',
                'validation_utility_guaranteed': False}


@contextmanager
def installed(core, config, mode='projected', deadline_epoch=None):
    native = core.finite_step
    step = ProjectedStep(config, mode, deadline_epoch)
    core.finite_step = step
    try:
        yield step
    finally:
        core.finite_step = native
