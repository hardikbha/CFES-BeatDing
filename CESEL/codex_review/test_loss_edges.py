import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from cfes import ces_losses as L
from cfes.ces_state import CESState
from cfes.ces_model import CueHead
from types import SimpleNamespace

torch.manual_seed(1)
for n in (0, 3, 8):
    u = torch.ones(n, 4, requires_grad=True)
    y = torch.zeros(n, dtype=torch.long)
    g = torch.arange(n) % 2
    for mask in (torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool)):
        loss, stats = L.floor_loss(u, y, g, y, mask, {'n_min': 4, 'delta': .1})
        assert torch.isfinite(loss) and loss.item() == 0
        loss.backward()
        assert torch.isfinite(u.grad).all()
print('PASS floor: empty pool, missing class/groups, undersized cells, empty cue mask; finite zero gradients')
st = CESState(4, 3, 3, {})
st.cls_mean[0].fill_(2); st.cls_mean[1].fill_(9)
st.grp_mean[0, 1].fill_(5); st.grp_init[0, 1] = True
C = torch.randn(6, 4, 3, requires_grad=True)
y = torch.tensor([1, 1, 1, 1, 0, 1]); g = torch.tensor([1, 0, -1, 9, 1, 1])
R = L.replacement(C, y, st, torch.tensor([-1,-1,-1,-1,-1,4]), g=g)
assert not R.requires_grad
assert torch.equal(R[:,0,0].detach()[:5], torch.tensor([5.,2.,2.,2.,9.]))
assert torch.equal(R[5], C[4].detach())
print('PASS replacement: own-group EMA, absent/invalid group fallback, paired override, detached reference')
p, has = L.demographic_partner(torch.zeros(6, dtype=torch.long), torch.zeros(6, dtype=torch.long), torch.Generator())
assert not has.any()
assert torch.equal(L.demographic_swap(C, st, p, has), C)
assert not L.swap_validity(st, C, torch.zeros(6, dtype=torch.long), p, has).any()
z = torch.tensor([-1e30,-400.,0.,400.,1e30], requires_grad=True)
m = L.margin(z, torch.ones(5), {})
assert torch.isfinite(m).all(); m.sum().backward(); assert torch.isfinite(z.grad).all()
print('PASS swap without donor is identity/invalid; asinh finite at +/-400 and +/-1e30 (extreme gradients may underflow)')
head = CueHead(4, 3)
model = SimpleNamespace(head=head, score_from=head.score_from)
u, sm = L.utilities(model, C, model.score_from(C), y, R, {'margin':'linear'})
assert torch.allclose(u.sum(1), L.margin(model.score_from(C),y,{'margin':'linear'})-L.margin(model.score_from(R),y,{'margin':'linear'}),atol=2e-6)
print('PASS linear utility additivity with mixed paired/group/fallback replacements')
