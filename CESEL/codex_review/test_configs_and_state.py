import sys, json, hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from cfes.config import load_config
from cfes.ces_state import CESState
from cfes.ces_validation import load_validated_family
from cfes.ces_model import CUES
from cfes.ces_losses import sens_loss
paths=sorted((ROOT/'configs').rglob('*.yaml'))
for p in paths:
    c=load_config(str(p))
    if p.parent.name in ('ffpp','ff2','ff2_public'):
        assert c['ces']['validation']['per_family'] == (p.parent.name!='ffpp')
        assert ('fairforge2_public' in c['data']['train']['csv']) == (p.parent.name=='ff2_public')
    print('PASS load',p.relative_to(ROOT))
print(f'PASS all {len(paths)} configs; _base_ composition and FF2-only per_family/public train paths')
c=load_config(str(ROOT/'configs/ff2_public/full.yaml'))
d={'cues':list(CUES),'valid':dict.fromkeys(CUES,True),'target':dict.fromkeys(CUES,.5)}
d['per_family']={'families':c['data']['families'],'valid':{},'target':{}}
for name in c['data']['families']:
    d['per_family']['valid'][name]=dict.fromkeys(CUES,True)
    d['per_family']['target'][name]=dict.fromkeys(CUES,.5)
d['per_family']['valid']['synth']['bound']=False
d['per_family']['target']['synth']['bound']=None
p=ROOT/'review_scratch/family_fixture.json'; p.write_text(json.dumps(d)); c['ces']['validated_json']=str(p)
V,T=load_validated_family(c,9)
assert V[0].all() and V[8].all() and not V[6,2] and V[7,2]
u=torch.zeros(3,4,requires_grad=True); loss,_=sens_loss(u,V[torch.tensor([0,6,8])],T[torch.tensor([0,6,8])]);loss.backward()
assert u.grad[1,2]==0 and u.grad[0,2]<0 and u.grad[2,2]<0
print('PASS FF2-public family mask: synth boundary excluded; reals/unmapped fakes retain global mask')
st=CESState(4,3,8,c); torch.manual_seed(0)
C=torch.randn(16,4,3); y=torch.arange(16)%2; g=torch.arange(16)%8
st.update(C,y,g); st.probe_step(C,g)
p=ROOT/'review_scratch/state.pt'; torch.save(st.full_state(),p)
st2=CESState(4,3,8,c); st2.load_full_state(torch.load(p,weights_only=False))
for obj in (st,st2): obj.update(C+1,y,g); obj.probe_step(C+1,g)
assert all(torch.equal(v,st2.state_dict()[k]) for k,v in st.state_dict().items())
print('PASS CES buffers/probe/Adam state save-load-next-update exact on CPU')
# Coverage manifest, hashes also allow verifying code was not changed after this test.
files=sorted(p for folder in ('cfes','scripts','run','configs','tests') for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
(ROOT/'review_scratch/reviewed_files.sha256').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in files))
print(f'Coverage inventory: {len(files)} files')
