"""CPU synthetic checks; no dataset/teacher checkpoint or GPU required."""
import copy,importlib.util,json,tempfile
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('st_run',HERE/'run.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def cache(root):
    root.mkdir();rng=np.random.default_rng(17);head=torch.nn.Linear(4,1)
    with torch.no_grad():head.weight.copy_(torch.tensor([[1.,-.2,.1,.3]]));head.bias.zero_()
    files=[];counts={}
    for split,n in [('train',64),('val',32)]:
        x=rng.normal(size=(n,4)).astype('float32');y=np.array([0]*(n//2)+[1]*(n//2));x[:,0]+=y*2-1
        with torch.no_grad():z=head(torch.from_numpy(x)).squeeze(1).numpy()
        rows=[{'image':f'{split}_{i}','label':int(y[i]),'identity_ids':f'{split}_{i}','source_ids':f'{split}_{i}','family':'fixture',
               **{a:str(i%2) for a in m.core.AXES}} for i in range(n)]
        np.save(root/f'{split}_features.npy',x);np.save(root/f'{split}_teacher_logits.npy',z)
        m.core.write_json(root/f'{split}_rows_PRIVATE.json',rows);counts[split]=n
        files += [f'{split}_features.npy',f'{split}_teacher_logits.npy',f'{split}_rows_PRIVATE.json']
        if split=='train':
            src=np.where(y==0,np.arange(n)%2,-1).astype('int64');np.save(root/'real_sources.npy',src)
            ref=m.objective.fit_train_reference(torch.from_numpy(z[y==0]))
    m.core.write_json(root/'reference.json',{k:v.tolist() if torch.is_tensor(v) else v for k,v in ref.items()})
    torch.save(head.state_dict(),root/'teacher_head.pt');files+=['real_sources.npy','reference.json','teacher_head.pt']
    m.core.write_json(root/'CACHE_VERIFIED.json',{'format':'SOURCE_TAIL_FEATURE_CACHE_v1','fixture':True,'source_hashes':m.hashes(),
        'teacher_checkpoint_sha256':'synthetic','manifest_sha256':{'train':'fixture','val':'fixture'},'rows':counts,
        'files':{n:m.core.sha(root/n) for n in files}})

if __name__=='__main__':
    torch.manual_seed(17);torch.set_num_threads(2);device=torch.device('cpu')
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);cache(root/'cache');c=m.config();c.update(batch_size=8,min_support=2,min_auc=0.,max_fnr=1.,auxiliary_per_source=4)
        try:m.checked_cache(root/'cache')
        except ValueError:pass
        else:raise AssertionError('Fixture accepted as production')
        print('PASS fixture/production isolation')
        c0=copy.deepcopy(c)
        control=m.train(c0,root/'cache',root/'control','control',device,fixture=True)
        treatment=m.train(c,root/'cache',root/'full','treatment',device,fixture=True)
        partial=m.train(c,root/'cache',root/'resume','treatment',device,fixture=True,stop_steps=5)
        assert partial['status']=='INCOMPLETE_RESUMABLE'
        resumed=m.train(c,root/'cache',root/'resume','treatment',device,resume=True,fixture=True)
        a=torch.load(root/'full/last.pt',weights_only=True);b=torch.load(root/'resume/last.pt',weights_only=True)
        for k in a['head']:torch.testing.assert_close(a['head'][k],b['head'][k],rtol=0,atol=0)
        assert a['history']==b['history'] and treatment['validation']==resumed['validation']
        ctr=torch.load(root/'control/last.pt',weights_only=True)
        assert any(not torch.equal(a['head'][k],ctr['head'][k]) for k in a['head'])
        assert a['steps']==24 and all(h['original_count']==64 and h['originals_once_each'] for h in a['history'])
        print('PASS full three-epoch lifecycle, exact interrupted/resumed head+history, native selected-score replay, original coverage')
        print('PASS source-tail coefficient changes actual learned parameters versus matched control')
        before=m.checked_cache(root/'cache',fixture=True)
        assert before['files']['teacher_head.pt']==m.core.sha(root/'cache/teacher_head.pt')
        assert before['files']['train_features.npy']==m.core.sha(root/'cache/train_features.npy')
        print('PASS encoder features and teacher head immutable; head-only optimization')
        with (root/'cache/real_sources.npy').open('ab') as f:f.write(b'changed')
        try:m.checked_cache(root/'cache',fixture=True)
        except ValueError:pass
        else:raise AssertionError('Tampered cache accepted')
        print('PASS altered cache rejected')
    ref={'thresholds':[0.],'scale':1.,'targets':[.2],'temperature':1.}
    fn=m.objective.SourceTailObjective(ref,min_source_rows=2)
    z=torch.tensor([-1.,1.],requires_grad=True);teacher=torch.tensor([-1.,2.],requires_grad=True)
    real=torch.tensor([4.,4.,-8.,-8.],requires_grad=True)
    loss=fn(z,torch.tensor([0,1]),teacher,real,torch.tensor([0,0,1,1]));loss['total'].backward()
    assert (real.grad[:2]>0).all() and (real.grad[2:]==0).all() and teacher.grad is None and z.grad[1]<0
    print('PASS one-sided source gradient, fake preservation gradient, detached teacher')
    model=m.core.Detector(head_type='linear',policy='frozen').eval()
    image=torch.randn(2,3,224,224)
    with torch.inference_mode():
        f=model.encoder.features(image).mean((2,3));z=model.classifier(f).squeeze(1)
        torch.testing.assert_close(z,model(image)['logit'],rtol=0,atol=0)
    assert all(not p.requires_grad for p in model.encoder.parameters())
    print('PASS full-width native Xception feature-cache/complete-forward identity; encoder frozen')
    print('SOFTWARE_CHECKS_PASS; no benchmark result, private checkpoint or CUDA claim')
