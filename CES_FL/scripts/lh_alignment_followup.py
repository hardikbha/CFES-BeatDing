#!/usr/bin/env python3
"""Pinned TRAIN diagnostics and one conditional native-arm-C alignment ablation.

The scientific model/source and H1 config remain unchanged. This helper does
not allocate a GPU, generate review images, alter approvals or access TEST.
"""
import argparse
from collections import Counter, defaultdict
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT/'proposals/lh_alignment_ablation_20261105_r1'
SOURCE = ROOT/'src/ces_fl_nuisance_resnet50.py'
spec = importlib.util.spec_from_file_location('alignment_science', SOURCE)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def read(path): return json.loads(Path(path).read_text())


def protocol():
    p = read(PROPOSAL/'protocol.json')
    if m.sha(SOURCE) != p['scientific_source_sha256']: raise ValueError('Pinned scientific source changed')
    if m.sha(PROPOSAL/'C1.json') != p['config_sha256']['H1']: raise ValueError('C1 config is not identical to H1')
    return p


def fresh_json(path, data):
    if Path(path).exists(): raise ValueError('Preserve existing evidence: '+str(path))
    m.write_json(path, data)


def verify_reference(root, name, p):
    root = Path(root)
    if not (root/'DONE').is_file(): raise ValueError('Reference incomplete: '+name)
    if m.sha(root/'config.json') != p['config_sha256'][name]: raise ValueError('Reference config differs: '+name)
    if m.sha(root/'best.pt') != p['checkpoint_sha256'][name]: raise ValueError('Reference checkpoint differs: '+name)
    c, v, h, s = (read(root/f) for f in ('config.json','RUN_VERIFIED.json','history.json','validation_scores_PRIVATE.json'))
    run = read(root/'RUN.json')
    if v.get('status') != 'COMPLETED_VERIFIED' or v.get('synthetic_fixture') or v.get('test_evaluated'):
        raise ValueError('Invalid native reference verification')
    if v['source_sha256'] != p['scientific_source_sha256'] or v['config_digest'] != m.digest(c):
        raise ValueError('Reference source/config binding differs')
    if [r['epoch'] for r in h] != [1,2,3] or v['epochs_completed'] != 3: raise ValueError('Reference epoch budget differs')
    winner = max(h, key=lambda r:r['validation']['overall']['auc'])
    if v['selected_epoch'] != winner['epoch'] or winner['validation'] != v['validation']: raise ValueError('Reference selection differs')
    if s['split'] != 'validation' or s['test_read'] or s['checkpoint_sha256'] != p['checkpoint_sha256'][name]:
        raise ValueError('Invalid reference score cache')
    if s['manifest_sha256'] != c['val_sha256']: raise ValueError('Reference validation manifest differs')
    if m.metric_report(s['rows'],s['logits'],s['threshold'],c['min_eval_support']) != v['validation']:
        raise ValueError('Reference saved-score replay differs')
    return {'root':str(root.resolve()),'checkpoint_sha256':m.sha(root/'best.pt'),
            'config_sha256':m.sha(root/'config.json'),'cache_sha256':m.sha(root/'validation_scores_PRIVATE.json'),
            'verification_sha256':m.sha(root/'RUN_VERIFIED.json'),'history_sha256':m.sha(root/'history.json'),
            'run_receipt_sha256':m.sha(root/'RUN.json'),'torch':run['torch']}


def canonical_cases(c, rows):
    active, _ = m.group_inventory(rows, c)
    chosen = {min(cell[0], key=lambda ix:m.seed_for(c['seed'],axis,group,rows[ix]['image']))
              for axis, groups in active.items() for group, cell in groups.items()}
    expected = {}
    for ix in sorted(chosen):
        for k, op in enumerate(m.OPERATORS):
            for j, strength in enumerate(m.STRENGTHS):
                seed = m.seed_for(c['seed'],ix,k,j,'review')
                settings = {'operator':op,'strength':strength,'seed':seed,
                            'jpeg_quality':random.Random(seed).choice((85,95,100))}
                for nu in m.NUISANCE_BANK:
                    expected[f'case_{ix:06d}_{k}_{j}_{nu}'] = {
                        'image':rows[ix]['image'],'image_sha256':rows[ix]['image_sha256'],
                        'groups':{a:rows[ix][a] for a in c['axes']},
                        'settings':{**settings,'common_post_nuisance':nu}}
    return expected


def verify_case_inventory(manifest, review, expected):
    """Metadata coverage check; image/sheet hashes checked separately, no writes."""
    cases, decisions = manifest['cases'], review['cases']
    if not expected or len(cases) != len(expected): raise ValueError('Missing/empty canonical case inventory')
    ids = [r['id'] for r in cases]
    if len(set(ids)) != len(ids) or set(ids) != set(expected): raise ValueError('Duplicate/changed preview cases')
    rids = [r['id'] for r in decisions]
    if len(set(rids)) != len(rids) or set(rids) != set(expected): raise ValueError('Duplicate/missing review decisions')
    for item in cases:
        for key, value in expected[item['id']].items():
            if item[key] != value: raise ValueError('Canonical case metadata changed')
    flags = ('approved','face_crop_suitable','control_processing_acceptable',
             'intended_local_change_present','demographic_appearance_preserved')
    for r in decisions:
        if any(r.get(k) is not True for k in flags) or not r.get('reviewer','').strip() or not r.get('evidence_notes','').strip():
            raise ValueError('Existing review incomplete/rejected; do not invent or change approvals')


def verify_review(c, rows, path, p):
    m.check_qualification(c, path)
    q = read(path); manifest = read(q['preview_file']); review = read(q['review_file'])
    if manifest['synthetic_fixture'] or manifest['source_sha256'] != m.sha(SOURCE) or manifest['config_digest'] != m.digest(c):
        raise ValueError('Review manifest binding differs')
    if review['preview_sha256'] != m.sha(q['preview_file']): raise ValueError('Review hash differs')
    expected = canonical_cases(c, rows)
    if len(expected) != p['qualification']['expected_cases']: raise ValueError('Canonical 352-case coverage changed')
    verify_case_inventory(manifest, review, expected)
    if q['case_count'] != len(expected): raise ValueError('Qualification case count differs')
    for item in manifest['cases']:
        if m.sha(item['sheet']) != item['sheet_sha256'] or m.sha(item['image']) != item['image_sha256']:
            raise ValueError('Previously reviewed bytes changed')
    return {'qualification_sha256':m.sha(path),'cases':len(expected),
            'preview_sha256':m.sha(q['preview_file']),'review_sha256':m.sha(q['review_file']),
            'existing_review_only':True,'new_approval_created':False,'real_forgery_validity_proven':False}


def bind(refs_path, out):
    p = protocol(); refs = read(refs_path)
    keys = {'H0_run','H1_run','A1_run','H1_preflight','H1_qualification'}
    if set(refs) != keys: raise ValueError('Reference path fields differ from template')
    refs = {k:str(Path(v).resolve()) for k,v in refs.items()}
    records = {n:verify_reference(refs[n+'_run'],n,p) for n in ('H0','H1','A1')}
    if len({r['torch'] for r in records.values()})!=1: raise ValueError('Reference runtime versions differ')
    c = m.load_config(PROPOSAL/'C1.json')
    rows, _, _ = m.checked_inputs(c, refs['H1_preflight'])
    checked = verify_review(c, rows, refs['H1_qualification'], p)
    fresh_json(out, {'status':'BOUND','protocol_sha256':m.sha(PROPOSAL/'protocol.json'),
        'helper_sha256':m.sha(__file__),'scientific_source_sha256':m.sha(SOURCE),
        'refs':refs,'references':records,'qualification':checked,
        'preflight_sha256':m.sha(Path(refs['H1_preflight'])/'PREFLIGHT.json'),
        'config_sha256':m.sha(PROPOSAL/'C1.json'),'test_read':False,'training_started':False})
    print('BOUND_EXISTING_352_CASE_REVIEW_AND_REFERENCES')


def load_binding(path):
    p = protocol(); b = read(path)
    if b['status'] != 'BOUND' or b['protocol_sha256'] != m.sha(PROPOSAL/'protocol.json') or b['helper_sha256'] != m.sha(__file__):
        raise ValueError('Binding belongs to another helper/protocol')
    if b['scientific_source_sha256'] != m.sha(SOURCE) or b['config_sha256'] != m.sha(PROPOSAL/'C1.json'):
        raise ValueError('Scientific source/config changed')
    for name in ('H0','H1','A1'):
        if verify_reference(b['refs'][name+'_run'],name,p) != b['references'][name]: raise ValueError('Reference artifacts changed')
    c = m.load_config(PROPOSAL/'C1.json')
    if m.sha(Path(b['refs']['H1_preflight'])/'PREFLIGHT.json') != b['preflight_sha256']: raise ValueError('Preflight changed')
    rows, _, _ = m.checked_inputs(c, b['refs']['H1_preflight'])
    if verify_review(c, rows, b['refs']['H1_qualification'], p) != b['qualification']: raise ValueError('Review changed')
    return p, b, c, rows


def main_indices(rows, event):
    if event<0 or (event+1)*32>len(rows): raise ValueError('Insufficient distinct TRAIN rows for fixed diagnostic plan')
    generator=torch.Generator().manual_seed(m.seed_for(17,'alignment-uniform-main-v1'))
    order=torch.randperm(len(rows),generator=generator).tolist()
    # Uniform rows with the native inverse-cell BCE weights, not an additional
    # class/group balancing step that would change the effective objective.
    return order[event*32:(event+1)*32]


def transform_batch(batch, condition):
    if condition == 'identity': return batch
    if condition != 'resize90': raise ValueError('Unplanned diagnostic transform')
    result = dict(batch)
    mean=torch.tensor([.485,.456,.406])[:,None,None]
    std=torch.tensor([.229,.224,.225])[:,None,None]
    for key in ('x','s','t'):
        # Existing normalized views -> RGB -> same resize -> normalization.
        raw=(batch[key]*std+mean).clamp(0,1)
        result[key]=torch.stack([m.normalize(m.nuisance_view(x,'resize90')) for x in raw])
    return result


def batch_views(rows, active, c, event, condition):
    c={**c,'nuisance_mode':'none','amp':False}
    indices=main_indices(rows,event)
    samples=[m.Originals(rows,c,epoch=0,pairs=True,training=False)[i] for i in indices]
    batch={k:torch.stack([r[k] for r in samples]) if torch.is_tensor(samples[0][k])
           else torch.tensor([r[k] for r in samples]) for k in samples[0]}
    aux=m.auxiliary(rows,active,c,event,0)
    return transform_batch(batch,condition),transform_batch(aux,condition),indices


def weighted_parts(losses,c):
    k=c['aux_every']
    return {'BCE':losses['classification'],
        'fairness':k*c['lambda_fair']*losses['error_fairness'],
        'sensitivity':c['lambda_sensitivity']*(losses['original_sensitivity']+k*losses['aux_sensitivity']),
        'sham_control':c['lambda_sensitivity']*c['sham_weight']*(losses['original_control_stability']+k*losses['aux_control_stability']),
        'alignment':k*c['lambda_alignment']*losses['response_alignment']}


def gradient_stats(grads, baseline, names):
    result={}
    for scope in ('all','encoder','head'):
        ix=[i for i,n in enumerate(names) if scope=='all' or (n.startswith('encoder.') if scope=='encoder' else not n.startswith('encoder.'))]
        norm=sum(float(grads[i].float().square().sum()) for i in ix if grads[i] is not None)**.5
        bn=sum(float(baseline[i].float().square().sum()) for i in ix if baseline[i] is not None)**.5
        dot=sum(float((grads[i].float()*baseline[i].float()).sum()) for i in ix if grads[i] is not None and baseline[i] is not None)
        result[scope]={'gradient_norm':norm,'cosine_to_BCE':max(-1.,min(1.,dot/(norm*bn))) if norm*bn else None,
                       'norm_ratio_to_BCE':norm/bn if bn else None}
    return result


def numbers(v):
    a=np.asarray(v,dtype=float)
    return {'n':len(a),'mean':float(a.mean()),'median':float(np.median(a)),
            'p10':float(np.quantile(a,.1)),'p90':float(np.quantile(a,.9))}


def profile_batch(model,b,aux,weights,c,threshold,device):
    c={**c,'amp':False}; captured=[]
    handle=model.register_forward_hook(lambda _m,_x,out:captured.append(out['logit'].detach().float().cpu()))
    model.train()
    try:
        losses,_=m.train_step(model,b,aux,weights,c,'D',threshold,device,0)
        if any(not torch.isfinite(value) for value in losses.values()): raise RuntimeError('Nonfinite diagnostic loss')
        parts=weighted_parts(losses,c)
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        names,params=zip(*named)
        base=torch.autograd.grad(parts['BCE'],params,retain_graph=True,allow_unused=True)
        info={}
        for name,loss in parts.items():
            gg=base if name=='BCE' else torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            if any(g is not None and not torch.isfinite(g).all() for g in gg): raise RuntimeError('Nonfinite gradient')
            info[name]={'weighted_loss':float(loss.detach()),**gradient_stats(gg,base,names)}
        real=b['y']==0
        expected_forwards=6 if real.any() else 4
        if len(captured)!=expected_forwards: raise ValueError('Unexpected native H forward structure')
        raw_delta=torch.cat((captured[2]-captured[1],captured[5]-captured[4])) if real.any() else captured[3]-captured[2]
        r=torch.asinh(raw_delta)
        mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
        # No-mask average is a perturbation-strength diagnostic, not forensic validity.
        pixel=torch.cat((((b['t'][real]-b['s'][real])*std).abs().mean((1,2,3)),
                         ((aux['t']-aux['s'])*std).abs().mean((1,2,3))))
        headnorm=float(model.classifier.weight.detach().float().norm())
        return {'gradients':info,'procedural_response':numbers(r.numpy()),
            'raw_response':numbers(raw_delta.numpy()),'margin_satisfied_fraction':float((r>=c['margin']).float().mean()),
            'head_weight_norm':headnorm,'head_gain_normalized_delta':numbers(raw_delta.numpy()/headnorm) if headnorm else None,
            'pixel_gap':numbers(pixel.numpy()),'absolute_original_logits':numbers(captured[0].abs().numpy()),
            'optimizer_updates':0}
    finally:
        handle.remove();model.zero_grad(set_to_none=True);model.eval()


def assess(records, trigger):
    summary={}; sufficient=True; positive=False
    for condition in ('identity','resize90'):
        rows=[r for r in records if r['model']=='H1' and r['condition']==condition]
        valid=[r['gradients']['alignment']['all'] for r in rows
               if r['gradients']['alignment']['all']['cosine_to_BCE'] is not None
               and r['gradients']['alignment']['all']['norm_ratio_to_BCE'] is not None
               and math.isfinite(r['gradients']['alignment']['all']['cosine_to_BCE'])
               and math.isfinite(r['gradients']['alignment']['all']['norm_ratio_to_BCE'])]
        enough=len(valid)>=trigger['minimum_valid_batches_per_condition']
        sufficient=sufficient and enough
        neg=sum(r['cosine_to_BCE']<0 for r in valid)/len(valid) if valid else None
        ratio=float(np.median([r['norm_ratio_to_BCE'] for r in valid])) if valid else None
        hit=bool(enough and neg>=trigger['minimum_negative_cosine_fraction'] and ratio>=trigger['minimum_median_alignment_to_BCE_gradient_norm_ratio'])
        summary[condition]={'valid_batches':len(valid),'negative_fraction':neg,'median_norm_ratio':ratio,'trigger_met':hit}
        positive=positive or hit
    return {'run_C1':bool(sufficient and positive),'conditions':summary,
            'scope':'Prespecified engineering triage; does not establish causal harm or significance.'}


def gpu_guard(device):
    if device.type!='cuda': return None
    receipt=os.environ.get('CES_GPU_BINDING_RECEIPT')
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or not receipt or not Path(receipt).is_file():
        raise ValueError('Set verified allocation/UUID binding before Python and provide CES_GPU_BINDING_RECEIPT')
    if torch.cuda.device_count()!=1: raise ValueError('Exactly one bound CUDA device required')
    # Ownership and physical mapping must be verified by the external allocation
    # controller. File existence and CUDA visibility are not ownership proof.
    return {'receipt_sha256':m.sha(receipt),'receipt_path':str(Path(receipt).resolve()),
            'external_allocation_verification_required':True}


def audit_train(binding,out,device):
    started=time.monotonic();p,b,c,rows=load_binding(binding)
    allocation=gpu_guard(device);out=m.fresh_dir(out)
    if device.type=='cuda' and str(torch.__version__)!=b['references']['H1']['torch']:
        raise ValueError('Use the original H1 PyTorch/CUDA runtime for the controlled diagnostic')
    active,_=m.group_inventory(rows,c);weights=m.original_weights(rows,c['balance_axis']);records=[]
    plan={event:main_indices(rows,event) for event in p['diagnostic']['events']}
    m.write_json(out/'TRAIN_PLAN_PRIVATE.json',{'rows':{str(e):[rows[i]['image'] for i in ix] for e,ix in plan.items()},
        'train_manifest_sha256':c['train_sha256'],'seed':17,'test_read':False})
    for name in p['diagnostic']['checkpoints']:
        m.seed_all(c['seed'])
        model,ck=m.load_trained(Path(b['refs'][name+'_run'])/'best.pt',device)
        if ck['arm']!='D':raise ValueError('Reference H checkpoint is not native D')
        before={n:v.detach().clone() for n,v in model.named_buffers()}
        reals=sorted([r for r in rows if r['label']==0],key=lambda r:m.seed_for(c['seed'],'train-calibration',r['image']))[:c['train_calibration_size']]
        threshold=m.calibrate(m.predict(model,reals,c,device),c['target_fpr'])
        for event in p['diagnostic']['events']:
            for condition in p['diagnostic']['conditions']:
                batch,aux,ix=batch_views(rows,active,c,event,condition)
                record={'model':name,'event':event,'condition':condition,'axis':aux['axis'],
                    'operator':aux['operator'],'strength':aux['strength'],
                    'main_plan_digest':m.digest([rows[i]['image'] for i in ix]),
                    'main_class_counts':dict(Counter(str(rows[i]['label']) for i in ix)),
                    'main_joint_class_counts':dict(Counter(f"{rows[i]['gender_race']}:{rows[i]['label']}" for i in ix)),
                    'train_threshold':threshold,
                    **profile_batch(model,batch,aux,weights,c,threshold['threshold_logit'],device)}
                records.append(record)
                print(json.dumps({'model':name,'event':event,'condition':condition,
                    'alignment_cosine':record['gradients']['alignment']['all']['cosine_to_BCE']}),flush=True)
        if any(not torch.equal(v,dict(model.named_buffers())[n]) for n,v in before.items()): raise RuntimeError('Diagnostic changed BN buffers')
        del model,before
        if device.type=='cuda':torch.cuda.empty_cache()
    result={'status':'COMPLETE','binding_sha256':m.sha(binding),'helper_sha256':m.sha(__file__),
        'protocol_sha256':m.sha(PROPOSAL/'protocol.json'),'records':records,
        'decision':assess(records,p['diagnostic']['trigger']),'allocation':allocation,
        'runtime':{'torch':str(torch.__version__),'device':str(device),
                   'cudnn_deterministic':torch.backends.cudnn.deterministic,
                   'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
                   'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32},
        'elapsed_seconds':time.monotonic()-started,'optimizer_updates':0,'test_read':False,
        'scope':'TRAIN samples and procedural processing only. Pixel retention is not forensic validation.'}
    m.write_json(out/'DIAGNOSTIC.json',result);print(json.dumps(result['decision']))


def diagnostic_gate(binding,path,p):
    d=read(path)
    if d['status']!='COMPLETE' or d['binding_sha256']!=m.sha(binding) or d['helper_sha256']!=m.sha(__file__) or d['protocol_sha256']!=m.sha(PROPOSAL/'protocol.json'):
        raise ValueError('Diagnostic binding changed')
    if d.get('optimizer_updates')!=0 or d.get('test_read') is not False:raise ValueError('Diagnostic scope changed')
    expected={(model,event,condition) for model in p['diagnostic']['checkpoints'] for event in p['diagnostic']['events'] for condition in p['diagnostic']['conditions']}
    keys=[(r['model'],r['event'],r['condition']) for r in d['records']]
    if len(keys)!=len(expected) or set(keys)!=expected:raise ValueError('Missing/duplicate diagnostic batches')
    for event in p['diagnostic']['events']:
        rows=[r for r in d['records'] if r['event']==event]
        if len({r['main_plan_digest'] for r in rows})!=1:raise ValueError('Diagnostic source draws are not paired')
        axis=('gender','race','age_bin','skin_bin','gender_race')[event%5]
        op,strength=[(op,s) for op in m.OPERATORS for s in m.STRENGTHS][event//5]
        if any(r['axis']!=axis or r['operator']!=op or r['strength']!=strength or r['optimizer_updates']!=0 for r in rows):
            raise ValueError('Diagnostic coverage/procedural plan changed')
    if d['decision']!=assess(d['records'],p['diagnostic']['trigger']) or not d['decision']['run_C1']:
        raise ValueError('Prespecified diagnostic concern not supported; stop without C1 training')
    return d


def cuda_check(binding,diagnostic,out):
    device=torch.device('cuda')
    p,b,c,_=load_binding(binding);diagnostic_gate(binding,diagnostic,p)
    gpu_guard(device)
    if str(torch.__version__)!=b['references']['H1']['torch']:raise ValueError('Use the original H1 runtime')
    m.dry_run(c,b['refs']['H1_preflight'],out,'C',device)


def train_c1(binding,diagnostic,dry,software,out,budget_hours):
    device=torch.device('cuda')
    p,b,c,_=load_binding(binding);d=diagnostic_gate(binding,diagnostic,p)
    allocation=gpu_guard(device)
    if str(torch.__version__)!=b['references']['H1']['torch']:raise ValueError('Use the original H1 runtime')
    if not math.isfinite(budget_hours) or not 0<budget_hours<=p['budget']['native_training_cap_hours']:
        raise ValueError('Invalid verified remaining training allowance')
    for path,kind in [(dry,'dry'),(software,'software')]:
        r=read(path)
        if r.get('status')!='PASS' or r.get('source_sha256')!=m.sha(SOURCE) or r.get('device')!='cuda' or r.get('head_type')!='linear':
            raise ValueError('Stale/wrong CUDA check')
        if r.get('torch')!=str(torch.__version__):raise ValueError('CUDA receipt runtime differs')
        if kind=='dry' and (r.get('arm')!='C' or r.get('config_digest')!=m.digest(c)):
            raise ValueError('Fresh native C backward receipt required')
        if kind=='dry' and r['losses']['response_alignment']!=0:raise ValueError('Candidate backward still includes alignment')
        if kind=='software' and r.get('nuisance_mode')!='matched':raise ValueError('Wrong nuisance smoke receipt')
    out=Path(out)
    if out.exists():raise ValueError('Preserve prior candidate output; no automatic resume')
    launch={'candidate':'C1','native_arm':'C','binding_sha256':m.sha(binding),'diagnostic_sha256':m.sha(diagnostic),
        'diagnostic_path':str(Path(diagnostic).resolve()),
        'helper_sha256':m.sha(__file__),'scientific_source_sha256':m.sha(SOURCE),'allocation':allocation,
        'budget_hours':budget_hours,'initialization':'original P50 with fresh seed17 head','test_read':False}
    fresh_json(out.parent/(out.name+'_LAUNCH.json'),launch)
    m.train(c,b['refs']['H1_preflight'],out,'C',device,qualification=b['refs']['H1_qualification'],budget_hours=budget_hours)


def report(binding,candidate,out):
    p,b,c,_=load_binding(binding);root=Path(candidate)
    launch=read(root.parent/(root.name+'_LAUNCH.json'))
    if launch['binding_sha256']!=m.sha(binding) or launch['helper_sha256']!=m.sha(__file__) or launch['native_arm']!='C':
        raise ValueError('Candidate launch does not match the bound follow-up')
    if launch['diagnostic_sha256']!=m.sha(launch['diagnostic_path']):raise ValueError('Launch diagnostic changed')
    diagnostic_gate(binding,launch['diagnostic_path'],p)
    if not (root/'DONE').is_file():raise ValueError('Candidate incomplete')
    v,h,s=(read(root/n) for n in ('RUN_VERIFIED.json','history.json','validation_scores_PRIVATE.json'))
    if v.get('status')!='COMPLETED_VERIFIED' or v.get('synthetic_fixture') or v.get('test_evaluated') or v.get('arm')!='C':
        raise ValueError('Invalid C1 production receipt')
    if v['source_sha256']!=m.sha(SOURCE) or v['config_digest']!=m.digest(c) or m.sha(root/'config.json')!=p['config_sha256']['H1']:
        raise ValueError('Candidate settings differ beyond native C/D arm')
    if [r['epoch'] for r in h]!=[1,2,3] or v['epochs_completed']!=3:raise ValueError('Candidate epoch budget differs')
    winner=max(h,key=lambda r:r['validation']['overall']['auc'])
    if winner['epoch']!=v['selected_epoch'] or winner['validation']!=v['validation']:raise ValueError('Selection changed')
    if m.sha(root/'best.pt')!=v['selected_checkpoint_sha256'] or s['checkpoint_sha256']!=v['selected_checkpoint_sha256']:raise ValueError('Candidate checkpoint changed')
    if s['split']!='validation' or s['test_read'] or s['manifest_sha256']!=c['val_sha256']:raise ValueError('Not the fixed validation cache')
    if m.metric_report(s['rows'],s['logits'],s['threshold'],c['min_eval_support'])!=v['validation']:raise ValueError('Saved-score replay mismatch')
    baseline_h=read(Path(b['refs']['H1_run'])/'history.json')
    for x,y in zip(h,baseline_h):
        for k in ('steps','unique_originals','originals_once_each','original_real_pairs','auxiliary_schedule',
                  'auxiliary_planned_draws','main_nuisance_counts','main_nuisance_by_axis_group_class','aux_nuisance_events'):
            if x[k]!=y[k]:raise ValueError('Exposure/nuisance draws differ: '+k)
        if x['losses_step_mean']['response_alignment']!=0:raise ValueError('C1 still optimizes alignment')
    out=m.fresh_dir(out);contrasts={}
    for name in ('H1','A1'):
        cache=Path(b['refs'][name+'_run'])/'validation_scores_PRIVATE.json'
        r=read(cache)
        if r['rows']!=s['rows']:raise ValueError('Reference/candidate validation rows differ')
        m.compare(root/'validation_scores_PRIVATE.json',cache,out/f'C1_minus_{name}.json',p['guardrails']['bootstrap_draws'],17)
        contrasts[name]=read(out/f'C1_minus_{name}.json')
    rr=contrasts['H1']['intervals'];primary=rr['gender_race/F_FPR']
    favorable=primary['estimate']<0 and rr['overall/auc']['estimate']>=0 and rr['overall/fnr']['estimate']<=0
    enough=primary['valid_replicates']>=p['guardrails']['minimum_usable']
    assessment={'point_guardrails_met':favorable,'primary_uncertainty_supported':enough,
        'decision':'INCONCLUSIVE_PRIMARY_UNCERTAINTY' if favorable and not enough else 'NUMERICALLY_FAVORABLE_REQUIRES_CONFIRMATION' if favorable else 'NUMERICAL_GUARD_FAILED_OR_MIXED',
        'primary_difference':primary,'scope':'One seed, source-validation only; no cross-domain or disentanglement claim.'}
    m.write_json(out/'RESULT.json',{'candidate':v,'history':h,'contrasts':contrasts,'assessment':assessment,
        'fixed_probability_0_5':m.metric_report(s['rows'],s['logits'],0.,c['min_eval_support']),
        'test_read':False,'new_inference':False})
    lines=['# C1 alignment ablation — FF++ validation only','',
        '| Model | Selected epoch | AUC % | FPR % | FNR % | Gender F_FPR ×100 | Race F_FPR ×100 | Joint F_FPR ×100 | Age F_FPR ×100 | Skin F_FPR ×100 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ('H1','A1','C1'):
        rv=v if name=='C1' else read(Path(b['refs'][name+'_run'])/'RUN_VERIFIED.json')
        r=rv['validation'];o=r['overall'];axes=r['axes']
        values=[100*o[k] for k in ('auc','fpr','fnr')]+[100*axes[k]['F_FPR'] for k in ('gender','race','gender_race','age_bin','skin_bin')]
        lines.append('|'+name+'|'+str(rv['selected_epoch'])+'|'+'|'.join(f'{value:.6f}' for value in values)+'|')
    lines+=['','Assessment: `'+assessment['decision']+'`.','',
        'All comparisons use native clean-validation-selected checkpoints and their own stored validation-real10%FPR thresholds.',
        'RESULT.json contains complete group metrics, all contrasts, paired intervals/usable counts and separate fixed0.5 diagnostics.',
        'Unsupported primary uncertainty stays inconclusive. No new TEST inference, cross-domain win or feature-disentanglement claim.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(assessment))


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    q=sub.add_parser('bind');q.add_argument('--references',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('diagnose');q.add_argument('--binding',required=True);q.add_argument('--out',required=True);q.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    q=sub.add_parser('cuda-check');q.add_argument('--binding',required=True);q.add_argument('--diagnostic',required=True);q.add_argument('--out',required=True)
    q=sub.add_parser('train');q.add_argument('--binding',required=True);q.add_argument('--diagnostic',required=True);q.add_argument('--dry-run-receipt',required=True);q.add_argument('--software-receipt',required=True);q.add_argument('--out',required=True);q.add_argument('--budget-hours',type=float,required=True)
    q=sub.add_parser('report');q.add_argument('--binding',required=True);q.add_argument('--candidate',required=True);q.add_argument('--out',required=True)
    a=parser.parse_args();torch.set_num_threads(2)
    if a.command=='bind':bind(a.references,a.out)
    elif a.command=='diagnose':audit_train(a.binding,a.out,torch.device(a.device))
    elif a.command=='cuda-check':cuda_check(a.binding,a.diagnostic,a.out)
    elif a.command=='train':train_c1(a.binding,a.diagnostic,a.dry_run_receipt,a.software_receipt,a.out,a.budget_hours)
    else:report(a.binding,a.candidate,a.out)


if __name__=='__main__':main()
