"""Separate prospective CDF/response study; no historical trainer is patched.

Only explicit subcommands execute. Production training requires a signed external
release receipt, exact input hashes, and preflight. No TEST/evaluation command.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import ces_fl_counterfactual_5050 as base
import ces_fl_cdf_objectives as losses

VERSION = 'CESFL_CDF_RESPONSE_20261102_R1'
BASE_SHA = '3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab'
VIEWS = ('original', 'resize', 'x0', 'xa', 't0', 'ta')


def hashes():
    if base.sha(base.__file__) != BASE_SHA:
        raise ValueError('Historical architecture dependency changed')
    return {Path(p).name: base.sha(p) for p in [__file__, losses.__file__, base.__file__]}


def read_config(path):
    c = json.loads(Path(path).read_text())
    if c['format'] != VERSION or c['dataset'] != 'ding_ffpp' or c['selection'] != 'validation_auc_only' or c['fairness_axis'] != 'gender_race':
        raise ValueError('Invalid study/selection/group definition')
    if c['backbone'] not in ('xception', 'resnet50') or c['seed'] not in c['declared_seeds']:
        raise ValueError('Undeclared backbone/seed')
    if c['epochs'] != 50 or c['pretrain_epochs'] != 50 or c['size'] != 224:
        raise ValueError('Production budget/preprocessing changed')
    if c['batch_size'] != 2*len(c['groups'])*c['per_cell'] or c['per_cell'] < 2:
        raise ValueError('Complete label×group quota batch required')
    if (c['pretrain_batch_size'] != 16 or c['target_fpr'] != .1 or c['min_support'] != 20
            or c['map_consistency'] is not False or c['test_manifest'] is not None):
        raise ValueError('Protocol field changed or TEST supplied to training')
    for key in ['lr', 'encoder_lr', 'pretrain_encoder_lr', 'weight_decay']:
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError('Invalid optimizer setting')
    hashes()
    return c


class QuotaBatches:
    """Same finite replacement-sampling schedule in every downstream arm."""
    def __init__(self, rows, groups, per_cell, seed, epoch):
        self.cells = [[i for i, r in enumerate(rows) if r['label'] == y and r['gender_race'] == g]
                      for y in (0, 1) for g in groups]
        if any(not cell for cell in self.cells):
            raise ValueError('Missing TRAIN label×group cell')
        self.per_cell, self.seed, self.epoch = per_cell, seed, epoch
        self.steps = math.ceil(len(rows)/(len(self.cells)*per_cell))

    def __len__(self): return self.steps

    def __iter__(self):
        rng = random.Random(f'quota:{self.seed}:{self.epoch}')
        for _ in range(self.steps):
            batch = [rng.choice(cell) for cell in self.cells for _ in range(self.per_cell)]
            rng.shuffle(batch)
            yield batch


class Originals(Dataset):
    def __init__(self, rows, c):
        self.rows, self.c = rows, c
        self.reader = base.PairImages([], size=c['size'], backbone=c['backbone'])

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {'x': self.reader.load(r['image']), 'y': r['label'],
                'g': self.c['groups'].index(r['gender_race']), 'index': i}


def audit_bank(c, rows, val, *, synthetic=False):
    """Recorded approvals are checked, never manufactured or scientifically certified."""
    if not c['bank'] or base.sha(c['bank']) != c['bank_sha256']:
        raise ValueError('BLOCKED_INPUTS: reviewed matched quartet bank missing/changed')
    b = json.loads(Path(c['bank']).read_text())
    if b['format'] != VERSION or b['synthetic_fixture'] != synthetic:
        raise ValueError('Bank provenance/fixture role mismatch')
    if b['train_sha256'] != c['input_hashes']['train']:
        raise ValueError('Wrong TRAIN membership')
    if not c['heldout_exclusions'] or base.sha(c['heldout_exclusions']) != c['heldout_exclusions_sha256']:
        raise ValueError('BLOCKED_INPUTS: independent frozen held-out exclusion receipt required')
    excluded = json.loads(Path(c['heldout_exclusions']).read_text())
    if excluded['status'] != 'PASS' or excluded['test_scores_used'] or excluded['synthetic_fixture'] != synthetic:
        raise ValueError('Exclusion receipt must be metadata-only and independently verified')
    index = {r['image']: r for r in rows}
    held_ids = set(excluded['heldout_contributor_ids']) | {v for r in val for k in ['identity_ids', 'source_ids'] for v in r[k].split(';')}
    train_ids = {v for r in rows for k in ['identity_ids', 'source_ids'] for v in r[k].split(';')}
    seen, source_directions, quota, operators = set(), {}, Counter(), set()
    source_operations, components, component_sources, contributor_components = {}, {}, {}, {}
    operator_quota = Counter()
    shared_assets = {}
    for q in b['quartets']:
        if q['id'] in seen or q['direction'] not in ('older', 'younger'):
            raise ValueError('Duplicate quartet/wrong age direction')
        seen.add(q['id']); source = q['source_image']
        if source not in index or index[source]['label'] != 0 or base.sha(source) != q['source_sha256']:
            raise ValueError('Source must be an unchanged real TRAIN original')
        group = index[source]['gender_race']
        if q['group'] != group:
            raise ValueError('Source group metadata mismatch')
        pair = (source, q['direction'])
        if pair in source_directions:
            raise ValueError('Repeated source/direction is not coverage')
        source_directions[pair] = q['id']; quota[(group, q['direction'])] += 1
        contributors = set(q['contributors'])
        required = {v for k in ['identity_ids', 'source_ids'] for v in index[source][k].split(';')}
        if not required <= contributors or not contributors <= train_ids or contributors & held_ids:
            raise ValueError('Missing/held-out/non-TRAIN contributor')
        reviews = q['reviews']
        if len(reviews) != 2 or len({v['reviewer'] for v in reviews}) != 2:
            raise ValueError('Two independent reviewers required')
        for review in reviews:
            fields = ['approved', 'identity_preserved', 'age_direction_valid', 'operation_valid',
                      'matched_processing', 'forensic_evidence_assessed', 'unrelated_changes_assessed',
                      'authenticity_preservation_supported', 'forensic_preservation_supported']
            if any(review.get(k) is not True for k in fields) or not review['evidence'] or not review['reviewer'].strip() or review['reviewer'] in q['generators']:
                raise ValueError('Uncertain/unreviewed/self-reviewed quartet')
            evidence = review['evidence']
            if base.sha(evidence['path']) != evidence['sha256']:
                raise ValueError('Changed/missing independent review evidence')
            if review['image_hashes'] != {v:q['views'][v]['sha256'] for v in VIEWS}:
                raise ValueError('Review is not bound to the actual six assets')
        for receipt in [q['processing_receipt'], q['operator_receipt']]:
            if base.sha(receipt['path']) != receipt['sha256']:
                raise ValueError('Changed/missing processing/operation provenance')
        operators.add(q['operator_family'])
        component = q['source_component_id']
        if excluded['train_component_by_image_sha256'].get(q['source_sha256']) != component:
            raise ValueError('Source component not independently mapped')
        for contributor in contributors:
            if contributor in contributor_components and contributor_components[contributor] != component:
                raise ValueError('Shared source/donor contributor split across claimed independent components')
            contributor_components[contributor] = component
        if component in components and components[component] != group:
            raise ValueError('Component group labels require adjudication')
        components[component] = group
        component_sources.setdefault(component,set()).add(source)
        operation = (q['operator_family'],q['views']['t0']['operator_code_sha256'],q['views']['t0']['settings_sha256'],q['views']['t0']['donor_sha256'],q['views']['t0']['seed'])
        if source in source_operations and source_operations[source] != operation:
            raise ValueError('Age direction confounded with operation family/donor/settings')
        source_operations[source] = operation
        operator_quota[(group,q['direction'],q['operator_family'])] += 1
        source_size = None
        for role in VIEWS:
            a = q['views'][role]; image = str(Path(a['image']).resolve())
            if base.sha(image) != a['sha256'] or (role == 'original' and image != source):
                raise ValueError('Changed/misaligned asset')
            if image in shared_assets and shared_assets[image] != (source, role, a['sha256']):
                raise ValueError('Asset reused across unrelated sources or roles')
            shared_assets[image] = (source, role, a['sha256'])
            with base.Image.open(image) as im:
                im.load()
                if role == 'original': source_size = im.size
                elif im.size != source_size: raise ValueError('Canvas mismatch')
        if q['views']['x0']['parent_sha256'] != q['source_sha256'] or q['views']['xa']['parent_sha256'] != q['source_sha256']:
            raise ValueError('Edits must independently originate from original')
        if q['views']['x0']['condition'] != 'nochange' or q['views']['xa']['condition'] != q['direction']:
            raise ValueError('Wrong no-change/demographic edit condition')
        for role, parent in [('t0', 'x0'), ('ta', 'xa')]:
            if q['views'][role]['parent_sha256'] != q['views'][parent]['sha256']:
                raise ValueError('Wrong operation parent')
        for field in ['operator_code_sha256', 'settings_sha256', 'donor_sha256', 'seed']:
            if not q['views']['t0'].get(field) or q['views']['t0'][field] != q['views']['ta'].get(field):
                raise ValueError('Unmatched operator/donor/settings')
        for field in ['editor_revision', 'settings_sha256', 'seed']:
            if q['views']['x0'].get(field) != q['views']['xa'].get(field) or q['views']['x0'].get(field) is None:
                raise ValueError('Unmatched editor settings/seed')
        if any(q['views'][role]['sha256'] == q['views'][role]['parent_sha256'] for role in ['xa', 't0', 'ta']):
            raise ValueError('Byte-identical intervention')
    per_group = 1 if synthetic else 40
    if any(quota[(g, d)] < per_group for g in c['groups'] for d in ['older', 'younger']):
        raise ValueError('BLOCKED_INPUTS: subgroup×direction source coverage below fixed floor')
    sources = {source for source, _ in source_directions}
    if any((source, d) not in source_directions for source in sources for d in ['older', 'younger']):
        raise ValueError('Both directions required per source')
    if not synthetic and (len(sources) < 320 or len(operators) < 2):
        raise ValueError('BLOCKED_INPUTS: need320sources and2validated operator families')
    if not synthetic and (len(operators)!=2 or any(operator_quota[(g,d,o)]<20 for g in c['groups'] for d in ['older','younger'] for o in operators)
            or any(sum(group==g for group in components.values())<5 for g in c['groups'])
            or any(len(v)>8 for v in component_sources.values())):
        raise ValueError('BLOCKED_INPUTS: operation/group coverage or independent-source/reuse floor failed')
    return sorted(b['quartets'], key=lambda q: (q['group'], q['direction'], q['id']))


def data_preflight(c, arm, *, synthetic=False):
    for k, expected in c['input_hashes'].items():
        if base.sha(c[k]) != expected: raise ValueError('Input hash changed: '+k)
    rows, val = base.read_rows(c['train']), base.read_rows(c['val'])
    split = base.check_splits(rows, val, c['protocol_overlap_file'])
    counts = Counter((r['label'], r['gender_race']) for r in rows)
    if any(r['ids8'].strip().lower() in ('unknown','','none','nan') for r in rows):
        raise ValueError('Unknown pretraining-balancing metadata')
    if set(g for _, g in counts) != set(c['groups']) or any(counts[(y, g)] < (2 if synthetic else 20) for y in (0, 1) for g in c['groups']):
        raise ValueError('Wrong/inadequate TRAIN group coverage')
    priors = [counts[(0, g)]/sum(counts[(0, h)] for h in c['groups']) for g in c['groups']]
    bank = audit_bank(c, rows, val, synthetic=synthetic) if arm in ('nochange', 'score', 'response') else []
    return rows, val, priors, split, bank


def release_check(config_path, c, approval):
    if not approval: raise ValueError('DO NOT RUN: independent review/launch release not supplied')
    r = json.loads(Path(approval).read_text())
    if (r.get('approved') is not True or not r.get('reviewer') or not r.get('evidence')
            or r.get('config_sha256') != base.sha(config_path) or r.get('source_hashes') != hashes()):
        raise ValueError('Changed/missing independent review release')
    if r.get('fairforge2_release') or c['dataset'] != 'ding_ffpp':
        raise ValueError('FairForge-2 is outside this release')


def make_model(c, device, checkpoint=None, *, synthetic=False):
    base.seed_all(c['seed'])
    m = base.DiagramCES(c['backbone'], c['weights'], hidden=c['hidden'], gate_hidden=c['gate_hidden'],
                      pretrained=checkpoint is None and not synthetic, encoder_policy='trainable',
                      bn_policy='update' if checkpoint is None else 'frozen', head_seed=c['seed'])
    if checkpoint is not None:
        m.encoder.load_state_dict({k[8:]: v for k, v in checkpoint['model'].items() if k.startswith('encoder.')}, strict=True)
    return m.to(device)


def load_pretrain(c, *, synthetic=False):
    if not c['pretrain_run'] or base.sha(Path(c['pretrain_run'])/'best.pt') != c['pretrain_sha256']:
        raise ValueError('BLOCKED_INPUTS: bind same verified pretraining checkpoint per backbone/seed')
    root = Path(c['pretrain_run'])
    receipt=json.loads((root/'RUN_VERIFIED.json').read_text())
    # Native P50 records its budget in summary/history, not RUN_VERIFIED.
    # Check those actual records rather than manufacturing a compatibility key.
    if 'completed_epochs' in receipt:
        completed = receipt['completed_epochs']
    else:
        summary=json.loads((root/'summary.json').read_text())
        history=[json.loads(line) for line in (root/'history.jsonl').read_text().splitlines()]
        completed=summary['completed_epochs']
        if (summary['status'] != 'COMPLETED_VERIFIED' or summary['stage'] != 'pretrain'
                or [v['epoch'] for v in history] != list(range(1,completed+1))
                or not receipt['validation_replayed'] or receipt['test_read']):
            raise ValueError('Native P50 summary/history/replay mismatch')
    if (not (root/'DONE').exists() or receipt['status'] != 'PASS' or completed != c['pretrain_epochs']
            or receipt['checkpoint_sha256'] != c['pretrain_sha256']):
        raise ValueError('Pretraining not verified')
    ck = torch.load(root/'best.pt', map_location='cpu', weights_only=True)
    if ck['stage'] != 'pretrain' or ck['synthetic_fixture'] != synthetic or ck['config']['seed'] != c['seed'] or ck['config']['backbone'] != c['backbone']:
        raise ValueError('Pretraining seed/backbone/role mismatch')
    for key in ['train', 'val', 'weights']:
        if ck['config']['input_hashes'][key] != c['input_hashes'][key]: raise ValueError('Wrong pretraining inputs')
    for key in ['size','hidden','gate_hidden','lr','pretrain_encoder_lr','weight_decay']:
        if ck['config'][key] != c[key]: raise ValueError('Wrong pretraining model/optimizer settings')
    if ck['format'] not in [VERSION, base.VERSION]: raise ValueError('Unknown pretrain format')
    if ck['format'] == base.VERSION and (ck['source_sha256'] != BASE_SHA or ck['config']['pretrain_epochs'] != 50):
        raise ValueError('Wrong nativeP50 provenance')
    if ck['format'] == VERSION and ck['hashes'] != hashes(): raise ValueError('Changed new pretrain source')
    return ck


def run(c, arm, out, device, *, dry=False, synthetic=False):
    if c.get('test_manifest') is not None:
        raise ValueError('TEST cannot be supplied to a training/dry-run path')
    rows, val, prior, split, bank = data_preflight(c, arm, synthetic=synthetic)
    pre = None if arm == 'pretrain' else load_pretrain(c, synthetic=synthetic)
    model = make_model(c, device, pre, synthetic=synthetic); del pre
    pconfig = {**c, 'encoder_lr': c['pretrain_encoder_lr'] if arm == 'pretrain' else c['encoder_lr']}
    optimizer = base.build_optimizer(model, pconfig)
    dro = losses.CompleteGroupDRO(2*len(prior), c['dro_eta']).to(device)
    priors = torch.tensor(prior, device=device)
    out = base.fresh(out); base.write(out/'config.json', c)
    heads = {k: v for k, v in model.state_dict().items() if not k.startswith('encoder.')}
    initial_heads = hashlib.sha256(b''.join(v.detach().cpu().contiguous().numpy().tobytes() for _, v in sorted(heads.items()))).hexdigest()
    base.write(out/'provenance.json', {'hashes': hashes(), 'parameters': model.counts(), 'splits': split,
        'arm': arm, 'heads_initial_sha256': initial_heads, 'TRAIN_real_priors': prior,
        'optimizer_groups': [{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups],
        'synthetic_fixture': synthetic, 'test_read': False, 'bank_unique_sources': len({q['source_image'] for q in bank})})
    epochs = c['pretrain_epochs'] if arm == 'pretrain' else c['epochs']
    best = -math.inf; usage = Counter()
    for epoch in range(epochs):
        model.train(); base.seed_all(c['seed']+epoch)
        pair_order=list(bank);random.Random(f'pairs:{c["seed"]}:{epoch}').shuffle(pair_order)
        if arm == 'pretrain':
            batches = base.EpochBatches(len(rows), c['pretrain_batch_size'], c['seed'], epoch)
        else:
            batches = QuotaBatches(rows, c['groups'], c['per_cell'], c['seed'], epoch)
        loader = DataLoader(Originals(rows,c), batch_sampler=batches, num_workers=c['workers'],
                            generator=torch.Generator().manual_seed(c['seed']+epoch))
        totals=Counter(); exposures=Counter(); counts=Counter(); steps=0
        cell_counts=Counter((r['label'],r['ids8']) for r in rows)
        for step,b in enumerate(loader):
            z=model(b['x'].to(device))['logit']; y=b['y'].float().to(device); groups=b['g'].to(device)
            q=None
            chosen=[pair_order[(step*c['pair_batch']+j)%len(pair_order)] for j in range(c['pair_batch'])] if bank else []
            if chosen:
                reader=base.PairImages([],size=c['size'],backbone=c['backbone'])
                # Same six forwards, order and source schedule in all paired arms.
                images=torch.stack([reader.load(x['views'][v]['image']) for v in VIEWS for x in chosen]).to(device)
                scores=model(images)['logit']; q={v:scores[i*len(chosen):(i+1)*len(chosen)] for i,v in enumerate(VIEWS)}
                usage.update(x['id'] for x in chosen)
            if arm=='pretrain':
                weights=z.new_tensor([len(rows)/(len(cell_counts)*cell_counts[(rows[i]['label'],rows[i]['ids8'])]) for i in b['index'].tolist()])
                classification=(weights*torch.nn.functional.binary_cross_entropy_with_logits(z,y,reduction='none')).mean()
                values={'total':classification,'classification':classification}
            else:
                values=losses.objective(z,y,groups,priors,arm,dro=dro,q=q,**c['loss'])
            optimizer.zero_grad(set_to_none=True); values['total'].backward()
            if not torch.isfinite(values['total']) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError('Nonfinite/missing parameter gradients')
            if dry:
                base.write(out/'DRY_RUN.json',{'status':'PASS','arm':arm,'device':device,'optimizer_steps':0,
                    'hashes':hashes(),'losses':{k:float(v.detach()) for k,v in values.items()},'paired_views':len(chosen)*6,'synthetic_fixture':synthetic})
                return
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();steps+=1
            exposures.update(b['index'].tolist());counts.update(zip(b['y'].tolist(),b['g'].tolist()))
            for k,v in values.items():totals[k]+=float(v.detach())
        z=base.predict(model,val,device,c['validation_batch_size'],c['workers'],c['size'])
        cal=base.calibrate(z[np.array([r['label']==0 for r in val])],c['target_fpr'])
        report=base.metrics(val,z,cal['threshold_logit'],c['min_support'])
        ck={'format':VERSION,'stage':'pretrain' if arm=='pretrain' else 'downstream','config':c,'arm':arm,
            'epoch':epoch+1,'hashes':hashes(),'model':model.state_dict(),'optimizer':optimizer.state_dict(),
            'dro':dro.state_dict(),'calibration':cal,'validation':report,'synthetic_fixture':synthetic}
        if report['overall']['auc']>best:best=report['overall']['auc'];base.atomic_checkpoint(out/'best.pt',ck)
        base.atomic_checkpoint(out/'last.pt',ck)
        record={'epoch':epoch+1,'validation':report,'calibration':cal,'optimizer_steps':steps,
            'original_draws':sum(exposures.values()),'unique_originals':len(exposures),'label_group_draws':{str(k):v for k,v in counts.items()},
            'losses':{k:v/steps for k,v in totals.items()},'pair_draws':sum(usage.values()),'unique_quartets':len(usage)}
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        if bank:
            base.write(out/'pair_usage.json',{'quartet_draws':dict(usage),'unique_quartets':len(usage),
                'unique_source_images':len({q['source_image'] for q in bank if usage[q['id']]}),
                'unique_recorded_source_components':len({q['source_component_id'] for q in bank if usage[q['id']]})})
        print(json.dumps({'epoch':epoch+1,'arm':arm,'val_auc':report['overall']['auc']}),flush=True)
    selected=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(selected['model'],strict=True)
    actual=base.metrics(val,base.predict(model,val,device,c['validation_batch_size'],c['workers'],c['size']),selected['calibration']['threshold_logit'],c['min_support'])
    if actual!=selected['validation']:raise RuntimeError('Selected checkpoint replay mismatch')
    base.write(out/'RUN_VERIFIED.json',{'status':'PASS','checkpoint_sha256':base.sha(out/'best.pt'),'completed_epochs':epochs,
        'selected_epoch':selected['epoch'],'hashes':hashes(),'test_read':False,'validation_replayed':True,'synthetic_fixture':synthetic})
    (out/'DONE').write_text('Completed and VAL replay verified; no TEST evaluation.\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','dry-run','train','bind-pretrain','bind-bank'])
    p.add_argument('--config',required=True);p.add_argument('--arm',choices=['pretrain',*losses.ARMS],required=True)
    p.add_argument('--out',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--approval')
    p.add_argument('--pretrain-run');p.add_argument('--bank');p.add_argument('--heldout-exclusions')
    a=p.parse_args();c=read_config(a.config)
    if a.command=='bind-pretrain':
        if not a.pretrain_run:raise ValueError('Specify actual pretrain output')
        c['pretrain_run']=str(Path(a.pretrain_run).resolve());c['pretrain_sha256']=base.sha(Path(c['pretrain_run'])/'best.pt')
        load_pretrain(c);base.write(base.fresh(a.out)/'bound_config.json',c);return
    if a.command=='bind-bank':
        if not a.bank or not a.heldout_exclusions:raise ValueError('Specify actual bank/exclusion metadata receipt')
        c['bank']=str(Path(a.bank).resolve());c['bank_sha256']=base.sha(c['bank'])
        c['heldout_exclusions']=str(Path(a.heldout_exclusions).resolve());c['heldout_exclusions_sha256']=base.sha(c['heldout_exclusions'])
        data_preflight(c,'response');load_pretrain(c);base.write(base.fresh(a.out)/'bound_config.json',c);return
    if a.command=='train':release_check(a.config,c,a.approval)
    if a.command=='preflight':
        rows,val,prior,split,bank=data_preflight(c,a.arm)
        if a.arm!='pretrain':load_pretrain(c)
        base.write(base.fresh(a.out)/'PREFLIGHT.json',{'status':'PASS','hashes':hashes(),'config_sha256':base.sha(a.config),'train':len(rows),'val':len(val),'priors':prior,'splits':split,'quartets':len(bank)})
    else:run(c,a.arm,a.out,a.device,dry=a.command=='dry-run')


if __name__=='__main__':main()
