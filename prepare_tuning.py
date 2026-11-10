"""CPU-only fresh input audit and binding for four fixed schedule candidates."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

SOURCE=Path(__file__).resolve().parent
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_response_tuning_20261110_r1')
PARENT=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_fully_matched_20261110_r1')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def audit():
    cache=load(SOURCE/'data_cache.py','fresh_schedule_cache')
    cache.ROOT=ROOT
    if (ROOT/'shared_input_audit').exists():raise RuntimeError('Preserve any existing fresh input audit')
    cache.main()


def bind():
    m=load(SOURCE/'schedule_tuning.py','tuning_setup')
    old_setup=load(SOURCE/'setup.py','parent_recipe_qualification')
    record=json.loads((ROOT/'shared_input_audit/PREFLIGHT.json').read_text());assert record['status']=='PASS' and record['decoded_images']==249373
    assert m.core.sha(ROOT/'shared_input_audit/inventory_PRIVATE.json')==record['inventory_sha256']
    prior=json.loads((PARENT/'shared_input_audit/PREFLIGHT.json').read_text())
    assert record['inventory_sha256']==prior['inventory_sha256']
    cache=json.loads(Path(record['row_cache_path']).read_text());train=cache['train'];lookup={r['image']:r['image_sha256'] for r in train}
    template=json.loads((PARENT/'lh_s17/training/config.json').read_text());jobs=[]
    for name,ceiling in m.CEILINGS.items():
        c=dict(template,max_gpu_hours=4.,tuning_format=m.VERSION,tuning_name=name,tuning_source_sha256=m.core.sha(SOURCE/'schedule_tuning.py'),
            response_schedule=dict(delay_epochs=2,ramp_epochs=6,ceiling=ceiling,shape='smoothstep'))
        for key in ('test','test_sha256','sfd_file'):c.pop(key,None)
        c['suite_method']='lh';m.check(c)
        folder=ROOT/name;folder.mkdir(exist_ok=False);pre=folder/'preflight';pre.mkdir()
        (pre/'inventory_PRIVATE.json').symlink_to(ROOT/'shared_input_audit/inventory_PRIVATE.json')
        _,support=m.core.group_inventory(train,c)
        m.core.write_json(pre/'PREFLIGHT.json',dict(record,config_digest=m.core.digest(c),groups=support,
            derivation='Fresh identical TRAIN/SELECT bytes; unchanged model/operator; fixed response schedule only',
            parent_audit_sha256=m.core.sha(ROOT/'shared_input_audit/PREFLIGHT.json')))
        old_setup.qualify_recipe(c,folder,lookup)
        config=SOURCE/'proposals/ff2_response_tuning_20261110_r1'/f'{name}.json';m.core.write_json(config,c)
        m.core.write_json(folder/'CONFIG_BINDING.json',dict(config_path=str(config),config_sha256=m.core.sha(config),
            qualification_sha256=m.core.sha(folder/'qualification.json'),source_sha256=c['source_sha256'],tuning_source_sha256=c['tuning_source_sha256']))
        jobs.append(dict(name=name,ceiling=ceiling,state='PREPARED',seed=17,epochs=10))
    compare=[]
    for job in jobs:
        c=json.loads((SOURCE/'proposals/ff2_response_tuning_20261110_r1'/f"{job['name']}.json").read_text())
        c.pop('tuning_name');c['response_schedule'].pop('ceiling');compare.append(c)
    assert all(c==compare[0] for c in compare)
    m.core.write_json(ROOT/'MATRIX_VERIFIED.json',dict(status='PASS',runs=4,seed=17,epochs=10,only_scientific_difference='response ceiling',
        common_delay_epochs=2,common_smooth_ramp_epochs=6,ceilings=list(m.CEILINGS.values()),no_TEST_CAL_dependency=True))
    m.core.write_json(ROOT/'QUEUE.json',dict(jobs=jobs))
    m.core.write_json(ROOT/'PREPARATION.json',dict(status='PASS',fresh_images_hashed_decoded=249373,
        inventory_matches_prior=True,recipe_panel='88 unchanged PS-v2 cases with original review decisions; no new human or per-image approval',
        qualification_reuse_explicit=True,TEST_CAL_access=False))
    print('PREPARED_FOUR_FIXED_SCHEDULES',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('audit','bind'));a=p.parse_args()
    audit() if a.stage=='audit' else bind()
