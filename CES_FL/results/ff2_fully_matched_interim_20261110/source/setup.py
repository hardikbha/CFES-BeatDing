"""Bind the FF2 common-recipe matrix and its unchanged procedural recipe panel."""
import copy
import importlib.util
import json
from pathlib import Path

SOURCE=Path(__file__).resolve().parent
ROOT=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_fully_matched_20261110_r1')
OLD=Path('/home/sachin.chaudhary/ces_fl_runs/ff2_lh_ps_20261106_r1')
spec=importlib.util.spec_from_file_location('shared_setup',SOURCE/'src/shared.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def qualify_recipe(c,folder,train_lookup):
    """Explicit recipe-panel reuse, not invented human/per-training-image review."""
    q=json.loads((OLD/'qualification.json').read_text())
    assert q['source_sha256']==m.core.sha(m.core.__file__)
    for key in ('preview','review'):assert m.core.sha(q[key+'_file'])==q[key+'_sha256']
    manifest=json.loads(Path(q['preview_file']).read_text())
    original_review=json.loads(Path(q['review_file']).read_text())
    assert manifest['source_sha256']==q['source_sha256'] and len(manifest['cases'])==88
    assert original_review['preview_sha256']==q['preview_sha256']
    for item in manifest['cases']:
        assert train_lookup[item['image']]==item['image_sha256']
        assert m.core.sha(item['sheet'])==item['sheet_sha256']
    # All 88 actual reviewed assets/settings stay unchanged. The panel itself
    # intentionally uses fixed seed17, independently of downstream training seed.
    manifest=copy.deepcopy(manifest)
    manifest['config_digest']=m.core.digest(c)
    manifest['panel_reuse']={'policy':'fixed_seed17_method_neutral_PS_v2_recipe_panel',
        'original_qualification':str(OLD/'qualification.json'),
        'original_qualification_sha256':m.core.sha(OLD/'qualification.json'),
        'training_seed':c['seed'],'panel_seed':17,'new_visual_review_performed':False,
        'scope':'Unchanged procedural operator recipe and parameter ranges; not every training image or real-forgery validity'}
    preview=folder/'preview';preview.mkdir()
    m.core.write_json(preview/'preview_PRIVATE.json',manifest)
    review=copy.deepcopy(original_review)
    review['preview_sha256']=m.core.sha(preview/'preview_PRIVATE.json')
    review['reuse_provenance']=manifest['panel_reuse']
    m.core.write_json(folder/'review_PRIVATE.json',review)
    m.core.qualify(preview,folder/'review_PRIVATE.json',folder/'qualification.json')

def main():
    audit=ROOT/'shared_input_audit'
    receipt=json.loads((audit/'PREFLIGHT.json').read_text())
    assert receipt['status']=='PASS' and receipt['decoded_images']==249373
    inventory=audit/'inventory_PRIVATE.json'
    assert m.core.sha(inventory)==receipt['inventory_sha256']
    inv=json.loads(inventory.read_text());train_lookup={r['image']:r['sha256'] for r in inv if r['split']=='train'}
    template=json.loads((OLD/'L_H_PS/config.json').read_text())
    rows=json.loads(Path(receipt['row_cache_path']).read_text());train,val=rows['train'],rows['val']
    assert len(train)==218266 and len(val)==31107
    groups=sorted({r['gender_race'] for r in train});assert len(groups)==8 and 'unknown' not in groups
    test=Path('/home/sachin.chaudhary/ces_fl_pipeline/manifests/ff2/test.csv')
    configs=[];recipe=None
    for seed in m.SEEDS:
        for method in m.METHODS:
            c=dict(template,seed=seed,axes=['gender_race'],workers=4,max_gpu_hours=6.,
                suite_format=m.VERSION,suite_method=method,suite_hashes=m.hashes(),
                common_classification='unweighted_BCE',group_names=groups,
                ding_lambda=.005,sinkhorn_blur=.0005,dag_alpha=.5,daw_inner_alpha=.9,daw_outer_alpha=.5,
                sfd_file=str(ROOT/'SFD_FROZEN.json'),test=str(test),test_sha256=m.core.sha(test),
                operator_review_policy='fixed_seed17_method_neutral_PS_v2_recipe_panel')
            m.check(c)
            comparable={k:v for k,v in c.items() if k not in ('suite_method','seed')}
            if recipe is None:recipe=comparable
            else:assert recipe==comparable
            name=f'{method}_s{seed}';folder=ROOT/name;folder.mkdir()
            cfg=SOURCE/'proposals/ff2_fully_matched_20261110_r1'/f'{name}.json'
            m.core.write_json(cfg,c)
            pre=folder/'preflight';pre.mkdir();(pre/'inventory_PRIVATE.json').symlink_to(inventory)
            _,support=m.core.group_inventory(train,c)
            bound=dict(receipt,config_digest=m.core.digest(c),groups=support,
                derivation='Shared fresh full image audit; source-identical pixels/recipe; common joint auxiliary axis and new suite configuration binding',
                parent_audit_sha256=m.core.sha(audit/'PREFLIGHT.json'))
            m.core.write_json(pre/'PREFLIGHT.json',bound)
            qualify_recipe(c,folder,train_lookup)
            m.core.write_json(folder/'CONFIG_BINDING.json',{'method':method,'seed':seed,'config_path':str(cfg),
                'config_sha256':m.core.sha(cfg),'suite_hashes':m.hashes(),'qualification_sha256':m.core.sha(folder/'qualification.json')})
            configs.append({'name':name,'seed':seed,'method':method,'state':'PREPARED'})
            print('PREPARED',name,flush=True)
    m.core.write_json(ROOT/'COMMON_RECIPE.json',recipe)
    m.core.write_json(ROOT/'MATRIX_VERIFIED.json',{'status':'PASS','runs':21,'methods':list(m.METHODS),
        'seeds':list(m.SEEDS),'only_different_fields':['seed','suite_method'],
        'shared_recipe_digest':m.core.digest(recipe),'test_predictions_read':False})
    m.core.write_json(ROOT/'QUEUE.json',{'jobs':[{'name':'SFD','seed':17,'method':'sfd','state':'PREPARED'}]+configs})
    m.core.write_json(ROOT/'PREPARATION.json',{'status':'PASS','configs':21,'image_audit':receipt['decoded_images'],
        'shared_encoder_pretraining_seed':17,'downstream_seeds':list(m.SEEDS),'suite_hashes':m.hashes(),
        'operator_panel':'88 unchanged PS-v2 recipe cases; explicit reuse; no new human approval',
        'test_access':'manifest hash frozen; no predictions read'})

if __name__=='__main__':main()
