"""TRAIN support/sampling and VAL component metadata; never opens TEST/images."""
import argparse,importlib.util,json,random,math
from pathlib import Path
from collections import Counter

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 s=importlib.util.spec_from_file_location('cesfl',Path(a.source)/'CES_FL/src/ces_fl_forensic_response.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
 c=m.load_config(a.config);train=m.read_manifest(c['train']);val=m.read_manifest(c['val']);active,support=m.group_inventory(train,c);weights=m.original_weights(train,c['balance_axis']);inventory={}
 for axis,groups in active.items():
  inventory[axis]={g:{str(y):dict(rows=len(cell[y]),recorded_tuples=len({m.recorded_contributor(train[i]) for i in cell[y]}),sum_original_BCE_weights=float(weights[cell[y]].sum())) for y in (0,1)} for g,cell in groups.items()}
 planned=Counter();n=c['group_per_class'];events=math.ceil(math.ceil(len(train)/c['batch_size'])/c['aux_every'])
 for event in range(events):
  axis=c['axes'][event%len(c['axes'])];rng=random.Random(m.seed_for(c['seed'],0,event,'aux'))
  for g,cell in active[axis].items():
   for label in (0,1):
    pool=cell[label];chosen=rng.sample(pool,n) if len(pool)>=n else rng.choices(pool,k=n);u=len({m.recorded_contributor(train[i]) for i in chosen});planned[f'{axis}:{g}:{label}:U={u}/n={n}']+=1
 components=m.components(val);sizes=sorted(map(len,components),reverse=True);vgroup={}
 for axis in c['axes']:
  groups=sorted({r[axis] for r in val});vgroup[axis]={}
  for g in groups:
   vgroup[axis][g]={str(label):dict(rows=sum(r[axis]==g and r['label']==label for r in val),components_with_rows=sum(any(val[i][axis]==g and val[i]['label']==label for i in comp) for comp in components)) for label in (0,1)}
 result=dict(status='PASS',scientific_source_sha256=m.sha(m.__file__),config_digest=m.digest(c),images_decoded=False,test_read=False,train_support=inventory,planned_epoch0_auxiliary_events=events,planned_contributor_diversity_histogram=dict(planned),validation_components=dict(number=len(components),largest_rows=sizes[0],largest_fraction=sizes[0]/len(val),top10_sizes=sizes[:10],group_class_support=vgroup),warning='Recorded contributor components/tuples do not certify biological independence; histograms are plans, not consumed training exposures')
 with Path(a.out).open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(status='PASS',train_rows=len(train),val_rows=len(val),validation_components=len(components),largest_component_rows=sizes[0],test_read=False)))
if __name__=='__main__':main()
