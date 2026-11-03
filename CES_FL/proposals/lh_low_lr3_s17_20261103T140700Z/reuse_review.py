"""Copy genuine named decisions only after full case/settings/byte equivalence."""
import argparse,copy,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('original_review_guard',HERE.parent/'plain_head_s17_20261103T101418Z/reuse_review.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior);m=prior.m
def main():
 p=argparse.ArgumentParser();p.add_argument('--preview',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 oldq=Path('/home/sachin.chaudhary/ces_fl_runs/plain_head_s17_20261103T101418Z/review_H/qualification.json')
 oldc=json.loads((HERE.parent/'plain_head_s17_20261103T101418Z/L_H_plain_hierarchical.json').read_text())
 newc=m.load_config(HERE/'config.json');assert {k for k in newc if oldc[k]!=newc[k]}=={'head_lr','encoder_lr','epochs'}
 m.check_qualification(oldc,oldq);q=json.loads(oldq.read_text());old=json.loads(Path(q['preview_file']).read_text())
 pv=Path(a.preview)/'preview_PRIVATE.json';new=json.loads(pv.read_text())
 assert new['config_digest']==m.digest(newc) and new['source_sha256']==m.sha(m.__file__)
 prior.equivalent_cases(old,new)
 review=json.loads(Path(q['review_file']).read_text());assert review['preview_sha256']==q['preview_sha256']
 copied=copy.deepcopy(review);copied['preview_sha256']=m.sha(pv)
 copied['reuse_provenance']=dict(qualification_path=str(oldq),qualification_sha256=m.sha(oldq),basis='Every case ID, settings, source image bytes and sheet bytes identical; all named decisions/flags/notes preserved verbatim',new_visual_review_performed=False)
 out=m.fresh_dir(a.out);m.write_json(out/'review_PRIVATE.json',copied);m.qualify(a.preview,out/'review_PRIVATE.json',out/'qualification.json')
 print('REVIEW_REUSE_PASS',len(copied['cases']),'cases; named AI scope preserved')
if __name__=='__main__':main()
