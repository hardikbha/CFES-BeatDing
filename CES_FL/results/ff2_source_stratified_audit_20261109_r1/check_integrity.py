"""Post-analysis byte-integrity check only; no predictions/models/TEST/CAL."""
import json
import subprocess
from datetime import datetime,timezone
from pathlib import Path
from audit import sha,codehashes

p=Path(__file__).resolve().parent
f=json.loads((p/'FREEZE.json').read_text());b=json.loads((p/'BINDINGS.json').read_text())
assert f['helper_sha256']==codehashes()
for path,h in {**f['inputs_sha256'],**b['input_sha256']}.items():assert sha(path)==h
checks=[]
for item in b['bindings']:
    c=json.loads((Path(item['run_directory'])/'config.json').read_text())
    actual=sha(c['init_checkpoint']);assert actual==c['init_sha256']
    checks.append({'model':item['name'],'init_checkpoint_path':c['init_checkpoint'],'init_checkpoint_sha256':actual,'status':'PASS'})
source=f['detached_source_checkout']
assert subprocess.check_output(['git','-C',source,'rev-parse','HEAD'],text=True).strip()==f['documentation_commit']
assert not subprocess.check_output(['git','-C',source,'status','--porcelain'],text=True).strip()
result={'status':'PASS','verified_utc':datetime.now(timezone.utc).isoformat(),'frozen_analysis_helpers_unchanged':True,
        'all_input_hashes_rechecked':True,'detached_source_clean':True,'initialization_bindings':checks}
(p/'FINAL_INTEGRITY.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: frozen helpers, all recorded inputs, initial encoder and detached scientific source remain unchanged.')
