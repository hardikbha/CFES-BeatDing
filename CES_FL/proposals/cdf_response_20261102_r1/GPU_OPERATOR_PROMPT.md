# Operator handoff — no full launch authorized by this preparation task

Read `CES_FL/AGENTS.md`, this proposal's README/PREREGISTRATION/DATA_GATE and actual evidence. Preserve every running source, existing result and failed receipt. Do not use main's old RUNNING snapshot as a live scheduler query. Current research verdict is **DO NOT RUN the full study**. There is no approved production release in this branch. Template approval is false. Release receipts record reviewed decisions bound to hashes; they are not cryptographic signatures or evidence of image validity.

## 1. Fetch a fresh detached source; verify the handoff commit

The final handoff supplies the exact published commit. Set CES_REVIEWED_COMMIT to that value; do not substitute a later branch tip automatically.

```bash
cd /home/sachin.chaudhary/ces_fl_sources/cesfl-tail-response-20261102-r1
git fetch origin refs/heads/proposal/cesfl-tail-response-20261102-r1
: "${CES_REVIEWED_COMMIT:?set exact commit from the reviewed handoff}"
git cat-file -e "${CES_REVIEWED_COMMIT}^{commit}"
CES_SOURCE_DIR="/home/sachin.chaudhary/ces_fl_sources/cdf-response-${CES_REVIEWED_COMMIT:0:12}"
git worktree add --detach "$CES_SOURCE_DIR" "$CES_REVIEWED_COMMIT"
cd "$CES_SOURCE_DIR"
test "$(git rev-parse HEAD)" = "$CES_REVIEWED_COMMIT"
test -z "$(git status --porcelain)"
CES_PY=/home/sachin.chaudhary/.conda/envs/qwen38/bin/python
CES_PROP=CES_FL/proposals/cdf_response_20261102_r1
"$CES_PY" - <<'PY'
from importlib.metadata import version
for name,expected in [('torch','2.10.0+cu128'),('torchvision','0.25.0+cu128'),('geomloss','0.3.1')]:
    assert version(name)==expected,(name,version(name),'requires a new environment review/smoke')
print('TESTED_ENVIRONMENT_VERSIONS_PASS')
PY
"$CES_PY" - <<'PY'
import hashlib,json
from pathlib import Path
m=json.loads(Path('CES_FL/proposals/cdf_response_20261102_r1/SOURCE_MANIFEST.json').read_text())
for name,sha in m['protected_source_and_config_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha,name
print('SOURCE_CONFIG_HASHES_PASS')
PY
```

Never pull over a source directory used by running jobs. Output roots must be new. Pushing this branch does not start any work. No model downloads, generation, automatic TEST evaluation, cancellation or FF2 follow-on is attached.

## 2. Inspect resources and obtain legitimate allocation

```bash
qstat -u sachin.chaudhary
ssh gpu01 nvidia-smi
ssh gpu02 nvidia-smi
export CES_SOURCE_DIR CES_PY CES_PROP CES_REVIEWED_COMMIT
qsub -I -q gpu -v CES_SOURCE_DIR,CES_PY,CES_PROP,CES_REVIEWED_COMMIT \
  -l select=1:ncpus=8:ngpus=1:mem=48gb -l walltime=12:00:00
```

The interactive qsub command is a **future operator command**, not submitted during preparation. Use it only when authorized to allocate. Within the actual job, record PBS_JOBID, `qstat -f "$PBS_JOBID"`, hostname and the site's assigned GPU mapping. A low-utilization index or CUDA visibility alone does not establish ownership. Site binding must identify the assigned UUID; do not guess it from the observation snapshot. Run subsequent GPU commands inside that allocation. User-authorized direct SSH software checks in the published receipts are explicitly labeled as such and are not PBS allocations.

```bash
: "${PBS_JOBID:?allocated job required}"
: "${CUDA_VISIBLE_DEVICES:?site must bind the assigned device}"
: "${CES_EXPECTED_GPU_UUID:?set from verified job-specific assignment}"
export CES_EXPECTED_GPU_UUID
"$CES_PY" - <<'PY'
import os,torch
assert torch.cuda.is_available() and torch.cuda.device_count()==1
actual=str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-')
assert actual==os.environ['CES_EXPECTED_GPU_UUID'].removeprefix('GPU-'),(actual,'assignment mismatch')
print({'visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'uuid':actual,'device':torch.cuda.get_device_name(0),
       'free_total_bytes':torch.cuda.mem_get_info(),'torch':torch.__version__})
PY
```

Do not share another user's GPU, kill existing processes or bypass failed checks. No blank/fabricated assignment receipt. The existing FairForge-2 original-only job may finish; no new FF2 work.

## 3. Recheck software and bound seed17 real inputs

```bash
cd "$CES_SOURCE_DIR"
CES_RUN_ROOT="/home/sachin.chaudhary/ces_fl_runs/cdf_response_checks_$(date -u +%Y%m%dT%H%M%SZ)"
test ! -e "$CES_RUN_ROOT"
mkdir -p "$CES_RUN_ROOT"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$CES_PY" -u CES_FL/tests/smoke_cdf_proposal.py \
  --device cpu --out "$CES_RUN_ROOT/cpu" > "$CES_RUN_ROOT/cpu.log" 2>&1
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$CES_PY" -u CES_FL/tests/smoke_cdf_proposal.py \
  --device cuda --out "$CES_RUN_ROOT/cuda" > "$CES_RUN_ROOT/cuda.log" 2>&1
CES_CFG="$CES_PROP/configs/xception_s17.json"
"$CES_PY" -u CES_FL/src/ces_fl_cdf_study.py preflight \
  --config "$CES_CFG" --arm cdf_fair --device cuda --out "$CES_RUN_ROOT/preflight_s17"
"$CES_PY" -u CES_FL/src/ces_fl_cdf_study.py dry-run \
  --config "$CES_CFG" --arm cdf_fair --device cuda --out "$CES_RUN_ROOT/dry_s17"
```

Require exits0, matching source/config/input/checkpoint hashes, finite losses/gradients, DRY_RUN optimizer_steps0 and recorded binding. Recheck existing jobs/output ledger to prevent duplicates. This does not validate an edit bank or release training. Paired `preflight` must currently reject the null bank with BLOCKED_INPUTS; do not fill it with Qwen candidates or synthetic unit-test fixtures.

## 4. Future reviewed original-only matrix (conditional commands, do not run now)

The existing Xception seed17 P50 is bound and reusable. Other backbone/seed P50 bindings are null intentionally. If independently authorized, obtain a source/config-bound external release for each concrete P50 config and perform its preflight and own dry-run first. Example seed7 future pretraining:

```bash
: "${CES_P7_RELEASE:?actual reviewed config/source-bound release required}"
: "${CES_P7_OUT:?new absolute output directory required}"
"$CES_PY" -u CES_FL/src/ces_fl_cdf_study.py train \
  --config "$CES_PROP/configs/xception_s7.json" --arm pretrain \
  --device cuda --approval "$CES_P7_RELEASE" --out "$CES_P7_OUT"
```

After P50 DONE and independent validation replay, bind its **actual** SHA and path to a new derived config (do not edit committed configs or old results):

```bash
: "${CES_P7_BIND_OUT:?new absolute binding-output directory}"
"$CES_PY" CES_FL/src/ces_fl_cdf_study.py bind-pretrain \
  --config "$CES_PROP/configs/xception_s7.json" --arm bce \
  --pretrain-run "$CES_P7_OUT" --out "$CES_P7_BIND_OUT"
```

The bound config needs its own actual external review release because its hash changed. Repeat forseed27/ResNet only when authorized; do not rerun the existing Xception seed17 P50. Begin the staged Xception three-seed four-original-arm matrix, using separate outputs and releases bound to the exact config/source. Each arm needs its own preflight/backward PASS. For an actually reviewed seed17 original-only arm, the conditional command is:

```bash
: "${CES_REVIEWED_ARM:?one of bce dro mean_fair cdf_fair}"
case "$CES_REVIEWED_ARM" in bce|dro|mean_fair|cdf_fair) ;; *) exit 2 ;; esac
: "${CES_ARM_RELEASE:?actual source/config-bound review receipt}"
: "${CES_ARM_OUT:?new absolute per-arm output directory}"
"$CES_PY" -u CES_FL/src/ces_fl_cdf_study.py train \
  --config "$CES_PROP/configs/xception_s17.json" --arm "$CES_REVIEWED_ARM" \
  --device cuda --approval "$CES_ARM_RELEASE" --out "$CES_ARM_OUT"
```

Do not manufacture release files or interpret this conditional form as permission. Publish actual job/allocation/device, source/config hashes, initial head hash, per-epoch original/group exposure, VAL selection/calibration, failures and checkpoint SHA. No auto TEST call exists. Corrected Ding production-stage launches remain blocked until the unimplemented wrapper/SFD binding is reviewed and tested.

## 5. Paired arms remain blocked

Only after [DATA_GATE.md](DATA_GATE.md) passes, bind actual bank and independent held-out metadata exclusions using `bind-bank --bank ACTUAL_REVIEWED_BANK --heldout-exclusions ACTUAL_VERIFIED_RECEIPT` into a **new** output. Review that exact bound config. Run separate nochange/score/response preflight and dry-run, then obtain their actual release before training. Do not lower coverage, infer approvals or relabel old six-source experiments as this bank. Publish unique originals, recorded components, per-group/direction/operator coverage, rejection reasons and repeated pair draws.

Stop any launch on a failed gate and preserve its failure log. Keep validation and test isolated. FairForge-2 remains a separate held study with only its already-running P finish exception.
