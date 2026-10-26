# CESEL_MA: CESel-MA (multi-axis evidence floor) + the priority runner

R = `~/ICLR_2027/research/cvpr_readiness/r20261009`. Spec: `R/decisions/2026-10-26-final-model-spec.md`; implementation
choices and the added baselines: `R/decisions/2026-10-25-cesel-plan.md` Amendment 6. This folder is a COPY of
`R/agents/CESEL` (code only, Codex-review fixes kept); CESEL and CLIPFAIR were not modified (running / pending jobs
import them). The CESel algorithm, flags and files are documented in `R/agents/CESEL/README.md`; only the additions are
listed here. Copy differences besides the additions: `A` / STOP paths point here (`run/cesel_run.sh`, `run/pick_gpu.sh`,
`tests/test_runlock_f1.sh`), FF2-public configs use `CESEL_MA/runs_ff2public`, the screen driver / PBS files / its
test were not copied, `scripts/evaluate_cfes.py` = CFES `scripts/evaluate.py` importing this package.

## CESel-MA (additions)

| File | What |
|---|---|
| `cfes/ma_labels.py` | per-image axes (gender, age bin, continuous ITA, ITA bin, gender x race) attached to train / val frames |
| `cfes/ma_losses.py` | `floor_ma_loss` (L_floor-MA), `freq_aux_factor` (warm-up), `ma_gaps` (selection gap) |
| `cfes/ces_engine.py` | terms `floor_ma`, `freq_aux`; MA pool labels; `attach_ma`, `ma_context`, `freq_warmup_steps` |
| `cfes/ces_data.py`, `cfes/ces_model.py` | batches carry `ma` (pairs `fake_ma` / `real_ma`); aux head `freq_aux` on c_freq |
| `scripts/train.py`, `scripts/train_ces.py` | `oppoint_ma` in validation, selection metric `val_oppoint_ma`, step-scheduled warm-up |
| `configs/arms/ma.yaml`, `configs/ma_ffpp.yaml` | CESel-MA (Full base): L_det + 0.5 L_sens + 1.0 L_floor-MA + 0.5 L_inv + 0.1 L_orth + warm-up |
| `configs/arms/ma_floor.yaml`, `configs/ma_floor_ffpp.yaml` | +floor base (used only if only +floor passes the gate) |
| `configs/ma_ff2public.yaml`, `configs/ma_floor_ff2public.yaml` | FF2-public; inherit the provenance guard (phase 2: needs PUBLIC_P1) |
| `scripts/gate_check.py` | step-1 gate -> `results/gate.json` (validation only) |
| `scripts/ma_compare.py` | MA s7 vs single-axis winner -> `results/ma_vs_single_s7.json` |
| `tests/ma_checks.py` | MA unit + end-to-end CPU checks |

Equations (u = cue utilities of the CES pool, as CESel; delta 0.1, n_min 4, validated cues):
```
L_floor-MA = sum_c w_c [ GxR_c + D_gender,c + D_age,c + S_c ],  w_real = 2, w_fake = 1
  GxR_c  = CESel floor cells (class c x condition x gender-race group), cleaner group labels
  D_a,c  = sum_{v: n>=4} sum_k relu( sg(ubar_{k,c}) - ubar_{v,k,c} - delta )^2          ubar_{k,c} = class mean
  S_c    = sum_{t_j: n_eff>=4} sum_k relu( sg(ubar_{k,c}) - sum_i w_i(t_j) u_ik / sum_i w_i(t_j) - delta )^2
           w_i(t) = exp(-(ITA_i - t)^2 / 2h^2), t_j = linspace(lo, hi, 8) over the TRAIN ITA range, h = (hi-lo)/8
L_freq_aux = balanced BCE( a^T c_freq + a0, y ) over the pool, weight 0.2 for step < 1 x steps_per_epoch, then 0
selection = val AUC - 0.5 max_a gap_a  (a in gender, age bin, ITA bin, gender x race; groups >= 2 real val videos),
            eligibility (TPR >= 80, FPR <= 20, start-referenced checks on the Lin groups) unchanged
```

### Label contract (agents/DEMO_AXES, README there)
- Files: `R/agents/DEMO_AXES/labels/labels_ffpp_train.csv`, `labels_ffpp_val.csv` (exist, 76,139 / 25,386 rows, all
  rows labelled); FF2-public: `labels_ff2public_train.csv`, `labels_ff2public_val.csv` (NOT produced yet; same columns;
  the FF2 final-model task waits for them); edges `ita_bin_edges.json` (`ffpp_protocol.ita_gw.edges`, `ff2.ita_gw.edges`).
- Key `img_path` = the training / validation CSV path. Columns used (configs `ma.labels.columns`): `cl_male` (1 = male;
  gender = 1 - cl_male), `cl_race` (asian / white / black / others -> group = 4 * gender + race code), `age4_if`
  (0..3 = <30 / 30-44 / 45-59 / 60+), `ita_vid` (FF++; per-video median ita_gw) or `ita_gw` (FF2; per image), `ita4`
  (declared TRAIN-quartile bin). Missing values -> the image is left out of that axis only.
- Coverage on FF++ (checked): gender / age / group 100 %, ITA 98.4 % (train), 98.5 % (val); cleaner group = Lin group
  on 89.6 % of train rows (val is Xu-only); val real videos per group: age 60+ = 6, ITA bins 52-71, GxR 2-104.

## Priority runner

`scripts/priority_runner.py` (aiface env) + `run/ma_run.sh` (per-task wrapper, pulid env for training) + the task
list `run/tasks.yaml`. Two PBS runners share the list: `run/pbs_priority_gpu01.sh` (GPUs {0,1,2,3,4,6,7}, <= 4) and
`run/pbs_priority_gpu02.sh` (GPUs {6,7} only, MPS off, <= 2). Design = the fixed CESel runner: task RUNLOCK
(`work/tasklocks/<task>.RUNLOCK`, token + atomic-rename CAS, heartbeat, fencing; exit 4 = held), free GPU = >= 60 GB
free and < 10 % util and not locked by CESEL_MA / CESEL / CLIPFAIR, <= 4 failed attempts per task across runners with a
10-20 min randomised wait (resume from last.pt), STOP file `work/STOP_MA` (2 h start grace), HOLD (`HOLD_UNTIL_FILES`:
no launch until both CESel screen jobs' PBS output files exist; `touch work/HOLD_OFF` releases it), idle exit after 2 h with nothing launchable.

Scheduling every round (60 s): re-read `run/tasks.yaml` and its include files; re-evaluate the gate while undecided;
ready tasks sorted by (priority, file order); a higher-priority ready task always launches first; same priority runs
concurrently. `reserve: true` tasks that only wait for a running / launching dependency keep one free GPU each away from
lower-priority tasks for at most 1 h (BD arms behind bd_smoke). A failed / skipped dependency skips its dependants only.

Current order: P5 BEAT_DING include -> P10 CESel-MA s7 -> compare -> s17 / s27 (gate PASS; not worse) -> P20 CLIP
baselines s7 (PG-FDD, DAW, DAG, FairReL [done], SFD -> Ding, Ding GDA connected) -> P30 the same s17 / s27 -> P40 Ding
official repro s17 / s27, Arm E w10 s17 / s27 -> P90+ FF2-public phase 1 -> CFES-D, B0 -> validated JSON -> final model.

### Task file format (`run/tasks.yaml`; edit a copy and `mv` it over the file)
```yaml
include:                       # external task files (below)
- {file: '{R}/agents/BEAT_DING/run/bd_tasks.json', priority: 5, prefix: '', optional: true, reserve: true}
tasks:
- name: ma_s7                  # unique, [A-Za-z0-9_.-]
  priority: 10                 # lower = first; leave gaps for new tasks
  run: [ma, '{ma_config}', '7']            # kind + args of run/ma_run.sh (kinds listed in its header)
  done: ['{MA}/runs/ffpp_{ma_config}_s7/TRAINED', ...]   # ALL must exist = done (never re-run)
  deps: [other_task]           # optional; must be done first
  when: gate_pass              # always | gate_pass | gate_fail | gate_decided | ma_not_worse | ma_worse
  needs: ['{LABELS}/labels_ffpp_train.csv']   # optional files required before launch
  gpu: true                    # false = CPU task (no GPU slot)
  hosts: [gpu01]               # optional
  enabled: true                # false = never launched (coordinator switch)
- name: my_arm                 # generic command task (kind cmd / cmd_cpu)
  priority: 15
  run: [cmd]
  cmd: "bash /abs/path/run.sh arm1 7"     # runs in cwd (default CESEL_MA) with CUDA_VISIBLE_DEVICES = one GPU
  cwd: /abs/path
  env: {KEY: value}
  done: ['/abs/path/runs/arm1_s7/DONE']
```
Placeholders: `{MA} {CESEL} {CFES} {CLIPFAIR} {FAIRREL} {DING} {LABELS} {R}`, `{ma_config}` / `{single_axis_arm}`
(from `results/gate.json`; unresolved -> the task waits).

### External task files (for other agents, e.g. BEAT_DING `run/bd_tasks.json`)
JSON (or YAML), re-read whenever it changes (write it atomically: temp file + `mv`). A missing file = no tasks yet; a
file that fails to parse / validate keeps its previous version (logged in `logs/priority_<host>.log`).
```json
{"tasks": [
  {"name": "bd_smoke", "cmd": ["bash", "/abs/R/agents/BEAT_DING/run/bd_run.sh", "smoke"],
   "done": ["/abs/R/agents/BEAT_DING/runs/smoke/SMOKE_OK"]},
  {"name": "BD0_s7", "cmd": "bash /abs/R/agents/BEAT_DING/run/bd_run.sh train BD0 7", "deps": ["bd_smoke"],
   "done": ["/abs/.../BD0_s7/TRAINED", "/abs/.../BD0_s7/eval/summary_calibrated.csv"]},
  {"name": "bd_report", "cmd": "bash /abs/.../bd_run.sh report", "gpu": false, "deps": ["BD0_s7"],
   "done": ["/abs/.../results/bd_report.md"]}
]}
```
Fields: `name` (unique across all files; the include's `prefix` is prepended to names and deps), `cmd` (string or argv
list; run with `bash -c` in `cwd`, default CESEL_MA), `done` (required, absolute paths or placeholders; ALL must exist),
optional `deps` (names in the same file), `needs`, `gpu` (default true; one GPU via CUDA_VISIBLE_DEVICES), `cwd`,
`env`, `when`, `enabled`, `hosts`, `reserve`. Priority = the include's (BEAT_DING: 5 = first); file order breaks ties.
The runner takes the task RUNLOCK before running `cmd`; the command itself should be resumable and exit 0 only when its
`done` files exist (exit != 0 = failed attempt, retried up to 4 times; 4 = held). Winner seeds resolved later: append
the tasks to the file when known (or add them with `enabled: false` and flip it).

## Commands
```
~/.conda/envs/pulid/bin/python tests/ma_checks.py                 # CESel-MA checks, ALL_PASS
~/.conda/envs/pulid/bin/python tests/ces_checks.py                # copied CESel checks (incl. one real CLIP pass)
bash tests/test_runlock_f1.sh run/ma_run.sh                       # RUNLOCK protocol of the task wrapper
~/.conda/envs/pulid/bin/python tests/dry_priority_runner.py       # 2-runner CPU dry run: gate PASS / FAIL / smoke fail
~/.conda/envs/aiface/bin/python scripts/gate_check.py             # gate now (PENDING until the screen report is complete)
qsub -W depend=afterany:44105 run/pbs_priority_gpu01.sh ; qsub -W depend=afterany:44104 run/pbs_priority_gpu02.sh
touch work/STOP_MA                                                # stop new launches
```
Status: `results/priority_status_<host>.json`, `results/gate.json`, `logs/priority_<host>.log`, `logs/ma_runs/<task>.log`,
attempts `runs/attempts/<task>.jsonl`, MA runs `runs/ffpp_<ma|ma_floor>_s<seed>/`, FF2 runs `runs_ff2public/`.
