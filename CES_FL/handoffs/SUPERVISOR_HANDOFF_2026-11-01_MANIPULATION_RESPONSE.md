# Copyable supervisor / research-engineer prompt: current CES-FL project

Snapshot verified **1 November 2026, 14:11:36 UTC / 19:41:36 IST**. This is a dated observation, not a claim of current liveness when you read it. Live source artifacts and newer status timestamps take precedence. The named source commit, completed checks and historical results must remain associated with their exact versions.

Copy the prompt below to the reviewer or their research assistant. They can use GitHub without cluster access; private data, images, reviews and binary checkpoints require authorized cluster access. Do not share credentials.

---

You are taking over the **review and supervised continuation of CES-FL**, a research project investigating demographic disparities in deepfake detection. Work from executable code, exact configurations, native checkpoints, logs and recorded reviews. Treat this handoff as an index and dated observation to verify. Your first action is a read-only state audit, not a new training launch.

## 1. Repository, active source and reading order

Canonical repository: **https://github.com/hardikbha/CFES-BeatDing**.

The current experiment is the new **controlled manipulation-response investigation**, separate from the older frozen-encoder CES-FL pilots, the original full-coverage counterfactual proposal, and FairForge-2 comparisons.

Exact active research source commit:

```text
58773bbef9d9fe79dd313610f694ddeea12c6037
```

It was fetched from `hardiksharma6555/CFES-BeatDing`, branch `upgrade/manipulation-response-20261101`. A byte-identical source commit is available in the canonical repository on branch **`source/manipulation-response-58773bb`**. The canonical `main` also contains the reviewed code/config/test files plus operational evidence and timestamped updates. Do not assume `main` HEAD is the running experiment's source revision.

Active immutable detached source worktree:

```text
/home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb
```

Read these in order:

1. `CES_FL/AGENTS.md` and `CES_FL/WORKFLOW.md`.
2. `CES_FL/proposals/manipulation_response_20261101_r1/README.md`.
3. Its `request.json`, `config.json`, `GPU_OPERATOR_PROMPT.txt`, and `evidence/source_manifest.json`.
4. Its `execution_manip_response_ssh_20261101T115720Z/README.md`, actual check receipts and logs.
5. `CES_FL/status/manip_response_ssh_20261101T115720Z_live.md` and `CES_FL/results/manip_response_ssh_20261101T115720Z/status.json`; check observation times.
6. `CES_FL/proposals/manipulation_response_quartets_20261101_r1/README.md`, `CANDIDATE_COVERAGE.json`, operator-discovery evidence and proposed settings.
7. `CES_FL/status/qwen_quality32_20261101T095157Z_live.md` and its machine-readable result, for the separate completed generator pilot.
8. Read historical result directories only after identifying their method, source, dataset and evaluation protocol. General README/STATUS sections include older preserved snapshots; do not mistake them for current state.

If you need a review checkout, create a new one; do not pull into a source tree read by a running job:

```bash
git clone https://github.com/hardikbha/CFES-BeatDing.git CESFL-review
cd CESFL-review
git fetch origin refs/heads/source/manipulation-response-58773bb
git cat-file -t 58773bbef9d9fe79dd313610f694ddeea12c6037
git worktree add --detach ../CESFL-review-source-58773bb \
  58773bbef9d9fe79dd313610f694ddeea12c6037
```

Do not rerun that worktree-add command on an existing path. The source worktree shows the exact science revision; the `main` review checkout has later execution/status evidence.

## 2. Research question and what this version actually tests

The wider program studies gender, age and skin-tone disparities. This **new seed-17 Xception experiment is age-intervention only**, with an original-image fairness surrogate over **gender × race** groups. The code also reports age/skin/gender/race metrics where manifest labels exist, but that does not mean all three counterfactual training pipelines are validated or running.

The motivation is that generative editing can modify texture, reconstruction quality and manipulation evidence even under a no-change instruction. Therefore, demanding equality of original/edited detector scores may suppress useful forensic information or learn editor-specific shortcuts.

The proposed alternative asks whether the detector responds positively and similarly to a **known added manipulation** applied to matched no-change and age-edited bases. This is an unproven research hypothesis, not established counterfactual fairness, a validated causal estimator, demonstrated novelty, or a win over Ding.

Distinguish:

- **Fairness evaluation:** measure performance across groups on held-out original benchmark images using their recorded labels/provenance.
- **Synthetic auxiliary training:** use independently reviewed generated/manipulated candidates with explicitly stated assumptions. Synthetic appearance is not demographic ground truth, and an edited real source is not automatically an authentic image.
- **Software smoke tests:** verify mathematics, gradients, masks, checkpointing and guards using fixtures. They cannot validate actual faces, manipulations, fairness claims or scientific generalization.

The current model is **not** the earlier minimal one-encoder/one-linear-head CES pseudocode. Its architecture has one shared encoder, global and spatial heads, learned gates and additive fusion. The new response objective is also distinct from the earlier single-image forensic-suppression margin proposal. Do not conflate these versions.

## 3. Verified state at the snapshot

| Component / arm | Verified status at 19:41 IST | Evidence / next dependency |
|---|---|---|
| New CUDA smoke | COMPLETED — all six test groups passed | `smoke_cuda/RESULT.json`, `logs/smoke.log`, native check receipt |
| Real TRAIN/VAL readiness | COMPLETED | All 101,525 original images decoded/hashed; audit receipt |
| Configured real backward check | COMPLETED | 16 original images, finite loss/gradients, zero optimizer steps |
| P: shared original-only pretraining | **RUNNING**, 31/50 epochs complete, epoch 32 underway | Native `P/history.jsonl`; train PID 1393109 on gpu02 GPU5 |
| bce | **BLOCKED_INPUTS** | Verified P50, exact binding and configured BCE backward check |
| fair | **BLOCKED_INPUTS** | Same P50, exact binding and configured fair backward check |
| nochange | **BLOCKED_INPUTS** | P50 plus valid reviewed matched quartet bank, audit/bind and own backward check |
| score | **BLOCKED_INPUTS** | Same bank/P50 and score backward check |
| response | **BLOCKED_INPUTS** | Same bank/P50 and response backward check |
| Qwen32 quality generation | COMPLETED GENERATION, NOT VALIDATED | 224/224 outputs, 32 sources; reviews pending |
| Qualified manipulation quartets | **BLOCKED_INPUTS** | No actual validated bank or independently approved quartets |
| New FairForge-2 comparison | DO NOT RUN | Its separate review blockers remain unresolved |

At this snapshot, P's best **recorded validation AUC** is **98.405989% at epoch 29**. The latest completed epoch is 31. This is not a final verified selection, a test result, accuracy, or evidence that the proposed response method improves fairness: only original-image P is training.

At that AUC-best recorded epoch, gender_race worst-group AUC is **94.109001%**, FPR max–min gap **30.984848 percentage points**, and the implemented Ding-style summed F_FPR is **0.5075766324** in fraction units. The gap and summed metric are different quantities. Higher detection AUC is not proof of lower disparity. Small subgroup support, including only 30 real Female-Black validation images, makes these disparities unstable. Selection must remain validation AUC only; do not switch to another epoch because its fairness looks better.

The recent estimate was P completion around 20:45–21:00 IST on 1 November, based on approximately 193 seconds per completed epoch. This is a forecast, not an artifact or guaranteed deadline; recompute from current native history and include final validation replay.

## 4. Current live processes, paths and monitors

Current run root:

```text
/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z
```

Verified at the snapshot:

- Host `gpu02`, physical GPU **5**, NVIDIA H100 80GB.
- CUDA UUID `GPU-bff97893-385b-6f3f-831b-113284c2e1f1`.
- Milestone worker PID **1390084**; actual training PID **1393109**.
- Read-only observation/publication PID **980934** on the management node.
- Training log `logs/P.log`; wrapper log `logs/P_worker.log`.
- Native outputs `P/config.json`, `P/provenance.json`, `P/history.jsonl`, `P/best.pt`, `P/last.pt` as created.
- Final completion requires all 50 epochs, `P/RUN_VERIFIED.json`, `P/summary.json` and `P/DONE`, with consistent source/config/checkpoint hashes and independent validation replay. An intermediate checkpoint is not completion.
- First two stage receipts: `smoke_cuda/RESULT.json`, `real_inputs/DECODE_AUDIT.json`, `P_real_dry/RESULT.json`; launch/device/failure records are in `checks/`.

The direct SSH route was explicitly authorized by the user. It superseded PBS smoke **44963.mgmt01**, cancelled while still queued with no execution vnode/host. It must not be described as a CUDA failure or a running allocation. Previous queued-attempt evidence remains under `manip_response_20261101T104925Z`.

Each SSH milestone holds the shared dispatcher exclusion plus project lease, requires 20GiB free headroom, applies a **15% Torch allocator cap**, verifies full UUID binding and preserves existing processes. There is **no claimed PBS allocation or exclusive GPU ownership**. A host name, low utilization or free VRAM alone does not grant ownership. Do not remove another task's lock or preempt work. The separate Qwen run used different resource settings; do not copy its 55% allocator policy into this trainer silently.

The finite observer normally publishes about every ten minutes and on state changes, for at most 72 hours. It **cannot launch, retry, approve, cancel, chain stages or evaluate TEST**. The original-only downstream arms will not start just because a status monitor exists. Reconcile actual completion and operate each authorized milestone explicitly. No unattended run-all executor is installed.

Read-only verification commands on the cluster:

```bash
qstat -u sachin.chaudhary
ssh gpu02 'ps -p 1390084,1393109 -o pid,stat,etimes,args'
ssh gpu02 'nvidia-smi -i 5 --query-gpu=index,uuid,name,memory.free,utilization.gpu --format=csv'
tail -n 12 /home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/logs/P.log
```

PIDs and device use may change; verify command lines rather than trusting a PID reused by another process. If a command fails, report the failure; do not infer that training passed or stopped from a stale status file.

## 5. Complete current-model code map

These are the files needed to understand the current model; `CODE_INDEX_2026-11-01_MANIPULATION_RESPONSE.json` alongside this handoff records actual hashes of the verified code/config files.

| Repository file / directory | Purpose |
|---|---|
| `CES_FL/src/ces_fl_manipulation_response.py` | Current downstream objectives, quartet/provenance audit, P binding, configured dry runs, five-arm training, validation replay, epoch resume and CLI |
| `CES_FL/src/ces_fl_counterfactual_5050.py` | Required adjacent architecture/pretrainer dependency; Xception, `DiagramCES`, image/manifest handling, weighting, optimizer, metrics, calibration, checkpoints, original-image P, and parent Qwen generator |
| `CES_FL/tests/smoke_manipulation_response.py` | Exact current six-group CPU/CUDA software suite |
| `CES_FL/proposals/manipulation_response_20261101_r1/config.json` | Exact seed, model, budgets, optimizer, manifest/weight hashes and fixed objective weights |
| Same directory: `request.json`, `README.md`, `GPU_OPERATOR_PROMPT.txt` | Hypothesis, advancement criterion, interpretation and milestone commands |
| Same directory: `quartet_schema_example.json` | Deliberately non-runnable format example; approvals false, not real bank data |
| Same directory: `evidence/source_manifest.json` | Exact trainer, dependency, test and config hashes |
| Same directory: `execution_manip_response_ssh_20261101T115720Z/` | Actual SSH operational wrappers, memory policy, guard tests, commands, logs, source/device receipts and real input/backward evidence |
| `CES_FL/proposals/manipulation_response_quartets_20261101_r1/` | Additive prospective target/donor coverage and cached manipulation-operator discovery; no valid completed bank |
| `CES_FL/proposals/qwen_bounded_quality_20261101_r2/` | Actual Qwen32 scope, exact source selection, operational worker/profiler, tests and provenance |
| `CES_FL/results/manip_response_ssh_20261101T115720Z/` | Aggregate current run observations and software evidence, not private images or checkpoints |
| `CES_FL/status/manip_response_ssh_20261101T115720Z_live.md` | Timestamped per-arm status |

Useful executable-source entry locations in commit58773bb:

- `ces_fl_counterfactual_5050.py:317`: `DiagramCES`; line391 manifests, line410 split checks, line450 calibration, line460 metrics, line494 optimizer, line595 generation planning, line624 Qwen generation, line897 independent P validation replay, line919 P trainer.
- `ces_fl_manipulation_response.py:56`: quartet audit; line144 dependencies/preflight; line170 objectives; line215 quartet forward order; line224 model initialization; line233 downstream training/dry run; line323 CLI.
- `smoke_manipulation_response.py:27`: six-group software test.

Exact scientific hashes:

```text
trainer:      0e82a522a22e641dc5588b6a15317e6842896ab2bdcc5364833af25e207f257e
architecture: 3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab
smoke test:   748f4a883af8b39b6edd0f992f929f12f3045ba85734fcf1e542b67f61895654
config:       d31c78ab2913e629ebbd7e7a860a0dd4bea90d2c920b0d77719b042a7537dab0
```

The current response script imports the adjacent architecture file and enforces its hash. It is **not a self-contained one-file distribution**. Do not copy only the response script or load its new-format checkpoints through an old evaluator.

The historical `ces_fl_diagram_model_v021.py` and `ces_fl_upgrade.py`, their configurations and tests are also in `CES_FL/`. They document prior experiments; they are **not the active response trainer**. Preserve them instead of replacing them with this version. Older Qwen working scripts outside Git are not assumed synchronized with these immutable sources.

## 6. Architecture and training policy

The model uses **one shared Xception encoder**, ImageNet initialization for P, a global MLP, shared spatial/patch MLP, sigmoid image and patch gates, and softplus-positive fusion weights plus scalar bias. The unused Ding adaptation classifier/channel block is removed from the encoder feature path. There is no demographic adversarial classifier or new learned attribution head.

Its final scalar logit obeys:

```text
z = global_contribution + mean(signed_local_contribution_map) + fusion_bias
```

The gates are **learned weights**, not proven reliability estimates. Spatial maps are **signed local contributions**, not validated forgery masks or pixelwise ground truth.

The current full-width configuration has **21,594,159 parameters, all trainable in P and downstream**. This count was measured in the actual configured CUDA backward check and is recorded in native P provenance. The CPU module-count receipt alongside this handoff provides the independent component breakdown; it uses `pretrained=False` solely to count parameters without loading weights or training.

| Component | Actually measured parameters |
|---|---:|
| Xception encoder | 20,806,952 |
| Global head | 262,401 |
| Shared patch head | 262,401 |
| Image gate | 131,201 |
| Patch gate | 131,201 |
| Fusion | 3 |
| **Total / current trainable** | **21,594,159 / 21,594,159** |

Fixed configuration:

- Seed17 and head seed17, Xception; hidden128, gate_hidden64; classifier input224px.
- P50: full encoder/head/gate/fusion training; BN running statistics update.
- Downstream each50: full parameter training with **BN running buffers frozen**.
- AdamW, batch16, workers4, weight decay1e-4.
- Head/gate/fusion LR1e-3; P encoder LR1e-4; downstream encoder LR1e-5.
- Every original appears once per epoch, shuffled without replacement; fixed inverse label×ids8 cell loss weights. No silent drop/replacement of difficult cases.
- After P finishes, all downstream arms reuse its same validation-selected encoder with identical fresh head/gate/fusion initialization, original order, weighting, optimizer budget, selection and calibration. Paired arms add forward compute; optimizer steps are matched, FLOPs are not identical.
- 50 shared P +5×50 downstream = **300 study epochs** if all arms become valid, not 50 total. Selected ancestral P weights can come from an earlier epoch even though P completed its full50-epoch budget.

## 7. Exact auxiliary mathematics and comparisons

Only original TRAIN images receive benchmark real/fake BCE labels. The edited bases and manipulated quartet outputs are **not assigned presumed benchmark authenticity labels**.

For each original real TRAIN source, define independently generated editor no-change base `x0` and age-edited base `xa`, with the same editor/model/settings/seed apart from instruction. Apply a recorded face-manipulation operator `T` with matched code/settings/donor/seed:

```text
t0 = T(x0)
ta = T(xa)
m0 = asinh(f(t0) - f(x0))
ma = asinh(f(ta) - f(xa))
L_response = mean(abs(m0 - ma))
             + 0.5 * mean(relu(0.5 - m0) + relu(0.5 - ma))
```

`asinh` is applied **after the within-pair logit difference**, not separately to its two logits. A common additive offset cancels within that difference, but arbitrary editor effects do not. Positive response margins discourage a constant-response solution without proving useful features or generalization. Operator shortcuts and target-age destruction must be investigated.

The fairness baseline uses TRAIN real-image mean sigmoid scores at fixed logit 0 / temperature 1: mean absolute deviations of observed real-group means from their pooled real-image mean. With fewer than two observed real groups, its contribution is zero and batch group counts are logged. It is a differentiable minibatch surrogate, not the true population F_FPR or the separately calibrated-threshold disparity.

| Arm | Actual objective |
|---|---|
| `bce` | Original class/group-weighted BCE |
| `fair` | BCE +0.1 × TRAIN soft real-group disparity |
| `nochange` | fair objective +0.1 × mean abs(asinh(f(original))−asinh(f(x0))) |
| `score` | fair objective +0.1 × mean abs(asinh(f(original))−asinh(f(xa))) |
| `response` | fair objective +0.1 × L_response |

The bounded bank is sampled deterministically with up to four quartets per original minibatch. Repeated exposure is logged and must not be called full source coverage. Map consistency, Group-DRO, joint MMD, forensic-suppression branches and other unlisted losses are not silently included.

Advancement criterion already recorded in request.json: response must lower the primary validation gender_race F_FPR against **both fair and score**, with no more than0.5 percentage-point AUC loss relative to fair. Keep this fixed. It is a development criterion, not a significance claim or evidence against Ding's published test numbers. Future multi-seed/backbone/unseen-operator confirmation requires separately specified configurations.

## 8. Dataset, leakage qualifications and evaluation

Exact current manifests:

```text
/home/sachin.chaudhary/ces_fl_pipeline/manifests/ding_ffpp/train.csv
/home/sachin.chaudhary/ces_fl_pipeline/manifests/ding_ffpp/val.csv
/home/sachin.chaudhary/ces_fl_pipeline/manifests/ding_ffpp/protocol_source_overlap.json
```

Current stage: FF++ with recorded Ding/Lin membership, **76,139 TRAIN /25,386 VAL**. TRAIN counts12,926real/63,213fake; VAL 4,092 real / 21,294 fake. Classifier updates use TRAIN only. No TEST manifest is read by this new experiment. Do not invent a new test split/path.

The real-input audit decoded and hashed all 101,525 images, all 380×380, with **zero identical image-byte overlap** across TRAIN/VAL. The preserved parent benchmark declares **163 overlapping identity/source-video proxies**. That exception is documented and must not become a claim of person-disjoint splitting or guaranteed identity independence. Do not alter splits to make the checks look better. Quartet contributors use a stricter rule below.

The manifest's age and skin-related annotations include automated proxies; recorded labels are not new human ground truth. `ids8` weighting cells and `gender_race` fairness cells are **different partitions**; Qwen32 coverage exposed that difference. Report supports and missingness separately.

All checkpoint selection is **validation AUC only**. The selected checkpoint freezes an empirical threshold from validation real images with target FPR 0.1. The rule uses finite-sample budget `floor(target*n_real)`, the next representable float above the descending-score budget boundary, and predicts fake when `logit >= threshold`. Ties can produce fewer false positives than budget. Always report achieved validation FPR and disclose that test FPR is not guaranteed; do not recalibrate on TEST.

Metrics include overall/per-group AUC, FPR/FNR and support, worst eligible group AUC, max–min gaps, and implemented Ding-style quantities. Eligibility requires both classes to have at least20 examples and excludes unknown labels from eligible fairness aggregates. Definitions in code:

```text
F_FPR = sum_g abs(FPR_g - overall_FPR)
F_DP = max_g positive_rate_g - min_g positive_rate_g
es_AUC = overall_AUC / (1 + sum_g abs(AUC_g - overall_AUC))
```

Native numbers are fractions, not percentages. Summed F_FPR is not FPR max–min gap and can exceed1. Small groups require uncertainty, not confident population claims. New per-group/paired-component confidence intervals and a locked final evaluator are not completed in this package. There is no TEST command in the response trainer. Prior project test results were already viewed; disclose development history and seek untouched confirmation evidence. Never compare our validation AUC directly with Ding's FF++ test AUC, and keep paper values, archived reproductions and new runs distinct.

## 9. Qwen32: completed candidates, not a valid response bank

Private run:

```text
/home/sachin.chaudhary/ces_fl_runs/qwen_quality32_20261101T095157Z
```

Verified generation completed: **32 originals** (16 real / 16 fake), **224 diffusion outputs** (seven per source), plus32 geometry controls. Conditions: no-change, younger, older, feminine presentation, masculine presentation, lighter tone, darker tone. Each candidate starts independently from its original. No automatic approvals were created.

Generator: Qwen-Image-Edit-2511, pinned cached revision `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`; BF16,40 steps,true CFG 4,1MP geometry. Complete call13310.21seconds (about3.70hours); mean59.21seconds per output excluding load; peak allocated39.15GiB. It ran on gpu01GPU3 and completed; do not claim the finished generator is still training.

Actual private candidates/provenance: `generation/items/<source_id>/record.json` and images; pending decisions: `quality_review_pending.csv`. Public status/profile record generation, not semantic validity. Earlier user review acknowledgements are not completed per-image forensic-validity evidence and must not be copied into this new224-row review artifact.

Earlier no-change controls visibly reconstructed/sharpened faces and some edits altered unrelated features; frozen-detector scores shifted strongly. These are documented confound flags, not automatic rejection or approval rules. Realism, target attainment, successful decoding and detector agreement alone cannot establish preserved identity/authenticity/forensic evidence.

This is a separate quality diagnostic, not the parent 76,139-source full-bank protocol. Production full-coverage binding was actually tested and rejects this subset. Do not silently substitute it for the full TRAIN bank or scale the original532,973-output generation. That full-bank recipe remains held; no huge follow-on generation is authorized by this handoff.

## 10. Matched quartet blockers and prospective preparation

The initial bank needs **16 unique original real TRAIN sources**: one older and one younger quartet for every eight real TRAIN gender_race groups. Source and donor contributors must be TRAIN and have **zero recorded VAL-contributor overlap**. This is stricter than the parent benchmark's allowed overlap. All quartet source/donor paths, hashes and eventual operation receipts stay private.

Existing preparation:

```text
/home/sachin.chaudhary/ces_fl_runs/manip_response_20261101T104925Z/quartet_preparation/prospective_quartets_PRIVATE.csv
```

It has16 prospective unique targets and16 donors, with zero recorded VAL-contributor overlap, one fixed same-gender_race donor per quartet and no donor reuse. Proxy distinctness is not confirmed biological person identity. Approval/reviewer fields are blank/0; `t0/ta` do not exist. This is **not** bank.json.

Qwen32 supplies only15 eligible real-source candidates covering five gender_race groups; eight sources are potentially reusable for the required quotas **after individual review**. Eight additional source slots are proposed: Female-Black 2, Female-Others 1, Male-Asian 2, Male-Black 2, Male-Others 1. Do not relax quotas or silently skip these groups. Additional bounded editing must be explicitly recorded/tested and must not disturb the completed Qwen plan.

A cached **SimSwap-224 people** pipeline was located at `/home/sachin.chaudhary/models_external/SimSwap`, GitHEAD `bd7b7686a17f41dd11cfcd5d82f7e4c5eb94b780`. Checkpoint hashes were actually measured:

```text
generator: 24caf144e9aabd5acd1127b06f13ed3528240adb9e747d77a94ac3f33e672330
ArcFace:   52ea5ce4902017b77a2bb811dd9a82f57dee3883e06c18a42d288437263c7a20
```

Its shared checkout has detector patches; preserve them and freeze actual code bytes before any adapter run. No target quartet manipulation has yet been validated. The old dataset drivers and control scripts include unrelated dataset/source/donor pools, including TEST/VAL in one driver. **Do not run them as-is.** A new immutable FF++ TRAIN-only adapter, matched no-generator geometry/blend controls, actual transform/landmark/encoding receipts and independent review are still required. The candidate generator environment differs from classifier aiface; do not install over either environment or treat metadata inspection as an executed-forward PASS.

Review base identity preservation between original/x0/xa, intended age change, donor transfer and operator validity in t0/ta, matched processing and unrelated changes. A genuine face swap intentionally changes identity relative to the target; clarify role-specific review semantics rather than requiring all four images to retain the same identity or blindly setting a generic flag. Ensure T does not erase the intended age contrast or introduce a shortcut. Reviewers should be blinded to detector scores; independent adjudication is preferred. Preserve uncertain/rejected examples and missingness; no automatic all-approve or fictional reviewer.

Only after actual qualified assets/reviews exist can the source's strict bank audit, binding and arm-specific dry runs unlock nochange / score / response. JPEG, blur, resize, arbitrary pasted pixels and Qwen no-change controls are not substitutes for genuine validated t0/ta.

## 11. Safe next milestones: inspect before acting

For the first response to this prompt, provide a read-only verification report. **Do not launch, stop, resume or change training merely because this handoff contains commands.** Report what the prior user already authorized and which concrete prerequisites are satisfied; no replacement configuration or test tuning is permitted.

If assigned the cluster-operator role to continue the existing authorized run:

1. Verify current P state and avoid duplicates. Do not restart or run its pretrain command again if it is active/completed.
2. After all 50-epochs and native independent validation replay, inspect DONE/summary/RUN_VERIFIED, checkpoint hashes and current config. Preserve every partial/failure artifact.
3. Bind that same exact verified P to the original-only downstream config; do not substitute legacy P or fabricate its future hash.
4. Run the configured actual CUDA BCE and fair backward checks separately, then run those matched50-epoch arms on appropriately reserved/verified resources. They do not need a quartet bank. Recheck available resources on both nodes without altering unrelated jobs.
5. Prepare and independently review the missing manipulation pipeline/quartets in a separate additive record. Block paired arms until all bank/coverage/validity checks pass.
6. Only then bind the actual bank and P, run nochange / score / response dry runs separately, and launch corresponding approved arms after PASS.
7. No whole-handoff unattended executor, automatic test evaluation, new seeds, backbone switch, FairForge-2 run, large model download or large image-bank generation follows from this prompt.

Exact later-stage CLI forms, **not commands to execute while P is running**:

```bash
MR_SOURCE=/home/sachin.chaudhary/ces_fl_sources/cesfl-response-58773bb
MR_RUN=/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z
MR_PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
MR_SCRIPT="$MR_SOURCE/CES_FL/src/ces_fl_manipulation_response.py"
MR_CONFIG="$MR_SOURCE/CES_FL/proposals/manipulation_response_20261101_r1/config.json"

# Only after native P50 completion/verification and no existing bound output:
"$MR_PY" "$MR_SCRIPT" bind --config "$MR_CONFIG" \
  --pretrain-run "$MR_RUN/P" --out "$MR_RUN/bound_originals"

# Inside a verified, appropriately reserved GPU session, one milestone at a time:
"$MR_PY" "$MR_SCRIPT" dry-run --config "$MR_RUN/bound_originals/bound_config.json" \
  --arm bce --device cuda --out "$MR_RUN/dry_bce"
"$MR_PY" "$MR_SCRIPT" dry-run --config "$MR_RUN/bound_originals/bound_config.json" \
  --arm fair --device cuda --out "$MR_RUN/dry_fair"

# Each arm requires its own backward PASS and duplicate/output checks first:
"$MR_PY" "$MR_SCRIPT" train --config "$MR_RUN/bound_originals/bound_config.json" \
  --arm bce --device cuda --out "$MR_RUN/bce"
"$MR_PY" "$MR_SCRIPT" train --config "$MR_RUN/bound_originals/bound_config.json" \
  --arm fair --device cuda --out "$MR_RUN/fair"
```

These are documented milestone forms, not a copy/paste run-all script. Existing SSH launch helpers are operational evidence, not permission to bypass locks or invoke an unchecked stage. Paired binding additionally requires the actual reviewed bank path/hash; no imaginary bank path is provided here. Use epoch-boundary `--resume` only after exact source/config/stage/history/selected-checkpoint reconciliation; never delete partial evidence to force a fresh run.

## 12. Historical studies and comparisons to keep separate

- `CES_FL/results/completed_seed17/` and `CES_FL/configs/completed_seed17/`: old P/A/B/C pilot; A ImageNet-frozen BCE, B detection-pretrained frozen BCE, C same frozen encoder plus invariance on36 variants from six sources. B/C denotes a comparison, not another arm. Not the current full-finetuning response study.
- `CES_FL/results/upgrade_tests_s17_20261101T040407Z/`: completed prior R0/R1/R2/RN/RD ten-epoch arms and25 authorized evaluations. Those published test results were viewed. Do not recycle them as this experiment's measurements.
- `CES_FL/proposals/counterfactual_5050_20261101_r1/`: separate full-coverage own-source-demographic-invariance plan. New original-only P uses its unchanged base-trainer/config dependency; the full approved-image bank and its B/N/C study were not launched. This does not mean the new original-only P is stopped.
- Prior BD0/CLIP/CFES/Ding-reproduction numbers are separate algorithms/protocols. Do not rerun Ding reproduction without a specific authorized reason or call those historical results current CES-FL response results.
- `CES_FL/proposals/ff2_supervisor_review_20261101_r1/REVIEW.md`: independent FairForge-2 full-comparison blockers. Do not substitute FF++ config for FF2; matched/corrected Ding baseline, score adapter, paired controls, group mapping and uncertainty work remain separate. Do not claim there are no historical FF2 results at all; distinguish historical in-domain results, transfer evaluations and the blocked new controlled comparison.
- The earlier preregistered CESel-MA evidence-floor gate remains **FAIL**. This exploratory response study does not amend or retroactively pass it.

## 13. Review deliverables and publication discipline

Return:

1. A concise plain-language explanation of the current architecture, hypothesis, objectives and how they differ from earlier CES-FL versions.
2. A verification table for **P, bce, fair, nochange, score, response**, each labeled RUNNING, COMPLETED, FAILED or BLOCKED_INPUTS with timestamps and real evidence paths. Describe queued/startup/unknown process state explicitly without pretending a scientific arm is running.
3. Exact source/config/input/checkpoint hashes; actual host/device and worker/child PID or scheduler ID; native epoch progress, selected validation metrics, subgroup support, calibration, quartet exposure and completion/failure evidence.
4. A distinction between measured findings, software tests, proposals, hypotheses and unresolved validity assumptions. State what you cannot inspect without cluster access.
5. A minimal dependency-ordered next action, proposed patches only if concrete issues exist, and an honest estimate from actual throughput. Do not promise beating Ding or improving fairness.
6. Scoped versioned GitHub updates and a local Markdown log. Do not overwrite existing results or live-read sources. Publish permitted aggregate metrics, textual logs, configs and source hashes; keep faces, row-level reviews/data, credentials and model/checkpoint binaries private.

Make code changes only in a new review branch/worktree, with exact request/config/source and appropriate software tests. Old smoke evidence never certifies changed code. A Git push is not a job launch. User authorization persists within its approved scope, but code/config replacement, skipped validity checks, expanded generation and test tuning are not implied by a status request.

If you have GitHub access only, report the last published observation and its age; do not claim to have SSH, inspected private checkpoints or verified current liveness. Ask the authorized cluster operator for the concrete missing receipts. GitHub write access and cluster authorization are separate.

First response: **verify current state read-only, explain what is running versus blocked, and identify exactly one next dependency. Preserve the active experiment.**

---

Cluster copy of this prompt: `/home/sachin.chaudhary/cesel_ma_audit_logs/CES_FL_SUPERVISOR_HANDOFF_2026-11-01.md`.
