# FairForge-2 public: Ding architecture, original-only seed17 baseline

User authorized training now, requested two GPUs and native Ding image/global
batch sizes, and explicitly selected direct SSH on gpu01/gpu02. This releases
only this original-only baseline. It does not release CES-FL counterfactual arms,
image generation, the old failed CESel-MA gate, or unrelated FF2 studies.

## Scientific identity and declared deviations

This is **Ding Xception classification → native-SFD → corrected class-conditioned
empirical-OT GDA on FairForge-2 public**. It is not an assertion that the public
GDA reproduces the paper, and not a replacement for archived results. Archived
GDA freezes all but last_linear, runs total/fairness construction under no_grad,
and permits encoder BN-buffer updates in train mode. The new GDA is explicitly
differentiable, class-conditional, head-only, and freezes encoder BN buffers.
Native architecture, initialization weights, original-image sampling, global
batch exposure, SGD hyperparameters and 50+50 epoch budgets are preserved.

The empirical OT uses fake softmax probabilities and sums the real- and
fake-class OT costs, comparing one uniformly selected demographic group with
the global main batch. In expectation, uniform selection estimates the mean
over eight groups. SamplesLoss0.3.1 uses p=2, blur=sqrt(0.0005), diameter=1,
debias=False and tensorized backend. GeomLoss's half-squared cost, empirical
samples rather than KDE, epsilon interpretation and BN freeze are disclosed
differences from the archived implementation/paper. Lambda=0.005. This task
does not modify the prospective CES-FL CDF proposal or its preregistration.

## Correction: real executable resolution and batch sizes

train_ori_model.py imports src.transform.resnet_default_data_transforms;
GDA.py and SFD.py import the top-level dictionary of the same name. Their
actual input is direct Resize((224,224)), RGB, ToTensor and mean/std0.5.
256x256 is only the archived FLOPs dummy, not the training loader. Earlier
summaries that interpreted the dummy as the classification resolution were
incorrect. Native test.py uses Resize(short side224), CenterCrop224 with the
same normalization. This new run uses that native test geometry on VAL; TEST
inference is absent from the training command.

| Stage | Epochs | Global main exposure/update | Auxiliary exposure/update | Trainability |
|---|---:|---:|---:|---|
| Classification | 50 | 64real+64fake=128 | 0 | All21,858,090 parameters; synchronized global-batch BN |
| SFD | One full TRAIN traversal | 64 originals | 0 | Inference only, final classification epoch50 |
| Corrected GDA | 50 | 32real+32fake=64 | 32real+32fake=64 from one uniformly selected group | Only1,026 last_linear parameters; encoder/BN frozen |

Global fake originals are shuffled once per epoch; real images are drawn with
replacement. GDA auxiliary real/fake samples are drawn independently with
replacement within the selected group. No demographic quota sampler is
substituted. NumPy's deterministic sampler differs from the archive's Python
random stream; no bitwise trajectory identity is claimed. Last batches are
preserved, distributed without duplicate padding, with correctly weighted CE.
Autograd gathering preserves auxiliary/main roles and uneven final-rank sizes.

SFD exactly preserves the archived Sequential(children[:-2]) helper and signed
cosine-neighborhood implementation (self included only in the numerator),
verified against the executable AST-extracted reference. It is not silently
replaced with backbone.features, absolute ranks, or a paper-inspired alternative.
The lowest2% (strict native threshold, normally10/512) output channels receive
a pruning mask. Tied/degenerate channel selection fails closed. No structural
parameter removal is claimed; all21,858,090 parameters remain registered.

## Inputs, roles, selection and uncertainty

Exact paths and SHA256 values are in config.json. Public canonical counts:
TRAIN218,266 (16,777real/201,489fake), VAL31,107 (2,438real/28,669fake),
TEST87,564 (5,441real/82,123fake). Native train.csv membership/labels and its
eight-group mapping must match the canonical TRAIN; nonexistent realtrain.csv
and faketrain.csv are not assumed. TRAIN/VAL/TEST metadata are checked for path,
recorded identity and source overlap. These records cannot establish hidden
donor or true-person independence. Stratified TRAIN/VAL decoding is a smoke
check; not every image is decoded before launch. Subsequent decoding failures
stop the run instead of skipping rows.

Only TRAIN produces gradients and SFD ranks. VAL AUC (earliest tie) selects
reported checkpoints. Native final classification epoch50 initializes SFD/GDA,
separately from the VAL-selected classification baseline. No TEST pixels or
scores are read; TEST metadata are used only to prevent split overlap.

Threshold: target real-VAL FPR0.1; n real scores, k=floor(0.1n), threshold is
nextafter(descending_scores[k],+infinity), predict fake at score>=threshold.
Ties are excluded as a whole; report actual achieved VAL FPR, never a TEST FPR
guarantee. The selected checkpoint and threshold are independently replayed and
frozen before any future TEST command. This task attaches no automatic TEST
evaluation. Report fixed probability0.5 separately in later approved evaluation.

VAL reports overall AUC, FPR/FNR, support, per-group statistics, worst eligible
AUC, gaps and Ding-style fractions for gender/race/intersection/age/skin/ids8.
Age/skin are source automatic proxies, not human demographic ground truth.
Single seed17 is an initial baseline, not sufficient evidence of superiority.
No seed-level or cluster-aware significance claim is supported by this run alone.

## Resource route and gates

Direct SSH is explicitly user authorized, with no PBS ngpus allocation claimed.
The operator cooperates with the existing pool-v2 registry under its host mutex,
claims two compatible shared GPUs atomically, respects all live reservations and
exclusive locks, and never removes another job/lease. It reserves20,000MiB/card
plus4,096MiB packing margin, sets a0.22 allocator fraction, and verifies fit with
a disposable full-native-batch two-rank CUDA smoke before training. Both GPUs
may be shared; no exclusive GPU or CPU ownership/speed guarantee is made.
Six CPU-only DataLoader workers and two compute threads per rank bound the
requested16-CPU budget. CPU allocation is not claimed as a scheduler grant.

Launch gates: source detached/clean, exact hashes, input preflight PASS,
source-matched CPU checks PASS, two-GPU CUDA/backward/BN/checkpoint smoke PASS.
CUDA failure stops training; no automatic batch reduction or CPU fallback.
Output directories are new. Existing jobs/checkpoints are never terminated.
Native per-epoch states, last/best optimizer checkpoints and histories persist.

## Exact commands

Private run root:
`/home/sachin.chaudhary/ces_fl_runs/ding_ff2_public_s17_20261102T130051Z`

From the pinned source checkout (record actual commit in execution evidence):

```bash
/home/sachin.chaudhary/.conda/envs/qwen38/bin/python -u CES_FL/tests/test_ding_ff2_public.py
/home/sachin.chaudhary/.conda/envs/qwen38/bin/python -u CES_FL/src/ding_ff2_public.py preflight --config CES_FL/proposals/ding_ff2_public_s17_20261102T130051Z/config.json --out /home/sachin.chaudhary/ces_fl_runs/ding_ff2_public_s17_20261102T130051Z/preflight_v2
```

These checks already create their evidence; do not repeat into existing output
directories. The operator is invoked ONCE over SSH after source/receipts freeze:

```text
ssh gpu01
nohup /home/sachin.chaudhary/.conda/envs/qwen38/bin/python -u /home/sachin.chaudhary/ces_fl_sources/ding_ff2_public_s17_20261102T130051Z/CES_FL/scripts/run_ding_ff2_ssh.py --source /home/sachin.chaudhary/ces_fl_sources/ding_ff2_public_s17_20261102T130051Z --config /home/sachin.chaudhary/ces_fl_sources/ding_ff2_public_s17_20261102T130051Z/CES_FL/proposals/ding_ff2_public_s17_20261102T130051Z/config.json --run /home/sachin.chaudhary/ces_fl_runs/ding_ff2_public_s17_20261102T130051Z --python /home/sachin.chaudhary/.conda/envs/qwen38/bin/python --preflight preflight_v2 > /home/sachin.chaudhary/ces_fl_runs/ding_ff2_public_s17_20261102T130051Z/logs/controller.log 2>&1 < /dev/null &
```

The operator first executes torch.distributed.run --standalone
--nproc-per-node=2 ... ding_ff2_public.py cuda-smoke, then the identical wrapper
with train only if CUDA_SMOKE_PASS.json matches preflight source hashes.
Actual commands/device UUIDs/PIDs and status are recorded in STATUS.json,
cuda_smoke/START.json, training/START.json, logs/cuda_smoke.log and
logs/training.log. No launch/result is inferred from this document.
