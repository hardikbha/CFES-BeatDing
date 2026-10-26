# DINGREPRO_LIN verification (16 Oct 2026)

Paper: Ding et al., "Decoupling Bias, Aligning Distributions", CVPR'26, arXiv 2511.10150 (PDF /tmp/ding2.pdf).

## Step 1: dataset matches the paper

| Split | Paper Table A.1 | Ours | Match |
|---|---|---|---|
| FF++ train | 76,139 | 76,139 (12,926 real / 63,213 fake) | exact |
| FF++ val | 25,386 | 25,386 (4,092 real + 21,294 fake) | exact |
| FF++ test | 25,401 | 25,401 (4,538 real / 20,863 fake) | exact |
| DFDC test | 22,857 | 22,857 (14,699 / 8,158) | exact, row-level |
| Celeb-DF test | 28,458 | 28,458 (5,713 / 22,745) | exact |
| DFD test | 9,385 | 9,385 (3,092 / 6,293) | exact |

- DFDC: Ding's repo ships its own DFDC manifest (`dataset/test.csv`). Ours equals it row for row, same order;
  0 mismatches in label, ismale, isasian, iswhite, isblack, intersec_label. Both contain the same 1,476 duplicate rows.
- FF++, Celeb-DF, DFD: Ding released no manifest, so these are count-verified only. Source is the same public
  Lin et al. SharePoint release the Ding README links to, with "Others" kept as a race (as in Table A.1).
- Subgroup schemes match Table A.1: 8 groups for FF++/DFDC, 6 for Celeb-DF/DFD (no Asian).
- intersec_label agrees with the gender/race flags on every row of every file; every image exists on disk.
- Inherited from Lin's release: 174/965 FF++ test video IDs (2,571 frames, all real youtube) also appear in train.
  Effect at epoch-2 checkpoint: AUC 97.45 all vs 97.02 without them. Ding used the same release.

## Step 2: protocol vs paper (what we run)

| Setting | Paper | Official code | Ours |
|---|---|---|---|
| Backbone | Xception | XceptionDetector | same (ImageNet weights from timm cache; Baidu link unreachable) |
| Base training | CE, 50 ep, SGD 1e-3, bs 64 | train_ori_model.py | same |
| SFD ratio | 2% | prc arg | 2% (10 of 512 channels) |
| Decoupling iterations | "third iteration" best | one SFD pass in README | one pass (follows released code) |
| GDA | 50 ep, λ=0.005, ε=5e-4 | GDA.py | same; bs 32 real + 32 fake = 64 |
| Test transform | not stated | Resize 224 + CenterCrop 224 | same |
| Threshold | not stated | 0.5 on softmax | same |
| Metrics | Eq. 8-10 | test.py | eval_ding_protocol.py, same formulas |
| Epoch reported | not stated | test.py logs every epoch | final epoch (49) pre-registered; per-epoch curve kept |

Known defects in the released code (disclosed patches):
- `ImageDataset_Test` reads `intersec_label = row[4]` for all four datasets; in every released CSV column 4 is
  iswhite/isblack. Groups come out empty or wrong. DFDC branch has the correct `row[6]` commented out.
  Our eval reads the column by name.
- PyTorch >= 2.6 needs `weights_only=False` to load GDA's pickled model objects.

Paper targets (Xception, "Ours" row): FF++ 97.71 (Table I.3); DFDC 61.86; Celeb-DF 76.75; DFD 81.46 (Table 2).
"Ori" row: FF++ 92.69; DFDC 56.13; Celeb-DF 69.18; DFD 74.32.

## Results

Filled in from results/eval_ding_protocol/*.json as checkpoints are scored.

### Baseline (DING_base = paper's "Ori"), seed 7, epoch 49 — scored 16 Oct 00:42-00:56
| Dataset | Paper Ori AUC | Ours AUC | Diff |
|---|---|---|---|
| FF++ | 92.69 | 89.01 | -3.68 |
| DFDC | 56.13 | 54.41 | -1.72 |
| Celeb-DF | 69.18 | 56.94 | -12.24 |
| DFD | 74.32 | 57.19 | -17.13 |
- All test images are 380x380 square, so the test-time Resize(224)+CenterCrop keeps the whole face; aspect ratio ruled out.
- The paper's Ori/Xception row equals the "Xception" column of Tables I.3-I.6, which the README says come from the
  AAAI'25 fairness benchmark (liudan193). Likely a different training pipeline from train_ori_model.py; unconfirmed.
- Single seed. Seeds 17/27 being scored.

### Baseline seed 17, epoch 49
FF++ 96.32 | DFDC 55.60 | Celeb-DF 69.16 | DFD 68.01. Seed spread vs seed 7 is 7-12 AUC; seed 17 matches paper Ori
on Celeb-DF (69.18) and DFDC (56.13). 2-seed mean: FF++ 92.67, DFDC 55.01, CDF 63.05, DFD 62.60.

### DING_full seed 7, GDA epoch 6 of 50 (interim, not a verdict)
| Dataset | Paper Ours | Ours ep6 | Our base s7 |
|---|---|---|---|
| FF++ | 97.71 | 97.45 | 89.01 |
| DFDC | 61.86 | 56.07 | 54.41 |
| Celeb-DF | 76.75 | 71.86 | 56.94 |
| DFD | 81.46 | 73.32 | 57.19 |
GDA lifts AUC over its own base on every set (+1.7 to +16). Threshold-0.5 fairness (F_FPR/F_DP) is unstable mid-training
(Celeb-DF race F_FPR 47.2 vs paper 16.6); es-AUC improves over base everywhere.

### Baseline ("Ori") 3 seeds, epoch 49 — COMPLETE (16 Oct ~01:50)
| Dataset | Paper Ori | s7 | s17 | s27 | Mean ± sd | Paper within 1 sd? |
|---|---|---|---|---|---|---|
| FF++ | 92.69 | 89.01 | 96.32 | 96.99 | 94.11 ± 4.43 | yes |
| DFDC | 56.13 | 54.41 | 55.60 | 54.04 | 54.68 ± 0.81 | no (-1.45, ~1.8 sd) |
| Celeb-DF | 69.18 | 56.94 | 69.16 | 75.32 | 67.14 ± 9.35 | yes |
| DFD | 74.32 | 57.19 | 68.01 | 75.21 | 66.80 ± 9.07 | yes |
Verdict: baseline reproduces (paper's single numbers inside our seed spread; DFDC 1.5 below, very tight spread).
Cross-domain seed sd is ~9 AUC: single-run differences under ~10 AUC on CDF/DFD are not interpretable.

## MAJOR FINDING 16 Oct: the released GDA fairness loss is inert
Official GDA.py (lines ~211-230, unchanged in Ding_mod) computes the Sinkhorn fairness loss AND the total loss inside
`with torch.no_grad():`. geomloss leaves autograd on for the sum, so total_loss.backward() runs, but the fairness term
is detached (fair_loss.requires_grad = False). Verified on a toy layer with the exact pattern: gradients with
lambda=1000 vs lambda=0 differ by 0.0; moving the loss outside no_grad gives a 122.9 difference.
=> Released "GDA" = plain cross-entropy retrain of last_linear (1,026 params) on balanced 32 real + 32 fake batches,
   with 10/512 adjust_channel outputs masked (SFD). Our DING_full runs are exactly this.
Also: GDA.py ignores any seed argument and always uses seed 121 (aux_funcs.get_random_seed); our 3 seeds differ only
in the base checkpoint. Original GDA.py hardcodes channels [497,275,493,95,396,13,10,391,65,431] (from their run);
ours read SFD's output for each seed.
Implication: the paper's Ori -> Ours gain cannot come from the Sinkhorn loss as released. It must come from the
balanced head refit and/or channel masking, or from unreleased code.
