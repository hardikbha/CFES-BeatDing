# ResNet-50 Table 3: Amendment 4 aggregates (06 Nov 2026)

Aggregates only: no faces, image paths, per-image scores or checkpoints.

## Arms
ResNet-50 is trained with Ding's recipe: SGD lr 1e-3, momentum 0.9, weight decay 5e-3, 64 real + 64 fake per step, no augmentation, 50 epochs, SWA over epochs 40-49, and BN recomputed on FF++ train.

- **R0-SWA:** Ding's CE path, unchanged.
- **R3-SWA:** R0-SWA plus, every 4th step, 4 x [0.5 CORAL + 0.02 real-tail] on a separate balanced fair batch.

Seeds 37, 47 and 57. The primary metric is op05 vs Ding published Table 3.

## Results
- **R0-SWA:** 14.0 +/- 1.0 of 40 per seed. The mean-metric row wins 15/40.
- **R3-SWA:** all 3 seeds were killed in epoch 1 by the pre-registered aux>CE gradient rule. Amendment 5 drops that rule as degenerate.

## Files
- `SCORECARD_R3_3seed.md` and `SCORECARD_R3_3seed.json`: per-seed rows, means, criteria and kill status.
- `percol_r0swa_means.txt`: per-column op05 values for R0-SWA, with the seed values, mean and wins vs Ding.
- `decisions_amendments_4_5.md`: the pre-registered Amendments 4 and 5.
- `codex_resnet_strategy_20261106.md`: the strategy memo.

## Code and config hashes (sha256)
```
fe256d604d6988f31fbc35f0dfc1bce93d6ad11e6269ab7d5a606e296d72027d  train_r3.py
599898cbd49f8b6d5958dec5cf90c095289d00af257c0210e8ae12f4084092a5  r2_losses.py
7711484f16b8095247068ba66b9d005fb1f7783f37fa613ff8599f1fd8e34faf  train_samebb.py
45033fca09619edb4cc6255e09a782f49556aaaa689299aef16fa814ccbecae9  samebb_score.py
71542db80e8d947cf702431c391b4017243d12f931edcb2783efea3ffae350ee  samebb_scorecard.py
9ccdd3cbc3f145d62b91ab46ee49153ef107c15c349e67e8b99f4430e333b1d8  r3_aggregate.py
ee2d3b3229ed0ce96bf3e5517e488059419d1fae411f9d05739808e81ea6a7d0  train_ori_model.py
2ee306814f0ba1baaf18a14fce30ed3c23aa63bdc6134ca21076723d489f0468  detectors/res50_detector.py
54c3753d73a71966be45a0d1961690ed08b8c2570cc0dfbabe9ae64623d800f7  run/run_r3.sh
206556d9903082b736a3b0f19f6e7d81778f8d5247a6ddfaaf16793e2db4622e  watchdog/r3_watchdog.sh
a431846f3ee97814cedece1c77fa187444264dedcf8b23d35769c4fa9bba2e5c  paper_cvpr27/best_vs_ding/best_vs_ding.md
```

Run config (R0_s37/args_train.txt):
```json
{
  "seed": 37,
  "num_epochs": 50,
  "lr": 0.001,
  "result_dir": "<local path>",
  "dataset_dir": "<local path>",
  "batch_size": 64,
  "workers": 6,
  "model_name": "Res50Detector",
  "arm": "R0",
  "group_manifest": "<local path>",
  "fair_every": 4,
  "fair_per_cell": 8,
  "lambda_coral": 0.5,
  "lambda_tail": 0.02,
  "tail_margin": 1.0,
  "fair_workers": 3,
  "fair_seed_offset": 1000003,
  "swa_start": 40,
  "val_epochs": "10,15",
  "resume": true,
  "max_steps": -1,
  "diag_every": 0,
  "skip_swa_bn": false
}```

## Update: SBI arms (Amendment 5), final 3 seeds
- `SCORECARD_SBI_3seed.md` and `SCORECARD_SBI_3seed.json` cover R0-SWA, R0-SBI-SWA and R3-SBI-SWA for seeds 37, 47 and 57, at op05 and valFPR10, for SWA and raw epoch 49. They include the paired diffs and the Celeb-DF per-column table.
- `TABLE3_FULL_op05_SWA.md` is the full Table 3 per-column view at op05 for the SWA models, with per-seed values and means vs Ding.
- Headline (op05, SWA), mean wins per seed:
  - R0-SWA: 14.0 +/- 1.0.
  - R0-SBI-SWA: 13.7 +/- 3.2.
  - R3-SBI-SWA: 17.7 +/- 3.1 (per seed 15, 17 and 21); mean AUC FF++ 96.01, DFDC 62.23, Celeb-DF 80.07, DFD 77.74.
  - Method evidence for R3-SBI vs R0-SBI: MET. Target (25/40 and AUC >= Ding on all 4 sets): NOT MET.
- `decisions_amendments_4_5_6.md`: the pre-registered Amendments 4, 5 and 6.
- SBI is the CERTA-2 F1 self-blend on InsightFace 106-point landmarks of FF++ train reals (12,784 of 12,926). No faces or landmarks are published.

Code sha256 (code_r3sbi, run script, watchdog):
```
1e45befacefa11eda07f19768f0d2b68be1583a36bffd683941ed09cbc3b9f65  RESNET_STRAT/watchdog_sbi/r3sbi_watchdog.sh
a765b5927c8adce375b606fe8d9bec8a4e3440daeb59917b2a51262fe6da0232  SAMEBB/code_r3sbi/run/run_r3sbi.sh
ccb57822cde4efdc004b2eb89710e81965e1ffe11cc23ec3b9c2230c6aca2134  SAMEBB/code_r3sbi/Ding_r3sbi/train_r3sbi.py
0c2230811fc14505735a81e37c53849a055f0171af0d5280f77b2b963beb4084  SAMEBB/code_r3sbi/Ding_r3sbi/sbi.py
d9ba7275e62dd28aa17af41b43e732b876307f7189e772083b5313498394c959  SAMEBB/code_r3sbi/Ding_r3sbi/r3sbi_aggregate.py
```

## Update: post-hoc variants (Amendment 8)
- `TABLE3_POSTHOC.md` and `TABLE3_POSTHOC.json` report each arm under SWA, SWA+TTA (hflip), SOUP (uniform average of 3 seeds, BN recomputed on FF++ train) and SOUP+TTA.
- Two operating points:
  - op05: Ding protocol, primary.
  - valFPR1: the 99th percentile of the model's own FF++ val real logits. This is a frozen secondary rule and a deviation from Ding's protocol.
- The tables show per-seed rows, the mean and mean-row wins. The 'best seed' rows are selected post hoc on test and labelled.
- First pass covers R0-SWA, R0-SBI, R3-SBI (3 seeds + soup), R7 (s47, s57) and R0-BSBI (s37). The other arms are added as they finish.
- Code sha256:
```
1cba36bb29030930c56bf68c60b23c3a71e6ce7db0a48506cf77819623b7927f  posthoc/posthoc.py
d16f4982c54e3ddb5fcf7e52cdaa1bd0c16c96531bcd7d3e9e7a8d4d97d85023  posthoc/worker.sh
81f50993d0f8b11cba4714b6d1b1e4afde82e0d400be6cc0c68ff75ee2dc072b  posthoc/chain.sh
```

## Update: R4/R5 (Amendment 6) and refreshed post-hoc table (Amendment 8/8a)
- `SCORECARD_R4R5_3seed.md` and `SCORECARD_R4R5_3seed.json`, op05, SWA, mean wins per seed:
  - R4 (balanced SBI sources): 15.67 +/- 1.53.
  - R5 (R4 with L_R 0.10): 14.67 +/- 0.58.
  - Both are below R3-SBI at 17.67 +/- 3.06.
- `TABLE3_FULL_v2_op05_SWA.md`: the full per-column Table 3 including R4 and R5.
- `TABLE3_POSTHOC.md` and `TABLE3_POSTHOC.json` now also cover R4, R5 and R7 (3 seeds + soup) and R6 s37.
- Soup sanity on FF++ val AUC flags R3-SBI and R4 as degenerate (more than 2 pt below the seed mean).
- R6b (Amendment 8a: R3-SBI plus L_DP with coefficient 2.0) is running, seeds 37, 47 and 57.
