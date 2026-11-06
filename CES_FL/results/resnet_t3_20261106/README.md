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
