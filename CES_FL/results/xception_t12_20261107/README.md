# Xception Tables 1-2: extra seeds (Amendment 10) and Xception SBI arms (Amendment 11), 07 Nov 2026

Aggregates only: no faces, image paths, per-image scores or checkpoints. Status at push: seed 17 rows only (the original locked L_A/L_H tables); seeds 7 and 27 (stage P -> L_A/L_H) and the Xception SBI arms (R0-SBI-SWA-X / R3-SBI-SWA-X, seeds 37/47/57) are training.

## Pipeline (Amendment 10)
- Stage P = ces_fl_manipulation_response.py pretrain @ 58773bb with only pretrain.seed and pretrain.head_seed changed; L_A/L_H = ces_fl_head_ablation.py @ 0d699d2 with only seed and init_checkpoint/init_sha256 changed; scoring = the locked lh_tables logic (seed-17 config pin replaced by the seed's config; no bootstrap).

```
af6b42b59996c42bdb2929bfa7be68ce0646d12bea3fd2246f47e7daf1f3cdd9  stage_seed.sh
4ce87cd45dd2697010d8e8e6994a396ae2abed4d3ac0c956e8e9b08efdfc8e08  make_l_configs.py
f6cfc541ad633af1168f141ff1548057e7efc3cb7b65c19282c2710e3da3bac1  score_seed.py
18cbd639d4d7b806b84d2b318d8e69f240ca9fc93287ba992a6cb50beb993dfd  aggregate.py
4a34fa58d511d0e09aeea5fa018c1189e67296f71e949aba1aab425a65ee70db  watchdog/xseeds_watchdog.sh
e5e3e38c2002e25d007a5760db108d4c6a7db6c94f8491fee02a6d5f4b68c809  configs/P_s7.json
48a72f1a0ebc6cb684dc8c301686e8b5b48ad6dbd63ee7629970c9442e09ec41  configs/P_s27.json
0e82a522a22e641dc5588b6a15317e6842896ab2bdcc5364833af25e207f257e  ces_fl_sources/cesfl-response-58773bb/CES_FL/src/ces_fl_manipulation_response.py
3e75192a4261eceda122ccd0c616b82928a25e05bc5173b2f728265b37d811ab  ces_fl_sources/cesfl-response-58773bb/CES_FL/src/ces_fl_counterfactual_5050.py
08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177  ces_fl_sources/cesfl-plain-head-0d699d2/CES_FL/src/ces_fl_head_ablation.py
env: .conda/envs/aiface/bin/python (torch 2.7.0+cu126), as the s17 originals
```
