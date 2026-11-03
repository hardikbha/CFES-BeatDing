# Plain-head comparisons: FF++ validation, seed17

2026-11-03T11:02:26.115097+00:00

Pinned scientific commit `0d699d241e27f758ec0990fbe58493a15bdb0ff9`; source SHA `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`.

|Model/objective|Completed epochs|Selected epoch|AUC %|FPR %|FNR %|Gender F_FPR|Race F_FPR|Joint F_FPR|Age F_FPR|Skin F_FPR|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|gated_A|0|pending|||||||||
|linear_A|0|pending|||||||||
|gated_H|0|pending|||||||||
|linear_H|0|pending|||||||||

F_FPR is a sum of absolute group-versus-pooled FPR deviations, displayed in fraction units. Lower is better. Reference thresholds are preserved; each new selected model uses its own global VAL-real empirical10%FPR threshold. Selection is earliest highest VAL AUC after ten epochs.

Linear:20,809,001 total /6,790,433 trainable /2,049 head parameters. Gated:21,594,159 total /7,575,591 trainable /787,207 heads, gates and fusion parameters. Same Xception exit policy and frozen BN.

All group denominators, AUC/FPR/FNR, disparities/gaps, worst-group AUC, histories, actual exposure, source/input/checkpoint hashes and runtime are in AGGREGATE.json. Fixed0.5 reports and paired intervals are added after completion.

## Checks and evidence

CPU:11 head tests,13 forensic regression,13 hierarchy regression,9 support regression passed. Production input preflights and CUDA receipts are reported by actual file presence and content below.
- preflight_A/PREFLIGHT.json: PENDING
- preflight_H/PREFLIGHT.json: PENDING
- smoke_cuda/SMOKE_RESULT.json: PENDING
- dry_A/DRY_RUN.json: PENDING
- dry_H/DRY_RUN.json: PENDING
- review_H/qualification.json: PENDING

## Limitations

- Single training seed17; validation development comparison only.
- TRAIN/VAL163 declared recorded contributor overlaps; biological identity separation unverified.
- Age/skin metadata may be proxies; no appearance-based relabeling.
- Procedural local mismatch/sham recipes do not establish demographic or causal forgery validity.
- Conditional on the same P50 encoder pretrained with gated heads.
- 746/1000 usable primary component replicates; inherited900minimum cannot be met.
