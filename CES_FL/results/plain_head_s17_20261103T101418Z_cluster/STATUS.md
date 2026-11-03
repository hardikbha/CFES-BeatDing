# Plain-head comparisons: FF++ validation, seed17

2026-11-03T12:41:13.708711+00:00

Pinned scientific commit `0d699d241e27f758ec0990fbe58493a15bdb0ff9`; source SHA `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`.

|Model/objective|Completed epochs|Selected epoch|AUC %|FPR %|FNR %|Gender F_FPR|Race F_FPR|Joint F_FPR|Age F_FPR|Skin F_FPR|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|gated_A|10|4|98.426411|9.995112|4.146708|0.043699|0.087846|0.505265|0.228592|0.028039|
|linear_A|10|3|98.232624|9.995112|4.404997|0.055458|0.070055|0.436857|0.214782|0.028241|
|gated_H|10|1|98.197763|9.995112|4.132620|0.059377|0.078985|0.421326|0.291921|0.055875|
|linear_H|6|1|98.200410|9.995112|4.306377|0.031940|0.056300|0.410837|0.239855|0.019498|

F_FPR is a sum of absolute group-versus-pooled FPR deviations, displayed in fraction units. Lower is better. Reference thresholds are preserved; each new selected model uses its own global VAL-real empirical10%FPR threshold. Selection is earliest highest VAL AUC after ten epochs.

Linear:20,809,001 total /6,790,433 trainable /2,049 head parameters. Gated:21,594,159 total /7,575,591 trainable /787,207 heads, gates and fusion parameters. Same Xception exit policy and frozen BN.

All group denominators, AUC/FPR/FNR, disparities/gaps, worst-group AUC, histories, actual exposure, source/input/checkpoint hashes and runtime are in AGGREGATE.json. Fixed0.5 reports and paired intervals are added after completion.

## Checks and evidence

CPU:11 head tests,13 forensic regression,13 hierarchy regression,9 support regression passed. Production input preflights and CUDA receipts are reported by actual file presence and content below.
- preflight_A/PREFLIGHT.json: PASS
- preflight_H/PREFLIGHT.json: PASS
- smoke_cuda/SMOKE_RESULT.json: PASS
- dry_A/DRY_RUN.json: PASS
- dry_H/DRY_RUN.json: PASS
- review_H/qualification.json: OPERATOR_REVIEW_RECORDED

## Limitations

- Single training seed17; validation development comparison only.
- TRAIN/VAL163 declared recorded contributor overlaps; biological identity separation unverified.
- Age/skin metadata may be proxies; no appearance-based relabeling.
- Procedural local mismatch/sham recipes do not establish demographic or causal forgery validity.
- Conditional on the same P50 encoder pretrained with gated heads.
- 746/1000 usable primary component replicates; inherited900minimum cannot be met.
