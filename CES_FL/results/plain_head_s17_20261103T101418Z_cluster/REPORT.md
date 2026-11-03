# Plain-head comparisons: FF++ validation, seed17

2026-11-03T12:50:41.982607+00:00

Pinned scientific commit `0d699d241e27f758ec0990fbe58493a15bdb0ff9`; source SHA `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`.

|Model/objective|Completed epochs|Selected epoch|AUC %|FPR %|FNR %|Gender F_FPR|Race F_FPR|Joint F_FPR|Age F_FPR|Skin F_FPR|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|gated_A|10|4|98.426411|9.995112|4.146708|0.043699|0.087846|0.505265|0.228592|0.028039|
|linear_A|10|3|98.232624|9.995112|4.404997|0.055458|0.070055|0.436857|0.214782|0.028241|
|gated_H|10|1|98.197763|9.995112|4.132620|0.059377|0.078985|0.421326|0.291921|0.055875|
|linear_H|10|1|98.200410|9.995112|4.306377|0.031940|0.056300|0.410837|0.239855|0.019498|

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

## plain_minus_gated_A

|Metric|First−second, fraction|95%CI|Usable/1000|
|---|---:|---|---:|
|overall/auc|-0.00193787|[-0.003153730950798997, -0.0008461241183232802]|1000|
|overall/fnr|0.00258289|[0.0002165752438544279, 0.005944932731695374]|1000|
|gender/F_FPR|0.01175886|[-0.003749145108901282, 0.021453092482697996]|1000|
|race/F_FPR|-0.01779115|[-0.09063267632878014, 0.04840902882944202]|998|
|gender_race/F_FPR|-0.06840858|[-0.12978633194588982, 0.049236386133546334]|746|
|age_bin/F_FPR|-0.01380973|[-0.04399868920999349, 0.01598960078788717]|976|
|skin_bin/F_FPR|0.00020144|[-0.028056451309245476, 0.027709624513281447]|1000|

## plain_minus_gated_H

|Metric|First−second, fraction|95%CI|Usable/1000|
|---|---:|---|---:|
|overall/auc|0.00002646|[-0.0013834034467089889, 0.001399306255334909]|1000|
|overall/fnr|0.00173758|[-0.0032779668814195365, 0.008205710784385286]|1000|
|gender/F_FPR|-0.02743734|[-0.048484273490629295, 0.014876648534982431]|1000|
|race/F_FPR|-0.02268451|[-0.07126871304796818, 0.045275811920857945]|998|
|gender_race/F_FPR|-0.01048940|[-0.12249989356586984, 0.059809723501164286]|746|
|age_bin/F_FPR|-0.05206561|[-0.21275549446872816, 0.03319271951288348]|976|
|skin_bin/F_FPR|-0.03637774|[-0.06874632970753039, 0.024170747120540247]|1000|

## H_minus_A_plain

|Metric|First−second, fraction|95%CI|Usable/1000|
|---|---:|---|---:|
|overall/auc|-0.00032214|[-0.0015520814136091078, 0.000980385362271163]|1000|
|overall/fnr|-0.00098619|[-0.005348730559416664, 0.002456691177722103]|1000|
|gender/F_FPR|-0.02351772|[-0.042220599098780925, 0.015959668069666928]|1000|
|race/F_FPR|-0.01375503|[-0.06994578642120575, 0.06240650600682544]|998|
|gender_race/F_FPR|-0.02602001|[-0.11686636391710192, 0.08296659515647456]|746|
|age_bin/F_FPR|0.02507321|[-0.020406267563877825, 0.05518774852217837]|976|
|skin_bin/F_FPR|-0.00874294|[-0.05531524683628504, 0.04515648170132786]|1000|

## H_minus_A_gated

|Metric|First−second, fraction|95%CI|Usable/1000|
|---|---:|---|---:|
|overall/auc|-0.00228648|[-0.004214435971476332, -0.00047353503188738463]|1000|
|overall/fnr|-0.00014088|[-0.010259830853881364, 0.007174286781236589]|1000|
|gender/F_FPR|0.01567848|[-0.014793540878182996, 0.0409702597042128]|1000|
|race/F_FPR|-0.00886168|[-0.0987982276104842, 0.07653780812876004]|998|
|gender_race/F_FPR|-0.08393920|[-0.18879629945583423, 0.09051814377758333]|746|
|age_bin/F_FPR|0.06332910|[-0.03438070174722755, 0.2056838425823091]|976|
|skin_bin/F_FPR|0.02783624|[-0.03588393614946428, 0.09016307692282725]|1000|

Head effects are descriptive conditional comparisons. No binary noninferiority gate was preregistered. Point improvements with intervals crossing zero are inconclusive, and single-seed VAL results cannot establish a Ding TEST win.
