# Four-arm head comparison

FF++ validation only, seed17. Native earliest maximum-AUC checkpoints, ten epochs each. Each selected model keeps its own global validation-calibrated threshold. Reference thresholds were reused verbatim.

|Model/objective|Selected epoch|Total/trainable parameters|Native train minutes|AUC %|FPR %|FNR %|Gender F_FPR|Race F_FPR|Joint F_FPR|Age F_FPR|Skin F_FPR|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|gated_A|4|21,594,159/7,575,591|60.98|98.426411|9.995112|4.146708|0.043699|0.087846|0.505265|0.228592|0.028039|
|linear_A|3|20,809,001/6,790,433|42.70|98.232624|9.995112|4.404997|0.055458|0.070055|0.436857|0.214782|0.028241|
|gated_H|1|21,594,159/7,575,591|60.89|98.197763|9.995112|4.132620|0.059377|0.078985|0.421326|0.291921|0.055875|
|linear_H|1|20,809,001/6,790,433|51.97|98.200410|9.995112|4.306377|0.031940|0.056300|0.410837|0.239855|0.019498|

F_FPR is a sum of absolute deviations in fraction units. Runtime is native accumulated training time; it excludes some software checks and replay overhead. Shared-node contention and different observation times prevent an architecture-only speed claim. Total allocated GPU time is recorded separately.

This allocation used 1.670782 GPU-hours, including checks and idle overhead within the guard. Source `0d699d241e27f758ec0990fbe58493a15bdb0ff9`, SHA `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`. See ALLOCATION_END and final scheduler accounting for any additional scheduler startup/cleanup seconds.

## gender

|Model|FPR gap pp|FNR gap pp|AUC gap pp|Worst-group AUC %|es_AUC %|F_DP fraction|
|---|---:|---:|---:|---:|---:|---:|
|gated_A|4.369893|3.578779|2.192993|97.224503|96.314246|0.018285|
|linear_A|5.545779|3.866441|2.514353|96.861281|95.823288|0.018820|
|gated_H|5.937741|4.094974|2.477126|96.814269|95.824080|0.020091|
|linear_H|3.194007|4.208504|2.463444|96.870294|95.839458|0.025470|

## race

|Model|FPR gap pp|FNR gap pp|AUC gap pp|Worst-group AUC %|es_AUC %|F_DP fraction|
|---|---:|---:|---:|---:|---:|---:|
|gated_A|7.087615|2.325221|2.581425|96.166697|94.086909|0.031452|
|linear_A|4.720893|2.705556|2.734380|95.997937|93.598829|0.042130|
|gated_H|7.601626|2.190773|2.278104|96.190718|93.993864|0.031860|
|linear_H|4.554469|2.731263|2.600630|96.044592|94.122714|0.025359|

## gender_race

|Model|FPR gap pp|FNR gap pp|AUC gap pp|Worst-group AUC %|es_AUC %|F_DP fraction|
|---|---:|---:|---:|---:|---:|---:|
|gated_A|33.839133|9.230769|5.600754|93.959518|89.593071|0.141150|
|linear_A|24.602273|9.401709|5.758828|93.746737|88.255314|0.149483|
|gated_H|24.352539|9.572650|5.825544|93.808247|88.258933|0.134807|
|linear_H|24.034091|10.085470|5.795891|93.688088|87.992240|0.141079|

## age_bin

|Model|FPR gap pp|FNR gap pp|AUC gap pp|Worst-group AUC %|es_AUC %|F_DP fraction|
|---|---:|---:|---:|---:|---:|---:|
|gated_A|22.135247|8.215874|1.237037|97.603678|96.500492|0.084348|
|linear_A|19.765166|9.800181|1.510603|97.244687|95.969614|0.092599|
|gated_H|27.937797|8.924451|1.803928|96.872557|95.870539|0.099475|
|linear_H|22.330942|9.216361|1.249351|97.423897|96.339115|0.092805|

## skin_bin

|Model|FPR gap pp|FNR gap pp|AUC gap pp|Worst-group AUC %|es_AUC %|F_DP fraction|
|---|---:|---:|---:|---:|---:|---:|
|gated_A|1.902805|3.019232|1.150046|97.794184|96.650306|0.018605|
|linear_A|2.247151|3.341183|1.259052|97.566192|96.296954|0.021540|
|gated_H|3.715984|1.691740|1.371283|97.435187|96.211685|0.029966|
|linear_H|1.179037|2.677522|1.425714|97.428098|96.266452|0.018108|

## ids8

|Model|FPR gap pp|FNR gap pp|AUC gap pp|Worst-group AUC %|es_AUC %|F_DP fraction|
|---|---:|---:|---:|---:|---:|---:|
|gated_A|25.274576|7.735264|3.943182|95.674276|90.930236|0.068480|
|linear_A|26.097621|8.232003|4.517781|95.143147|90.440219|0.070001|
|gated_H|26.849808|7.918381|4.326934|95.120988|90.945367|0.085216|
|linear_H|20.066498|8.435712|4.583796|95.111185|88.843625|0.071809|

All per-group real/fake denominators, AUC/FPR/FNR and fixed0.5 diagnostics are in GROUP_COUNTS_AND_RATES.csv. The complete four-row wide table, including all gaps and worst-group metrics, is FOUR_ARM_TABLE.csv.

Paired connected-contributor bootstrap intervals and usable counts are in REPORT.md and AGGREGATE.json. Missing-group draws remain unavailable. The inherited primary support is746/1000, below900. These are dataset-sampling intervals conditional on one trained seed, not training-seed variability or independent TEST confirmation.

The H-minus-A differences also change auxiliary forwards; they are descriptive objective-package comparisons. Both head types share the same encoder pretrained with the gated head. Removing the complete gated package does not isolate the contribution of gates alone. Procedural probes are not validated demographic counterfactuals or causal forgery removal.
