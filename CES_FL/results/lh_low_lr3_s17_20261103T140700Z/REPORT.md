# L_H lower-learning-rate three-epoch validation experiment

2026-11-03T14:58:26.522361+00:00

All new scientific computation uses TRAIN/VAL only. Exactly three config fields changed. Reference models/histories/checkpoints are preserved.

|Model/epoch|AUC %|Achieved FPR %|FNR %|Gender F_FPR|Race F_FPR|Joint F_FPR|Age F_FPR|Skin F_FPR|ids8 F_FPR|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|original_L_H_epoch1|98.200410|9.995112|4.306377|0.031940|0.056300|0.410837|0.239855|0.019498|0.438923|
|original_L_H_epoch2|98.131094|9.995112|4.456654|0.041739|0.077410|0.448156|0.272217|0.022574|0.520293|
|original_L_H_epoch3|98.046415|9.995112|4.442566|0.046639|0.071185|0.446238|0.303103|0.032821|0.507762|
|original_retained_best_after10|98.200410|9.995112|4.306377|0.031940|0.056300|0.410837|0.239855|0.019498|0.438923|
|low_lr_epoch1|97.874579|9.995112|5.466329|0.042719|0.101704|0.413609|0.118314|0.034494|0.336776|
|low_lr_epoch2|98.096682|9.995112|4.775993|0.040759|0.091281|0.407493|0.140089|0.037547|0.341237|
|low_lr_epoch3|98.185660|9.995112|4.611628|0.043699|0.094555|0.418939|0.192721|0.036871|0.337456|
|low_lr_selected_after3|98.185660|9.995112|4.611628|0.043699|0.094555|0.418939|0.192721|0.036871|0.337456|

## gender

|Model|F_DP|es-AUC %|Worst AUC %|FPR gap pp|FNR gap pp|AUC gap pp|
|---|---:|---:|---:|---:|---:|---:|
|original_L_H_epoch1|0.025470|95.839458|96.870294|3.194007|4.208504|2.463444|
|original_L_H_epoch2|0.020346|95.711796|96.759038|4.173912|3.784849|2.527690|
|original_L_H_epoch3|0.018051|95.516544|96.602595|4.663864|3.605423|2.648621|
|original_retained_best_after10|0.025470|95.839458|96.870294|3.194007|4.208504|2.463444|
|low_lr_epoch1|0.032015|95.190574|96.351952|4.271902|5.187936|2.819612|
|low_lr_epoch2|0.026671|95.572451|96.667782|4.075921|4.517882|2.641170|
|low_lr_epoch3|0.025872|95.711399|96.783099|4.369893|4.480102|2.585127|
|low_lr_selected_after3|0.025872|95.711399|96.783099|4.369893|4.480102|2.585127|

## race

|Model|F_DP|es-AUC %|Worst AUC %|FPR gap pp|FNR gap pp|AUC gap pp|
|---|---:|---:|---:|---:|---:|---:|
|original_L_H_epoch1|0.025359|94.122714|96.044592|4.554469|2.731263|2.600630|
|original_L_H_epoch2|0.024992|94.297719|96.101718|5.325203|2.784708|2.417283|
|original_L_H_epoch3|0.029140|94.384879|96.104798|4.753342|2.380667|2.322466|
|original_retained_best_after10|0.025359|94.122714|96.044592|4.554469|2.731263|2.600630|
|low_lr_epoch1|0.065931|92.870229|95.500000|7.822967|5.656109|2.947656|
|low_lr_epoch2|0.046167|93.540638|95.926337|7.185008|3.315804|2.713912|
|low_lr_epoch3|0.043425|93.812145|96.033198|6.866029|3.096898|2.701772|
|low_lr_selected_after3|0.043425|93.812145|96.033198|6.866029|3.096898|2.701772|

## age_bin

|Model|F_DP|es-AUC %|Worst AUC %|FPR gap pp|FNR gap pp|AUC gap pp|
|---|---:|---:|---:|---:|---:|---:|
|original_L_H_epoch1|0.092805|96.339115|97.423897|22.330942|9.216361|1.249351|
|original_L_H_epoch2|0.099750|96.134863|97.455234|25.309850|9.299764|1.150510|
|original_L_H_epoch3|0.104081|95.834301|97.304587|27.679930|9.258062|1.225507|
|original_retained_best_after10|0.092805|96.339115|97.423897|22.330942|9.216361|1.249351|
|low_lr_epoch1|0.076668|95.155189|96.586359|11.162661|9.384209|1.871023|
|low_lr_epoch2|0.078366|95.601845|96.934905|12.828876|9.800181|1.700928|
|low_lr_epoch3|0.088611|95.743527|97.041648|17.827609|9.841882|1.672545|
|low_lr_selected_after3|0.088611|95.743527|97.041648|17.827609|9.841882|1.672545|

## skin_bin

|Model|F_DP|es-AUC %|Worst AUC %|FPR gap pp|FNR gap pp|AUC gap pp|
|---|---:|---:|---:|---:|---:|---:|
|original_L_H_epoch1|0.018108|96.266452|97.428098|1.179037|2.677522|1.425714|
|original_L_H_epoch2|0.017290|96.196476|97.400167|1.344698|3.040323|1.402637|
|original_L_H_epoch3|0.014349|96.038077|97.379178|2.815951|2.917433|1.367537|
|original_retained_best_after10|0.018108|96.266452|97.428098|1.179037|2.677522|1.425714|
|low_lr_epoch1|0.025920|95.477364|97.070642|2.785010|3.270851|1.537130|
|low_lr_epoch2|0.024138|95.955580|97.347183|3.333568|3.088203|1.412759|
|low_lr_epoch3|0.022672|96.180556|97.471358|2.989038|3.010019|1.340213|
|low_lr_selected_after3|0.022672|96.180556|97.471358|2.989038|3.010019|1.340213|

## gender_race

|Model|F_DP|es-AUC %|Worst AUC %|FPR gap pp|FNR gap pp|AUC gap pp|
|---|---:|---:|---:|---:|---:|---:|
|original_L_H_epoch1|0.141079|87.992240|93.688088|24.034091|10.085470|5.795891|
|original_L_H_epoch2|0.132275|88.272481|93.717412|23.953223|9.658120|5.747146|
|original_L_H_epoch3|0.123635|88.301843|93.730286|24.181403|8.803419|5.721787|
|original_retained_best_after10|0.141079|87.992240|93.688088|24.034091|10.085470|5.795891|
|low_lr_epoch1|0.190038|86.658019|93.441333|26.022727|13.105413|5.880998|
|low_lr_epoch2|0.156704|87.584117|93.754604|26.022727|9.506173|5.705792|
|low_lr_epoch3|0.152538|87.875782|93.872260|25.738636|9.591643|5.645010|
|low_lr_selected_after3|0.152538|87.875782|93.872260|25.738636|9.591643|5.645010|

## ids8

|Model|F_DP|es-AUC %|Worst AUC %|FPR gap pp|FNR gap pp|AUC gap pp|
|---|---:|---:|---:|---:|---:|---:|
|original_L_H_epoch1|0.071809|88.843625|95.111185|20.066498|8.435712|4.583796|
|original_L_H_epoch2|0.080747|88.669476|94.962514|24.067991|8.565044|4.725221|
|original_L_H_epoch3|0.084931|88.562903|94.766353|23.934658|8.694377|4.905443|
|original_retained_best_after10|0.071809|88.843625|95.111185|20.066498|8.435712|4.583796|
|low_lr_epoch1|0.067625|88.911327|94.622645|20.799608|8.532652|4.887947|
|low_lr_epoch2|0.068861|89.622321|94.933497|20.638814|8.468025|4.707146|
|low_lr_epoch3|0.067625|89.764452|95.025965|19.314311|8.542401|4.658148|
|low_lr_selected_after3|0.067625|89.764452|95.025965|19.314311|8.542401|4.658148|

The actual loss is unchanged. Lower transformed sham differences alone do not establish lower raw differences or causal improvement. No TEST result or best fairness checkpoint is substituted.

Selected new checkpoint epoch3, SHA `59143159eea223d97bc436b510f13ca90b750f00851071a860f91ec9231d4a06`. Source/config/commands/provenance, complete histories and runtime are in COMPARISON.json; all group support/rates and every metric are in the CSV exports.

Single-seed validation development only. No significance conclusion or overall fairness win follows without supported uncertainty. Every regression is retained in the same tables.
