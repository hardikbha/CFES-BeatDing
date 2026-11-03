# Interim all-axis validation metrics and group support

Management observation: 2026-11-03T06:29:34.905000+00:00. New candidate values are not final results. Select by unchanged highest validation AUC, never by a favorable fairness epoch. New candidate source8d5f099/SHA0eeafd; old controls sourcec332fd34/SHA895300.

F_FPR uses fractions. FPR/FNR gaps and AUC columns use percent / percentage points. Lower F_FPR and gaps, higher worst-group AUC are preferable; compare all axes and utility together.

|Arm|State|Complete epochs|Selected epoch|Overall AUC %|Pooled FPR %|Pooled fake FNR %|
|---|---|---:|---:|---:|---:|---:|
|A|COMPLETED_VERIFIED|10/10|4|98.426411|9.995112|4.146708|
|B|COMPLETED_VERIFIED|10/10|2|98.406797|9.995112|4.235935|
|C|COMPLETED_VERIFIED|10/10|1|98.307659|9.995112|3.709965|
|D|COMPLETED_VERIFIED|10/10|1|98.252564|9.995112|3.963558|
|E_support|RUNNING;selected-so-far|2/10|2|98.228795|9.995112|4.146708|
|F_support_ramp3|RUNNING;selected-so-far|5/10|1|98.319331|9.995112|3.897812|

## gender

|Arm|F_FPR fraction|FPR gap pp|FNR gap pp|Worst-group AUC %|
|---|---:|---:|---:|---:|
|A|0.043699|4.369893|3.578779|97.224503|
|B|0.041739|4.173912|3.818987|97.177320|
|C|0.043699|4.369893|3.600326|97.048280|
|D|0.053498|5.349798|4.085121|96.855934|
|E_support|0.054478|5.447788|3.747984|96.932023|
|F_support_ramp3|0.048598|4.859845|3.949685|97.061978|

|Group|Real validation rows|Fake validation rows|Eligible at minimum20perclass|
|---|---:|---:|---|
|Female|2147|10957|True|
|Male|1945|10337|True|

## race

|Arm|F_FPR fraction|FPR gap pp|FNR gap pp|Worst-group AUC %|
|---|---:|---:|---:|---:|
|A|0.087846|7.087615|2.325221|96.166697|
|B|0.079181|6.221136|2.452000|96.165312|
|C|0.077373|7.296748|3.400172|96.127587|
|D|0.072905|6.686992|2.491652|96.205654|
|E_support|0.098695|8.516260|2.370878|96.193952|
|F_support_ramp3|0.099822|9.878049|2.833290|96.169935|

|Group|Real validation rows|Fake validation rows|Eligible at minimum20perclass|
|---|---:|---:|---|
|Asian|627|2886|True|
|Black|120|510|True|
|Others|328|1980|True|
|White|3017|15918|True|

## gender_race

|Arm|F_FPR fraction|FPR gap pp|FNR gap pp|Worst-group AUC %|
|---|---:|---:|---:|---:|
|A|0.505265|33.839133|9.230769|93.959518|
|B|0.468638|30.416667|9.572650|93.967743|
|C|0.480534|30.984848|10.769231|93.620499|
|D|0.379345|23.839133|9.572650|93.851876|
|E_support|0.468281|27.628827|9.401709|93.844008|
|F_support_ramp3|0.495578|30.562845|10.170940|93.770339|

|Group|Real validation rows|Fake validation rows|Eligible at minimum20perclass|
|---|---:|---:|---|
|Female-Asian|275|1153|True|
|Female-Black|30|120|True|
|Female-Others|89|810|True|
|Female-White|1753|8874|True|
|Male-Asian|352|1733|True|
|Male-Black|90|390|True|
|Male-Others|239|1170|True|
|Male-White|1264|7044|True|

## age_bin

|Arm|F_FPR fraction|FPR gap pp|FNR gap pp|Worst-group AUC %|
|---|---:|---:|---:|---:|
|A|0.228592|22.135247|8.215874|97.603678|
|B|0.230098|22.330942|8.841396|97.566547|
|C|0.276733|24.176816|5.338129|97.563609|
|D|0.303616|29.028932|8.757645|96.871904|
|E_support|0.312441|29.254462|8.465735|97.555856|
|F_support_ramp3|0.296048|29.119144|7.214693|97.519133|

|Group|Real validation rows|Fake validation rows|Eligible at minimum20perclass|
|---|---:|---:|---|
|30-44|1301|6200|True|
|45-59|511|2398|True|
|60+|63|240|True|
|<30|2217|12456|True|

## skin_bin

|Arm|F_FPR fraction|FPR gap pp|FNR gap pp|Worst-group AUC %|
|---|---:|---:|---:|---:|
|A|0.028039|1.902805|3.019232|97.794184|
|B|0.026288|2.059358|3.102774|97.771914|
|C|0.073475|5.244368|1.224957|97.561534|
|D|0.051027|3.764232|1.601167|97.561349|
|E_support|0.051145|3.212065|1.940864|97.421064|
|F_support_ramp3|0.058251|4.232780|1.264690|97.491940|

|Group|Real validation rows|Fake validation rows|Eligible at minimum20perclass|
|---|---:|---:|---|
|0|1161|7027|True|
|1|1010|5105|True|
|2|1063|5069|True|
|3|798|3764|True|
|unknown|60|329|False|

Both candidates are worse than oldD on the primary selected-so-far disparity at this observation. A favorable other-axis change does not offset the preregistered primary/utility/age/gender guards. Final comparisons and paired intervals await full ten-epoch completion and replay. No TEST inference, new Ding-model computation or FairForge-2 experiment. Race/gender labels follow inherited benchmark metadata; age/skin quality is not newly human-verified. Known contributor overlap and sparse support remain disclosed.
