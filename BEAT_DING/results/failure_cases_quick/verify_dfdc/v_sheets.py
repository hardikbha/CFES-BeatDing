import pandas as pd, json
from PIL import Image
V="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/results/failure_cases_quick/verify_dfdc"
D="/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/results/failure_cases/dfdc"
o=json.load(open(f"{V}/v_main.json"))["sheets"]; df=pd.read_pickle(f"{V}/df.pkl").drop_duplicates("path").set_index("path")
for k in ["fp","fn"]:
    it=pd.read_csv(f"{D}/{k}_sheet_items.csv")
    print(k,"n items",len(it),"same paths same order:",it.img_path.tolist()==o[f"{k}_paths"])
    print(" logit max absdiff",(it.logit.values-df.loc[it.img_path,"lg"].values).__abs__().max(),"ding absdiff",(it.ding_logit.values-df.loc[it.img_path,"dl"].values).__abs__().max())
    print(" ding_pred col sum",int(it.ding_pred.sum()))
    print(" groups:",it.grp.value_counts().to_dict(), "cl:", (it.cl_gender_vid+"-"+it.cl_race_vid).value_counts().to_dict())
    print(" sheet",Image.open(f"{D}/{k}_sheet.png").size)
it=pd.read_csv(f"{D}/fp_sheet_items.csv")
for v in ["0_7406","0_4516","0_4543","0_7434"]:
    r=it[it.video==v]; print(r.to_string(index=False)) 
