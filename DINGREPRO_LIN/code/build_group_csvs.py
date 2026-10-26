# Pre-split per-group real/fake train CSVs for GDA.py's create_group_dataloaders, which expects
# "{group}_realtrain.csv" / "{group}_faketrain.csv" files, group in range(NUM_GROUPS).
# DINGREPRO_LIN DISCLOSED DEVIATION: NUM_GROUPS=6 (F-A,F-B,F-W,M-A,M-B,M-W), not Ding's original 8
# (Lin et al.'s annotation has 3 races, not 4 -- see PREREG.txt section 2).
import pandas as pd, os
A = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/data/ff++_ding"
for split in ["train"]:
    for kind in ["real", "fake"]:
        df = pd.read_csv(os.path.join(A, f"{kind}{split}.csv"))
        for g in sorted(df['intersec_label'].unique()):
            sub = df[df['intersec_label'] == g]
            sub.to_csv(os.path.join(A, f"{g}_{kind}{split}.csv"), index=False)
            print(g, kind, len(sub))
