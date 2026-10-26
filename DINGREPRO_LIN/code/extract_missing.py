import zipfile, os, sys
from concurrent.futures import ThreadPoolExecutor

LIN = os.path.expanduser("~/ICLR_2027/related_work/Lin_FairnessGen_Dataset")
EM = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/EXACTMATCH_plan"
FR = "/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/FAIRREL_REPRO"

jobs = [
    ("/tmp/missing_ffpp_train.txt", f"{LIN}/ff++.zip", "ff++/crop_img/", f"{FR}/data/extracted/ffpp_train"),
    ("/tmp/missing_ffpp_val.txt",   f"{LIN}/ff++.zip", "ff++/crop_img/", f"{EM}/work/extracted/ffpp_val"),
    ("/tmp/missing_ffpp_test.txt",  f"{LIN}/ff++.zip", "ff++/crop_img/", f"{EM}/work/extracted/ffpp"),
    ("/tmp/missing_dfd.txt",        f"{LIN}/dfd.zip",  "dfd/crop_img/",  f"{EM}/work/extracted/dfd"),
    ("/tmp/missing_dfdc.txt",       f"{LIN}/dfdc.zip", "dfdc/crop_img/", f"{EM}/work/extracted/dfdc"),
    ("/tmp/missing_celebdf.txt",    f"{LIN}/celebdf.zip", None,          f"{EM}/work/extracted/celebdf"),
]

for missf, zpath, prefix, outdir in jobs:
    with open(missf) as f:
        names = [l.strip() for l in f if l.strip()]
    if not names:
        print(missf, "nothing to extract"); continue
    with zipfile.ZipFile(zpath) as zf:
        zipnames = set(zf.namelist())
        if prefix is None:
            # find prefix for celebdf dynamically
            cands = [n for n in zipnames if n.endswith(names[0])]
            if not cands:
                print("CANNOT FIND PREFIX for", missf, "sample", names[0])
                continue
            prefix = cands[0][:-len(names[0])]
            print("celebdf detected prefix:", prefix)
        found, notfound = 0, 0
        os.makedirs(outdir, exist_ok=True)
        for nm in names:
            zn = prefix + nm
            if zn not in zipnames:
                notfound += 1
                continue
            with zf.open(zn) as src, open(os.path.join(outdir, nm), "wb") as dst:
                dst.write(src.read())
            found += 1
        print(missf, "-> extracted", found, "not_found_in_zip", notfound, "of", len(names))
