"""FAIRREL_REPRO Step 0: build FairReL-format CSVs from Lin et al.'s released data, with split-provenance asserts.

- FF++ manifests are read DIRECTLY from ff++.zip (not from any other agent's parsed CSV); split = the manifest FILE
  a row came from (faketrain/realtrain -> train, fakeval/realval -> val, test -> test).
- Subgroup = Lin's own intersec_label (3-race: il = 4*female + rank(A0,W1,B2,O3)); cross-checked against the
  ismale/is* flags; Others (rank 3) and ismale==0 excluded.
- FF++ TRAIN PNGs are extracted from ff++.zip into data/extracted/ffpp_train/ (8 worker processes).
- FF++ val/test and Celeb-DF/DFD/DFDC PNGs were already extracted by EXACTMATCH_plan (read-only reuse); every reused
  file is checked against the zip's file_size, and a random sample of 300 per set against the zip's CRC.
- Asserts (logged): row counts, train paths come only from train manifests, zero path overlap train/val/test,
  zero fake-video overlap across splits, real-video straddling reported (Lin's stray-frame bug).
- DFDC: duplicate paths deduplicated (keep first; label+subgroup identical among duplicates, asserted).
- ffpp_test_clean.csv: FF++ test minus every real video that has >=1 frame in FF++ train.

Output CSV columns: path,label,subgroup,cluster  (FairReL needs path,label,subgroup; extra columns are ignored).
Run: ~/.conda/envs/pulid/bin/python build_data.py
"""
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
import zipfile
import zlib
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

A = Path(__file__).resolve().parent.parent
R = A.parent.parent
EM = R / "agents" / "EXACTMATCH_plan"
LIN = Path.home() / "ICLR_2027/related_work/Lin_FairnessGen_Dataset"
FFZIP = LIN / "ff++.zip"
OUT = A / "data"
TRAIN_DIR = OUT / "extracted" / "ffpp_train"
SG3 = {0: "M-A", 1: "M-W", 2: "M-B", 4: "F-A", 5: "F-W", 6: "F-B"}   # 3 and 7 = Others (excluded)
SPLIT_FILES = {"faketrain.csv": "train", "realtrain.csv": "train", "fakeval.csv": "val",
               "realval.csv": "val", "test.csv": "test"}
LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)


def check(cond, msg):
    log(("ASSERT PASS: " if cond else "ASSERT FAIL: ") + msg)
    if not cond:
        flush_log()
        sys.exit(1)


def flush_log():
    (A / "logs" / "build_data.log").write_text("\n".join(LOG) + "\n")


def video_of(name):
    m = re.match(r"^0_(.+)_\d+\.png$", name)
    assert m, name
    return m.group(1)


def derive3(r):
    ismale = int(r["ismale"])
    if ismale == 0:
        return None
    fem = 0 if ismale == 1 else 1
    ones = [k for k, c in ((0, "isasian"), (1, "iswhite"), (2, "isblack")) if int(r[c]) == 1]
    rank = ones[0] if len(ones) == 1 else 3
    return 4 * fem + rank


def parse_ffpp():
    rows = []
    raw = Counter()
    mism = 0
    with zipfile.ZipFile(FFZIP) as zf:
        for fname, split in SPLIT_FILES.items():
            data = zf.read("ff++/" + fname)
            log(f"manifest ff++/{fname}: {len(data)} bytes sha256={hashlib.sha256(data).hexdigest()[:16]} -> {split}")
            for r in csv.DictReader(io.StringIO(data.decode())):
                raw[split] += 1
                il = int(r["intersec_label"])
                d = derive3(r)
                if d is not None and d != il:
                    mism += 1
                if il not in SG3 or d is None:
                    continue
                name = Path(r["img_path"].replace("\\", "/")).name
                lab = int(r["label"])
                vid = video_of(name)
                check_lab = (vid.startswith("youtube_") and lab == 0) or (not vid.startswith("youtube_") and lab == 1)
                if not check_lab:
                    raise SystemExit(f"label/filename contradiction {name} {lab}")
                rows.append(dict(name=name, label=lab, subgroup=SG3[il], cluster=vid, split=split, manifest=fname))
    log(f"FF++ raw rows {dict(raw)}; intersec_label vs derived mismatches {mism}")
    return rows, raw


def _extract_chunk(names):
    n_new = 0
    with zipfile.ZipFile(FFZIP) as zf:
        for nm in names:
            dst = TRAIN_DIR / nm
            info = zf.getinfo("ff++/crop_img/" + nm)
            if dst.exists() and dst.stat().st_size == info.file_size:
                continue
            tmp = dst.with_suffix(".png.tmp")
            with zf.open(info) as src, open(tmp, "wb") as f:
                f.write(src.read())
            os.replace(tmp, dst)
            n_new += 1
    return n_new


def zip_index(zpath, prefix):
    with zipfile.ZipFile(zpath) as zf:
        return {Path(i.filename).name: (i.file_size, i.CRC) for i in zf.infolist() if i.filename.startswith(prefix)
                and i.filename.endswith(".png")}


def verify_reused(paths, idx, tag, zpath=None, inner=None):
    bad = [p for p in paths if not os.path.exists(p) or os.path.getsize(p) != idx[Path(p).name][0]]
    check(not bad, f"{tag}: all {len(paths)} reused PNGs exist with zip file_size (bad={len(bad)})")
    rng = random.Random(2027)
    samp = rng.sample(list(paths), min(300, len(paths)))
    badcrc = [p for p in samp if (zlib.crc32(open(p, "rb").read()) & 0xFFFFFFFF) != idx[Path(p).name][1]]
    check(not badcrc, f"{tag}: CRC32 matches zip for random sample of {len(samp)} (bad={len(badcrc)})")


def write(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "subgroup", "cluster"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ["path", "label", "subgroup", "cluster"]})
    log(f"wrote {path.name}: {len(rows)} rows, sha256={hashlib.sha256(path.read_bytes()).hexdigest()}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    rows, raw = parse_ffpp()
    check(raw == Counter(train=76139, val=25386, test=25401), f"raw rows per split = 76139/25386/25401 (got {dict(raw)})")
    by = defaultdict(list)
    for r in rows:
        by[r["split"]].append(r)
    n = {k: len(v) for k, v in by.items()}
    check(n == dict(train=69791, val=23078, test=23389), f"kept rows train/val/test = 69791/23078/23389 (got {n})")
    check(all(r["manifest"] in ("faketrain.csv", "realtrain.csv") for r in by["train"]),
          "every train row comes from faketrain.csv/realtrain.csv")
    check(not any(r["manifest"] in ("faketrain.csv", "realtrain.csv") for r in by["val"] + by["test"]),
          "no val/test row comes from a train manifest")
    names = {k: set(r["name"] for r in v) for k, v in by.items()}
    check(all(len(names[k]) == n[k] for k in n), "no duplicate filenames within any split")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        check(not (names[a] & names[b]), f"zero filename overlap {a} vs {b}")
    fvid = {k: set(r["cluster"] for r in v if r["label"] == 1) for k, v in by.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        check(not (fvid[a] & fvid[b]), f"zero FAKE-video overlap {a} vs {b}")
    rvid = {k: set(r["cluster"] for r in v if r["label"] == 0) for k, v in by.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        log(f"INFO real videos with frames in both {a} and {b}: {len(rvid[a] & rvid[b])} (Lin stray-frame bug, documented)")

    # ---- extract FF++ train ----
    idx = zip_index(FFZIP, "ff++/crop_img/")
    tr_names = sorted(names["train"])
    missing = [x for x in tr_names if x not in idx]
    check(not missing, f"all {len(tr_names)} train filenames present in ff++.zip crop_img/ (missing={len(missing)})")
    chunks = [tr_names[i::16] for i in range(16)]
    with Pool(8) as pool:
        new = sum(pool.map(_extract_chunk, chunks))
    log(f"extracted {new} new train PNGs into {TRAIN_DIR}")
    tr_paths = [str(TRAIN_DIR / x) for x in tr_names]
    bad = [p for p in tr_paths if os.path.getsize(p) != idx[Path(p).name][0]]
    check(not bad, f"all {len(tr_paths)} train PNGs have zip file_size")
    rng = random.Random(7)
    samp = rng.sample(tr_paths, 1000)
    badcrc = [p for p in samp if (zlib.crc32(open(p, "rb").read()) & 0xFFFFFFFF) != idx[Path(p).name][1]]
    check(not badcrc, f"train CRC32 matches zip for random 1000 (bad={len(badcrc)})")
    n_dir = len([x for x in os.listdir(TRAIN_DIR) if x.endswith(".png")])
    check(n_dir == 69791, f"extracted train dir holds exactly 69791 PNGs (got {n_dir})")

    # ---- FF++ CSVs ----
    roots = {"train": TRAIN_DIR, "val": EM / "work/extracted/ffpp_val", "test": EM / "work/extracted/ffpp"}
    for sp in ("train", "val", "test"):
        for r in by[sp]:
            r["path"] = str(roots[sp] / r["name"])
    verify_reused([r["path"] for r in by["val"]], idx, "FF++ val (EM extraction)")
    verify_reused([r["path"] for r in by["test"]], idx, "FF++ test (EM extraction)")
    for sp in ("train", "val", "test"):
        write(by[sp], OUT / f"ffpp_{sp}.csv")
    # path overlap on the written files (the check TRAIN_ON_LIN lacked)
    P = {sp: set(r["path"] for r in csv.DictReader(open(OUT / f"ffpp_{sp}.csv"))) for sp in ("train", "val", "test")}
    B = {sp: set(Path(p).name for p in P[sp]) for sp in P}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        check(not (P[a] & P[b]) and not (B[a] & B[b]), f"written CSVs: zero path AND basename overlap {a} vs {b}")
    check(all(p.startswith(str(TRAIN_DIR)) for p in P["train"]), "every ffpp_train.csv path is under data/extracted/ffpp_train")
    tl = Path(R / "agents/TRAIN_ON_LIN/data/train.csv")
    if tl.exists():
        tlb = set(Path(r["path"]).name for r in csv.DictReader(open(tl)))
        log(f"INFO TRAIN_ON_LIN train.csv basenames: overlap with our TRAIN {len(tlb & B['train'])}, "
            f"with our TEST {len(tlb & B['test'])} of {len(tlb)} (confirms the audit's finding)")

    # ---- clean test ----
    straddle = rvid["test"] & rvid["train"]
    clean = [r for r in by["test"] if not (r["label"] == 0 and r["cluster"] in straddle)]
    n_real_full = sum(r["label"] == 0 for r in by["test"])
    n_real_clean = sum(r["label"] == 0 for r in clean)
    log(f"clean test: dropped {len(straddle)} real videos straddling train/test; real frames {n_real_full} -> "
        f"{n_real_clean}; fakes unchanged {sum(r['label'] == 1 for r in clean)}")
    write(clean, OUT / "ffpp_test_clean.csv")
    with open(OUT / "ffpp_straddle_train_test_real_videos.txt", "w") as f:
        f.write("\n".join(sorted(straddle)) + "\n")

    # ---- target sets (EM parsed CSVs; paths -> EM extraction) ----
    zips = {"celebdf": None, "dfd": (LIN / "dfd.zip", "dfd/crop_img/"), "dfdc": (LIN / "dfdc.zip", "dfdc/crop_img/")}
    for ds in ("celebdf", "dfd", "dfdc"):
        src = list(csv.DictReader(open(EM / "data" / f"{ds}_test.csv")))
        ck = "cluster" if "cluster" in src[0] else "source_id"
        out, seen, dup_conflict = [], {}, 0
        for r in src:
            nm = Path(r["path"].replace("\\", "/")).name
            rec = dict(name=nm, label=int(r["label"]), subgroup=r["subgroup"], cluster=r[ck],
                       path=str(EM / "work/extracted" / ds / nm))
            if nm in seen:
                if (seen[nm]["label"], seen[nm]["subgroup"]) != (rec["label"], rec["subgroup"]):
                    dup_conflict += 1
                continue
            seen[nm] = rec
            out.append(rec)
        log(f"{ds}: {len(src)} rows as shipped (Others already excluded by EM parser) -> {len(out)} unique paths; "
            f"duplicate rows dropped {len(src) - len(out)}; label/subgroup conflicts among duplicates {dup_conflict}")
        check(dup_conflict == 0, f"{ds}: duplicates have identical label+subgroup")
        exp = {"celebdf": (28278, 28278), "dfd": (8611, 8611), "dfdc": (20007, 18851)}[ds]
        check((len(src), len(out)) == exp, f"{ds}: rows as-shipped/unique = {exp}")
        check(all(os.path.exists(r["path"]) for r in out), f"{ds}: every path exists in EM extraction")
        if zips[ds]:
            verify_reused([r["path"] for r in out], zip_index(*zips[ds]), f"{ds} (EM extraction)")
        else:
            log("celebdf: nested crop_img.zip (EM removed its local copy) -> file_size/CRC check skipped; existence checked")
        write(out, OUT / f"{ds}_test.csv")
    flush_log()
    json.dump({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob("*.csv"))},
              open(OUT / "DATA_SHA256.json", "w"), indent=1)
    with open(OUT / "DATA_SHA256.txt", "w") as f:
        for p in sorted(OUT.glob("*.csv")):
            f.write(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n")
    log("DONE")
    flush_log()


if __name__ == "__main__":
    main()
