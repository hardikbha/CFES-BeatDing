#!/usr/bin/env python3
"""Create a deterministic TRAIN-only leave-one-family-out source fold.

This utility reads only the pinned FF++ TRAIN manifest. It writes the fit and
development manifests under a private output directory; never publish those
row-level files. Recorded contributor IDs are provenance proxies, not proof of
true-person independence.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_TRAIN_SHA256 = "6ee9326fe8361dc3aee0f803c49b5eada169be5cf3bc3cb44ad3d934ba5ca657"
P50_SOURCE_SHA256 = "2d66966b1733feec222fe3a80ff9a6cd7181a6ba28cb9f767ad980968d337098"
SEED = 17
DEV_COMPONENT_FRACTION = 0.20
KNOWN_FAKE_FAMILIES = {"Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"}
AXES = ("gender", "race", "age_bin", "skin_bin", "gender_race", "ids8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_ids(value):
    return [v.strip() for v in str(value or "").split(";") if v.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    source = a.train.resolve()
    if sha(source) != EXPECTED_TRAIN_SHA256:
        raise SystemExit("Pinned TRAIN manifest hash mismatch")
    if a.out.exists():
        raise SystemExit("Output exists; preserve prior fold artifacts")

    with source.open(newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        required = {"image", "label", "identity_ids", "source_ids", "family", *AXES}
        if not required.issubset(fields):
            raise SystemExit(f"Missing required TRAIN metadata columns: {sorted(required-set(fields))}")
        rows = list(reader)
    if not rows:
        raise SystemExit("Empty TRAIN manifest")
    for row in rows:
        if row["label"] not in {"0", "1"}:
            raise SystemExit("Invalid TRAIN label")
        image = Path(row["image"])
        if not image.is_absolute():
            image = (source.parent / image).resolve()
        row["image"] = str(image)
        if not image.is_file() or not split_ids(row["identity_ids"]) or not split_ids(row["source_ids"]):
            raise SystemExit("Missing image or recorded contributor provenance")
    if len({r["image"] for r in rows}) != len(rows):
        raise SystemExit("Duplicate image rows in TRAIN")

    # Join original rows by either recorded identity or source IDs. Namespace
    # the two ID types so coincident strings do not create false links.
    parent = list(range(len(rows)))
    seen = {}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, row in enumerate(rows):
        for field in ("identity_ids", "source_ids"):
            for value in split_ids(row[field]):
                token = (field, value)
                j = seen.setdefault(token, i)
                ri, rj = root(i), root(j)
                if ri != rj:
                    parent[ri] = rj
    components = defaultdict(list)
    for i in range(len(rows)):
        components[root(i)].append(i)

    signatures = {}
    for cid, ix in components.items():
        tokens = sorted({(field, token) for i in ix for field in ("identity_ids", "source_ids")
                         for token in split_ids(rows[i][field])})
        signatures[cid] = hashlib.sha256(json.dumps(tokens, ensure_ascii=True,
                                                      separators=(",", ":")).encode()).hexdigest()

    fake_counts = Counter(r["family"] for r in rows if r["label"] == "1")
    family_components = {
        family: {root(i) for i, row in enumerate(rows)
                 if row["label"] == "1" and row["family"] == family}
        for family in fake_counts
    }
    supported = [f for f in sorted(KNOWN_FAKE_FAMILIES)
                 if fake_counts[f] >= 20 and len(family_components.get(f, ())) >= 2]
    if not supported:
        raise SystemExit("No supported known fake family under the frozen support rule")
    held_family = supported[0]
    held_components = family_components[held_family]
    eligible_components = [c for c in components if c not in held_components]
    if len(eligible_components) < 10:
        raise SystemExit("Insufficient non-held-out recorded-contributor components")

    def dev_key(cid):
        payload = f"CESFL-SG-20261105-seed{SEED}:{signatures[cid]}"
        return hashlib.sha256(payload.encode()).hexdigest()

    ranked = sorted(eligible_components, key=dev_key)
    dev_count = math.ceil(DEV_COMPONENT_FRACTION * len(ranked))
    dev_components = set(ranked[:dev_count])
    fit_components = set(eligible_components) - dev_components
    fit_ix = [i for i, row in enumerate(rows) if root(i) in fit_components]
    # The held-out family is evaluated only on its fakes. Independently selected
    # development contributors contribute real examples. Other rows are unused.
    dev_ix = [i for i, row in enumerate(rows)
              if (row["label"] == "1" and row["family"] == held_family)
              or (row["label"] == "0" and root(i) in dev_components)]
    used = set(fit_ix) | set(dev_ix)
    unused_ix = sorted(set(range(len(rows))) - used)

    fit_group_counts = Counter((rows[i]["label"], rows[i]["ids8"]) for i in fit_ix)
    ids8 = sorted({rows[i]["ids8"] for i in fit_ix})
    missing_cells = [(g, label) for g in ids8 for label in ("0", "1")
                     if fit_group_counts[(label, g)] == 0]
    if missing_cells:
        raise SystemExit(f"Pinned P50 ids8 balancing loses class cells: {missing_cells}")
    fit_comps_used = {root(i) for i in fit_ix}
    dev_real_comps = {root(i) for i in dev_ix if rows[i]["label"] == "0"}
    if fit_comps_used & dev_real_comps or fit_comps_used & held_components:
        raise SystemExit("Recorded contributor component leakage")
    if any(rows[i]["family"] == held_family for i in fit_ix):
        raise SystemExit("Held-out family entered detector fitting")
    if not any(rows[i]["label"] == "0" for i in dev_ix) or not any(rows[i]["label"] == "1" for i in dev_ix):
        raise SystemExit("Development set lacks a class")

    a.out.mkdir(parents=True)
    for name, indices in (("fit_train.csv", fit_ix), ("development.csv", dev_ix)):
        target = a.out / name
        with target.open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[i] for i in indices)
    dev_counts = {axis: {group: {"real": 0, "fake": 0} for group in sorted({rows[i][axis] for i in dev_ix})}
                  for axis in AXES}
    for i in dev_ix:
        row = rows[i]
        for axis in AXES:
            dev_counts[axis][row[axis]]["fake" if row["label"] == "1" else "real"] += 1
    audit = {
        "status": "PASS_WITH_SUPPORT_LIMITATIONS",
        "train_manifest_sha256": sha(source),
        "p50_native_source_sha256": P50_SOURCE_SHA256,
        "seed": SEED,
        "selection_rule": "lexicographically first known fake family with >=20 fake rows in >=2 recorded-contributor components",
        "held_out_family": held_family,
        "dev_component_fraction": DEV_COMPONENT_FRACTION,
        "dev_component_rule": "lowest SHA256(seed-tag || canonical recorded contributor component digest)",
        "counts": {
            "train_rows": len(rows), "recorded_components_total": len(components),
            "heldout_family_fake_rows": fake_counts[held_family],
            "heldout_family_components": len(held_components),
            "eligible_components": len(eligible_components),
            "fit_components": len(fit_components), "dev_real_components": len(dev_components),
            "fit_rows": len(fit_ix), "fit_real": sum(rows[i]["label"] == "0" for i in fit_ix),
            "fit_fake": sum(rows[i]["label"] == "1" for i in fit_ix),
            "development_rows": len(dev_ix), "development_real": sum(rows[i]["label"] == "0" for i in dev_ix),
            "development_fake": sum(rows[i]["label"] == "1" for i in dev_ix), "unused_rows": len(unused_ix),
            "heldout_family_support": {f: {"fake_rows": fake_counts[f],
                "recorded_components": len(family_components[f])} for f in sorted(fake_counts)},
            "fit_ids8_cell_counts": {g: {"real": fit_group_counts[("0", g)],
                                                "fake": fit_group_counts[("1", g)]} for g in ids8},
            "development_group_counts": dev_counts,
        },
        "hashes": {"fit_train.csv": sha(a.out / "fit_train.csv"),
                   "development.csv": sha(a.out / "development.csv")},
        "leakage_checks": {"fit_vs_dev_real_recorded_component_overlap": 0,
                           "heldout_family_fit_rows": 0, "heldout_family_fit_components": 0,
                           "test_manifest_read": False, "validation_manifest_read": False,
                           "true_person_independence_verified": False},
        "interpretation_limit": "The component graph is based on recorded IDs. Demographic group support in development is descriptive and may be too sparse for fairness claims.",
    }
    (a.out / "FOLD_AUDIT_PRIVATE.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status": audit["status"], "held_out_family": held_family,
                      "counts": audit["counts"], "hashes": audit["hashes"],
                      "test_manifest_read": False}, indent=2))


if __name__ == "__main__":
    main()
