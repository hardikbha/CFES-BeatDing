"""Automatic checks for an MFER matched-set file, and the validation sidecar that training requires.

A matched-set CSV (set_id, group, fake_path, real_path) is only used for training when a sidecar
<csv>.validation.json exists, its sha256 matches the CSV, and its status allows the use:

  status "smoke_only" : code-path fixtures. Training accepts it only with train.py --smoke (output dir smoke_*).
  status "validated"  : the automatic checks below passed AND a person signed off a matching-quality audit
                        (scripts/validate_matched_sets.py --approve ... --audit_doc ...).
The automatic checks cannot establish that a set is a fair comparison across groups (comparable manipulation
strength and acquisition conditions); that is what the audit is for.
"""
import hashlib
import json
import os
import re
import time

import pandas as pd
from PIL import Image

from .data import cluster_key, ffpp_identities

REQUIRED = ("set_id", "group", "fake_path", "real_path")
_FAKE = re.compile(r"^\d+_([^_]+)_(\d+)_(\d+)_(\d+)\.png$")      # label_Method_target_source_frame.png
_REAL = re.compile(r"^\d+_youtube_(\d+)_(\d+)\.png$")            # label_youtube_video_frame.png


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_path(csv):
    return csv + ".validation.json"


def ffpp_correspondence(fake_path, real_path):
    """FF++ (Lin et al. frames): the authentic control of a fake is the TARGET video's original frame with the
    same frame index. Returns None if consistent, else the reason."""
    f, r = _FAKE.match(os.path.basename(fake_path)), _REAL.match(os.path.basename(real_path))
    if not f:
        return "fake name not FF++ <label>_<Method>_<target>_<source>_<frame>"
    if not r:
        return "real name not FF++ <label>_youtube_<video>_<frame>"
    if f.group(2) != r.group(1):
        return f"real video {r.group(1)} is not the fake's target {f.group(2)}"
    if f.group(4) != r.group(2):
        return f"frame index differs (fake {f.group(4)}, real {r.group(2)})"
    return None


def check_matched_sets(df, train_df, holdout, n_groups, min_groups=2, correspondence="ffpp", check_images=True):
    """df: raw matched-set table. train_df / holdout[name]: read_split() frames (img_path, label, group, cluster).
    Returns (report, errors). errors is empty only if every hard check passed."""
    errors, rep = [], {"rows": int(len(df)), "sets": int(df["set_id"].nunique()) if "set_id" in df else 0}
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return rep, [f"missing columns {missing}"]
    na = df[list(REQUIRED)].isna().any(axis=1)
    rep["rows_with_missing_values"] = int(na.sum())
    if na.any():
        errors.append(f"{int(na.sum())} rows with a missing set_id/group/fake_path/real_path")
    d = df[~na].copy()
    d["group"] = pd.to_numeric(d["group"], errors="coerce")
    badg = d["group"].isna() | ~d["group"].between(0, n_groups - 1) | (d["group"] % 1 != 0)
    rep["rows_with_invalid_group"] = int(badg.sum())
    if badg.any():
        errors.append(f"{int(badg.sum())} rows with a group label outside 0..{n_groups - 1}")
    d = d[~badg]
    d["group"] = d["group"].astype(int)
    dup_rows = d.duplicated(["fake_path", "real_path"])
    dup_group = d.duplicated(["set_id", "group"])
    rep["duplicate_pairs"], rep["duplicate_group_in_set"] = int(dup_rows.sum()), int(dup_group.sum())
    if dup_rows.any():
        errors.append(f"{int(dup_rows.sum())} duplicated (fake, real) records")
    if dup_group.any():
        errors.append(f"{int(dup_group.sum())} sets list the same group twice")
    ng = d.groupby("set_id")["group"].nunique()
    rep["sets_below_min_groups"] = int((ng < min_groups).sum())
    rep["groups_per_set"] = {int(k): int(v) for k, v in ng.value_counts().sort_index().items()}

    # images readable
    paths = pd.unique(pd.concat([d["fake_path"], d["real_path"]]))
    unreadable = []
    for p in paths:
        try:
            if check_images:
                with Image.open(p) as im:
                    im.verify()
            elif not os.path.exists(p):
                raise FileNotFoundError(p)
        except Exception as e:      # noqa: BLE001 - any failure to read is reported
            unreadable.append(f"{p}: {type(e).__name__}")
    rep["unreadable_images"] = len(unreadable)
    if unreadable:
        errors.append(f"{len(unreadable)} unreadable images, e.g. {unreadable[:3]}")

    # labels and groups against the training manifest
    tr = train_df.set_index("img_path")
    in_train_f, in_train_r = d["fake_path"].isin(tr.index), d["real_path"].isin(tr.index)
    rep["fakes_not_in_train_manifest"] = int((~in_train_f).sum())
    rep["reals_not_in_train_manifest"] = int((~in_train_r).sum())
    if (~in_train_f).any() or (~in_train_r).any():
        errors.append("matched images must all come from the training manifest "
                      f"({int((~in_train_f).sum())} fakes, {int((~in_train_r).sum())} reals are not in it)")
    k = d[in_train_f & in_train_r]
    lab_f, lab_r = tr.loc[k["fake_path"], "label"].values, tr.loc[k["real_path"], "label"].values
    rep["fake_not_labelled_fake"], rep["real_not_labelled_real"] = int((lab_f != 1).sum()), int((lab_r != 0).sum())
    if (lab_f != 1).any() or (lab_r != 0).any():
        errors.append("a fake_path is not labelled fake or a real_path is not labelled real in the manifest")
    grp_f, grp_r = tr.loc[k["fake_path"], "group"].values, tr.loc[k["real_path"], "group"].values
    rep["group_differs_from_fake_manifest_label"] = int((grp_f != k["group"].values).sum())
    rep["real_control_group_differs"] = int((grp_r != k["group"].values).sum())
    if rep["group_differs_from_fake_manifest_label"]:
        errors.append(f"{rep['group_differs_from_fake_manifest_label']} rows whose group is not the fake's manifest group")

    # real-fake correspondence
    if correspondence == "ffpp":
        bad = [(f, r, why) for f, r in zip(d["fake_path"], d["real_path"]) if (why := ffpp_correspondence(f, r))]
    elif correspondence == "column":
        if not {"fake_source", "real_source"} <= set(d.columns):
            bad = [("", "", "correspondence=column needs fake_source and real_source columns")]
        else:
            bad = [(f, r, "source ids differ") for f, r, a, b in
                   zip(d["fake_path"], d["real_path"], d["fake_source"], d["real_source"]) if a != b]
    else:
        bad = [("", "", f"unknown correspondence rule {correspondence}")]
    rep["correspondence_rule"], rep["correspondence_failures"] = correspondence, len(bad)
    if bad:
        errors.append(f"{len(bad)} fake/real pairs fail the {correspondence} correspondence rule, e.g. {bad[:2]}")

    # split boundaries
    clusters = set(d["fake_path"].map(cluster_key)) | set(d["real_path"].map(cluster_key))
    idents = set().union(*[ffpp_identities(p) for p in paths]) if len(paths) else set()
    for name, h in holdout.items():
        hp = set(h["img_path"])
        n_path = int(d["fake_path"].isin(hp).sum() + d["real_path"].isin(hp).sum())
        n_clu = len(clusters & set(h["cluster"]))
        h_ids = set().union(*[ffpp_identities(p) for p in h["img_path"]]) if len(h) else set()
        rep[f"overlap_{name}"] = {"images": n_path, "videos": n_clu, "identities": len(idents & h_ids)}
        if n_path:
            errors.append(f"{n_path} matched images also appear in {name}")
        if n_clu:
            errors.append(f"{n_clu} matched source videos also appear in {name}")
    rep["identities_in_sets"] = len(idents)
    return rep, errors


def write_sidecar(csv, status, report, errors, approve=None, audit_doc=None, code_hash=None):
    if status not in ("smoke_only", "validated", "failed"):
        raise ValueError(status)
    if status == "validated" and (errors or not approve or not audit_doc):
        raise ValueError("status validated needs zero automatic errors, --approve and --audit_doc")
    side = {"csv": os.path.abspath(csv), "csv_sha256": sha256_file(csv), "status": status, "errors": errors,
            "report": report, "approved_by": approve, "audit_doc": audit_doc, "code_hash": code_hash,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with open(sidecar_path(csv), "w") as f:
        json.dump(side, f, indent=1, default=str)
    return side


def require_usable(csv, smoke):
    """Raises unless the sidecar allows this use. Returns the sidecar."""
    sp = sidecar_path(csv)
    if not os.path.exists(sp):
        raise RuntimeError(f"MFER training blocked: {csv} has no validation sidecar {sp}. "
                           "Run scripts/validate_matched_sets.py first.")
    side = json.load(open(sp))
    if side["csv_sha256"] != sha256_file(csv):
        raise RuntimeError(f"MFER training blocked: {csv} changed after validation (sha256 mismatch); re-validate.")
    if side["status"] == "validated":
        return side
    if side["status"] == "smoke_only" and smoke:
        return side
    raise RuntimeError(f"MFER training blocked: {csv} has status '{side['status']}'"
                       + (" (allowed only with train.py --smoke)" if side["status"] == "smoke_only" else ""))
