"""FF2-public provenance guard (Codex review F5, 25 Oct 2026; decisions/2026-10-25-cesel-plan.md Amendment 4).

Every FF2-public config carries
    guard: {public_only: true, role: phase1|phase2, public_train_csv: <frozen public TRAIN csv>, frozen_sha256: <sha>}
and scripts/train.py / scripts/train_ces.py call check() before building anything. A guarded run REFUSES to start
unless
  * data.train.csv is exactly the frozen public TRAIN csv and its sha256 equals frozen_sha256;
  * role phase1: no initial checkpoint at all (CLIP weights only); the finished run writes <out>/PUBLIC_P1
    (train csv sha + sha256 of last.pt / best.pt) after TRAINED;
  * role phase2 (CFES-D, CESel B0 and arms): every non-null model.init_ckpt / model.init_backbone_ckpt lives in a run
    directory holding PUBLIC_P1 (phase-1 trained on the same frozen public TRAIN; the checkpoint's sha256 must equal the
    one recorded) or PUBLIC_LINEAGE (a guarded phase-2 run that itself started from public lineage), and at least one
    such checkpoint is given. The run then writes <out>/PUBLIC_LINEAGE so that runs started from it (CESel arms from
    B0 best.pt) pass the same check.
Configs without guard.public_only are unaffected.
"""
import hashlib
import json
import os
import time


def _sha(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _refuse(msg):
    raise SystemExit(f"REFUSE (FF2-public provenance guard, cfes/public_guard.py): {msg}")


def check(cfg):
    """-> None (no guard) or the provenance record to write; raises SystemExit when the run must not start."""
    gd = cfg.get("guard") or {}
    if not gd.get("public_only"):
        return None
    pub = gd.get("public_train_csv")
    if not pub or not os.path.exists(pub):
        _refuse(f"guard.public_train_csv missing or not found: {pub}")
    tr = cfg["data"]["train"]["csv"]
    if isinstance(tr, list) or os.path.realpath(tr) != os.path.realpath(pub):
        _refuse(f"data.train.csv = {tr} is not the frozen public TRAIN {pub}")
    sha = _sha(pub)
    if sha != gd.get("frozen_sha256"):
        _refuse(f"public TRAIN sha256 {sha} != guard.frozen_sha256 {gd.get('frozen_sha256')} (split changed?)")
    role = gd.get("role", "phase2")
    m = cfg.get("model", {})
    inits = {k: m.get(k) for k in ("init_ckpt", "init_backbone_ckpt", "init_from") if m.get(k)}
    rec = {"role": role, "public_train_csv": os.path.realpath(pub), "train_csv_sha256": sha,
           "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "inits": {}}
    if role == "phase1":
        if inits:
            _refuse(f"phase-1 public run must start from the pretrained backbone only, got {inits}")
        return rec
    if role != "phase2":
        _refuse(f"unknown guard.role {role!r}")
    inits.pop("init_from", None)
    if not inits:
        _refuse("phase-2 public run needs model.init_ckpt or model.init_backbone_ckpt from a public phase-1 run "
                "(PUBLIC_P1) or a guarded public run (PUBLIC_LINEAGE); none given")
    for k, p in inits.items():
        if not os.path.exists(p):
            _refuse(f"model.{k} = {p} does not exist")
        d = os.path.dirname(os.path.abspath(p))
        p1, lin = os.path.join(d, "PUBLIC_P1"), os.path.join(d, "PUBLIC_LINEAGE")
        if os.path.exists(p1):
            mk = json.load(open(p1))
            want = (mk.get("checkpoints") or {}).get(os.path.basename(p))
            if want is None:
                _refuse(f"{p1} records no sha256 for {os.path.basename(p)}")
            if _sha(p) != want:
                _refuse(f"{p} sha256 differs from the one recorded in {p1} (checkpoint replaced?)")
            kind = "PUBLIC_P1"
        elif os.path.exists(lin):
            mk = json.load(open(lin))
            kind = "PUBLIC_LINEAGE"
        else:
            _refuse(f"model.{k} = {p}: its run directory has neither PUBLIC_P1 nor PUBLIC_LINEAGE (not trained on the "
                    f"frozen public TRAIN; e.g. the FF2-FULL phase-1 runs_ff2full/clip_s7_p1 saw RFW)")
        if mk.get("train_csv_sha256") != sha:
            _refuse(f"{kind} of {p} was written for TRAIN sha {mk.get('train_csv_sha256')}, not {sha}")
        rec["inits"][k] = {"path": os.path.abspath(p), "marker": kind}
    return rec


def write_lineage(out, rec):
    """phase-2 guarded run: mark <out> as public lineage (written at start, before any checkpoint exists)."""
    if rec and rec["role"] == "phase2":
        json.dump(rec, open(os.path.join(out, "PUBLIC_LINEAGE"), "w"), indent=1)


def write_p1(out, rec):
    """phase-1 guarded run, after TRAINED: PUBLIC_P1 with the sha256 of the final checkpoints."""
    if rec and rec["role"] == "phase1":
        rec = dict(rec, finished=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   checkpoints={c: _sha(os.path.join(out, c)) for c in ("last.pt", "best.pt")
                                if os.path.exists(os.path.join(out, c))})
        json.dump(rec, open(os.path.join(out, "PUBLIC_P1"), "w"), indent=1)
