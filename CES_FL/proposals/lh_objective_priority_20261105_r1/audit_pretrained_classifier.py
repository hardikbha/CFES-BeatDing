#!/usr/bin/env python3
"""Zero-update classifier attribution audit on FF++ TRAIN and clean VAL only.

The audit loads the complete native P50 classifier and the verified A0/B0/C0/H0
checkpoints. It does not optimize, select, calibrate on TEST, or save row scores.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

P50_SHA = "2d66966b1733feec222fe3a80ff9a6cd7181a6ba28cb9f767ad980968d337098"
CES_SHA = "1997eebf093aa3e37de1cbbe58302a1539ac2e517f966a4607aae89dc695a8e0"
P50_CKPT_SHA = "c28bbb9cfcd06a11894027dc9369c5fd674f640de6c182c28b6d6f0aa8d77f55"
P50_TRAIN_SHA = "6ee9326fe8361dc3aee0f803c49b5eada169be5cf3bc3cb44ad3d934ba5ca657"
P50_VAL_SHA = "52f4844cffe5a03adb08cecde8829244bb50e05eeead36f6d659599f1c24f5b1"
EXPECTED_CHECKPOINTS = {
    "A0": "59ea7bbaf7c47c41da7429091ab3d22a40855e03dae8bd3b90fd23d6efcef56a",
    "B0": "d5206ed8fe6a5a0f058aaa7a20bb0b37965959d6567c93406f0fe019ed21339a",
    "C0": "3e06092d4983d94d071d2cbc01aca2e583416a01382551837e9519f1c284afef",
    "H0": "40802bb6bc8f7a9ec315a697d74d2be8b510fedcd42966abd2b64b1be0260e42",
}
AXES = ("gender", "race", "age_bin", "skin_bin", "gender_race", "ids8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def state_sha(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_module(name, path, expected):
    if sha(path) != expected:
        raise RuntimeError(f"{name} source hash mismatch")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_cache(cache_path, rows, expected_checkpoint, manifest_sha):
    data = json.loads(Path(cache_path).read_text())
    if data["checkpoint_sha256"] != expected_checkpoint or data["manifest_sha256"] != manifest_sha:
        raise RuntimeError(f"Saved score cache provenance mismatch: {cache_path}")
    if len(data["rows"]) != len(rows):
        raise RuntimeError("Cache/manifest length mismatch")
    for old, new in zip(data["rows"], rows):
        if old["image"] != new["image"] or int(old["label"]) != int(new["label"]):
            raise RuntimeError("Cache row order/membership mismatch")
        if any(str(old.get(a, "unknown")) != str(new.get(a, "unknown")) for a in AXES):
            raise RuntimeError("Cache demographic metadata mismatch")
    if len(data["logits"]) != len(rows) or not np.isfinite(data["logits"]).all():
        raise RuntimeError("Invalid cached logits")
    return data


def scalar_metrics(nuisance, rows, z, threshold, min_support):
    return {
        "calibrated": nuisance.metric_report(rows, z, threshold, min_support),
        "fixed_logit_0": nuisance.metric_report(rows, z, 0.0, min_support),
    }


def distribution(rows, logits):
    z = np.asarray(logits, np.float64)
    y = np.asarray([r["label"] for r in rows], int)
    out = {}
    for label, name in ((0, "real"), (1, "fake")):
        q = np.quantile(z[y == label], [0, .1, .25, .5, .75, .9, 1])
        out[name] = {"n": int((y == label).sum()), "mean": float(z[y == label].mean()),
                     "std": float(z[y == label].std()),
                     "min_p10_p25_median_p75_p90_max": [float(v) for v in q]}
    return out


def cell_batches(rows, axis, seed=17, per_cell=2, repeats=2):
    groups = sorted({str(r.get(axis, "unknown")) for r in rows if str(r.get(axis, "unknown")).lower() != "unknown"})
    indices = defaultdict(list)
    for i, row in enumerate(rows):
        g = str(row.get(axis, "unknown"))
        if g.lower() == "unknown":
            continue
        indices[(g, int(row["label"]))].append(i)
    groups = [g for g in groups if all(len(indices[(g, y)]) >= per_cell * repeats for y in (0, 1))]
    if len(groups) < 2:
        return [], groups
    batches = []
    for rep in range(repeats):
        chosen = []
        for g in groups:
            for y in (0, 1):
                order = sorted(indices[(g, y)], key=lambda i: hashlib.sha256(
                    f"{seed}:classifier-audit:{axis}:{rep}:{rows[i]['image']}".encode()).hexdigest())
                chosen.extend(order[rep * per_cell:(rep + 1) * per_cell])
        batches.append((groups, chosen))
    return batches, groups


@torch.inference_mode()
def score_module(nuisance, model, rows, device, config=None, batch_size=16):
    model.eval()
    if config is not None:
        batch_size = int(config["batch_size"])
    logits = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        x = torch.stack([nuisance.normalize(nuisance.load_pixels(row, 224)) for row in batch]).to(device)
        logits.extend(model(x)["logit"].float().cpu().tolist())
    z = np.asarray(logits, np.float64)
    if len(z) != len(rows) or not np.isfinite(z).all():
        raise RuntimeError("Nonfinite or misaligned model predictions")
    return z


def p50_replay_difference(native_metrics, common_metrics):
    diffs = []
    def add(a, b):
        if a is None and b is None:
            return
        if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)):
            diffs.append(abs(float(a) - float(b)))
    for key in ("n", "n_real", "n_fake", "auc", "fpr", "fnr", "positive_rate"):
        add(native_metrics["overall"].get(key), common_metrics["calibrated"]["overall"].get(key))
    for axis in AXES:
        left, right = native_metrics["axes"][axis], common_metrics["calibrated"]["axes"][axis]
        for key in ("worst_group_auc", "fpr_gap", "fnr_gap", "auc_gap"):
            add(left.get(key), right.get(key))
        for group in set(left["groups"]) & set(right["groups"]):
            for key in ("n", "n_real", "n_fake", "auc", "fpr", "fnr", "positive_rate"):
                add(left["groups"][group].get(key), right["groups"][group].get(key))
        legacy = left.get("ding_style_fractions") or {}
        for key in ("F_FPR", "F_DP", "es_AUC"):
            add(legacy.get(key), right.get(key))
    return max(diffs, default=0.0)


def load_p50(p50, checkpoint, device):
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if ck.get("format") != p50.VERSION or ck.get("stage") != "pretrain" or ck.get("source_sha256") != P50_SHA:
        raise RuntimeError("Native P50 checkpoint format/source/stage mismatch")
    model = p50.DiagramCES(ck["config"]["backbone"], hidden=ck["config"]["hidden"],
                           gate_hidden=ck["config"]["gate_hidden"], pretrained=False,
                           encoder_policy="trainable", bn_policy="frozen", head_seed=ck["config"]["head_seed"])
    model.load_state_dict(ck["model"], strict=True)
    return model.to(device).eval(), ck


def grad_stats(nuisance, model, rows, train_weights, axis, batches, threshold, temperature, device):
    params = [p for p in model.parameters() if p.requires_grad]
    results = []
    model.eval()
    for rep, (groups, ix) in enumerate(batches):
        pixels = torch.stack([nuisance.normalize(nuisance.load_pixels(rows[i], 224)) for i in ix]).to(device)
        y = torch.tensor([int(rows[i]["label"]) for i in ix], device=device, dtype=torch.float32)
        z = model(pixels)["logit"].float()
        group_to_code = {g: j for j, g in enumerate(groups)}
        codes = torch.tensor([group_to_code[str(rows[i][axis])] for i in ix], device=device)
        priors = torch.zeros((len(groups), 2), dtype=torch.float32)
        for j, g in enumerate(groups):
            for yy in (0, 1):
                priors[j, yy] = sum(1 for r in rows if str(r.get(axis, "unknown")) == g and int(r["label"]) == yy)
        priors[:, 0] /= priors[:, 0].sum()
        priors[:, 1] /= priors[:, 1].sum()
        ww = train_weights[torch.tensor(ix)].to(device)
        bce = (ww * F.binary_cross_entropy_with_logits(z, y, reduction="none")).mean()
        fair = nuisance.soft_error_disparity(z, y, codes, priors.to(device), threshold, temperature)
        gb = torch.autograd.grad(bce, params, retain_graph=True, allow_unused=True)
        gf = torch.autograd.grad(fair, params, allow_unused=True)
        dot = nb = nf = 0.0
        for a, b in zip(gb, gf):
            if a is None and b is None:
                continue
            if a is None:
                a = torch.zeros_like(b)
            if b is None:
                b = torch.zeros_like(a)
            af, bf = a.detach().float(), b.detach().float()
            dot += float((af * bf).sum())
            nb += float((af * af).sum())
            nf += float((bf * bf).sum())
        results.append({"axis": axis, "batch": rep, "n": len(ix), "groups": len(groups),
                        "weighted_bce": float(bce.detach()), "soft_error_disparity": float(fair.detach()),
                        "weighted_bce_grad_norm": math.sqrt(nb), "soft_error_grad_norm": math.sqrt(nf),
                        "gradient_cosine": dot / math.sqrt(nb * nf) if nb > 0 and nf > 0 else None,
                        "threshold_source": "independently derived from fixed TRAIN-real calibration subset",
                        "diagnostic_only_not_optimizer_update": True})
        del pixels, z, bce, fair, gb, gf
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("smoke", "audit"), required=True)
    p.add_argument("--p50-source", required=True, type=Path)
    p.add_argument("--nuisance-source", required=True, type=Path)
    p.add_argument("--p50-checkpoint", required=True, type=Path)
    p.add_argument("--p50-config", required=True, type=Path)
    p.add_argument("--train", type=Path)
    p.add_argument("--val", type=Path)
    p.add_argument("--arms-root", type=Path)
    p.add_argument("--controls-root", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    a = p.parse_args()
    if a.out.exists():
        raise SystemExit("Output exists; preserve prior audit")
    p50 = load_module("p50_native", a.p50_source, P50_SHA)
    nuisance = load_module("ces_nuisance", a.nuisance_source, CES_SHA)
    if sha(a.p50_checkpoint) != P50_CKPT_SHA:
        raise RuntimeError("Pinned P50 checkpoint hash mismatch")
    pc = json.loads(a.p50_config.read_text())
    if pc["input_hashes"]["train"] != P50_TRAIN_SHA or pc["input_hashes"]["val"] != P50_VAL_SHA:
        raise RuntimeError("P50 config manifest hashes differ from pinned benchmark")
    if sha(a.p50_config) != sha(Path(a.p50_checkpoint).with_name("config.json")):
        raise RuntimeError("P50 checkpoint/config source directory mismatch")
    if a.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Expected exactly one verified CUDA device")
    device = torch.device(a.device)
    torch.set_num_threads(4)
    a.out.mkdir(parents=True)
    if a.mode == "smoke":
        model = p50.DiagramCES("resnet50", weights=pc["weights"], hidden=128, gate_hidden=64,
                               pretrained=True, encoder_policy="trainable", bn_policy="frozen", head_seed=17).to(device).eval()
        before = state_sha(model)
        x = torch.rand((2, 3, 224, 224), device=device)
        mean = x.new_tensor((.485, .456, .406))[:, None, None]
        std = x.new_tensor((.229, .224, .225))[:, None, None]
        z = model((x - mean) / std)
        loss = F.binary_cross_entropy_with_logits(z["logit"].float(), torch.tensor([0., 1.], device=device))
        loss.backward()
        if not torch.isfinite(loss) or not all(p.grad is not None and torch.isfinite(p.grad).all()
                                               for p in model.parameters() if p.requires_grad):
            raise RuntimeError("P50 classifier smoke backward failed")
        if state_sha(model) != before:
            raise RuntimeError("Zero-update smoke changed model parameters/buffers")
        if not torch.allclose(z["logit"], z["global_contribution"] + z["local_contribution"] + z["fusion_bias"],
                              rtol=1e-5, atol=1e-5):
            raise RuntimeError("Native P50 E-box forward decomposition failed")
        result = {"status": "PASS", "device": str(device), "torch": str(torch.__version__),
                  "p50_source_sha256": P50_SHA, "nuisance_source_sha256": CES_SHA,
                  "p50_checkpoint_sha256": P50_CKPT_SHA, "parameter_count": model.counts(),
                  "finite_backward": True, "state_unchanged": True, "test_read": False}
        (a.out / "SMOKE_PASS.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    required = (a.train, a.val, a.arms_root, a.controls_root)
    if any(x is None for x in required):
        raise SystemExit("Audit mode requires pinned TRAIN/VAL and all verified arm roots")
    if sha(a.train) != P50_TRAIN_SHA or sha(a.val) != P50_VAL_SHA:
        raise RuntimeError("Audit TRAIN/VAL manifest hash mismatch")
    if sha(a.p50_checkpoint) != P50_CKPT_SHA:
        raise RuntimeError("P50 checkpoint changed")
    train_rows = p50.read_rows(a.train)
    val_rows = p50.read_rows(a.val)
    model_specs = {"P50_complete": ("p50", a.p50_checkpoint, pc)}
    scores_private = {}
    for arm in ("A0", "B0", "C0", "H0"):
        root = (a.arms_root / arm) if arm in ("B0", "C0") else (a.controls_root / arm)
        verify = json.loads((root / "RUN_VERIFIED.json").read_text())
        if verify.get("status") != "COMPLETED_VERIFIED" or verify.get("test_evaluated") is not False:
            raise RuntimeError(f"{arm} completion/test-isolation receipt invalid")
        ckpt = root / "best.pt"
        if sha(ckpt) != EXPECTED_CHECKPOINTS[arm] or verify.get("selected_checkpoint_sha256") != EXPECTED_CHECKPOINTS[arm]:
            raise RuntimeError(f"{arm} selected checkpoint hash mismatch")
        run = json.loads((root / "RUN.json").read_text())
        if run.get("source_sha256") != CES_SHA:
            raise RuntimeError(f"{arm} source differs from pinned classifier adapter")
        cache = root / "validation_scores_PRIVATE.json"
        scores_private[arm] = check_cache(cache, val_rows, EXPECTED_CHECKPOINTS[arm], P50_VAL_SHA)
        model_specs[arm] = ("nuisance", ckpt, run.get("configuration", {}))

    outputs = {}
    state_digests_before = {}
    models = {}
    for name, (kind, checkpoint, conf) in model_specs.items():
        if kind == "p50":
            model, ck = load_p50(p50, checkpoint, device)
            threshold = float(ck["calibration"]["threshold_logit"])
            expected_auc = ck["validation"]["overall"]["auc"]
            config = ck["config"]
        else:
            model, ck = nuisance.load_trained(checkpoint, device)
            threshold = float(ck["calibration"]["threshold_logit"])
            expected_auc = ck["validation"]["overall"]["auc"]
            config = ck["config"]
        state_digests_before[name] = state_sha(model)
        models[name] = (model, kind, config, threshold, expected_auc)
        z = score_module(nuisance, model, val_rows, device, config if kind == "nuisance" else None)
        report = scalar_metrics(nuisance, val_rows, z, threshold, 20)
        if abs(report["calibrated"]["overall"]["auc"] - expected_auc) > 1e-5:
            raise RuntimeError(f"{name} clean-VAL replay does not match selected checkpoint")
        native_replay_diff = None
        if name == "P50_complete":
            native_replay = p50.metrics(val_rows, z, threshold, 20)
            native_replay_diff = p50_replay_difference(native_replay, report)
            if native_replay_diff > 1e-5:
                raise RuntimeError("Full native P50 validation metrics failed replay")
        cache_diff = None
        if name in scores_private:
            cache_diff = float(np.max(np.abs(z - np.asarray(scores_private[name]["logits"], np.float64))))
            if cache_diff > 1e-5:
                raise RuntimeError(f"{name} checkpoint replay differs from stored validation scores")
        outputs[name] = {"checkpoint_sha256": sha(checkpoint), "threshold_logit": threshold,
                         "validation": report, "score_distribution": distribution(val_rows, z),
                         "saved_cache_max_abs_logit_difference": cache_diff,
                         "native_p50_full_metric_replay_max_abs_difference": native_replay_diff}
        del z

    # Derive one TRAIN-only calibration subset per model. It is an audit probe,
    # not a new model threshold and is never used to change checkpoints.
    selected_real = [r for r in train_rows if int(r["label"]) == 0]
    selected_real = sorted(selected_real,
                           key=lambda r: nuisance.seed_for(17, "train-calibration", r["image"]))[:2048]
    gradient_results = {}
    train_weights = nuisance.original_weights(train_rows, "ids8")
    for name, (model, kind, config, val_threshold, expected_auc) in models.items():
        zcal = score_module(nuisance, model, selected_real, device, config if kind == "nuisance" else None)
        train_threshold = nuisance.calibrate(zcal, .1)["threshold_logit"]
        temp = float(config.get("temperature", 1.0))
        diagnostics = []
        for axis in AXES:
            batches, eligible = cell_batches(train_rows, axis, seed=17, per_cell=2, repeats=2)
            if not batches:
                diagnostics.append({"axis": axis, "status": "INSUFFICIENT_TRAIN_CELL_SUPPORT", "eligible_groups": eligible})
                continue
            diagnostics.extend(grad_stats(nuisance, model, train_rows, train_weights, axis, batches,
                                           train_threshold, temp, device))
        gradient_results[name] = {"train_real_calibration_n": len(selected_real),
                                  "train_derived_diagnostic_threshold": train_threshold,
                                  "per_axis_microbatch_gradients": diagnostics}

    # Deterministic TRAIN-only nuisance bank response sample. These are
    # processing sensitivities, not evidence that a forensic cue is valid.
    sample = sorted(train_rows, key=lambda r: hashlib.sha256(
        f"seed17:fixed-nuisance-audit:{r['image']}".encode()).hexdigest())[:128]
    nuisance_results = {}
    for name, (model, kind, config, threshold, expected_auc) in models.items():
        baseline = []
        transformed = {case: [] for case in nuisance.NUISANCE_BANK}
        labels = []
        for row in sample:
            raw = nuisance.load_pixels(row, 224)
            labels.append(int(row["label"]))
            x = nuisance.normalize(raw)[None].to(device)
            baseline.append(float(model(x)["logit"].float().detach().cpu()[0]))
            for case in nuisance.NUISANCE_BANK:
                view = nuisance.nuisance_view(raw, case)
                transformed[case].append(float(model(nuisance.normalize(view)[None].to(device))["logit"].float().detach().cpu()[0]))
        base = np.asarray(baseline)
        details = {}
        for case, values in transformed.items():
            delta = np.asarray(values) - base
            details[case] = {"n": len(delta), "mean_signed_logit_delta": float(delta.mean()),
                             "median_abs_logit_delta": float(np.median(np.abs(delta))),
                             "p90_abs_logit_delta": float(np.quantile(np.abs(delta), .9)),
                             "real_median_abs_delta": float(np.median(np.abs(delta[np.asarray(labels) == 0]))),
                             "fake_median_abs_delta": float(np.median(np.abs(delta[np.asarray(labels) == 1])))}
        nuisance_results[name] = {"sample_n": len(sample), "selection": "fixed SHA256 seed-17 TRAIN sample",
                                  "transform_results": details,
                                  "validity_note": "Procedural nuisance responses do not prove label or forensic-evidence preservation."}

    state_digests_after = {name: state_sha(model) for name, (model, *_rest) in models.items()}
    if state_digests_before != state_digests_after:
        raise RuntimeError("Audit changed checkpoint model parameters or buffers")
    report = {"status": "PASS_ZERO_UPDATE_AUDIT", "p50_source_sha256": P50_SHA,
              "nuisance_source_sha256": CES_SHA, "p50_checkpoint_sha256": P50_CKPT_SHA,
              "train_manifest_sha256": sha(a.train), "val_manifest_sha256": sha(a.val),
              "train_rows": len(train_rows), "val_rows": len(val_rows), "models": outputs,
              "train_gradient_diagnostics": gradient_results,
              "train_nuisance_response": nuisance_results,
              "state_unchanged": True, "test_manifest_read": False, "test_scores_read": False,
              "training_updates": 0,
              "limitations": ["Recorded contributor IDs do not prove true-person independence.",
                              "TRAIN microbatch gradients are local diagnostics, not population-gradient estimates.",
                              "The generic per-axis soft-error diagnostic is not identical to every arm's hierarchical training term.",
                              "Nuisance transformations are not validated forensic interventions."]}
    (a.out / "AUDIT_PRIVATE.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "models": {k: {"auc": v["validation"]["calibrated"]["overall"]["auc"],
                    "fpr": v["validation"]["calibrated"]["overall"]["fpr"],
                    "fnr": v["validation"]["calibrated"]["overall"]["fnr"]} for k, v in outputs.items()},
        "train_gradient_batches": {k: len(v["per_axis_microbatch_gradients"]) for k, v in gradient_results.items()},
        "state_unchanged": True, "test_read": False}, indent=2))


if __name__ == "__main__":
    main()
