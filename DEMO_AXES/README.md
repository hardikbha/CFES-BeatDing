# DEMO_AXES: gender / race / age / skin-tone evaluation layer

CPU-only layer that adds age and skin-tone axes (and cleaner gender/race labels) to every saved model's per-image
logits. No training, no GPU, no existing result file is modified. Pre-registration this serves:
`decisions/2026-10-26-final-model-spec.md` (axes, bins, min-support rule, H1-H4). Results:
`results/MULTI_AXIS_RESULTS.md`.

Other agents' folders inside DEMO_AXES that are NOT part of this layer: `outputs/` + `code/{aiface_join,relabel,
groupci,groupsizes,consist,nvc2,skinage_pts,ff2age}.py` (Reviewer 2 audit scripts, 26 Oct), `labels_fairface/`,
`code/fairface_*.py`, `code/fairforensics_*.py`, `results/fairforensics_ids/` (FairForensics-protocol labels, separate
agent; `rescore.py` only READS `labels_fairface/*.csv` for the extra `age_fairface2` sensitivity axis).

## Run order (all with `~/.conda/envs/aiface/bin/python`, OMP/MKL threads <= 8)

| step | command | writes | time |
|---|---|---|---|
| 1 | `code/make_lists.py` | `labels/raw/lists/<set>.txt` (unique image paths) | s |
| 2 | `code/run_face_attrs.sh` (16 procs, 1 onnxruntime thread each) | `labels/raw/face_attrs_<set>.csv.gz` | ~1 h for 297k images |
| 3 | `code/build_labels.py` | `labels/labels_<set>.csv`, `ita_bin_edges.json`, `ita_stability.csv`, `ita_construct_checks.csv`, `coverage.csv`, `human_check_sample.csv` | min |
| 4 | `code/rescore.py --B 1000` | `results/rescore_axes.csv`, `rescore_groups.csv`, `thresholds.csv`, `sanity_checks.csv`, `boot_cache.npz`, `rescore_meta.json` | min |
| 5 | `code/mechanism.py --B 1000` | `results/mechanism_b0.csv`, `mechanism_b0_points.csv` | min |
| 6 | `code/make_report.py` | `results/MULTI_AXIS_RESULTS.md` (inserts the hand-written `results/_summary_block.md` as Section 0; update it when numbers change), `seedmean_axes.csv`, `descriptive_H1_H4.csv` | s |

`results/boot_cache.npz` (~90 MB, float32) holds the bootstrap replicates that `make_report.py` averages over seeds;
it is a cache, not a result. Axes in `rescore_axes.csv`: primary `gender`, `race`, `gender_x_race` (AI-Face cleaner,
frame level), `age`, `skin_ita4`; sensitivity `*_vidmaj` (video-majority cleaner labels), `*_xu` (protocol labels),
`age_if`, `age_v2`, `skin_ita4_{raw,gwexp,local}`, `skin_mst3_aif`; exploratory `gender_x_age`, `gender_x_skin`.

Steps 4-6 pick up new models automatically (CESel p1 arms, CLIPFAIR runs) once their `eval/{val,ffpp_test,dfdc,
celebdf,dfd}_scores.npz` exist; rerun 4-6 then. Step 3 must not be rerun after scores have been looked at with
different bin rules: the ITA edges in `labels/ita_bin_edges.json` are the declared ones.

## Code

- `face_attrs.py`: per image, InsightFace buffalo_l SCRFD on a zero-padded crop (landmarks), buffalo_l genderage
  (numeric age; the estimator behind FF2's `if_age`), and ITA of face-skin pixels (cheek + forehead boxes from the
  landmarks; ellipse-minus-eyes/mouth template when no face is found; YCrCb skin mask; 10-90 % L* trim; gray-world
  white balance from the crop surround, chromatic only). Three ITA variants: `ita_gw` (primary), `ita_raw`,
  `ita_gwexp` (plus exposure normalisation). Full definition in the docstring.
- `build_labels.py`: joins Xu labels (protocol), AI-Face cleaner v2 gender/race/age (Purdue release; FF++ val falls
  back to Xu, flagged), AI-Face v1 MST/age/gender, our InsightFace age and ITA; video aggregation; declares the
  ITA bin edges; stability, construct checks, rater sample. Column dictionary in the docstring.
- `multiaxis_metrics.py`: per-axis F_FPR, F_DP, es-AUC, FPR gap, TPR gap, equalised odds, worst-group rates, with
  FPR/TPR-specific eligibility and the min-support rule (>= 3 real videos for FPR terms, >= 3 fake videos for TPR
  terms; 2 for validation selection); count-based cluster bootstrap. Definitions = `CFES/cfes/metrics.py`.
- `rescore.py`: model registry, thresholds (t05 = deployed 0.5 rule; A1 = the 22-Oct validation rule, stored value
  where it exists), all axes x sets x thresholds, reproduction checks against `fair_oppoint.json`.
- `mechanism.py`: CESel B0 per-group cue deficits (recomputed from saved `s_k` with the training replacement rule) vs
  per-group FPR/FNR deviations, Spearman with video-cluster bootstrap.
- `make_report.py`: the results document.

## Labels (`labels/`)

`labels_<set>.csv` for `ffpp_train`, `ffpp_val`, `ffpp_test`, `dfdc`, `celebdf`, `dfd`, `ff2_test` (FF2-public TEST
rows of `data_ding_schema/fairforge2_full/test.csv`, `cell_valid` marks the 87,564 rows of the official public
benchmark). Key = `img_path`, identical to the score npz `img_path` (FF2: the absolute path used by FF2 score files).
One row per protocol row (DFDC keeps its 1,476 duplicate rows, as the score files do).

Licence: AI-Face annotations are CC BY-NC-ND; the label tables are for internal evaluation only and must not be
redistributed. ITA and InsightFace age are our own measurements (machine estimates of apparent attributes, not
self-reported identity); no human label exists yet for any of these axes. `human_check_sample.csv` lists 300 reals
for the planned rater check (not run).
