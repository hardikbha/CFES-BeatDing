# FairForensics / GenFace 8-IDS protocol: FairFace labels and metrics (secondary, descriptive)

26 Oct 2026; finished 12:30. All 585,107 images are labelled (16 CPU workers, 2.9 h). The CESel gate failed, so the
p1 arms are not scored. This layer is secondary and descriptive only. It was pre-registered in
`decisions/2026-10-26-final-model-spec.md` §4, which was appended before any model was re-scored on these labels.
- H1–H4 are unchanged.
- Nothing here selects a model, a checkpoint or a threshold.
- No significance claim is made.

Code: `agents/DEMO_AXES/code/fairface_annotate.py` (labels), `fairface_fidelity.py` (preprocessing check),
`fairface_agreement.py`, `fairforensics_ids.py` (metric function and re-scoring), `fairforensics_report.py` (the
generated tables below). It is separate from the multi-axis layer in the same folder (`../../README.md`): that
layer's files are only read, never written. Its `rescore.py` registry and `multiaxis_metrics.py` bootstrap are
imported.

## 1. The protocol (FairForensics, arXiv 2608.01661, Sec. IV, Table I)
- Demographic attributes are "automatically annotated using the FairFace algorithm [27]".
- The attributes are:
  - sex: Male / Female
  - age: 0–29 (Youth) / 30+ (Adult)
  - race: White / Non-White
- The 8 intersectional demographic subgroups (IDS) are sex × age × race, with IDS labels 0–7: MYW, FYW, MAW, FAW,
  MAN, FAN, MYN, FYN.
- The fairness metrics are computed over the 8 IDS: F_FPR, F_MEO, F_DP and F_OAE, with AUC for detection.
- The paper does not specify:
  - which FairFace checkpoint it used;
  - how the 9 age bins and the 7 races are collapsed;
  - formulas for F_MEO and F_OAE.

## 2. Labels (`agents/DEMO_AXES/labels_fairface/<set>.csv`)

### Model
- **Checkpoint.** Official FairFace `res34_fair_align_multi_7_20190809.pt`.
  - Repo: github.com/dchen236/FairFace, commit 74b4f93e (3 Nov 2021).
  - Google Drive file id `11y0Wi3YQf21a_VcspUV4FwqzhMcfaVAB`. The README's folder link returns 404 to gdown.
  - Downloaded on the login node mgmt01 with the `comp` env's gdown. Nothing was installed.
  - 85,283,042 bytes; sha256 `37f2e74c0e7da196f8edd7dae0c88cffdbef7923beebebcc27381f0ad0fdb2c1`; md5
    `b39c084e39ad4135eeb7eae349dc4b62`.
  - Loads strictly into torchvision ResNet-34 with fc 512→18 (218/218 keys).
  - Location: `~/models_external/fairface/fair_face_models/`. The repo clone, with its
    `dlib_models/{mmod_human_face_detector,shape_predictor_5_face_landmarks}.dat`, is in
    `~/models_external/fairface/repo/`.
- **Checkpoint caveat.** The repo's *current* `predict.py` loads `fairface_alldata_20191111.pt`, a later checkpoint
  trained on train+val. We use the 2019-08-09 7-race checkpoint as specified, which is the one the original
  predict.py used.
- **Environment.** `ffaa`: dlib 19.24.2, numpy 1.26.4, torch 2.1.2, CPU. The `FA` env's dlib 19.24.0 rejects
  numpy-2 arrays ("Unsupported image type"), and aiface/pulid have no dlib.

### Columns
- **Raw.** The 7 race softmax probabilities `p_race_{White, Black, Latino_Hispanic, East Asian, Southeast Asian,
  Indian, Middle Eastern}`, `p_male` / `p_female`, and the 9 age-bin probabilities
  `p_age_{0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+}`. Also `det_mode`, `det_conf`, `n_faces` and
  `ok` (0 = unreadable).
- **Derived.**
  - `sex` = Male iff P(male) ≥ 0.5.
  - `age9` = the argmax bin.
  - `age2` = 0–29 iff the argmax bin is in {0–2, 3–9, 10–19, 20–29}, else 30+.
  - `age4` ∈ {<30, 30–49, 50–69, 70+}. FairFace's decade bins cannot be cut at 45/60, so the spec §2 bins
    (<30 / 30–44 / 45–59 / 60+) cannot be built from this classifier.
  - `p_age_lt30` = the mass of the four youngest bins (sensitivity only).
  - `race7` = the argmax.
  - `race2` = White iff race7 = White, so Latino_Hispanic and Middle Eastern count as Non-White.
  - `ids8` (0–7, Table I order) and `ids8_name`.
- **Video sets** (FF++, DFDC, Celeb-DF, DFD).
  - `vid` = the video key: the file name without the leading `0_` and the frame number. It maps 1:1 to `video`
    in `labels/labels_<set>.csv`.
  - `vid_{sex, age2, race2, age4}` = the majority over the video's frames. On a tie, the same rule is applied to the
    video's mean probabilities. Ties are rare: ≤ 14 videos per attribute per set.
  - `vid_*_agree` = the majority share.
  - `vid_ids8(_name)` = the IDS built from the three majority attributes.
  - `vid_n_frames`.
- **Key.** `img_path`, byte-identical to the score npz `img_path`.
  - FF2 uses the absolute path of `data_ding_schema/fairforge2_public/*.csv` and also stores the release `rel_path`.
  - DFDC's 1,476 duplicate protocol rows share one row (21,381 unique paths), so join on `img_path`.
- **Sets.** FF++ train/val/test, DFDC, Celeb-DF, DFD (DINGREPRO_LIN protocol CSVs), and FF2-public TEST, SELECT,
  TRAIN and CAL (all rows of `public_manifest.csv` per split). Test sets were run first. Progress and throughput
  are in `labels_fairface/PROGRESS.json` and the run log in `labels_fairface/run.log`.

## 3. Deviations from FairFace `predict.py` and from the task brief
- **D1 Detector resolution.** predict.py resizes to a max side of 800 px and runs the mmod CNN detector with 1
  upsample. That costs 7.4 s/image on this CPU, which is infeasible for 0.6 M images. We run mmod at a max side
  of 300 px with no upsample (0.19 s). Our inputs are face crops, so faces fill most of the frame.
- **D2 One face per image** (the largest detection). predict.py writes every detected face as its own output.
- **D3 Fallbacks** when mmod finds no face:
  1. the dlib HOG detector (max side 400, 1 upsample);
  2. the whole image, inset 5% per side, as the landmark box.
  The rates are in the coverage table (≤ 0.6% per set).
- **D4 Landmarks and chip from the full-resolution image** (predict.py uses the 800-px copy). The chip itself is
  identical: `get_face_chip(size=300, padding=0.25)`, then Resize(224) + ImageNet normalisation, as in predict.py.
- **D5 Decoding with PIL** (predict.py uses `dlib.load_rgb_image`).
- **D6 We kept the alignment.** The brief allowed replacing FairFace's crop/align step with a plain 224 resize. We
  did not, for two reasons:
  - Our crops differ in margin across datasets: the 380-px Lin crops for FF++/DFDC/CDF/DFD versus FairFace/FFHQ
    crops in FF2.
  - The classifier was trained on padding-0.25 dlib chips.
  **Fidelity of D1–D5** against predict.py's exact pipeline: 800 px, 1 upsample, 9.3 s/image on 300 random test
  images (60 per test set), with the same weights (`fidelity_official_vs_ours*.{csv,json}`).
  - Label agreement: sex 99.0%, age2 100%, race2 98.3%, race7 97.0%, age9 98.0% and IDS 97.3%.
  - The median of the largest probability difference is 0.02, and predict.py found a face in all 300 images.
- **D7 Video majority.** Video sets are scored with video-majority labels: GenFace is an image dataset, and a
  video's subject should not change group between frames. Frame labels are reported as a sensitivity analysis.
- **D8 FF2 evaluation rows.** FF2-public TEST is evaluated on the 87,564 cell-valid rows of
  `data_ding_schema/fairforge2_public/test.csv`, which is the public benchmark. Labels exist for all 93,543 TEST
  rows.
  - FF2 clusters are persons (`person_id`).
  - The 15,267 GAN/T2I de-novo fakes have no person, so each is its own cluster. This is the convention of
    `labels/labels_ff2_test.csv`.

## 4. Metrics (`code/fairforensics_ids.py`)

### Library function
`ids8_metrics(y, z, t, ids, cluster, person)`. Decision: fake iff logit ≥ t. All values are in %.
- F_FPR = Σ_g |FPR_g − FPR| (Ding Eq. 8).
- F_DP = max_g PR_g − min_g PR_g (Ding Eq. 9).
- es-AUC = AUC / (1 + Σ_g |AUC_g − AUC|) (Ding Eq. 10).
- FPR gap = max−min FPR_g, and TPR gap likewise, with the worst-group FPR and TPR.
- F_MEO = max(FPR gap, TPR gap). This is FairReL's gap form; FairForensics gives no formula.
- F_OAE = max−min ACC_g.

### Checks
- On synthetic data these equal `agents/CFES/cfes/metrics.fairness_block`, which itself equals Ding's released code
  (verified 16 Oct) under the Ding rule.
- They also equal `multiaxis_metrics.axis_metrics` under the min-support rule, to machine precision.

### Eligibility
- **Primary: min-support** (spec H4). On video sets, a group enters the FPR terms only with ≥ 3 real videos and the
  TPR terms only with ≥ 3 fake videos. AUC/PR/ACC terms need both. On FF2 the clusters are persons.
- **Also reported: `ding_exact`.** Every group that holds both classes, with no minimum.
- A metric needs ≥ 2 eligible groups.

### Thresholds
These come from the multi-axis layer's `rescore.py` logic and were never re-chosen with FairFace labels.
- **t05:** the deployed P(fake) ≥ 0.5 rule. For CFES-D the FF++-validation intercept is folded in.
- **A1:** the 22-Oct validation rule (FF++ val, Xu labels).
  - Stored values (`fair_oppoint.json`, CESel `a1.json`) are used where they exist.
  - Otherwise the rule is re-run on the saved val logits.
  - All 23 thresholds equal `results/thresholds.csv` of the multi-axis layer, up to that file's 5-digit rounding
    (`thresholds_used.csv`).
- FF2-trained models have no FF++-val A1, so they get t05 only.

### Models
- **Spec §4 models:** CFES-D s7/17/27, arm C s7/17/27, arm E w1/w3/w10 s7, detection-only control s7/17/27, Ding
  repro SFD+GDA s7 and Ori s7/17/27, and CESel B0 s7. CESel B0 has no FF2 score file.
- **Historical rows** from the same registry (not in the spec list; descriptive): CFES A (rev4), CFES-ResNet, and
  the FF2-trained phase 1 / control / CFES runs (trained on the full FF2 TRAIN, including RFW).
- **Excluded on purpose:** the CESel Amendment-5 p1 arms and the CLIPFAIR baselines. Their selection is still
  running on validation.
  - One debug view of an early version briefly printed one p1-arm row (DFD, A1) before the filter was added.
  - That row is not used anywhere.

### Bootstrap
Video/person-cluster bootstrap, B = 1000 (rng 2026), with eligibility fixed from the full sample.
- It gives 95% intervals for F_FPR, the FPR gap and the TPR gap.
- These intervals are descriptive only.

## 5. Agreement (`agreement.csv`, `agreement_crosstabs.txt`, `video_consistency.csv`)
Full tables are below.
Ranges cover the frame and video levels and all sources.
- **Sex agrees well where all sources agree.**
  - FF++: 98.2–99.9% (κ 0.96–1.00).
  - Celeb-DF: 99.4–99.8%.
  - FF2 vs the manifest: 98.1% (κ 0.96).
  - It is lower on DFDC (85–95%, κ 0.69–0.90) and DFD (78–89%, κ 0.56–0.77). There FairFace agrees more with
    AI-Face v1/v2 than with Xu.
- **Race2 is the weak attribute on FF++.**
  - FF++: 77–80% (κ 0.48–0.56). FairFace calls 41.5% of FF++ test frames Non-White, against Xu 25.8% and AI-Face
    v2 22.1%. Most of the difference is Xu-"White" faces that FairFace labels Middle Eastern (3,603 frames) or
    Latino (714).
  - Other sets: DFDC 86–91%, Celeb-DF 96–99%, DFD 87–94%.
  - FF2 vs the manifest: 87.8% (κ 0.70). FairFace says 31.8% White against the manifest's 23.4%. Many manifest
    Latino/Middle Eastern/Asian faces are labelled White.
- **Age2 is noisy.**
  - Against InsightFace age < 30 (the same cut): 65–78% at frame level (κ 0.24–0.50), with Spearman of the
    expected age 0.65 on FF2.
  - FairFace calls more faces young than InsightFace does: Celeb-DF 83% vs 55%, FF2 TEST 26% vs 15%.
  - Against AI-Face v2 young vs middle+senior: 82–89% at frame level.
  - Against AI-Face v1 < 25 the agreement is poor, as expected from the 25-vs-30 cut mismatch.
- **Per-video consistency.**
  - The mean majority share is 92.5–99.5% for sex, 88–94% for age2 and 91–97% for race2.
  - Only 25–47% of videos keep one IDS across all frames. The video-majority rule moves 12–23% of frames to another
    IDS. This is the reason for D7.
  - On FF2, for persons with ≥ 2 images, 87% of a person's real images carry the person's majority IDS, and 75% of
    their generated fakes do.

## 6. Results: what the tables show (descriptive; no significance claims)
- **Support is thin on two sets.**
  - Celeb-DF: only 4 of 8 IDS have ≥ 3 real and ≥ 3 fake videos. MAW has no fakes.
  - DFD: FAW and MAN have < 3 real videos, and each IDS holds only 0–5 real and 1–8 fake persons.
  - Their 8-IDS numbers rest on very few identities. The bootstrap intervals are wide on every video set.
- **The IDS numbers are not the Xu-label numbers.**
  - The 8-IDS F_FPR differs from the gender × race F_FPR of the paper: the groups are different and there are 8 of
    them.
  - At A1, CFES-D's seed means on FF++/DFDC/CDF/DFD/FF2 are 24.9 / 14.9 / 32.7 / 50.3 / 25.1, with TPR 79.7 / 38.6 /
    72.9 / 75.6 / 69.1.
  - For comparison, its Xu-label intersectional F_FPR at A1 is 16.8 / 14.1 / 20.5 / 34.7 (`fair_oppoint`).
- **The threshold changes the picture, as in the main audit.** At t05, CFES-D's FF2 TEST FPR is 91% and its TPR 95%:
  the folded FF++ intercept makes it call nearly everything fake. Its low t05 F_FPR on FF2 (17.4) is therefore a
  degenerate operating point.
- **Read F_DP with the 24-Oct oracle caveat.** It is label-blind and penalises accuracy when the groups' fake rates
  differ.

## 7. Caveats
- These are machine estimates of apparent attributes from one classifier. They are not self-reported identity and
  have no human validation.
- The FairFace race head is known to move many Xu-"White" FF++ subjects into Middle Eastern or Latino.
- The checkpoint and the collapsing rules FairForensics used are not stated, so these are "FairForensics-style"
  labels, not their labels.

## 8. Re-running
```
cd agents/DEMO_AXES/code
OMP_NUM_THREADS=1 ~/.conda/envs/ffaa/bin/python fairface_annotate.py --workers 16      # resumable (labels_fairface/_parts)
OMP_NUM_THREADS=1 ~/.conda/envs/ffaa/bin/python fairface_annotate.py --rederive        # derived columns only
OMP_NUM_THREADS=1 ~/.conda/envs/ffaa/bin/python fairface_fidelity.py --n 60 --workers 16
~/.conda/envs/aiface/bin/python fairface_agreement.py
OMP_NUM_THREADS=4 ~/.conda/envs/aiface/bin/python fairforensics_ids.py --B 1000         # picks up new score files
~/.conda/envs/aiface/bin/python fairforensics_report.py                                 # refreshes the tables below
```
Outputs in this folder:
- `ids8_metrics.csv`: per model × seed × set × threshold × labels × rule, with CIs.
- `ids8_groups.csv`: per-IDS counts and rates, with FPR/TPR/PR/ACC/AUC in %.
- `ids8_group_sizes.csv`, `ids8_seedmean.csv`, `thresholds_used.csv`, `run_meta.json`.
- The agreement files and the fidelity files.

<!-- GENERATED BY code/fairforensics_report.py; edits below this line are overwritten -->

## Generated tables

### Coverage

| set | images (unique paths) | mmod300 % | hog400 % | fullrect % | unreadable | worker CPU-s / image |
|---|---|---|---|---|---|---|
| ffpp_test | 25401 | 99.85 | 0.00 | 0.15 | 0 | 0.29 |
| dfdc | 21381 | 99.38 | 0.02 | 0.60 | 0 | 0.31 |
| celebdf | 28458 | 99.99 | 0.00 | 0.01 | 0 | 0.32 |
| dfd | 9385 | 99.68 | 0.00 | 0.32 | 0 | 0.32 |
| ff2pub_test | 93543 | 99.56 | 0.03 | 0.41 | 0 | 0.36 |
| ff2pub_select | 34001 | 99.43 | 0.04 | 0.53 | 0 | 0.43 |
| ffpp_val | 25386 | 99.98 | 0.01 | 0.01 | 0 | 0.57 |
| ffpp_train | 76139 | 99.85 | 0.12 | 0.03 | 0 | 0.29 |
| ff2pub_train | 236163 | 99.49 | 0.04 | 0.47 | 0 | 0.20 |
| ff2pub_cal | 35250 | 99.41 | 0.05 | 0.54 | 0 | 0.21 |

Throughput: 56.3 images/s with 16 single-threaded CPU workers (585107 images, 173 min wall).

### Fidelity: our preprocessing vs predict.py's (same weights, same images)

| set | n | agree_sex | agree_age2 | agree_race2 | agree_race7 | agree_age9 | agree_ids8 | median_max_abs_prob_diff |
|---|---|---|---|---|---|---|---|---|
| all | 300 | 99.0 | 100.0 | 98.3 | 97.0 | 98.0 | 97.3 | 0.0 |
| celebdf | 60 | 98.3 | 100.0 | 98.3 | 96.7 | 100.0 | 96.7 | 0.0 |
| dfd | 60 | 100.0 | 100.0 | 98.3 | 98.3 | 96.7 | 98.3 | 0.0 |
| dfdc | 60 | 96.7 | 100.0 | 100.0 | 96.7 | 96.7 | 96.7 | 0.0 |
| ff2pub_test | 60 | 100.0 | 100.0 | 98.3 | 96.7 | 96.7 | 98.3 | 0.0 |
| ffpp_test | 60 | 100.0 | 100.0 | 96.7 | 96.7 | 100.0 | 96.7 | 0.0 |

300 images; predict.py's detector found no face in 0.

### Agreement with the other label sources

| set | level | source | attribute | n | agree % (kappa) | FairFace / other positive % | pos_class |
|---|---|---|---|---|---|---|---|
| ffpp_test | frame | Xu | sex | 25401 | 98.2 (0.96) | 46.7 / 47.5 | Male |
| ffpp_test | frame | Xu | race2 (White vs Non-White) | 25401 | 77.7 (0.51) | 58.5 / 74.2 | White |
| ffpp_test | frame | AIF-v2 | sex | 25401 | 99.5 (0.99) | 46.7 / 46.4 | Male |
| ffpp_test | frame | AIF-v2 | race2 (White vs Non-White) | 25401 | 76.5 (0.48) | 58.5 / 77.9 | White |
| ffpp_test | frame | AIF-v1 | sex | 25401 | 99.6 (0.99) | 46.7 / 46.4 | Male |
| ffpp_test | frame | InsightFace | age2 (<30 vs 30+) | 25398 | 75.5 (0.50) | 55.1 / 55.5 | 0-29 |
| ffpp_test | frame | AIF-v1 | age2 FairFace <30 vs AIF-v1 <25 (cut mismatch 30 vs 25) | 25401 | 61.8 (0.27) | 55.1 / 23.6 | <25 |
| ffpp_test | frame | AIF-v2 | age2 vs AIF-v2 young / middle+senior ('others' dropped; v2 cut not documented) | 20361 | 84.0 (0.60) | 66.3 / 79.6 | 0-29 |
| ffpp_test | video | Xu | sex | 965 | 98.8 (0.97) | 46.2 / 46.8 | Male |
| ffpp_test | video | Xu | race2 (White vs Non-White) | 965 | 79.6 (0.55) | 59.1 / 74.1 | White |
| ffpp_test | video | AIF-v2 | sex | 965 | 99.8 (1.00) | 46.2 / 46.0 | Male |
| ffpp_test | video | AIF-v2 | race2 (White vs Non-White) | 965 | 78.3 (0.52) | 59.1 / 78.0 | White |
| ffpp_test | video | AIF-v1 | sex | 965 | 99.9 (1.00) | 46.2 / 46.1 | Male |
| ffpp_test | video | InsightFace | age2 (<30 vs 30+) | 965 | 78.1 (0.56) | 55.5 / 55.2 | 0-29 |
| ffpp_test | video | AIF-v1 | age2 FairFace <30 vs AIF-v1 <25 (cut mismatch 30 vs 25) | 965 | 60.4 (0.26) | 55.6 / 21.7 | <25 |
| ffpp_test | video | AIF-v2 | age2 vs AIF-v2 young / middle+senior ('others' dropped; v2 cut not documented) | 872 | 82.8 (0.61) | 60.8 / 75.9 | 0-29 |
| dfdc | frame | Xu | sex | 21381 | 84.8 (0.69) | 41.6 / 48.2 | Male |
| dfdc | frame | Xu | race2 (White vs Non-White) | 21381 | 86.0 (0.72) | 51.8 / 57.2 | White |
| dfdc | frame | AIF-v2 | sex | 21381 | 92.6 (0.85) | 41.6 / 41.9 | Male |
| dfdc | frame | AIF-v2 | race2 (White vs Non-White) | 21381 | 88.4 (0.77) | 51.8 / 58.6 | White |
| dfdc | frame | AIF-v1 | sex | 21381 | 93.9 (0.87) | 41.6 / 38.2 | Male |
| dfdc | frame | InsightFace | age2 (<30 vs 30+) | 21353 | 71.5 (0.43) | 50.4 / 41.9 | 0-29 |
| dfdc | frame | AIF-v1 | age2 FairFace <30 vs AIF-v1 <25 (cut mismatch 30 vs 25) | 21381 | 59.0 (0.18) | 50.3 / 10.9 | <25 |
| dfdc | frame | AIF-v2 | age2 vs AIF-v2 young / middle+senior ('others' dropped; v2 cut not documented) | 14529 | 81.9 (0.59) | 61.7 / 74.3 | 0-29 |
| dfdc | video | Xu | sex | 757 | 87.1 (0.74) | 40.6 / 48.2 | Male |
| dfdc | video | Xu | race2 (White vs Non-White) | 757 | 88.9 (0.78) | 54.4 / 57.3 | White |
| dfdc | video | AIF-v2 | sex | 757 | 94.5 (0.89) | 40.6 / 41.3 | Male |
| dfdc | video | AIF-v2 | race2 (White vs Non-White) | 757 | 91.3 (0.82) | 54.4 / 59.7 | White |
| dfdc | video | AIF-v1 | sex | 757 | 95.1 (0.90) | 40.6 / 37.5 | Male |
| dfdc | video | InsightFace | age2 (<30 vs 30+) | 757 | 73.8 (0.48) | 50.9 / 41.1 | 0-29 |
| dfdc | video | AIF-v1 | age2 FairFace <30 vs AIF-v1 <25 (cut mismatch 30 vs 25) | 757 | 57.1 (0.15) | 50.9 / 8.5 | <25 |
| dfdc | video | AIF-v2 | age2 vs AIF-v2 young / middle+senior ('others' dropped; v2 cut not documented) | 668 | 77.7 (0.53) | 55.7 / 73.2 | 0-29 |
| celebdf | frame | Xu | sex | 28458 | 99.4 (0.97) | 10.6 / 10.2 | Male |
| celebdf | frame | Xu | race2 (White vs Non-White) | 28458 | 96.0 (0.76) | 88.9 / 92.6 | White |
| celebdf | frame | AIF-v2 | sex | 28458 | 99.5 (0.97) | 10.6 / 10.1 | Male |
| celebdf | frame | AIF-v2 | race2 (White vs Non-White) | 28458 | 96.1 (0.77) | 88.9 / 92.5 | White |
| celebdf | frame | AIF-v1 | sex | 28458 | 99.5 (0.97) | 10.6 / 10.1 | Male |
| celebdf | frame | InsightFace | age2 (<30 vs 30+) | 28458 | 64.8 (0.24) | 82.9 / 55.3 | 0-29 |
| celebdf | frame | AIF-v1 | age2 FairFace <30 vs AIF-v1 <25 (cut mismatch 30 vs 25) | 28458 | 28.8 (0.05) | 82.9 / 12.2 | <25 |
| celebdf | frame | AIF-v2 | age2 vs AIF-v2 young / middle+senior ('others' dropped; v2 cut not documented) | 26920 | 89.4 (0.36) | 86.4 / 96.0 | 0-29 |
| celebdf | video | Xu | sex | 956 | 99.7 (0.98) | 10.4 / 10.3 | Male |
| celebdf | video | Xu | race2 (White vs Non-White) | 956 | 98.6 (0.91) | 91.4 / 92.6 | White |
| celebdf | video | AIF-v2 | sex | 956 | 99.8 (0.99) | 10.4 / 10.1 | Male |
| celebdf | video | AIF-v2 | race2 (White vs Non-White) | 956 | 98.7 (0.92) | 91.4 / 92.5 | White |
| celebdf | video | AIF-v1 | sex | 956 | 99.8 (0.99) | 10.4 / 10.1 | Male |
| celebdf | video | InsightFace | age2 (<30 vs 30+) | 956 | 67.4 (0.27) | 86.6 / 57.1 | 0-29 |
| celebdf | video | AIF-v1 | age2 FairFace <30 vs AIF-v1 <25 (cut mismatch 30 vs 25) | 956 | 23.7 (0.03) | 86.5 / 10.5 | <25 |
| celebdf | video | AIF-v2 | age2 vs AIF-v2 young / middle+senior ('others' dropped; v2 cut not documented) | 930 | 94.1 (0.63) | 88.7 / 94.2 | 0-29 |
| dfd | frame | Xu | sex | 9385 | 78.0 (0.56) | 47.3 / 45.0 | Male |
| dfd | frame | Xu | race2 (White vs Non-White) | 9385 | 87.3 (0.65) | 76.2 / 75.5 | White |
| dfd | frame | AIF-v2 | sex | 9385 | 87.4 (0.75) | 47.3 / 40.2 | Male |
| dfd | frame | AIF-v2 | race2 (White vs Non-White) | 9385 | 91.8 (0.78) | 76.2 / 75.7 | White |
| dfd | frame | AIF-v1 | sex | 9385 | 85.2 (0.70) | 47.3 / 35.3 | Male |
| dfd | frame | InsightFace | age2 (<30 vs 30+) | 9383 | 78.2 (0.28) | 85.2 / 78.1 | 0-29 |
| dfd | frame | AIF-v1 | age2 FairFace <30 vs AIF-v1 <25 (cut mismatch 30 vs 25) | 9385 | 41.6 (0.10) | 85.2 / 29.1 | <25 |
| dfd | frame | AIF-v2 | age2 vs AIF-v2 young / middle+senior ('others' dropped; v2 cut not documented) | 8634 | 88.5 (0.22) | 87.6 / 96.8 | 0-29 |
| dfd | video | Xu | sex | 319 | 79.6 (0.59) | 45.1 / 44.8 | Male |
| dfd | video | Xu | race2 (White vs Non-White) | 319 | 87.5 (0.66) | 75.5 / 75.5 | White |
| dfd | video | AIF-v2 | sex | 319 | 89.0 (0.78) | 45.1 / 39.2 | Male |
| dfd | video | AIF-v2 | race2 (White vs Non-White) | 319 | 93.7 (0.83) | 75.5 / 74.9 | White |
| dfd | video | AIF-v1 | sex | 319 | 87.1 (0.73) | 45.1 / 34.2 | Male |
| dfd | video | InsightFace | age2 (<30 vs 30+) | 319 | 84.6 (0.30) | 90.6 / 84.6 | 0-29 |
| dfd | video | AIF-v1 | age2 FairFace <30 vs AIF-v1 <25 (cut mismatch 30 vs 25) | 319 | 36.7 (0.08) | 90.3 / 27.0 | <25 |
| dfd | video | AIF-v2 | age2 vs AIF-v2 young / middle+senior ('others' dropped; v2 cut not documented) | 314 | 92.7 (0.38) | 91.1 / 96.5 | 0-29 |
| ffpp_val | frame | Xu | sex | 25386 | 97.1 (0.94) | 46.3 / 48.4 | Male |
| ffpp_val | frame | Xu | race2 (White vs Non-White) | 25386 | 79.9 (0.55) | 60.8 / 74.6 | White |
| ffpp_val | frame | AIF-v1 | sex | 25386 | 99.3 (0.99) | 46.3 / 46.1 | Male |
| ffpp_val | frame | InsightFace | age2 (<30 vs 30+) | 25383 | 75.0 (0.49) | 55.4 / 56.4 | 0-29 |
| ffpp_val | frame | AIF-v1 | age2 FairFace <30 vs AIF-v1 <25 (cut mismatch 30 vs 25) | 25386 | 61.9 (0.29) | 55.4 / 19.6 | <25 |
| ffpp_val | video | Xu | sex | 963 | 97.4 (0.95) | 46.0 / 48.0 | Male |
| ffpp_val | video | Xu | race2 (White vs Non-White) | 963 | 80.2 (0.56) | 61.0 / 74.8 | White |
| ffpp_val | video | AIF-v1 | sex | 963 | 99.7 (0.99) | 46.0 / 45.9 | Male |
| ffpp_val | video | InsightFace | age2 (<30 vs 30+) | 963 | 76.8 (0.53) | 56.2 / 57.3 | 0-29 |
| ffpp_val | video | AIF-v1 | age2 FairFace <30 vs AIF-v1 <25 (cut mismatch 30 vs 25) | 963 | 62.3 (0.30) | 56.0 / 19.5 | <25 |
| ffpp_train | frame | Xu | sex | 76139 | 98.2 (0.96) | 43.7 / 44.1 | Male |
| ffpp_train | frame | Xu | race2 (White vs Non-White) | 76139 | 74.8 (0.48) | 52.6 / 74.5 | White |
| ffpp_train | frame | AIF-v2 | sex | 76139 | 99.1 (0.98) | 43.7 / 43.5 | Male |
| ffpp_train | frame | AIF-v2 | race2 (White vs Non-White) | 76139 | 74.6 (0.48) | 52.6 / 75.2 | White |
| ffpp_train | frame | AIF-v1 | sex | 76139 | 99.2 (0.98) | 43.7 / 43.5 | Male |
| ffpp_train | frame | InsightFace | age2 (<30 vs 30+) | 76139 | 78.2 (0.55) | 59.2 / 59.6 | 0-29 |
| ffpp_train | frame | AIF-v1 | age2 FairFace <30 vs AIF-v1 <25 (cut mismatch 30 vs 25) | 76139 | 59.6 (0.27) | 59.2 / 22.4 | <25 |
| ffpp_train | frame | AIF-v2 | age2 vs AIF-v2 young / middle+senior ('others' dropped; v2 cut not documented) | 63375 | 85.5 (0.62) | 68.8 / 80.9 | 0-29 |
| ffpp_train | video | Xu | sex | 2722 | 98.2 (0.96) | 43.4 / 44.0 | Male |
| ffpp_train | video | Xu | race2 (White vs Non-White) | 2722 | 76.0 (0.51) | 53.1 / 74.4 | White |
| ffpp_train | video | AIF-v2 | sex | 2722 | 99.3 (0.99) | 43.4 / 43.3 | Male |
| ffpp_train | video | AIF-v2 | race2 (White vs Non-White) | 2722 | 75.4 (0.49) | 53.1 / 75.5 | White |
| ffpp_train | video | AIF-v1 | sex | 2722 | 99.5 (0.99) | 43.4 / 43.2 | Male |
| ffpp_train | video | InsightFace | age2 (<30 vs 30+) | 2722 | 80.6 (0.59) | 60.1 / 60.8 | 0-29 |
| ffpp_train | video | AIF-v1 | age2 FairFace <30 vs AIF-v1 <25 (cut mismatch 30 vs 25) | 2722 | 59.0 (0.26) | 60.0 / 21.7 | <25 |
| ffpp_train | video | AIF-v2 | age2 vs AIF-v2 young / middle+senior ('others' dropped; v2 cut not documented) | 2507 | 85.0 (0.64) | 64.6 / 77.2 | 0-29 |
| ff2pub_test | image:all | FF2 manifest | sex | 93543 | 98.1 (0.96) | 58.4 / 58.3 | Male |
| ff2pub_test | image:all | FF2 manifest | race2 (White vs Non-White) | 93543 | 87.8 (0.70) | 31.8 / 23.4 | White |
| ff2pub_test | image:all | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 93543 | 78.4 (0.35) | 26.0 / 15.0 | 0-29 |
| ff2pub_test | image:real | FF2 manifest | sex | 5441 | 98.7 (0.97) | 59.7 / 59.9 | Male |
| ff2pub_test | image:real | FF2 manifest | race2 (White vs Non-White) | 5441 | 91.8 (0.78) | 26.5 / 21.7 | White |
| ff2pub_test | image:real | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 5441 | 75.7 (0.24) | 27.8 / 9.1 | 0-29 |
| ff2pub_test | image:fake | FF2 manifest | sex | 88102 | 98.0 (0.96) | 58.3 / 58.2 | Male |
| ff2pub_test | image:fake | FF2 manifest | race2 (White vs Non-White) | 88102 | 87.6 (0.69) | 32.2 / 23.5 | White |
| ff2pub_test | image:fake | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 88102 | 78.6 (0.36) | 25.9 / 15.4 | 0-29 |
| ff2pub_test | image:cell-valid | FF2 manifest | sex | 87564 | 98.1 (0.96) | 58.3 / 58.2 | Male |
| ff2pub_test | image:cell-valid | FF2 manifest | race2 (White vs Non-White) | 87564 | 89.2 (0.73) | 31.6 / 24.2 | White |
| ff2pub_test | image:cell-valid | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 87564 | 78.1 (0.34) | 26.3 / 14.6 | 0-29 |
| ff2pub_select | image:all | FF2 manifest | sex | 34001 | 97.7 (0.95) | 59.6 / 59.4 | Male |
| ff2pub_select | image:all | FF2 manifest | race2 (White vs Non-White) | 34001 | 87.3 (0.66) | 28.3 / 20.6 | White |
| ff2pub_select | image:all | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 34001 | 78.3 (0.34) | 25.0 / 16.1 | 0-29 |
| ff2pub_select | image:real | FF2 manifest | sex | 2438 | 98.6 (0.97) | 58.8 / 58.9 | Male |
| ff2pub_select | image:real | FF2 manifest | race2 (White vs Non-White) | 2438 | 92.0 (0.77) | 24.7 / 21.0 | White |
| ff2pub_select | image:real | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 2438 | 76.3 (0.27) | 27.5 / 10.7 | 0-29 |
| ff2pub_select | image:fake | FF2 manifest | sex | 31563 | 97.6 (0.95) | 59.7 / 59.4 | Male |
| ff2pub_select | image:fake | FF2 manifest | race2 (White vs Non-White) | 31563 | 86.9 (0.65) | 28.5 / 20.5 | White |
| ff2pub_select | image:fake | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 31563 | 78.5 (0.35) | 24.8 / 16.5 | 0-29 |
| ff2pub_select | image:cell-valid | FF2 manifest | sex | 31107 | 97.8 (0.95) | 59.7 / 59.4 | Male |
| ff2pub_select | image:cell-valid | FF2 manifest | race2 (White vs Non-White) | 31107 | 89.2 (0.71) | 27.5 / 21.4 | White |
| ff2pub_select | image:cell-valid | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 31107 | 77.9 (0.33) | 25.2 / 15.6 | 0-29 |
| ff2pub_train | image:all | FF2 manifest | sex | 236163 | 97.7 (0.95) | 59.9 / 59.6 | Male |
| ff2pub_train | image:all | FF2 manifest | race2 (White vs Non-White) | 236163 | 88.2 (0.69) | 29.4 / 21.4 | White |
| ff2pub_train | image:all | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 236163 | 78.1 (0.33) | 25.4 / 14.8 | 0-29 |
| ff2pub_train | image:real | FF2 manifest | sex | 16777 | 98.4 (0.97) | 59.3 / 59.6 | Male |
| ff2pub_train | image:real | FF2 manifest | race2 (White vs Non-White) | 16777 | 91.9 (0.78) | 26.4 / 21.4 | White |
| ff2pub_train | image:real | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 16777 | 75.1 (0.24) | 28.5 / 9.9 | 0-29 |
| ff2pub_train | image:fake | FF2 manifest | sex | 219386 | 97.7 (0.95) | 59.9 / 59.6 | Male |
| ff2pub_train | image:fake | FF2 manifest | race2 (White vs Non-White) | 219386 | 87.9 (0.69) | 29.6 / 21.4 | White |
| ff2pub_train | image:fake | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 219386 | 78.4 (0.34) | 25.1 / 15.2 | 0-29 |
| ff2pub_train | image:cell-valid | FF2 manifest | sex | 218266 | 97.8 (0.95) | 59.9 / 59.6 | Male |
| ff2pub_train | image:cell-valid | FF2 manifest | race2 (White vs Non-White) | 218266 | 89.8 (0.73) | 28.9 / 22.2 | White |
| ff2pub_train | image:cell-valid | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 218266 | 77.8 (0.32) | 25.6 / 14.3 | 0-29 |
| ff2pub_cal | image:all | FF2 manifest | sex | 35250 | 97.6 (0.95) | 60.9 / 60.3 | Male |
| ff2pub_cal | image:all | FF2 manifest | race2 (White vs Non-White) | 35250 | 87.6 (0.65) | 26.6 / 18.6 | White |
| ff2pub_cal | image:all | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 35250 | 79.2 (0.35) | 24.0 / 15.2 | 0-29 |
| ff2pub_cal | image:real | FF2 manifest | sex | 2509 | 98.4 (0.97) | 60.9 / 60.6 | Male |
| ff2pub_cal | image:real | FF2 manifest | race2 (White vs Non-White) | 2509 | 91.9 (0.76) | 22.5 / 18.9 | White |
| ff2pub_cal | image:real | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 2509 | 76.0 (0.26) | 27.9 / 10.4 | 0-29 |
| ff2pub_cal | image:fake | FF2 manifest | sex | 32741 | 97.5 (0.95) | 60.9 / 60.3 | Male |
| ff2pub_cal | image:fake | FF2 manifest | race2 (White vs Non-White) | 32741 | 87.2 (0.64) | 26.9 / 18.5 | White |
| ff2pub_cal | image:fake | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 32741 | 79.4 (0.35) | 23.7 / 15.5 | 0-29 |
| ff2pub_cal | image:cell-valid | FF2 manifest | sex | 32255 | 97.6 (0.95) | 61.1 / 60.4 | Male |
| ff2pub_cal | image:cell-valid | FF2 manifest | race2 (White vs Non-White) | 32255 | 89.5 (0.70) | 25.8 / 19.2 | White |
| ff2pub_cal | image:cell-valid | FF2 manifest if_age (InsightFace) | age2 (<30 vs 30+) | 32255 | 78.8 (0.33) | 24.1 / 14.5 | 0-29 |

Positive class: Male (sex), White (race2), 0-29 / <25 (age). Spearman of FairFace expected age:

| set | level | source | n | spearman_expected_age |
|---|---|---|---|---|
| ffpp_test | frame | InsightFace | 25398 | 0.652 |
| ffpp_test | video | InsightFace | 965 | 0.706 |
| dfdc | frame | InsightFace | 21353 | 0.586 |
| dfdc | video | InsightFace | 757 | 0.665 |
| celebdf | frame | InsightFace | 28458 | 0.491 |
| celebdf | video | InsightFace | 956 | 0.599 |
| dfd | frame | InsightFace | 9383 | 0.397 |
| dfd | video | InsightFace | 319 | 0.453 |
| ffpp_val | frame | InsightFace | 25383 | 0.644 |
| ffpp_val | video | InsightFace | 963 | 0.692 |
| ffpp_train | frame | InsightFace | 76139 | 0.661 |
| ffpp_train | video | InsightFace | 2722 | 0.706 |
| ff2pub_test | image:all | FF2 manifest if_age (InsightFace) | 93543 | 0.653 |
| ff2pub_test | image:all | FF2 manifest p_adult | 93543 | 0.631 |
| ff2pub_test | image:real | FF2 manifest if_age (InsightFace) | 5441 | 0.647 |
| ff2pub_test | image:real | FF2 manifest p_adult | 5441 | 0.638 |
| ff2pub_test | image:fake | FF2 manifest if_age (InsightFace) | 88102 | 0.655 |
| ff2pub_test | image:fake | FF2 manifest p_adult | 88102 | 0.631 |
| ff2pub_test | image:cell-valid | FF2 manifest if_age (InsightFace) | 87564 | 0.652 |
| ff2pub_test | image:cell-valid | FF2 manifest p_adult | 87564 | 0.631 |
| ff2pub_select | image:all | FF2 manifest if_age (InsightFace) | 34001 | 0.625 |
| ff2pub_select | image:all | FF2 manifest p_adult | 34001 | 0.612 |
| ff2pub_select | image:real | FF2 manifest if_age (InsightFace) | 2438 | 0.644 |
| ff2pub_select | image:real | FF2 manifest p_adult | 2438 | 0.638 |
| ff2pub_select | image:fake | FF2 manifest if_age (InsightFace) | 31563 | 0.626 |
| ff2pub_select | image:fake | FF2 manifest p_adult | 31563 | 0.610 |
| ff2pub_select | image:cell-valid | FF2 manifest if_age (InsightFace) | 31107 | 0.618 |
| ff2pub_select | image:cell-valid | FF2 manifest p_adult | 31107 | 0.608 |
| ff2pub_train | image:all | FF2 manifest if_age (InsightFace) | 236163 | 0.616 |
| ff2pub_train | image:all | FF2 manifest p_adult | 236163 | 0.613 |
| ff2pub_train | image:real | FF2 manifest if_age (InsightFace) | 16777 | 0.631 |
| ff2pub_train | image:real | FF2 manifest p_adult | 16777 | 0.643 |
| ff2pub_train | image:fake | FF2 manifest if_age (InsightFace) | 219386 | 0.618 |
| ff2pub_train | image:fake | FF2 manifest p_adult | 219386 | 0.610 |
| ff2pub_train | image:cell-valid | FF2 manifest if_age (InsightFace) | 218266 | 0.611 |
| ff2pub_train | image:cell-valid | FF2 manifest p_adult | 218266 | 0.611 |
| ff2pub_cal | image:all | FF2 manifest if_age (InsightFace) | 35250 | 0.620 |
| ff2pub_cal | image:all | FF2 manifest p_adult | 35250 | 0.585 |
| ff2pub_cal | image:real | FF2 manifest if_age (InsightFace) | 2509 | 0.642 |
| ff2pub_cal | image:real | FF2 manifest p_adult | 2509 | 0.621 |
| ff2pub_cal | image:fake | FF2 manifest if_age (InsightFace) | 32741 | 0.621 |
| ff2pub_cal | image:fake | FF2 manifest p_adult | 32741 | 0.582 |
| ff2pub_cal | image:cell-valid | FF2 manifest if_age (InsightFace) | 32255 | 0.612 |
| ff2pub_cal | image:cell-valid | FF2 manifest p_adult | 32255 | 0.582 |

### Per-video (FF2: per-person) consistency of FairFace labels

maj% = mean share of a unit's frames that carry its majority label; unan% = share of units whose frames all agree.

| set | unit | n | frames/unit | sex maj% | age2 maj% | race2 maj% | IDS maj% | sex unan% | age2 unan% | race2 unan% | IDS unan% | frames relabelled by majority % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ffpp_test | all | 965 | 30.0 | 99.5 | 94.5 | 93.3 | 88.2 | 96.1 | 69.7 | 67.3 | 47.4 | 13.5 |
| ffpp_test | real | 267 | 29.0 | 99.7 | 96.2 | 96.1 | 92.5 | 97.4 | 77.9 | 80.5 | 65.5 | 12.9 |
| ffpp_test | fake | 698 | 30.0 | 99.4 | 93.8 | 92.3 | 86.5 | 95.6 | 66.6 | 62.2 | 40.4 | 13.6 |
| dfdc | all | 757 | 30.0 | 93.2 | 87.9 | 90.8 | 77.1 | 67.8 | 45.3 | 57.1 | 24.6 | 23.2 |
| dfdc | real | 473 | 30.0 | 93.3 | 88.2 | 91.2 | 78.0 | 67.9 | 46.1 | 57.9 | 24.7 | 22.3 |
| dfdc | fake | 284 | 30.0 | 93.1 | 87.4 | 90.1 | 75.6 | 67.6 | 44.0 | 55.6 | 24.3 | 24.7 |
| celebdf | all | 956 | 30.0 | 99.5 | 91.7 | 96.6 | 88.4 | 95.4 | 51.3 | 76.3 | 38.0 | 11.7 |
| celebdf | real | 192 | 30.0 | 99.4 | 90.6 | 94.6 | 85.5 | 93.8 | 50.0 | 70.8 | 35.4 | 14.6 |
| celebdf | fake | 764 | 30.0 | 99.6 | 92.0 | 97.1 | 89.1 | 95.8 | 51.6 | 77.6 | 38.6 | 11.0 |
| dfd | all | 319 | 30.0 | 92.5 | 89.2 | 95.1 | 80.1 | 59.9 | 44.8 | 73.4 | 27.6 | 20.3 |
| dfd | real | 104 | 30.0 | 95.5 | 85.1 | 96.5 | 80.0 | 71.2 | 33.7 | 75.0 | 26.0 | 20.1 |
| dfd | fake | 215 | 30.0 | 91.0 | 91.1 | 94.4 | 80.2 | 54.4 | 50.2 | 72.6 | 28.4 | 20.4 |
| ffpp_val | all | 963 | 30.0 | 99.5 | 95.3 | 93.8 | 89.1 | 96.1 | 72.1 | 66.5 | 47.4 | 12.3 |
| ffpp_val | real | 248 | 28.0 | 99.9 | 96.6 | 96.1 | 92.9 | 97.6 | 79.4 | 77.0 | 62.9 | 12.5 |
| ffpp_val | fake | 715 | 30.0 | 99.3 | 94.8 | 93.0 | 87.8 | 95.5 | 69.5 | 62.8 | 42.0 | 12.2 |
| ffpp_train | all | 2722 | 30.0 | 99.5 | 94.8 | 93.1 | 88.1 | 96.7 | 71.0 | 63.9 | 45.2 | 12.7 |
| ffpp_train | real | 602 | 29.0 | 99.6 | 95.7 | 95.2 | 91.1 | 98.2 | 76.6 | 74.9 | 59.5 | 12.3 |
| ffpp_train | fake | 2120 | 30.0 | 99.5 | 94.5 | 92.5 | 87.2 | 96.3 | 69.4 | 60.7 | 41.2 | 12.8 |
| ff2pub_test | persons (real, >=2 images) | 534 | 2.0 | 99.4 | 90.8 | 96.2 | 86.9 | 98.5 | 77.9 | 89.7 | 69.1 | – |
| ff2pub_test | persons (fake, >=2 images) | 4520 | 15.0 | 97.9 | 84.4 | 90.6 | 75.5 | 82.7 | 30.2 | 49.9 | 12.2 | – |
| ff2pub_select | persons (real, >=2 images) | 229 | 2.0 | 99.2 | 90.7 | 95.9 | 86.7 | 98.3 | 77.3 | 88.6 | 67.7 | – |
| ff2pub_select | persons (fake, >=2 images) | 2037 | 14.0 | 97.6 | 84.5 | 89.7 | 74.7 | 80.8 | 30.9 | 47.6 | 12.4 | – |
| ff2pub_train | persons (real, >=2 images) | 1527 | 2.0 | 99.3 | 91.4 | 94.9 | 86.6 | 98.0 | 78.1 | 86.6 | 67.3 | – |
| ff2pub_train | persons (fake, >=2 images) | 13857 | 14.0 | 97.6 | 84.0 | 90.7 | 75.0 | 82.5 | 30.0 | 51.3 | 12.4 | – |
| ff2pub_cal | persons (real, >=2 images) | 246 | 2.0 | 99.4 | 91.1 | 94.9 | 86.3 | 98.4 | 77.2 | 87.0 | 66.7 | – |
| ff2pub_cal | persons (fake, >=2 images) | 2017 | 14.0 | 97.5 | 84.5 | 89.9 | 74.6 | 80.7 | 31.8 | 46.7 | 12.0 | – |

### Group sizes per IDS (main labels)

Cell = frames real/fake · videos real/fake · persons real/fake (FF2: images · persons · persons). '(no FPR)' = fewer than 3 real videos/persons, so the group is excluded from FPR terms under the min-support rule; '(no TPR)' likewise for fakes.

| ids_name | FF++ test | DFDC | Celeb-DF | DFD | FF2-public TEST |
|---|---|---|---|---|---|
| MYW | 149/480 · 7/16 · 7/12 | 1098/541 · 34/19 · 34/19 | 358/60 · 12/2 · 12/1 (no TPR) | 445/1733 · 15/59 · 2/6 | 198/2645 · 163/1171 · 163/1171 |
| FYW | 1245/6802 · 79/228 · 79/135 | 1901/1692 · 60/60 · 60/60 | 3332/20489 · 112/688 · 26/12 | 1489/2781 · 50/95 · 5/8 | 296/5177 · 274/2248 · 274/2248 |
| MAW | 835/4379 · 45/146 · 45/80 | 2137/1351 · 75/47 · 75/47 | 506/0 · 17/0 · 17/0 (no TPR) | 208/327 · 7/11 · 1/5 | 585/11472 · 515/4233 · 515/4233 |
| FAW | 301/957 · 17/32 · 17/23 | 1926/1616 · 63/54 · 63/54 | 447/834 · 15/28 · 7/6 | 30/81 · 1/3 · 1/3 (no FPR) | 361/6947 · 337/2888 · 337/2888 |
| MAN | 816/3689 · 51/123 · 51/76 | 1686/724 · 52/27 · 52/27 | 770/1095 · 26/37 · 5/3 | 0/59 · 0/2 · 0/1 (no FPR) | 1940/27483 · 1585/6081 · 1585/6081 |
| FAN | 150/180 · 9/6 · 9/6 | 1449/320 · 43/11 · 43/11 | 120/30 · 4/1 · 3/1 (no TPR) | 88/88 · 3/3 · 2/2 | 1041/14740 · 909/3886 · 909/3886 |
| MYN | 241/1289 · 15/43 · 15/28 | 1339/360 · 42/11 · 42/11 | 30/117 · 1/4 · 1/2 (no FPR) | 506/955 · 17/33 · 3/5 | 526/6239 · 467/2206 · 467/2206 |
| FYN | 801/3087 · 44/104 · 44/69 | 3163/1554 · 104/55 · 104/55 | 150/120 · 5/4 · 5/4 | 326/269 · 11/9 · 3/4 | 494/7420 · 451/3197 · 451/3197 |

### 8-IDS results, seed means (min-support rule; video-majority / image labels)

| set | threshold | model | seeds | AUC | TPR | FPR | F_FPR | F_DP | es_AUC | fpr_gap | tpr_gap | worst_fpr | F_MEO | F_OAE | elig F/T/B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FF++ test | t05 | CFES-D | 3 | 95.6 | 87.9 | 9.9 | 46.1 | 28.1 | 72.2 | 32.2 | 27.3 | 32.9 | 32.6 | 26.0 | 8/8/8 |
| FF++ test | t05 | arm C | 3 | 95.8 | 79.4 | 3.3 | 29.4 | 32.8 | 71.6 | 19.9 | 32.3 | 20.0 | 32.3 | 22.9 | 8/8/8 |
| FF++ test | t05 | arm E w1 | 1 | 97.0 | 85.3 | 2.8 | 17.3 | 47.6 | 73.7 | 12.7 | 47.6 | 12.7 | 47.6 | 28.9 | 8/8/8 |
| FF++ test | t05 | arm E w3 | 1 | 97.1 | 82.7 | 1.9 | 11.9 | 51.7 | 73.3 | 9.3 | 53.5 | 9.3 | 53.5 | 30.5 | 8/8/8 |
| FF++ test | t05 | arm E w10 | 1 | 96.7 | 65.6 | 0.3 | 2.4 | 43.2 | 72.2 | 1.3 | 46.6 | 1.3 | 46.6 | 19.5 | 8/8/8 |
| FF++ test | t05 | control (det-only) | 3 | 91.5 | 86.8 | 21.1 | 68.8 | 28.1 | 66.0 | 35.2 | 27.2 | 40.1 | 35.6 | 25.5 | 8/8/8 |
| FF++ test | t05 | CFES A (rev4) | 3 | 93.0 | 87.6 | 18.0 | 72.8 | 23.0 | 67.1 | 39.1 | 23.6 | 42.7 | 39.1 | 27.2 | 8/8/8 |
| FF++ test | t05 | CFES-ResNet | 3 | 92.5 | 88.3 | 19.6 | 47.9 | 27.0 | 75.0 | 19.8 | 19.2 | 31.1 | 26.9 | 16.3 | 8/8/8 |
| FF++ test | t05 | Ding repro (SFD+GDA) | 1 | 97.4 | 94.8 | 12.5 | 28.1 | 32.9 | 85.7 | 10.3 | 19.6 | 17.4 | 19.6 | 15.4 | 8/8/8 |
| FF++ test | t05 | Ding Ori repro (SFD) | 3 | 94.1 | 93.8 | 26.3 | 45.9 | 25.6 | 80.1 | 18.5 | 16.1 | 35.0 | 28.2 | 19.0 | 8/8/8 |
| FF++ test | t05 | CESel B0 | 1 | 93.3 | 88.2 | 18.1 | 92.0 | 16.4 | 67.8 | 50.0 | 16.4 | 50.0 | 50.0 | 27.2 | 8/8/8 |
| FF++ test | A1 | CFES-D | 3 | 95.6 | 79.7 | 3.9 | 24.7 | 37.9 | 72.2 | 16.4 | 37.8 | 16.4 | 37.8 | 23.9 | 8/8/8 |
| FF++ test | A1 | arm C | 3 | 95.8 | 79.7 | 3.2 | 30.4 | 32.9 | 71.6 | 21.0 | 32.7 | 21.1 | 32.9 | 23.7 | 8/8/8 |
| FF++ test | A1 | arm E w1 | 1 | 97.0 | 81.6 | 2.1 | 13.3 | 50.9 | 73.7 | 9.3 | 51.5 | 9.3 | 51.5 | 28.8 | 8/8/8 |
| FF++ test | A1 | arm E w3 | 1 | 97.1 | 80.4 | 1.6 | 8.7 | 52.5 | 73.3 | 6.7 | 53.0 | 6.7 | 53.0 | 28.2 | 8/8/8 |
| FF++ test | A1 | arm E w10 | 1 | 96.7 | 80.6 | 2.0 | 10.7 | 50.2 | 72.2 | 7.3 | 53.6 | 7.3 | 53.6 | 29.4 | 8/8/8 |
| FF++ test | A1 | control (det-only) | 3 | 91.5 | 76.8 | 10.0 | 46.5 | 27.5 | 66.0 | 24.1 | 25.9 | 25.6 | 30.3 | 19.3 | 8/8/8 |
| FF++ test | A1 | CFES A (rev4) | 3 | 93.0 | 79.0 | 8.7 | 50.3 | 24.8 | 67.1 | 30.8 | 24.4 | 32.0 | 30.8 | 22.3 | 8/8/8 |
| FF++ test | A1 | CFES-ResNet | 3 | 92.5 | 81.1 | 10.5 | 36.8 | 34.1 | 75.0 | 14.7 | 25.9 | 19.0 | 25.9 | 15.9 | 8/8/8 |
| FF++ test | A1 | Ding repro (SFD+GDA) | 1 | 97.4 | 83.4 | 1.9 | 7.6 | 45.6 | 85.7 | 4.0 | 36.6 | 4.0 | 36.6 | 17.1 | 8/8/8 |
| FF++ test | A1 | Ding Ori repro (SFD) | 3 | 94.1 | 82.6 | 6.8 | 18.5 | 40.9 | 80.1 | 8.0 | 34.5 | 10.2 | 34.7 | 19.0 | 8/8/8 |
| FF++ test | A1 | CESel B0 | 1 | 93.3 | 76.8 | 7.3 | 51.6 | 20.2 | 67.8 | 32.7 | 20.9 | 32.7 | 32.7 | 20.1 | 8/8/8 |
| DFDC | t05 | CFES-D | 3 | 76.9 | 61.2 | 21.8 | 50.2 | 17.7 | 54.6 | 22.7 | 32.4 | 37.8 | 32.4 | 16.2 | 8/8/8 |
| DFDC | t05 | arm C | 3 | 75.1 | 37.3 | 6.1 | 22.3 | 10.8 | 50.1 | 12.4 | 36.6 | 14.4 | 36.6 | 22.8 | 8/8/8 |
| DFDC | t05 | arm E w1 | 1 | 79.2 | 34.4 | 4.3 | 10.8 | 10.1 | 55.3 | 6.6 | 33.4 | 6.8 | 33.4 | 25.5 | 8/8/8 |
| DFDC | t05 | arm E w3 | 1 | 77.9 | 31.1 | 3.7 | 7.4 | 8.6 | 53.1 | 4.8 | 30.6 | 5.2 | 30.6 | 26.3 | 8/8/8 |
| DFDC | t05 | arm E w10 | 1 | 76.4 | 20.5 | 1.1 | 6.8 | 5.0 | 51.8 | 2.7 | 28.0 | 2.8 | 28.0 | 28.8 | 8/8/8 |
| DFDC | t05 | control (det-only) | 3 | 71.7 | 68.5 | 38.1 | 60.0 | 21.7 | 49.9 | 24.5 | 25.1 | 51.7 | 27.0 | 19.4 | 8/8/8 |
| DFDC | t05 | CFES A (rev4) | 3 | 73.2 | 63.9 | 29.9 | 56.1 | 21.1 | 51.1 | 26.4 | 25.5 | 43.6 | 30.7 | 18.2 | 8/8/8 |
| DFDC | t05 | CFES-ResNet | 3 | 58.7 | 46.9 | 35.0 | 51.9 | 16.6 | 37.8 | 25.8 | 29.7 | 44.4 | 29.8 | 20.2 | 8/8/8 |
| DFDC | t05 | Ding repro (SFD+GDA) | 1 | 56.0 | 55.6 | 49.1 | 49.0 | 12.1 | 37.1 | 21.2 | 31.1 | 59.7 | 31.1 | 14.0 | 8/8/8 |
| DFDC | t05 | Ding Ori repro (SFD) | 3 | 54.7 | 62.5 | 56.3 | 52.7 | 16.1 | 36.8 | 23.7 | 23.8 | 65.5 | 26.3 | 18.3 | 8/8/8 |
| DFDC | t05 | CESel B0 | 1 | 76.5 | 58.4 | 21.9 | 21.5 | 10.1 | 54.8 | 11.4 | 27.3 | 28.1 | 27.3 | 16.1 | 8/8/8 |
| DFDC | A1 | CFES-D | 3 | 76.9 | 38.6 | 6.7 | 14.9 | 8.7 | 54.6 | 8.2 | 33.1 | 11.6 | 33.1 | 21.8 | 8/8/8 |
| DFDC | A1 | arm C | 3 | 75.1 | 37.2 | 5.7 | 21.3 | 9.9 | 50.1 | 12.2 | 37.2 | 13.9 | 37.2 | 21.8 | 8/8/8 |
| DFDC | A1 | arm E w1 | 1 | 79.2 | 28.0 | 2.8 | 6.0 | 8.7 | 55.3 | 4.7 | 30.0 | 5.0 | 30.0 | 26.3 | 8/8/8 |
| DFDC | A1 | arm E w3 | 1 | 77.9 | 28.3 | 3.2 | 6.1 | 7.3 | 53.1 | 4.1 | 29.8 | 4.3 | 29.8 | 27.1 | 8/8/8 |
| DFDC | A1 | arm E w10 | 1 | 76.4 | 32.9 | 3.5 | 8.1 | 8.5 | 51.8 | 5.1 | 33.9 | 5.5 | 33.9 | 26.8 | 8/8/8 |
| DFDC | A1 | control (det-only) | 3 | 71.7 | 51.4 | 21.7 | 42.7 | 16.5 | 49.9 | 20.8 | 29.7 | 34.3 | 30.3 | 17.1 | 8/8/8 |
| DFDC | A1 | CFES A (rev4) | 3 | 73.2 | 48.2 | 16.4 | 37.8 | 15.6 | 51.1 | 19.1 | 28.9 | 27.4 | 30.8 | 17.4 | 8/8/8 |
| DFDC | A1 | CFES-ResNet | 3 | 58.7 | 29.9 | 19.5 | 35.2 | 13.5 | 37.8 | 17.5 | 22.6 | 26.7 | 23.1 | 21.5 | 8/8/8 |
| DFDC | A1 | Ding repro (SFD+GDA) | 1 | 56.0 | 14.5 | 7.7 | 18.2 | 6.2 | 37.1 | 8.1 | 19.2 | 12.4 | 19.2 | 30.1 | 8/8/8 |
| DFDC | A1 | Ding Ori repro (SFD) | 3 | 54.7 | 23.0 | 17.1 | 40.9 | 13.9 | 36.8 | 16.6 | 28.5 | 24.4 | 28.5 | 27.7 | 8/8/8 |
| DFDC | A1 | CESel B0 | 1 | 76.5 | 35.5 | 7.2 | 16.0 | 6.8 | 54.8 | 7.1 | 38.1 | 11.7 | 38.1 | 23.3 | 8/8/8 |
| Celeb-DF | t05 | CFES-D | 3 | 87.3 | 90.5 | 42.0 | 50.6 | 32.3 | 69.4 | 25.6 | 27.4 | 57.3 | 30.5 | 22.3 | 7/5/4 |
| Celeb-DF | t05 | arm C | 3 | 89.1 | 75.3 | 14.5 | 36.0 | 39.3 | 72.7 | 21.8 | 39.3 | 29.6 | 39.3 | 21.3 | 7/5/4 |
| Celeb-DF | t05 | arm E w1 | 1 | 87.6 | 75.8 | 17.1 | 39.1 | 34.3 | 70.3 | 31.1 | 30.5 | 38.7 | 31.1 | 19.4 | 7/5/4 |
| Celeb-DF | t05 | arm E w3 | 1 | 86.9 | 75.7 | 16.1 | 34.8 | 37.6 | 71.9 | 22.2 | 33.5 | 26.7 | 33.5 | 21.3 | 7/5/4 |
| Celeb-DF | t05 | arm E w10 | 1 | 84.9 | 43.2 | 4.3 | 17.5 | 22.0 | 68.1 | 8.5 | 14.4 | 9.2 | 14.4 | 18.4 | 7/5/4 |
| Celeb-DF | t05 | control (det-only) | 3 | 81.4 | 89.0 | 49.9 | 105.2 | 45.9 | 66.1 | 48.4 | 48.5 | 77.0 | 56.9 | 28.0 | 7/5/4 |
| Celeb-DF | t05 | CFES A (rev4) | 3 | 85.7 | 91.9 | 49.6 | 101.0 | 40.3 | 68.7 | 42.1 | 39.5 | 73.9 | 50.4 | 28.7 | 7/5/4 |
| Celeb-DF | t05 | CFES-ResNet | 3 | 71.4 | 82.9 | 60.1 | 53.5 | 23.6 | 56.1 | 29.7 | 24.6 | 74.2 | 37.3 | 25.3 | 7/5/4 |
| Celeb-DF | t05 | Ding repro (SFD+GDA) | 1 | 71.2 | 90.5 | 69.6 | 64.6 | 23.2 | 55.5 | 33.4 | 26.7 | 85.3 | 33.4 | 34.6 | 7/5/4 |
| Celeb-DF | t05 | Ding Ori repro (SFD) | 3 | 67.1 | 91.6 | 74.2 | 40.9 | 13.3 | 57.4 | 24.2 | 10.7 | 86.4 | 24.2 | 32.9 | 7/5/4 |
| Celeb-DF | t05 | CESel B0 | 1 | 85.0 | 92.4 | 50.2 | 65.0 | 47.7 | 65.4 | 39.0 | 45.4 | 62.0 | 45.4 | 23.3 | 7/5/4 |
| Celeb-DF | A1 | CFES-D | 3 | 87.3 | 72.9 | 15.3 | 32.7 | 33.5 | 69.4 | 18.3 | 33.8 | 25.3 | 33.8 | 20.9 | 7/5/4 |
| Celeb-DF | A1 | arm C | 3 | 89.1 | 76.1 | 14.3 | 36.9 | 40.1 | 72.7 | 22.9 | 38.6 | 30.4 | 38.6 | 21.1 | 7/5/4 |
| Celeb-DF | A1 | arm E w1 | 1 | 87.6 | 70.9 | 12.2 | 28.1 | 44.3 | 70.3 | 15.5 | 49.0 | 20.7 | 49.0 | 28.8 | 7/5/4 |
| Celeb-DF | A1 | arm E w3 | 1 | 86.9 | 72.6 | 13.9 | 31.0 | 40.1 | 71.9 | 16.9 | 37.4 | 20.7 | 37.4 | 22.7 | 7/5/4 |
| Celeb-DF | A1 | arm E w10 | 1 | 84.9 | 70.5 | 15.5 | 35.7 | 19.3 | 68.1 | 22.9 | 14.5 | 26.7 | 22.9 | 13.4 | 7/5/4 |
| Celeb-DF | A1 | control (det-only) | 3 | 81.4 | 75.2 | 28.9 | 84.2 | 49.0 | 66.1 | 45.9 | 56.1 | 57.5 | 62.0 | 24.4 | 7/5/4 |
| Celeb-DF | A1 | CFES A (rev4) | 3 | 85.7 | 81.5 | 31.9 | 85.8 | 45.2 | 68.7 | 40.0 | 47.8 | 56.1 | 58.5 | 26.9 | 7/5/4 |
| Celeb-DF | A1 | CFES-ResNet | 3 | 71.4 | 72.8 | 41.5 | 53.5 | 30.7 | 56.1 | 31.3 | 34.8 | 58.3 | 40.7 | 19.8 | 7/5/4 |
| Celeb-DF | A1 | Ding repro (SFD+GDA) | 1 | 71.2 | 58.2 | 25.9 | 59.4 | 46.0 | 55.5 | 38.8 | 57.3 | 44.7 | 57.3 | 21.4 | 7/5/4 |
| Celeb-DF | A1 | Ding Ori repro (SFD) | 3 | 67.1 | 68.7 | 40.4 | 56.6 | 36.1 | 57.4 | 37.3 | 44.6 | 59.3 | 46.6 | 20.7 | 7/5/4 |
| Celeb-DF | A1 | CESel B0 | 1 | 85.0 | 74.4 | 19.4 | 32.4 | 43.7 | 65.4 | 14.2 | 56.6 | 23.9 | 56.6 | 27.2 | 7/5/4 |
| DFD | t05 | CFES-D | 3 | 89.9 | 85.9 | 24.6 | 95.1 | 22.7 | 74.7 | 58.6 | 17.6 | 63.4 | 58.6 | 31.1 | 6/7/6 |
| DFD | t05 | arm C | 3 | 90.7 | 73.0 | 7.3 | 31.4 | 18.5 | 78.0 | 19.1 | 34.6 | 19.7 | 34.6 | 9.6 | 6/7/6 |
| DFD | t05 | arm E w1 | 1 | 89.7 | 66.0 | 4.8 | 10.4 | 28.3 | 74.1 | 5.1 | 59.2 | 7.1 | 59.2 | 12.9 | 6/7/6 |
| DFD | t05 | arm E w3 | 1 | 89.7 | 62.7 | 4.5 | 9.7 | 29.7 | 76.1 | 5.7 | 63.5 | 6.1 | 63.5 | 14.2 | 6/7/6 |
| DFD | t05 | arm E w10 | 1 | 90.0 | 50.5 | 1.6 | 5.5 | 26.2 | 77.8 | 3.7 | 59.6 | 3.7 | 59.6 | 19.2 | 6/7/6 |
| DFD | t05 | control (det-only) | 3 | 83.7 | 78.3 | 28.2 | 43.1 | 22.5 | 70.0 | 32.3 | 26.3 | 46.8 | 37.1 | 14.9 | 6/7/6 |
| DFD | t05 | CFES A (rev4) | 3 | 87.0 | 83.1 | 27.1 | 51.5 | 21.5 | 72.8 | 35.1 | 21.5 | 46.8 | 36.3 | 14.3 | 6/7/6 |
| DFD | t05 | CFES-ResNet | 3 | 71.4 | 81.1 | 52.0 | 53.6 | 17.5 | 50.2 | 35.8 | 47.9 | 63.6 | 47.9 | 17.9 | 6/7/6 |
| DFD | t05 | Ding repro (SFD+GDA) | 1 | 72.5 | 83.6 | 59.0 | 57.0 | 22.4 | 42.9 | 31.0 | 54.0 | 75.7 | 54.0 | 39.7 | 6/7/6 |
| DFD | t05 | Ding Ori repro (SFD) | 3 | 66.8 | 83.0 | 63.1 | 59.0 | 19.7 | 40.7 | 33.3 | 32.6 | 80.2 | 39.0 | 31.8 | 6/7/6 |
| DFD | t05 | CESel B0 | 1 | 88.0 | 81.9 | 22.5 | 47.2 | 34.1 | 74.5 | 35.6 | 14.6 | 45.8 | 35.6 | 12.2 | 6/7/6 |
| DFD | A1 | CFES-D | 3 | 89.9 | 75.6 | 9.9 | 50.0 | 19.9 | 74.7 | 30.5 | 31.7 | 31.7 | 45.8 | 16.6 | 6/7/6 |
| DFD | A1 | arm C | 3 | 90.7 | 73.4 | 7.5 | 33.1 | 18.8 | 78.0 | 21.0 | 34.8 | 21.6 | 34.8 | 10.5 | 6/7/6 |
| DFD | A1 | arm E w1 | 1 | 89.7 | 55.3 | 3.5 | 9.2 | 35.1 | 74.1 | 5.5 | 66.6 | 5.5 | 66.6 | 25.2 | 6/7/6 |
| DFD | A1 | arm E w3 | 1 | 89.7 | 59.2 | 3.8 | 10.1 | 33.1 | 76.1 | 6.1 | 65.4 | 6.1 | 65.4 | 19.5 | 6/7/6 |
| DFD | A1 | arm E w10 | 1 | 90.0 | 68.5 | 4.8 | 14.4 | 26.0 | 77.8 | 8.6 | 64.1 | 8.6 | 64.1 | 10.1 | 6/7/6 |
| DFD | A1 | control (det-only) | 3 | 83.7 | 67.0 | 15.1 | 34.3 | 17.6 | 70.0 | 22.8 | 35.5 | 28.9 | 37.2 | 16.2 | 6/7/6 |
| DFD | A1 | CFES A (rev4) | 3 | 87.0 | 74.0 | 16.6 | 37.6 | 20.2 | 72.8 | 22.7 | 29.0 | 28.8 | 33.2 | 14.8 | 6/7/6 |
| DFD | A1 | CFES-ResNet | 3 | 71.4 | 69.1 | 36.0 | 51.7 | 16.8 | 50.2 | 37.6 | 44.3 | 51.5 | 45.4 | 22.2 | 6/7/6 |
| DFD | A1 | Ding repro (SFD+GDA) | 1 | 72.5 | 63.7 | 25.2 | 51.4 | 24.4 | 42.9 | 28.9 | 43.4 | 43.1 | 43.4 | 28.8 | 6/7/6 |
| DFD | A1 | Ding Ori repro (SFD) | 3 | 66.8 | 66.7 | 37.4 | 51.1 | 25.4 | 40.7 | 31.2 | 42.3 | 56.6 | 47.6 | 29.9 | 6/7/6 |
| DFD | A1 | CESel B0 | 1 | 88.0 | 67.6 | 5.2 | 16.2 | 31.6 | 74.5 | 10.3 | 32.9 | 10.3 | 32.9 | 17.0 | 6/7/6 |
| FF2-public TEST | t05 | CFES-D | 3 | 71.0 | 95.0 | 91.0 | 17.4 | 4.5 | 54.8 | 11.0 | 4.5 | 93.5 | 11.0 | 4.3 | 8/8/8 |
| FF2-public TEST | t05 | arm C | 3 | 69.9 | 53.0 | 19.0 | 24.8 | 17.8 | 55.2 | 10.1 | 19.0 | 22.9 | 19.0 | 18.0 | 8/8/8 |
| FF2-public TEST | t05 | arm E w1 | 1 | 76.9 | 51.9 | 10.4 | 12.2 | 14.1 | 62.5 | 6.8 | 15.3 | 13.6 | 15.3 | 14.7 | 8/8/8 |
| FF2-public TEST | t05 | arm E w3 | 1 | 78.5 | 45.1 | 7.4 | 10.1 | 14.3 | 64.4 | 4.7 | 15.6 | 10.1 | 15.6 | 15.1 | 8/8/8 |
| FF2-public TEST | t05 | arm E w10 | 1 | 79.2 | 30.4 | 2.3 | 3.6 | 15.0 | 64.3 | 1.7 | 16.3 | 3.2 | 16.3 | 16.1 | 8/8/8 |
| FF2-public TEST | t05 | control (det-only) | 3 | 62.7 | 62.4 | 46.9 | 41.9 | 14.3 | 51.4 | 18.3 | 14.8 | 53.3 | 18.3 | 13.4 | 8/8/8 |
| FF2-public TEST | t05 | CFES A (rev4) | 3 | 66.6 | 65.4 | 43.3 | 36.6 | 14.5 | 53.9 | 14.6 | 15.2 | 48.3 | 16.8 | 13.9 | 8/8/8 |
| FF2-public TEST | t05 | CFES-ResNet | 3 | 38.5 | 36.3 | 53.8 | 46.9 | 14.9 | 34.9 | 19.1 | 14.7 | 61.2 | 19.1 | 13.3 | 8/8/8 |
| FF2-public TEST | t05 | Ding repro (SFD+GDA) | 1 | 36.4 | 43.3 | 65.5 | 60.9 | 15.7 | 32.5 | 20.4 | 15.5 | 72.1 | 20.4 | 13.9 | 8/8/8 |
| FF2-public TEST | t05 | Ding Ori repro (SFD) | 3 | 37.9 | 48.4 | 65.8 | 54.7 | 18.0 | 33.7 | 20.3 | 17.6 | 72.4 | 21.0 | 15.3 | 8/8/8 |
| FF2-public TEST | t05 | FF2-trained phase 1 | 3 | 97.1 | 96.2 | 20.2 | 24.3 | 3.4 | 93.6 | 9.9 | 2.0 | 25.9 | 9.9 | 2.2 | 8/8/8 |
| FF2-public TEST | t05 | FF2-trained control | 2 | 97.5 | 95.7 | 15.4 | 18.5 | 3.4 | 94.1 | 8.7 | 2.1 | 21.5 | 8.7 | 2.1 | 8/8/8 |
| FF2-public TEST | t05 | FF2-trained CFES | 1 | 97.5 | 89.5 | 3.5 | 16.2 | 4.8 | 93.8 | 6.8 | 3.9 | 8.9 | 6.8 | 3.8 | 8/8/8 |
| FF2-public TEST | A1 | CFES-D | 3 | 71.0 | 69.1 | 45.9 | 25.1 | 13.7 | 54.8 | 12.4 | 14.2 | 50.5 | 18.3 | 13.0 | 8/8/8 |
| FF2-public TEST | A1 | arm C | 3 | 69.9 | 53.2 | 19.0 | 25.0 | 17.9 | 55.2 | 11.6 | 19.1 | 23.3 | 19.1 | 17.9 | 8/8/8 |
| FF2-public TEST | A1 | arm E w1 | 1 | 76.9 | 39.3 | 5.9 | 6.8 | 12.3 | 62.5 | 4.0 | 13.4 | 7.7 | 13.4 | 12.9 | 8/8/8 |
| FF2-public TEST | A1 | arm E w3 | 1 | 78.5 | 41.0 | 5.8 | 7.6 | 13.2 | 64.4 | 3.1 | 14.4 | 7.5 | 14.4 | 14.1 | 8/8/8 |
| FF2-public TEST | A1 | arm E w10 | 1 | 79.2 | 49.0 | 9.2 | 16.1 | 17.6 | 64.3 | 7.3 | 19.1 | 13.8 | 19.1 | 18.4 | 8/8/8 |
| FF2-public TEST | A1 | control (det-only) | 3 | 62.7 | 50.5 | 30.5 | 36.0 | 15.6 | 51.4 | 17.0 | 16.4 | 37.7 | 17.6 | 15.3 | 8/8/8 |
| FF2-public TEST | A1 | CFES A (rev4) | 3 | 66.6 | 54.5 | 28.6 | 24.6 | 14.7 | 53.9 | 10.5 | 15.6 | 33.1 | 15.6 | 14.6 | 8/8/8 |
| FF2-public TEST | A1 | CFES-ResNet | 3 | 38.5 | 23.5 | 37.9 | 37.7 | 11.4 | 34.9 | 17.6 | 11.1 | 44.4 | 17.6 | 10.3 | 8/8/8 |
| FF2-public TEST | A1 | Ding repro (SFD+GDA) | 1 | 36.4 | 20.7 | 38.3 | 41.8 | 13.5 | 32.5 | 18.1 | 13.0 | 45.1 | 18.1 | 11.7 | 8/8/8 |
| FF2-public TEST | A1 | Ding Ori repro (SFD) | 3 | 37.9 | 25.1 | 43.4 | 44.0 | 14.5 | 33.7 | 19.6 | 14.0 | 49.9 | 19.6 | 12.5 | 8/8/8 |

### Sensitivity: Ding's exact rule (all groups holding both classes, no minimum)

| set | threshold | model | seeds | AUC | TPR | FPR | F_FPR | F_DP | es_AUC | fpr_gap | tpr_gap | worst_fpr | F_MEO | F_OAE | elig F/T/B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FF++ test | t05 | CFES-D | 3 | 95.6 | 87.9 | 9.9 | 46.1 | 28.1 | 72.2 | 32.2 | 27.3 | 32.9 | 32.6 | 26.0 | 8/8/8 |
| FF++ test | t05 | arm C | 3 | 95.8 | 79.4 | 3.3 | 29.4 | 32.8 | 71.6 | 19.9 | 32.3 | 20.0 | 32.3 | 22.9 | 8/8/8 |
| FF++ test | t05 | control (det-only) | 3 | 91.5 | 86.8 | 21.1 | 68.8 | 28.1 | 66.0 | 35.2 | 27.2 | 40.1 | 35.6 | 25.5 | 8/8/8 |
| FF++ test | t05 | Ding repro (SFD+GDA) | 1 | 97.4 | 94.8 | 12.5 | 28.1 | 32.9 | 85.7 | 10.3 | 19.6 | 17.4 | 19.6 | 15.4 | 8/8/8 |
| FF++ test | t05 | CESel B0 | 1 | 93.3 | 88.2 | 18.1 | 92.0 | 16.4 | 67.8 | 50.0 | 16.4 | 50.0 | 50.0 | 27.2 | 8/8/8 |
| FF++ test | A1 | CFES-D | 3 | 95.6 | 79.7 | 3.9 | 24.7 | 37.9 | 72.2 | 16.4 | 37.8 | 16.4 | 37.8 | 23.9 | 8/8/8 |
| FF++ test | A1 | arm C | 3 | 95.8 | 79.7 | 3.2 | 30.4 | 32.9 | 71.6 | 21.0 | 32.7 | 21.1 | 32.9 | 23.7 | 8/8/8 |
| FF++ test | A1 | control (det-only) | 3 | 91.5 | 76.8 | 10.0 | 46.5 | 27.5 | 66.0 | 24.1 | 25.9 | 25.6 | 30.3 | 19.3 | 8/8/8 |
| FF++ test | A1 | Ding repro (SFD+GDA) | 1 | 97.4 | 83.4 | 1.9 | 7.6 | 45.6 | 85.7 | 4.0 | 36.6 | 4.0 | 36.6 | 17.1 | 8/8/8 |
| FF++ test | A1 | CESel B0 | 1 | 93.3 | 76.8 | 7.3 | 51.6 | 20.2 | 67.8 | 32.7 | 20.9 | 32.7 | 32.7 | 20.1 | 8/8/8 |
| DFDC | t05 | CFES-D | 3 | 76.9 | 61.2 | 21.8 | 50.2 | 17.7 | 54.6 | 22.7 | 32.4 | 37.8 | 32.4 | 16.2 | 8/8/8 |
| DFDC | t05 | arm C | 3 | 75.1 | 37.3 | 6.1 | 22.3 | 10.8 | 50.1 | 12.4 | 36.6 | 14.4 | 36.6 | 22.8 | 8/8/8 |
| DFDC | t05 | control (det-only) | 3 | 71.7 | 68.5 | 38.1 | 60.0 | 21.7 | 49.9 | 24.5 | 25.1 | 51.7 | 27.0 | 19.4 | 8/8/8 |
| DFDC | t05 | Ding repro (SFD+GDA) | 1 | 56.0 | 55.6 | 49.1 | 49.0 | 12.1 | 37.1 | 21.2 | 31.1 | 59.7 | 31.1 | 14.0 | 8/8/8 |
| DFDC | t05 | CESel B0 | 1 | 76.5 | 58.4 | 21.9 | 21.5 | 10.1 | 54.8 | 11.4 | 27.3 | 28.1 | 27.3 | 16.1 | 8/8/8 |
| DFDC | A1 | CFES-D | 3 | 76.9 | 38.6 | 6.7 | 14.9 | 8.7 | 54.6 | 8.2 | 33.1 | 11.6 | 33.1 | 21.8 | 8/8/8 |
| DFDC | A1 | arm C | 3 | 75.1 | 37.2 | 5.7 | 21.3 | 9.9 | 50.1 | 12.2 | 37.2 | 13.9 | 37.2 | 21.8 | 8/8/8 |
| DFDC | A1 | control (det-only) | 3 | 71.7 | 51.4 | 21.7 | 42.7 | 16.5 | 49.9 | 20.8 | 29.7 | 34.3 | 30.3 | 17.1 | 8/8/8 |
| DFDC | A1 | Ding repro (SFD+GDA) | 1 | 56.0 | 14.5 | 7.7 | 18.2 | 6.2 | 37.1 | 8.1 | 19.2 | 12.4 | 19.2 | 30.1 | 8/8/8 |
| DFDC | A1 | CESel B0 | 1 | 76.5 | 35.5 | 7.2 | 16.0 | 6.8 | 54.8 | 7.1 | 38.1 | 11.7 | 38.1 | 23.3 | 8/8/8 |
| Celeb-DF | t05 | CFES-D | 3 | 87.3 | 90.5 | 42.0 | 56.6 | 39.7 | 60.7 | 25.6 | 30.5 | 57.3 | 32.7 | 31.3 | 7/7/7 |
| Celeb-DF | t05 | arm C | 3 | 89.1 | 75.3 | 14.5 | 40.3 | 45.8 | 58.9 | 21.8 | 48.4 | 29.6 | 48.4 | 28.8 | 7/7/7 |
| Celeb-DF | t05 | control (det-only) | 3 | 81.4 | 89.0 | 49.9 | 106.6 | 46.3 | 47.6 | 52.7 | 49.3 | 77.0 | 62.0 | 49.8 | 7/7/7 |
| Celeb-DF | t05 | Ding repro (SFD+GDA) | 1 | 71.2 | 90.5 | 69.6 | 67.7 | 23.9 | 36.7 | 33.4 | 32.5 | 85.3 | 33.4 | 40.8 | 7/7/7 |
| Celeb-DF | t05 | CESel B0 | 1 | 85.0 | 92.4 | 50.2 | 67.8 | 47.7 | 46.8 | 39.0 | 45.4 | 62.0 | 45.4 | 35.9 | 7/7/7 |
| Celeb-DF | A1 | CFES-D | 3 | 87.3 | 72.9 | 15.3 | 40.0 | 41.5 | 60.7 | 19.2 | 39.4 | 26.2 | 39.4 | 25.5 | 7/7/7 |
| Celeb-DF | A1 | arm C | 3 | 89.1 | 76.1 | 14.3 | 41.1 | 46.3 | 58.9 | 22.9 | 49.3 | 30.4 | 49.3 | 28.1 | 7/7/7 |
| Celeb-DF | A1 | control (det-only) | 3 | 81.4 | 75.2 | 28.9 | 97.1 | 49.0 | 47.6 | 50.3 | 63.7 | 57.5 | 65.6 | 42.9 | 7/7/7 |
| Celeb-DF | A1 | Ding repro (SFD+GDA) | 1 | 71.2 | 58.2 | 25.9 | 74.4 | 46.0 | 36.7 | 38.8 | 77.3 | 44.7 | 77.3 | 67.0 | 7/7/7 |
| Celeb-DF | A1 | CESel B0 | 1 | 85.0 | 74.4 | 19.4 | 45.1 | 48.1 | 46.8 | 26.9 | 63.3 | 36.7 | 63.3 | 38.8 | 7/7/7 |
| DFD | t05 | CFES-D | 3 | 89.9 | 85.9 | 24.6 | 104.1 | 22.7 | 73.2 | 58.9 | 17.6 | 63.4 | 58.9 | 31.1 | 7/7/7 |
| DFD | t05 | arm C | 3 | 90.7 | 73.0 | 7.3 | 38.7 | 28.8 | 77.5 | 19.7 | 34.6 | 19.7 | 34.6 | 27.5 | 7/7/7 |
| DFD | t05 | control (det-only) | 3 | 83.7 | 78.3 | 28.2 | 66.8 | 22.7 | 65.5 | 42.7 | 26.3 | 46.8 | 42.7 | 14.9 | 7/7/7 |
| DFD | t05 | Ding repro (SFD+GDA) | 1 | 72.5 | 83.6 | 59.0 | 79.3 | 44.6 | 37.5 | 39.0 | 54.0 | 75.7 | 54.0 | 39.7 | 7/7/7 |
| DFD | t05 | CESel B0 | 1 | 88.0 | 81.9 | 22.5 | 69.6 | 34.1 | 68.7 | 45.8 | 14.6 | 45.8 | 45.8 | 12.2 | 7/7/7 |
| DFD | A1 | CFES-D | 3 | 89.9 | 75.6 | 9.9 | 55.4 | 26.5 | 73.2 | 30.6 | 31.7 | 31.7 | 45.8 | 29.9 | 7/7/7 |
| DFD | A1 | arm C | 3 | 90.7 | 73.4 | 7.5 | 40.5 | 28.4 | 77.5 | 21.6 | 34.8 | 21.6 | 34.8 | 27.0 | 7/7/7 |
| DFD | A1 | control (det-only) | 3 | 83.7 | 67.0 | 15.1 | 48.3 | 20.4 | 65.5 | 28.3 | 35.5 | 28.9 | 37.5 | 21.0 | 7/7/7 |
| DFD | A1 | Ding repro (SFD+GDA) | 1 | 72.5 | 63.7 | 25.2 | 66.6 | 31.9 | 37.5 | 33.1 | 43.4 | 43.1 | 43.4 | 28.8 | 7/7/7 |
| DFD | A1 | CESel B0 | 1 | 88.0 | 67.6 | 5.2 | 21.4 | 31.6 | 68.7 | 10.3 | 32.9 | 10.3 | 32.9 | 26.6 | 7/7/7 |
| FF2-public TEST | t05 | CFES-D | 3 | 71.0 | 95.0 | 91.0 | 17.4 | 4.5 | 54.8 | 11.0 | 4.5 | 93.5 | 11.0 | 4.3 | 8/8/8 |
| FF2-public TEST | t05 | arm C | 3 | 69.9 | 53.0 | 19.0 | 24.8 | 17.8 | 55.2 | 10.1 | 19.0 | 22.9 | 19.0 | 18.0 | 8/8/8 |
| FF2-public TEST | t05 | control (det-only) | 3 | 62.7 | 62.4 | 46.9 | 41.9 | 14.3 | 51.4 | 18.3 | 14.8 | 53.3 | 18.3 | 13.4 | 8/8/8 |
| FF2-public TEST | t05 | Ding repro (SFD+GDA) | 1 | 36.4 | 43.3 | 65.5 | 60.9 | 15.7 | 32.5 | 20.4 | 15.5 | 72.1 | 20.4 | 13.9 | 8/8/8 |
| FF2-public TEST | A1 | CFES-D | 3 | 71.0 | 69.1 | 45.9 | 25.1 | 13.7 | 54.8 | 12.4 | 14.2 | 50.5 | 18.3 | 13.0 | 8/8/8 |
| FF2-public TEST | A1 | arm C | 3 | 69.9 | 53.2 | 19.0 | 25.0 | 17.9 | 55.2 | 11.6 | 19.1 | 23.3 | 19.1 | 17.9 | 8/8/8 |
| FF2-public TEST | A1 | control (det-only) | 3 | 62.7 | 50.5 | 30.5 | 36.0 | 15.6 | 51.4 | 17.0 | 16.4 | 37.7 | 17.6 | 15.3 | 8/8/8 |
| FF2-public TEST | A1 | Ding repro (SFD+GDA) | 1 | 36.4 | 20.7 | 38.3 | 41.8 | 13.5 | 32.5 | 18.1 | 13.0 | 45.1 | 18.1 | 11.7 | 8/8/8 |

### Sensitivity: frame-level FairFace labels (video sets; min-support)

| set | threshold | model | seeds | AUC | TPR | FPR | F_FPR | F_DP | es_AUC | fpr_gap | tpr_gap | worst_fpr | F_MEO | F_OAE | elig F/T/B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FF++ test | t05 | CFES-D | 3 | 95.6 | 87.9 | 9.9 | 31.0 | 19.3 | 80.7 | 18.3 | 17.2 | 20.2 | 19.0 | 15.1 | 8/8/8 |
| FF++ test | t05 | arm C | 3 | 95.8 | 79.4 | 3.3 | 23.0 | 19.5 | 78.8 | 14.5 | 19.0 | 15.0 | 19.0 | 15.6 | 8/8/8 |
| FF++ test | t05 | control (det-only) | 3 | 91.5 | 86.8 | 21.1 | 53.0 | 17.5 | 71.7 | 26.2 | 16.5 | 33.0 | 26.2 | 16.0 | 8/8/8 |
| FF++ test | t05 | Ding repro (SFD+GDA) | 1 | 97.4 | 94.8 | 12.5 | 23.1 | 20.5 | 89.6 | 8.8 | 13.7 | 17.0 | 13.7 | 9.3 | 8/8/8 |
| FF++ test | t05 | CESel B0 | 1 | 93.3 | 88.2 | 18.1 | 66.0 | 11.9 | 72.6 | 37.7 | 11.9 | 38.8 | 37.7 | 16.1 | 8/8/8 |
| FF++ test | A1 | CFES-D | 3 | 95.6 | 79.7 | 3.9 | 15.0 | 25.3 | 80.7 | 9.0 | 24.3 | 9.9 | 24.3 | 17.5 | 8/8/8 |
| FF++ test | A1 | arm C | 3 | 95.8 | 79.7 | 3.2 | 23.7 | 19.5 | 78.8 | 15.0 | 19.2 | 15.5 | 19.9 | 16.1 | 8/8/8 |
| FF++ test | A1 | control (det-only) | 3 | 91.5 | 76.8 | 10.0 | 38.1 | 17.1 | 71.7 | 18.7 | 17.4 | 20.7 | 19.8 | 14.1 | 8/8/8 |
| FF++ test | A1 | Ding repro (SFD+GDA) | 1 | 97.4 | 83.4 | 1.9 | 7.4 | 26.4 | 89.6 | 3.4 | 21.3 | 3.4 | 21.3 | 13.8 | 8/8/8 |
| FF++ test | A1 | CESel B0 | 1 | 93.3 | 76.8 | 7.3 | 39.7 | 15.1 | 72.6 | 26.0 | 18.0 | 26.4 | 26.0 | 16.9 | 8/8/8 |
| DFDC | t05 | CFES-D | 3 | 76.9 | 61.2 | 21.8 | 50.2 | 20.5 | 55.3 | 21.3 | 28.6 | 35.5 | 28.6 | 15.4 | 8/8/8 |
| DFDC | t05 | arm C | 3 | 75.1 | 37.3 | 6.1 | 24.0 | 11.7 | 51.2 | 10.5 | 26.9 | 12.9 | 26.9 | 22.2 | 8/8/8 |
| DFDC | t05 | control (det-only) | 3 | 71.7 | 68.5 | 38.1 | 64.3 | 24.9 | 49.4 | 27.5 | 23.9 | 51.6 | 27.6 | 20.2 | 8/8/8 |
| DFDC | t05 | Ding repro (SFD+GDA) | 1 | 56.0 | 55.6 | 49.1 | 41.1 | 12.4 | 41.9 | 20.5 | 13.7 | 58.2 | 20.5 | 12.7 | 8/8/8 |
| DFDC | t05 | CESel B0 | 1 | 76.5 | 58.4 | 21.9 | 27.5 | 15.6 | 55.2 | 11.5 | 26.0 | 27.4 | 26.0 | 16.7 | 8/8/8 |
| DFDC | A1 | CFES-D | 3 | 76.9 | 38.6 | 6.7 | 14.2 | 10.5 | 55.3 | 6.9 | 24.3 | 10.9 | 24.3 | 19.8 | 8/8/8 |
| DFDC | A1 | arm C | 3 | 75.1 | 37.2 | 5.7 | 22.3 | 10.3 | 51.2 | 9.9 | 27.3 | 12.1 | 27.3 | 21.9 | 8/8/8 |
| DFDC | A1 | control (det-only) | 3 | 71.7 | 51.4 | 21.7 | 47.6 | 19.6 | 49.4 | 20.9 | 27.7 | 34.2 | 27.9 | 19.3 | 8/8/8 |
| DFDC | A1 | Ding repro (SFD+GDA) | 1 | 56.0 | 14.5 | 7.7 | 19.7 | 8.1 | 41.9 | 6.5 | 9.9 | 10.4 | 9.9 | 24.3 | 8/8/8 |
| DFDC | A1 | CESel B0 | 1 | 76.5 | 35.5 | 7.2 | 19.6 | 8.0 | 55.2 | 8.5 | 26.1 | 12.5 | 26.1 | 23.4 | 8/8/8 |
| Celeb-DF | t05 | CFES-D | 3 | 87.3 | 90.5 | 42.0 | 49.6 | 46.7 | 49.9 | 22.3 | 50.7 | 54.4 | 50.7 | 25.9 | 8/8/8 |
| Celeb-DF | t05 | arm C | 3 | 89.1 | 75.3 | 14.5 | 28.0 | 55.8 | 51.8 | 13.8 | 65.2 | 22.0 | 65.2 | 23.7 | 8/8/8 |
| Celeb-DF | t05 | control (det-only) | 3 | 81.4 | 89.0 | 49.9 | 117.8 | 50.5 | 44.1 | 45.4 | 59.7 | 71.6 | 64.6 | 32.6 | 8/8/8 |
| Celeb-DF | t05 | Ding repro (SFD+GDA) | 1 | 71.2 | 90.5 | 69.6 | 50.6 | 22.6 | 39.2 | 27.1 | 22.3 | 81.8 | 27.1 | 60.0 | 8/8/8 |
| Celeb-DF | t05 | CESel B0 | 1 | 85.0 | 92.4 | 50.2 | 74.8 | 48.6 | 46.0 | 34.1 | 52.2 | 59.9 | 52.2 | 33.8 | 8/8/8 |
| Celeb-DF | A1 | CFES-D | 3 | 87.3 | 72.9 | 15.3 | 28.9 | 50.3 | 49.9 | 12.1 | 54.1 | 21.4 | 54.1 | 19.2 | 8/8/8 |
| Celeb-DF | A1 | arm C | 3 | 89.1 | 76.1 | 14.3 | 28.7 | 57.0 | 51.8 | 15.0 | 60.8 | 22.6 | 60.8 | 24.2 | 8/8/8 |
| Celeb-DF | A1 | control (det-only) | 3 | 81.4 | 75.2 | 28.9 | 87.3 | 50.2 | 44.1 | 37.1 | 55.8 | 48.6 | 58.0 | 31.2 | 8/8/8 |
| Celeb-DF | A1 | Ding repro (SFD+GDA) | 1 | 71.2 | 58.2 | 25.9 | 63.9 | 46.3 | 39.2 | 38.4 | 55.6 | 45.6 | 55.6 | 39.6 | 8/8/8 |
| Celeb-DF | A1 | CESel B0 | 1 | 85.0 | 74.4 | 19.4 | 32.9 | 46.0 | 46.0 | 14.7 | 65.2 | 24.9 | 65.2 | 28.4 | 8/8/8 |
| DFD | t05 | CFES-D | 3 | 89.9 | 85.9 | 24.6 | 112.8 | 32.4 | 70.6 | 65.4 | 20.8 | 71.8 | 65.4 | 30.5 | 8/8/8 |
| DFD | t05 | arm C | 3 | 90.7 | 73.0 | 7.3 | 40.3 | 24.5 | 73.1 | 22.4 | 33.2 | 23.4 | 33.2 | 20.7 | 8/8/8 |
| DFD | t05 | control (det-only) | 3 | 83.7 | 78.3 | 28.2 | 55.5 | 27.3 | 68.3 | 32.8 | 26.4 | 50.5 | 37.5 | 13.6 | 8/8/8 |
| DFD | t05 | Ding repro (SFD+GDA) | 1 | 72.5 | 83.6 | 59.0 | 77.6 | 30.4 | 46.9 | 41.9 | 42.9 | 76.7 | 42.9 | 26.1 | 8/8/8 |
| DFD | t05 | CESel B0 | 1 | 88.0 | 81.9 | 22.5 | 51.9 | 33.5 | 74.0 | 32.3 | 20.0 | 43.3 | 32.3 | 11.3 | 8/8/8 |
| DFD | A1 | CFES-D | 3 | 89.9 | 75.6 | 9.9 | 66.0 | 27.9 | 70.6 | 34.5 | 27.4 | 36.6 | 41.3 | 23.3 | 8/8/8 |
| DFD | A1 | arm C | 3 | 90.7 | 73.4 | 7.5 | 44.2 | 26.2 | 73.1 | 25.3 | 33.1 | 25.9 | 35.3 | 21.5 | 8/8/8 |
| DFD | A1 | control (det-only) | 3 | 83.7 | 67.0 | 15.1 | 39.1 | 25.6 | 68.3 | 21.0 | 31.6 | 30.5 | 34.1 | 17.8 | 8/8/8 |
| DFD | A1 | Ding repro (SFD+GDA) | 1 | 72.5 | 63.7 | 25.2 | 60.0 | 28.5 | 46.9 | 34.9 | 35.3 | 45.0 | 35.3 | 19.0 | 8/8/8 |
| DFD | A1 | CESel B0 | 1 | 88.0 | 67.6 | 5.2 | 24.2 | 28.0 | 74.0 | 8.5 | 26.2 | 9.8 | 26.2 | 19.0 | 8/8/8 |

### Per seed, with video/person-cluster bootstrap 95% intervals (descriptive)

| set | threshold | model | seed | AUC | TPR | FPR | F_FPR [95% CI] | FPR gap [95% CI] | TPR gap [95% CI] | F_DP | es_AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FF++ test | t05 | CFES-D | 7 | 95.9 | 87.7 | 8.7 | 49.0 [24.1, 94.5] | 36.3 [11.4, 72.9] | 26.2 [12.4, 60.9] | 25.1 | 71.9 |
| FF++ test | t05 | CFES-D | 17 | 94.0 | 84.5 | 12.0 | 51.7 [30.6, 91.7] | 34.3 [16.0, 59.7] | 28.4 [13.0, 64.8] | 28.3 | 70.1 |
| FF++ test | t05 | CFES-D | 27 | 97.0 | 91.4 | 9.1 | 37.7 [24.2, 83.9] | 26.1 [10.8, 61.7] | 27.1 [11.8, 55.4] | 31.0 | 74.4 |
| FF++ test | t05 | arm C | 7 | 95.8 | 82.5 | 4.1 | 34.9 [11.9, 73.1] | 25.3 [5.3, 58.3] | 29.8 [17.4, 63.3] | 30.1 | 71.0 |
| FF++ test | t05 | arm C | 17 | 95.3 | 75.8 | 2.5 | 30.9 [9.1, 59.4] | 20.7 [4.0, 46.9] | 29.5 [16.7, 62.4] | 29.9 | 71.0 |
| FF++ test | t05 | arm C | 27 | 96.3 | 80.0 | 3.3 | 22.3 [7.6, 56.0] | 13.6 [3.0, 40.2] | 37.6 [18.4, 65.8] | 38.4 | 72.8 |
| FF++ test | t05 | control (det-only) | 7 | 90.9 | 83.9 | 18.4 | 73.0 [45.1, 118.9] | 37.3 [19.9, 72.0] | 23.4 [15.6, 59.5] | 22.6 | 62.8 |
| FF++ test | t05 | control (det-only) | 17 | 93.0 | 88.9 | 21.4 | 78.9 [51.5, 129.0] | 39.3 [23.8, 77.6] | 27.9 [13.8, 68.0] | 25.4 | 68.7 |
| FF++ test | t05 | control (det-only) | 27 | 90.8 | 87.6 | 23.5 | 54.3 [50.3, 114.9] | 28.8 [23.3, 56.1] | 30.3 [11.9, 70.5] | 36.3 | 66.5 |
| FF++ test | t05 | Ding repro (SFD+GDA) | 7 | 97.4 | 94.8 | 12.5 | 28.1 [19.4, 63.6] | 10.3 [9.0, 33.1] | 19.6 [6.4, 40.4] | 32.9 | 85.7 |
| FF++ test | t05 | CESel B0 | 7 | 93.3 | 88.2 | 18.1 | 92.0 [57.3, 140.9] | 50.0 [25.1, 87.4] | 16.4 [11.1, 54.2] | 16.4 | 67.8 |
| FF++ test | A1 | CFES-D | 7 | 95.9 | 80.0 | 3.6 | 27.6 [11.0, 61.9] | 20.7 [4.9, 48.0] | 34.7 [15.9, 70.2] | 34.0 | 71.9 |
| FF++ test | A1 | CFES-D | 17 | 94.0 | 78.4 | 6.1 | 33.5 [17.8, 63.1] | 20.7 [9.0, 41.1] | 30.7 [14.9, 65.4] | 33.4 | 70.1 |
| FF++ test | A1 | CFES-D | 27 | 97.0 | 80.8 | 2.0 | 12.9 [6.3, 28.1] | 8.0 [2.2, 18.8] | 48.1 [22.7, 74.3] | 46.3 | 74.4 |
| FF++ test | A1 | arm C | 7 | 95.8 | 78.8 | 2.6 | 27.4 [8.0, 64.6] | 21.3 [3.2, 52.8] | 32.2 [19.3, 65.3] | 32.5 | 71.0 |
| FF++ test | A1 | arm C | 17 | 95.3 | 80.5 | 3.8 | 42.3 [14.2, 77.0] | 28.7 [6.1, 59.8] | 27.9 [15.6, 63.2] | 27.4 | 71.0 |
| FF++ test | A1 | arm C | 27 | 96.3 | 79.8 | 3.2 | 21.4 [7.4, 53.2] | 12.9 [2.9, 38.1] | 38.0 [18.3, 66.6] | 38.9 | 72.8 |
| FF++ test | A1 | control (det-only) | 7 | 90.9 | 76.9 | 11.9 | 63.2 [34.1, 106.2] | 33.3 [14.5, 68.1] | 22.6 [18.2, 62.5] | 21.9 | 62.8 |
| FF++ test | A1 | control (det-only) | 17 | 93.0 | 76.3 | 6.0 | 40.3 [16.8, 77.4] | 24.3 [7.9, 55.1] | 21.7 [15.1, 63.8] | 25.6 | 68.7 |
| FF++ test | A1 | control (det-only) | 27 | 90.8 | 77.3 | 12.0 | 36.1 [29.8, 87.7] | 14.8 [13.2, 48.7] | 33.4 [17.1, 74.2] | 35.1 | 66.5 |
| FF++ test | A1 | Ding repro (SFD+GDA) | 7 | 97.4 | 83.4 | 1.9 | 7.6 [5.1, 16.2] | 4.0 [2.3, 9.3] | 36.6 [14.3, 75.0] | 45.6 | 85.7 |
| FF++ test | A1 | CESel B0 | 7 | 93.3 | 76.8 | 7.3 | 51.6 [25.7, 94.4] | 32.7 [10.4, 67.0] | 20.9 [14.9, 57.4] | 20.2 | 67.8 |
| DFDC | t05 | CFES-D | 7 | 75.6 | 58.7 | 21.3 | 55.3 [39.3, 79.3] | 25.6 [17.7, 41.4] | 34.7 [27.6, 59.2] | 21.2 | 52.6 |
| DFDC | t05 | CFES-D | 17 | 77.4 | 63.4 | 23.9 | 40.4 [26.7, 65.3] | 18.7 [12.1, 33.0] | 31.7 [23.9, 51.6] | 14.4 | 55.1 |
| DFDC | t05 | CFES-D | 27 | 77.8 | 61.7 | 20.3 | 54.8 [39.2, 80.9] | 23.9 [17.1, 39.2] | 30.9 [22.0, 51.9] | 17.5 | 56.2 |
| DFDC | t05 | arm C | 7 | 74.5 | 41.9 | 8.8 | 27.7 [21.3, 45.7] | 16.8 [10.6, 25.9] | 38.8 [25.1, 63.6] | 13.1 | 49.3 |
| DFDC | t05 | arm C | 17 | 75.8 | 33.0 | 2.5 | 9.0 [7.1, 19.8] | 5.2 [3.1, 10.5] | 34.4 [24.2, 60.6] | 7.7 | 49.6 |
| DFDC | t05 | arm C | 27 | 74.9 | 36.9 | 7.1 | 30.0 [20.7, 47.1] | 15.3 [9.4, 24.2] | 36.5 [24.7, 63.5] | 11.6 | 51.3 |
| DFDC | t05 | control (det-only) | 7 | 69.2 | 61.4 | 34.7 | 77.9 [58.1, 101.5] | 30.8 [24.4, 45.7] | 32.3 [21.5, 50.9] | 27.0 | 49.4 |
| DFDC | t05 | control (det-only) | 17 | 74.6 | 67.6 | 32.3 | 37.3 [27.9, 67.7] | 18.8 [12.7, 33.6] | 24.6 [20.1, 43.5] | 18.0 | 51.7 |
| DFDC | t05 | control (det-only) | 27 | 71.4 | 76.5 | 47.2 | 64.7 [42.7, 93.7] | 24.0 [18.4, 41.1] | 18.3 [14.5, 49.7] | 20.1 | 48.8 |
| DFDC | t05 | Ding repro (SFD+GDA) | 7 | 56.0 | 55.6 | 49.1 | 49.0 [34.7, 82.8] | 21.2 [15.7, 38.7] | 31.1 [20.2, 55.8] | 12.1 | 37.1 |
| DFDC | t05 | CESel B0 | 7 | 76.5 | 58.4 | 21.9 | 21.5 [17.8, 49.8] | 11.4 [8.3, 24.6] | 27.3 [20.9, 54.0] | 10.1 | 54.8 |
| DFDC | A1 | CFES-D | 7 | 75.6 | 36.0 | 4.3 | 11.7 [8.7, 23.5] | 6.9 [4.2, 12.4] | 32.7 [22.7, 62.3] | 8.4 | 52.6 |
| DFDC | A1 | CFES-D | 17 | 77.4 | 51.3 | 13.5 | 23.0 [18.3, 44.8] | 13.0 [8.5, 22.3] | 36.2 [26.5, 61.6] | 11.9 | 55.1 |
| DFDC | A1 | CFES-D | 27 | 77.8 | 28.5 | 2.4 | 10.1 [8.2, 20.8] | 4.7 [3.3, 10.1] | 30.3 [18.9, 59.3] | 5.9 | 56.2 |
| DFDC | A1 | arm C | 7 | 74.5 | 35.4 | 5.4 | 16.4 [14.0, 31.9] | 11.1 [6.5, 18.1] | 37.3 [23.2, 64.2] | 9.5 | 49.3 |
| DFDC | A1 | arm C | 17 | 75.8 | 39.6 | 5.0 | 18.0 [13.8, 31.7] | 10.8 [6.4, 17.8] | 38.3 [26.4, 62.2] | 8.8 | 49.6 |
| DFDC | A1 | arm C | 27 | 74.9 | 36.6 | 6.9 | 29.3 [20.5, 45.7] | 14.8 [9.1, 23.3] | 36.0 [24.4, 63.1] | 11.3 | 51.3 |
| DFDC | A1 | control (det-only) | 7 | 69.2 | 50.2 | 23.8 | 65.3 [48.6, 87.2] | 27.4 [20.5, 39.8] | 34.5 [22.6, 52.8] | 21.7 | 49.4 |
| DFDC | A1 | control (det-only) | 17 | 74.6 | 44.8 | 12.0 | 13.3 [11.8, 36.7] | 9.5 [5.9, 19.9] | 30.9 [24.3, 54.1] | 9.9 | 51.7 |
| DFDC | A1 | control (det-only) | 27 | 71.4 | 59.1 | 29.3 | 49.4 [36.1, 80.8] | 25.5 [16.8, 40.2] | 23.6 [18.2, 55.9] | 18.0 | 48.8 |
| DFDC | A1 | Ding repro (SFD+GDA) | 7 | 56.0 | 14.5 | 7.7 | 18.2 [12.7, 34.0] | 8.1 [5.7, 17.2] | 19.2 [10.1, 43.8] | 6.2 | 37.1 |
| DFDC | A1 | CESel B0 | 7 | 76.5 | 35.5 | 7.2 | 16.0 [11.0, 31.5] | 7.1 [4.8, 15.4] | 38.1 [24.4, 62.1] | 6.8 | 54.8 |
| Celeb-DF | t05 | CFES-D | 7 | 87.8 | 90.0 | 38.6 | 38.1 [29.0, 100.8] | 17.8 [13.7, 64.2] | 30.0 [22.0, 45.7] | 36.0 | 71.7 |
| Celeb-DF | t05 | CFES-D | 17 | 84.7 | 88.0 | 43.4 | 70.7 [50.1, 124.2] | 31.9 [21.0, 68.5] | 34.3 [24.8, 48.1] | 37.7 | 64.1 |
| Celeb-DF | t05 | CFES-D | 27 | 89.5 | 93.6 | 44.0 | 43.1 [33.7, 100.3] | 27.1 [20.0, 58.4] | 17.8 [10.1, 26.8] | 23.1 | 72.4 |
| Celeb-DF | t05 | arm C | 7 | 88.4 | 79.4 | 19.2 | 42.5 [29.2, 83.9] | 23.0 [14.9, 46.7] | 41.8 [33.3, 61.4] | 42.9 | 72.3 |
| Celeb-DF | t05 | arm C | 17 | 88.7 | 65.2 | 7.3 | 25.7 [13.7, 61.5] | 18.0 [6.8, 48.2] | 36.2 [25.2, 55.4] | 34.0 | 68.5 |
| Celeb-DF | t05 | arm C | 27 | 90.1 | 81.5 | 17.1 | 39.8 [26.7, 107.5] | 24.4 [13.3, 72.7] | 40.0 [29.1, 54.7] | 40.9 | 77.3 |
| Celeb-DF | t05 | control (det-only) | 7 | 81.3 | 86.3 | 42.8 | 52.0 [33.3, 111.5] | 28.9 [16.8, 65.2] | 51.4 [37.2, 66.6] | 45.0 | 66.8 |
| Celeb-DF | t05 | control (det-only) | 17 | 81.7 | 88.8 | 49.2 | 121.5 [82.1, 166.9] | 47.6 [36.5, 79.8] | 50.5 [38.3, 61.4] | 47.4 | 63.7 |
| Celeb-DF | t05 | control (det-only) | 27 | 81.2 | 92.1 | 57.7 | 142.0 [108.9, 188.6] | 68.7 [60.2, 82.4] | 43.6 [34.0, 54.5] | 45.2 | 67.8 |
| Celeb-DF | t05 | Ding repro (SFD+GDA) | 7 | 71.2 | 90.5 | 69.6 | 64.6 [43.1, 110.6] | 33.4 [22.2, 58.8] | 26.7 [15.5, 75.7] | 23.2 | 55.5 |
| Celeb-DF | t05 | CESel B0 | 7 | 85.0 | 92.4 | 50.2 | 65.0 [56.4, 133.4] | 39.0 [29.2, 73.9] | 45.4 [34.2, 56.3] | 47.7 | 65.4 |
| Celeb-DF | A1 | CFES-D | 7 | 87.8 | 72.0 | 12.9 | 25.1 [18.4, 68.2] | 15.3 [9.2, 41.5] | 33.8 [24.9, 58.8] | 34.3 | 71.7 |
| Celeb-DF | A1 | CFES-D | 17 | 84.7 | 75.1 | 23.3 | 48.1 [35.5, 116.8] | 22.1 [16.0, 76.8] | 39.8 [30.8, 64.1] | 38.2 | 64.1 |
| Celeb-DF | A1 | CFES-D | 27 | 89.5 | 71.7 | 9.6 | 24.8 [18.6, 66.2] | 17.5 [10.3, 43.3] | 27.9 [15.4, 43.1] | 28.0 | 72.4 |
| Celeb-DF | A1 | arm C | 7 | 88.4 | 74.4 | 14.0 | 32.8 [25.2, 71.7] | 19.3 [12.9, 39.6] | 42.7 [35.0, 67.0] | 43.2 | 72.3 |
| Celeb-DF | A1 | arm C | 17 | 88.7 | 72.6 | 11.9 | 37.2 [19.5, 83.1] | 24.8 [9.4, 65.6] | 32.8 [24.8, 53.3] | 36.4 | 68.5 |
| Celeb-DF | A1 | arm C | 27 | 90.1 | 81.2 | 16.9 | 40.8 [26.8, 107.7] | 24.5 [13.3, 73.2] | 40.1 [29.6, 56.1] | 40.8 | 77.3 |
| Celeb-DF | A1 | control (det-only) | 7 | 81.3 | 76.6 | 29.0 | 35.6 [27.6, 95.9] | 24.4 [14.0, 60.0] | 55.4 [38.9, 74.1] | 45.7 | 66.8 |
| Celeb-DF | A1 | control (det-only) | 17 | 81.7 | 66.5 | 19.2 | 66.0 [40.3, 116.2] | 38.3 [17.3, 79.2] | 55.7 [44.6, 67.7] | 47.4 | 63.7 |
| Celeb-DF | A1 | control (det-only) | 27 | 81.2 | 82.6 | 38.6 | 150.8 [114.5, 199.8] | 74.9 [65.7, 89.5] | 57.3 [46.9, 70.3] | 53.8 | 67.8 |
| Celeb-DF | A1 | Ding repro (SFD+GDA) | 7 | 71.2 | 58.2 | 25.9 | 59.4 [41.4, 104.8] | 38.8 [23.1, 68.1] | 57.3 [50.2, 80.8] | 46.0 | 55.5 |
| Celeb-DF | A1 | CESel B0 | 7 | 85.0 | 74.4 | 19.4 | 32.4 [26.5, 69.4] | 14.2 [12.0, 36.2] | 56.6 [44.9, 67.1] | 43.7 | 65.4 |
| DFD | t05 | CFES-D | 7 | 89.4 | 84.3 | 21.6 | 100.2 [59.9, 156.1] | 61.8 [44.0, 88.0] | 20.2 [9.8, 60.7] | 23.0 | 73.4 |
| DFD | t05 | CFES-D | 17 | 89.6 | 86.9 | 30.3 | 106.5 [76.3, 161.3] | 68.2 [52.6, 91.6] | 14.6 [10.3, 24.5] | 27.1 | 75.0 |
| DFD | t05 | CFES-D | 27 | 90.7 | 86.4 | 21.9 | 78.7 [48.2, 133.5] | 45.7 [31.7, 83.3] | 18.1 [10.4, 50.5] | 18.1 | 75.8 |
| DFD | t05 | arm C | 7 | 90.4 | 75.5 | 8.4 | 35.1 [20.5, 74.4] | 19.0 [10.6, 49.4] | 31.6 [20.7, 66.6] | 17.5 | 77.5 |
| DFD | t05 | arm C | 17 | 91.3 | 70.2 | 5.1 | 25.3 [14.7, 40.7] | 18.7 [8.1, 28.9] | 48.4 [23.6, 87.2] | 18.2 | 79.4 |
| DFD | t05 | arm C | 27 | 90.3 | 73.4 | 8.4 | 33.6 [19.4, 76.9] | 19.5 [9.3, 59.5] | 23.7 [14.1, 74.5] | 19.9 | 76.9 |
| DFD | t05 | control (det-only) | 7 | 83.0 | 75.5 | 26.5 | 57.7 [41.2, 123.0] | 42.4 [26.4, 81.9] | 24.5 [16.7, 49.6] | 19.7 | 69.0 |
| DFD | t05 | control (det-only) | 17 | 86.2 | 78.2 | 22.1 | 41.2 [32.3, 78.2] | 37.6 [22.2, 54.2] | 23.3 [15.6, 42.6] | 19.9 | 75.2 |
| DFD | t05 | control (det-only) | 27 | 81.9 | 81.4 | 35.9 | 30.3 [29.3, 113.2] | 16.8 [16.1, 72.6] | 31.2 [16.3, 84.5] | 27.7 | 65.7 |
| DFD | t05 | Ding repro (SFD+GDA) | 7 | 72.5 | 83.6 | 59.0 | 57.0 [33.8, 107.6] | 31.0 [20.2, 63.4] | 54.0 [25.2, 88.5] | 22.4 | 42.9 |
| DFD | t05 | CESel B0 | 7 | 88.0 | 81.9 | 22.5 | 47.2 [30.3, 82.3] | 35.6 [19.6, 54.7] | 14.6 [12.7, 34.5] | 34.1 | 74.5 |
| DFD | A1 | CFES-D | 7 | 89.4 | 74.8 | 7.1 | 47.5 [19.3, 90.5] | 24.5 [11.2, 63.3] | 28.6 [13.9, 72.2] | 16.8 | 73.4 |
| DFD | A1 | CFES-D | 17 | 89.6 | 82.3 | 17.8 | 85.5 [59.9, 129.8] | 55.8 [40.0, 71.6] | 13.6 [10.4, 28.3] | 19.0 | 75.0 |
| DFD | A1 | CFES-D | 27 | 90.7 | 69.6 | 4.7 | 17.0 [8.5, 32.2] | 11.0 [4.6, 21.5] | 53.0 [26.7, 78.6] | 23.9 | 75.8 |
| DFD | A1 | arm C | 7 | 90.4 | 72.4 | 6.4 | 25.9 [14.2, 54.2] | 13.8 [7.3, 36.1] | 39.7 [26.6, 65.8] | 20.6 | 77.5 |
| DFD | A1 | arm C | 17 | 91.3 | 74.6 | 7.7 | 40.1 [25.8, 62.8] | 29.8 [15.9, 43.2] | 41.2 [13.6, 85.6] | 15.9 | 79.4 |
| DFD | A1 | arm C | 27 | 90.3 | 73.2 | 8.3 | 33.1 [19.0, 76.9] | 19.5 [9.0, 59.5] | 23.5 [14.1, 74.5] | 20.0 | 76.9 |
| DFD | A1 | control (det-only) | 7 | 83.0 | 67.5 | 17.1 | 47.7 [32.0, 102.1] | 35.2 [20.0, 65.4] | 29.9 [21.5, 73.6] | 17.5 | 69.0 |
| DFD | A1 | control (det-only) | 17 | 86.2 | 63.8 | 8.2 | 30.8 [16.9, 52.0] | 17.3 [10.1, 28.9] | 45.5 [27.7, 72.5] | 19.4 | 75.2 |
| DFD | A1 | control (det-only) | 27 | 81.9 | 69.6 | 20.1 | 24.3 [16.4, 105.2] | 16.0 [10.0, 83.2] | 31.0 [18.4, 80.7] | 15.7 | 65.7 |
| DFD | A1 | Ding repro (SFD+GDA) | 7 | 72.5 | 63.7 | 25.2 | 51.4 [31.5, 86.2] | 28.9 [18.5, 52.2] | 43.4 [26.2, 86.8] | 24.4 | 42.9 |
| DFD | A1 | CESel B0 | 7 | 88.0 | 67.6 | 5.2 | 16.2 [10.5, 31.6] | 10.3 [6.5, 19.2] | 32.9 [21.0, 72.3] | 31.6 | 74.5 |
| FF2-public TEST | t05 | CFES-D | 7 | 72.9 | 94.5 | 88.6 | 17.0 [12.4, 27.7] | 10.6 [5.8, 16.5] | 3.4 [2.8, 4.7] | 3.3 | 56.3 |
| FF2-public TEST | t05 | CFES-D | 17 | 67.3 | 95.1 | 94.6 | 16.9 [11.7, 24.6] | 11.1 [6.9, 16.7] | 4.3 [3.1, 5.7] | 4.6 | 52.5 |
| FF2-public TEST | t05 | CFES-D | 27 | 72.8 | 95.3 | 89.9 | 18.4 [13.3, 28.2] | 11.3 [6.4, 18.2] | 5.8 [5.0, 7.1] | 5.6 | 55.6 |
| FF2-public TEST | t05 | arm C | 7 | 74.1 | 60.3 | 21.0 | 23.3 [14.9, 34.4] | 8.7 [6.5, 15.1] | 14.7 [12.7, 17.0] | 13.4 | 59.9 |
| FF2-public TEST | t05 | arm C | 17 | 68.7 | 45.7 | 12.4 | 17.7 [12.2, 27.1] | 8.2 [5.7, 13.2] | 20.8 [18.7, 22.8] | 19.4 | 55.0 |
| FF2-public TEST | t05 | arm C | 27 | 66.9 | 52.9 | 23.5 | 33.5 [26.4, 46.1] | 13.4 [11.0, 19.6] | 21.6 [19.7, 24.1] | 20.6 | 50.7 |
| FF2-public TEST | t05 | control (det-only) | 7 | 64.3 | 59.0 | 39.3 | 37.8 [26.4, 51.9] | 16.8 [12.3, 23.9] | 14.4 [12.2, 16.5] | 13.7 | 53.0 |
| FF2-public TEST | t05 | control (det-only) | 17 | 64.6 | 63.6 | 46.2 | 45.6 [31.5, 59.8] | 18.1 [12.1, 26.8] | 13.3 [11.9, 15.9] | 12.8 | 53.8 |
| FF2-public TEST | t05 | control (det-only) | 27 | 59.2 | 64.4 | 55.3 | 42.2 [31.8, 55.4] | 20.0 [13.6, 28.3] | 16.9 [14.7, 19.2] | 16.3 | 47.3 |
| FF2-public TEST | t05 | Ding repro (SFD+GDA) | 7 | 36.4 | 43.3 | 65.5 | 60.9 [48.6, 75.4] | 20.4 [16.8, 28.2] | 15.5 [14.2, 18.3] | 15.7 | 32.5 |
| FF2-public TEST | A1 | CFES-D | 7 | 72.9 | 69.6 | 41.4 | 30.4 [20.6, 44.0] | 11.4 [9.0, 20.8] | 13.0 [10.9, 15.1] | 12.1 | 56.3 |
| FF2-public TEST | A1 | CFES-D | 17 | 67.3 | 88.5 | 84.8 | 27.5 [21.6, 39.8] | 19.3 [12.6, 26.7] | 7.2 [5.6, 9.2] | 7.8 | 52.5 |
| FF2-public TEST | A1 | CFES-D | 27 | 72.8 | 49.4 | 11.5 | 17.5 [12.5, 27.3] | 6.6 [5.0, 12.0] | 22.5 [20.4, 24.8] | 21.0 | 55.6 |
| FF2-public TEST | A1 | arm C | 7 | 74.1 | 55.9 | 15.1 | 17.4 [11.5, 28.3] | 8.3 [5.6, 13.5] | 14.9 [13.3, 17.5] | 13.6 | 59.9 |
| FF2-public TEST | A1 | arm C | 17 | 68.7 | 51.1 | 18.8 | 24.4 [18.5, 35.6] | 13.3 [8.7, 18.4] | 20.7 [18.6, 22.9] | 19.6 | 55.0 |
| FF2-public TEST | A1 | arm C | 27 | 66.9 | 52.5 | 23.1 | 33.2 [25.7, 45.1] | 13.1 [10.8, 19.3] | 21.7 [19.8, 24.1] | 20.7 | 50.7 |
| FF2-public TEST | A1 | control (det-only) | 7 | 64.3 | 51.7 | 29.7 | 34.5 [25.4, 47.8] | 17.9 [12.7, 23.9] | 14.6 [12.5, 16.7] | 14.0 | 53.0 |
| FF2-public TEST | A1 | control (det-only) | 17 | 64.6 | 47.1 | 23.7 | 30.5 [22.1, 43.2] | 13.9 [10.0, 20.0] | 15.8 [14.4, 18.3] | 15.0 | 53.8 |
| FF2-public TEST | A1 | control (det-only) | 27 | 59.2 | 52.8 | 38.2 | 43.1 [33.0, 56.5] | 19.1 [13.2, 26.7] | 18.7 [16.4, 20.9] | 17.9 | 47.3 |
| FF2-public TEST | A1 | Ding repro (SFD+GDA) | 7 | 36.4 | 20.7 | 38.3 | 41.8 [32.5, 56.6] | 18.1 [13.5, 24.7] | 13.0 [11.2, 14.9] | 13.5 | 32.5 |
