# FairForge-2: dataset and research-release inventory

Verified 8 November 2026. This is an inventory of actual artifacts, not a new experiment, a completed human study, or confirmation that the dataset is publicly downloadable. No training, inference, image generation, threshold change or access-setting change occurred.

**Answer:** FairForge-2 Public (FF2-Public) is our proposed image-based fairness benchmark for face-forgery detection. Most dataset components exist, but the complete publicly reproducible research release is not finished. Dataset assets and proposed CES-FL model artifacts are separate contributions. A dataset paper does not inherently require a new architecture; it needs clear data, annotation, protocol, baseline and reproducibility evidence. A method paper additionally needs evidence that the method improves its prespecified outcomes.

## 1. Exact dataset/version to describe

Use the frozen **FF2-Public 398,957-image** release (25 October 2026 freeze,26 October metadata amendment). Do not describe the old 80,280-image 12-arm v1.0 dataset card, the internal 837,220-image build, the RFW-based older hf_release folder, and this public candidate as one interchangeable version.

Freshly scanned `manifests/public_manifest.csv`:

- 398,957 image rows: **27,165 real /371,792 fake**; only 6.81% are real, so this is not a class-balanced dataset.
- 256×256 RGB JPEG face crops as documented. A fresh eight-item TRAIN-only spot check (real plus one per fake family) decoded 8/8 and matched that format. This does not verify every image anew.
- **34 generator labels /31 distinct generator models**, seven fake families. Labels such as swap/simswap can share a model; do not call these 34 independent architectures.
- **22,454 recorded real-person IDs**, including ArcFace-grouped FairFace pseudo-persons. They are not 22,454 independently verified human identities; synthetic GAN/T2I faces should not be counted as actual people.
- Sources: FFHQ4,641 real +80,755 fake; FairFace 22,524 real +291,037 fake. RFW-derived images and InSwapper outputs are excluded from this release candidate according to the recorded release decisions.
- Every row has a unique item ID, relative image path and recorded image SHA256. This task verified uniqueness of the manifest fields, not rehashed all 398,957 image files.
- Fresh manifest SHA256: `21f1b206fdd4ac0e838542b268de79ff103268fb888cbacda10b609a3ebd5d2a`.

| Fake family | Images | Example generators |
|---|---:|---|
| Face swap |82,608|SimSwap, BlendFace, REFace|
| Reenactment |40,581|LivePortrait, TPSMM|
| Identity-preserving generation |33,370|Arc2Face, InstantID, IP-Adapter, PhotoMaker, PuLID|
| Editing |78,805|IC-Light, inpainting, MagicBrush, OmniGen, Qwen editing|
| Restoration/super-resolution |77,197|CodeFormer, DifFace, GFPGAN, Real-ESRGAN|
| Text-to-image |41,498|Stable Cascade, FLUX variants, HART, Infinity, Kolors, PixArt, Playground|
| GAN |17,733|EG3D, Projected GAN, StyleGAN2, StyleSwin|

These existing benchmark fake-generation methods are distinct from the later Qwen counterfactual training candidates. Having Qwen-generated benchmark examples does not validate any demographic intervention bank.

## 2. Your release checklist

| Release component | Verified status | What exists and what remains |
|---|---|---|
| Dataset |Present locally; public access incomplete/unverified|Images and frozen manifest exist; historical upload verification records 20 WebDataset shards. Dataset GitHub is private; live Hugging Face query returned 401. No publicly verified download release in this audit.|
| Annotations |Present, with important validation limits|Binary labels, generator/family, demographic groups, provenance, source/donor links, splits, image hashes, adult-gate values; automatic age/sex/race and skin-tone estimates. No released pixel forgery masks or standard bbox columns were found. Human validation and human MST remain empty.|
| Dataset documentation |Substantial draft, unfinished release details|README, datasheet, freeze record, statistics, licence notices/evidence, DUA, takedown policy, human protocol and generator table. Author/contact/citation/approval placeholders and documented release blockers remain. Some root-level cards are stale.|
| Benchmark |Fixed split definitions present|TRAIN/SELECT/CAL/TEST, official cell-valid Ding-schema files and all-image pooled files; seen/unseen model metadata. Recorded ID separation is verifiable; true-person/hidden-donor independence is not guaranteed.|
| Baselines |Partial|Stored 80-detector public-subset benchmark and historical CFES-family results exist. This is not 80 matched public-TRAIN retrains. No complete matched public-split Ding/DAW/DAG/PG/CES-FL counterfactual leaderboard exists in the inspected evidence.|
| Proposed model |Implementation present; empirical validation incomplete|CES-FL code, objectives and configurations exist in the method repository. Current public-FF2 checkpoint is original-only BCE pretraining; the complete proposed counterfactual method has not been established on FF2.|
| Evaluation code |Present in research repository; not a portable complete dataset benchmark package|Ding-style metrics, AUC, FPR/FNR, group metrics, fixed/calibrated reporting, family summaries and uncertainty code exist. Dataset-only GitHub lacks the complete detector evaluation harness. Standardize and document protocols before a common leaderboard.|
| Training code |Present in method repository; release integration incomplete|CES-FL executable trainer and exact seed 17FF2 config exist. Dataset GitHub mainly ships construction/annotation/limited regeneration scripts; build scripts still require unshipped internal inputs. Portable paths, dependency lock, commands and per-baseline configurations remain to curate.|
| Model checkpoints |Present locally; not distributed with dataset repository|Verified FF2-trained Xception CES-FL BCE checkpoint exists, plus historical local artifacts. Dataset GitHub has no.pt/.pth/.safetensors/.onnxweights. No public final counterfactual model checkpoint is established.|

Masks and boxes are not required for a whole-image binary classification/fairness benchmark. If a paper claims pixel-localization accuracy, it needs genuine independent spatial ground truth; CES-FL signed contribution maps cannot be substituted for it.

## 3. Exact benchmark splits

Freshly counted both the master manifest and the split CSVs:

| Split | All release images | Real | Fake | Official group-valid/Ding rows |
|---|---:|---:|---:|---:|
|TRAIN|236,163|16,777|219,386|218,266|
|SELECT(model selection)|34,001|2,438|31,563|31,107|
|CAL(threshold calibration)|35,250|2,509|32,741|32,255|
|TEST|93,543|5,441|88,102|87,564|
|Total|398,957|27,165|371,792|369,192|

**Why the counts differ:**29,765 FairFace-derived T2I/identity-conditioned fakes have unconfirmed inherited group labels. They remain available for pooled detection analyses with `use_in_cell_metrics=False`, and must be excluded from per-group fairness metrics. Official `manifests/ding/` files omit them; `ding_pooled/` retains them. Do not silently mix these populations.

Current CES-FL adapter uses 218,266TRAIN/31,107VAL(SELECT)/87,564TEST, totaling 336,937 rows. Its separate CAL32,255 rows is not an input to the existing pilot. These are the public group-valid subset, not the whole 398,957-image dataset.

Release protocol: select checkpoint/hyperparameters on SELECT, calibrate a global threshold on CAL, then freeze it before TEST; supplementary Ding-style probability 0.5 results are separately defined. **Existing CES-FL pilot instead calibrated on SELECT/validation real images.** That is a documented experiment-specific difference, not a test-leakage claim or permission to overwrite its stored threshold. A dataset-standard comparison needs an explicit future version/config with the declared CAL protocol.

Fresh scan found zero cross-split overlap in recorded real `person_id` sets. Prior stats also record donor/driver and pixel/caption link checks. Pseudo-identities, hidden contributors, near-duplicates and external benchmark overlap remain limitations; no claim of independently confirmed true-person disjointness is made.

## 4. Annotation content and reliability

Main manifest has 32 columns, including `rel_path,label,method,family,generator_model,src_dataset,item_id,person_id,donor_person_id,source_real_rel_path,split,cell_native,race_native,gender_native,intersec_label,group,use_in_cell_metrics,cell_label_rule,p_adult,if_age,sha256`.

- **Authenticity/generator:** construction labels,0 real/1 fake. Label 1 includes AIrestoration and other processing as defined by the benchmark; it does not mean every item is a deceptive identity deepfake. Report forgery family and define the task.
- **Gender/race:** perceived labels, inherited/crowd/classifier-derived with provenance. Eight Ding groups: Male/Female×Asian/Black/White/Others. Race is not skin tone. FairFace pseudo-person labels use component aggregation, not self-identification. Generated faces can fail to render the source label.
- **Age:** FairFace 9-bin/4-bin/2-bin estimates plus InsightFace apparent age. All 398,957 have FairFace age and InsightFace age fields. These two models and bin definitions are distinct; declare the chosen axis. Passing an apparent-adult gate is not proof of true age.
- **Skin tone:** automatic ITA and variants.374,864/398,957 have ITA(93.961%);24,093 are missing. Four bins use fixed TRAIN-real quartiles; they are relative pixel-colour bins, affected by illumination/white balance/exposure, not human Monk categories or dermatological truth.
- **Human review:** all 398,957 rows have `human_validated=False`; all `human_mst` values are empty. Ten prepared rater sheets have 9,206 presentations and no filled race/gender/age/MST answers. Synthetic rater fixtures are software tests, not actual annotations. A human protocol/kit exists; this inventory cannot treat it as completed review.
- **Spatial labels:** no mask/bbox/landmark coordinates in the released master schema. Five-landmark alignment is preprocessing; it is not a shipped localization target. Face ROI detection modes and confidence values are diagnostic metadata.

Independent reviewers still need to resolve label quality, uncertain/minor flags, annotation agreement and the semantic scope of authenticity. The documented checklist flags 112 automatic very-young predictions for manual review; classifier predictions alone neither establish their true ages nor certify the dataset is minor-free. This audit did not adjudicate those images.

## 5. Baselines and evaluation evidence

`release_ff2_public/stats/benchmarks/panel_public.csv` contains 80 detector rows; stored aggregate metadata gives best panel AUC87.33%(c 2 pclip_ufd) on the public group-valid TEST population. These are existing scores filtered/recomputed for the public subset, not 80 fresh end-to-end in-domain trainings or a new result from this audit. The wider per-seed/subset CSV contains 672 rows across different training populations and subsets, not 672 independent models.

Historical public CFES-D seed 7TEST AUC98.2598% belongs to the older CLIP implementation and a different selection/calibration protocol. Older CFES models trained on the internal/full dataset and evaluated on public TEST remain labeled that way. The historical Ding 99.3573%AUC is on the internal 182,884-row TEST, not the current public 87,564-row benchmark. FF++→FF2 transfer is another separate experiment. None substitutes for matched public-only baselines.

The current FF2 original-only BCE/Xception checkpoint completed 10 epochs, selected 9, on public TRAIN/SELECT:

| Saved validation metric | Value |
|---|---:|
|AUC|99.810536%|
|Global FPR|9.967186%(243/2,438 real)|
|Global FNR|0.380202%|
|TEST evaluated|False|

Its saved native independent validation replay passed. This task freshly rehashed/CPU-loaded the checkpoint, verified selected epoch, finite tensors and 10 history records. It did not rerun full inference, compute new TEST metrics, establish a fairness gain, or launch training. The dataset remains strongly imbalanced, so AUC and per-class/group errors are more informative than accuracy alone.

## 6. Exact files and download/access links

**Dataset metadata/code GitHub:** https://github.com/hardikbha/FairForge2 — current authenticated API verified `private=true`,125 files, commit `c452b7a5ccd71c7628429ac86d95fd29338fdd91`. No image JPEGs or model weights in that tree. A collaborator with access can read it; unauthenticated 404 does not mean the repository is absent.

**Image host:** https://huggingface.co/datasets/hardiksharma6555/FairForge2 — historical `hf_verify.json` records 87 remote files,20 shards totaling 10,650,603,520 bytes(~10.65GB decimal), no missing/size/SHA mismatches and private=true/gated=false at that observation. **Live 8 November metadata request returned 401 RepositoryNotFoundError**, so current existence/access/revision cannot be confirmed with the credentials available to this process. Do not promise public downloading or claim it is currently gated/open. This task did not modify access settings or re-upload data. The old temporary staging directory has 0 of 20 tarshards present now; original TRAINimage samples still exist and decode. Historical upload receipts cannot establish current remote integrity.

**Method repository:** https://github.com/hardikbha/CFES-BeatDing — current GitHub API verifies public. CES_FL contains the trainer/model/config/results, while DEMO_AXES contains demographic estimation code. A dataset paper/release should link these precisely, not imply the dataset-only repository ships complete model training/evaluation.

Local authoritative release root:
`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/release_ff2_public/`

| Artifact | Absolute path |
|---|---|
|Full manifest|`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/release_ff2_public/manifests/public_manifest.csv`|
|Current datasheet|`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/release_ff2_public/docs/DATASHEET.md`|
|Release README|`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/release_ff2_public/docs/README.md`|
|Official split CSVs|`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/release_ff2_public/manifests/ding/`|
|Automatic annotations|`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/release_ff2_public/annotations/ff2_public_auto_annotations.csv.gz`|
|Annotation dictionary|`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/release_ff2_public/annotations/README_auto_annotations.md`|
|Human protocol|`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/release_ff2_public/docs/HUMAN_VALIDATION_PROTOCOL.md`|
|Existing baseline summaries|`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/release_ff2_public/stats/benchmarks/`|
|Benchmark recomputation script|`/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/release_ff2_public/recompute_benchmarks.py`|
|CES-FL model/trainer/evaluator|`/home/sachin.chaudhary/ces_fl_sources/ff2_p_s17_20261102_875bce0/CES_FL/src/ces_fl_diagram_model_v021.py`|
|Exact executed FF2 config|`/home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1/training/config.json`|
|Selected detector checkpoint|`/home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1/training/best.pt`|
|Checkpoint verification|`/home/sachin.chaudhary/ces_fl_runs/ff2_originals_s17_20261102_r1/training/RUN_VERIFIED.json`|

Selected checkpoint fresh SHA256:
`43152de1b98dce48801114efb2454cb0bc6aa7400445dd5f139759022229895a`.

It is a trained detector checkpoint containing model/config/optimizer/calibration state,259,577,189 bytes (247.55 MiB). The **ImageNet initializer** and the many **generator weights** used to build fakes are different artifacts. There is no need to redistribute every generator model for a detection benchmark; link original models and their terms. No public weights download URL is claimed for this local detector checkpoint. No image/weights file was uploaded by this task.

## 7. What is left before calling it a complete release

The on-disk checklist documents unresolved human validation, ethics/IRB, FairFace provenance/licence review, maintainer/takedown contacts, attribution/licence layering, and citation placeholders. This report records those project statuses; it does not independently resolve legal permissions or ethics decisions. Dataset GitHub citation still contains author/date/repository placeholders. Some datasheet wording says 32 models in its motivation while the actual frozen composition has 31; use the verified number and correct documentation through a separately reviewed change.

Historical checklist claims that nothing was uploaded are superseded by the later private upload receipt; current access is nevertheless unverified. Do not use old snapshots as evidence of current hosting. Do not treat unfilled author/contact placeholders as operational infrastructure.

The practical release work is: finish independent human validation/adjudication; resolve recorded access/attribution/approval items; publish a consistent dataset card/version/split/threshold protocol; verify the live host under the intended access policy; provide portable environment-pinned training/evaluation commands; curate comparable public-only baselines; and distribute an appropriately documented detector checkpoint/model card if desired. The complete CES-FL counterfactual model and matched Ding comparison retain their experimental gates. No further FairForge-2 training is authorized by this inventory request.

## 8. Evidence and reproducibility of this inventory

Fresh CPU-only commands:

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python -u /home/sachin.chaudhary/cesel_ma_audit_logs/fairforge2_release_inventory_20261108T141615Z/inspect_release_metadata.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /home/sachin.chaudhary/.conda/envs/qwen38/bin/python -u /home/sachin.chaudhary/cesel_ma_audit_logs/fairforge2_release_inventory_20261108T141615Z/inspect_model_and_samples.py
gh api repos/hardikbha/FairForge2 --jq '{full_name,private,html_url,default_branch,pushed_at}'
gh api repos/hardikbha/FairForge2/git/trees/main?recursive=1 --jq '{truncated,files:[.tree[] | select(.type=="blob") | .path]}'
```

Actual outcome: metadata scan 398,957 master+398,957 annotation rows;8/8TRAINsample decodes; checkpoint hash/finite tensors/history checks PASS; GitHub dataset metadata/tree accessible with authorized account and private; HuggingFace metadata 401(unverified). Existing test-score files were not opened; split TESTmetadata was counted only. No all-image redecoding/rehashing, test inference or new method result was claimed.

Detailed machine evidence is stored next to this report as ACTUAL_MANIFEST_INVENTORY.json, SPLIT_AND_SOURCE_CHECKS.json, HF_CURRENT_METADATA.json, DATASET_GITHUB_METADATA.json, DATASET_GITHUB_FILE_TREE.json, TRAIN_SAMPLE_DECODING.json and MODEL_CHECKPOINT_INVENTORY.json. Raw face data, row-level human labels and weights remain outside public GitHub.
