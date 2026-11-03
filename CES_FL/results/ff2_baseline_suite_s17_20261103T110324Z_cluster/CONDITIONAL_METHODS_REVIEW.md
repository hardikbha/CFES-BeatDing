# Conditional-method audit, 3 November2026

The first wave remains the exact BCE/DAG/DAW common recipe at scientific commit `815b4d4bc679c55d4ee6323cc7284335ce210568`. The following are concrete adapter gates, not missing results counted as zeros. No conditional method has been launched.

## PG-FDD — BLOCKED_ADAPTER / LABEL_MAPPING_REVIEW

Pinned official source: `Purdue-M2/Fairness-Generalization`, commit `e552a5422987897ab747e3b91b4b1b38d6531446`, privately fetched under `/home/sachin.chaudhary/ces_fl_dependencies/ff2_baselines_pinned_20261103/pg_fdd`.

`training/detectors/fair_df_detector.py:40` constructs three separate encoders; line63 fixes the specific-class head to6 outputs; line137 supplies the FF++ group counts `[2475,25443,1468,4163,8013,31281,1111,2185]`. Lines195–240 require `label_spe` and reconstruction losses. `training/train.py:208` enables BN running statistics for the first SAM pass, line215 executes `first_step`, line218 disables BN running statistics for the second pass, and line225 executes `second_step`. These cannot be replaced by the first-wave CVaR adapter.

Actual public FF2 TRAIN contains218,266rows:16,777real and201,489fake, with26`method` values including`real` and8 broad`family` values. The canonical manifest has no ready`label_spe` column. The native Ding-schema CSV header is exactly `img_path,label,ismale,isasian,iswhite,isblack,intersec_label` (seven columns). PG's `training/dataset/pair_dataset.py:41` reads fake specific labels from column index7; feeding this seven-column file directly would fail. Feeding canonical column index7 would incorrectly use `skin_bin` as the forgery-specific target.

Available method strings are a candidate provenance source, not an approved six-class mapping. A new versioned adapter must verify their provenance, derive a complete TRAIN-only mapping and output dimension, preserve all original membership, derive group counts from TRAIN in official order, retain all three encoders/reconstruction/contrastive terms and SAM/BN semantics, and pass objective/gradient/checkpoint tests. No such certified FF2 adapter/config currently exists in the supplied commit. We have not invented generator labels or truncated the data to make six outputs fit. No PG CUDA time is authorized until those gates and a measured cost check pass.

## FairAdapter — BLOCKED_ENTRYPOINT / DATA_ADAPTER

Pinned official source: `AppleDogDog/FairnessDetection`, commit `b81e1cc0205e79a0393513e72aa0208021452704`. Its README says only “FairnessDetection / Fairness Detection in AIGC”; it does not identify which of the different trainers reproduces the paper.

`training/train_clipFairness.py:77` selects `CLIP_Adapter` (`training/detectors/my_model.py`), while `training/train_clipfairness_loss.py:77` selects `CLIP_Fairness_loss` (`my_model_loss.py`). Their losses/calls and defaults differ. `train_clip_batchsize.py` is a third variant. Blindly choosing one would not certify a paper implementation.

`training/dataset/mydataset_train.py:44` reads a hardcoded external `bicycle_fake.csv` to define length. Lines85–94 hardcode20object categories such as airplane, bird, sofa and bicycle, then load real/fake views for each category. FF2's26generator/method strings and demographic groups are not those object-category labels. The source's length/domain/view semantics require an explicit defensible adaptation; mapping demographics to these categories would change the method. `train_clipFairness.py:73` also hardcodes `cuda:1`, which is invalid for a single-visible-GPU allocation. Line376 onward freezes named text/image encoders, while the Adam constructor includes all model parameters; aliasing, optimizer membership and actual gradients need verification. Its `amsgrad='false'` is a nonempty string, not booleanFalse, and must be interpreted explicitly rather than silently corrected.

Existing local OpenAI ViT-L/14 weights were found at `/home/sachin.chaudhary/.cache/clip/ViT-L-14.pt` (932,768,134bytes), SHA256 `b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`. Availability is verified; compatibility with a selected paper entrypoint, objective parity and an FF2 view adapter are still unverified. No weights were downloaded, no model inference was run for this check, and the generation-oriented top-level `main.py` was not executed.

Keep FairAdapter conditional until a paper-consistent entrypoint/data recipe is established and recorded in a new exact config, followed by CPU/CUDA/actual-TRAIN checks and a budget projection. Preserve its CLIP backbone in any future row.

## RSEF-FDD — BLOCKED_SOURCE / DATA

Remains blocked as explicitly instructed: no verified official implementation or validated redundant-sample/environment recipe is present in the accepted proposal. No substitute loss, extra data or author contact was introduced.
