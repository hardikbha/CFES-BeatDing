# Actual commands and execution scope

Documentation handoff was located on `hardiksharma6555/CFES-BeatDing`, branch `review/ff2-fairness-diagnosis-20261109-r1`, at **76dcb54e240dcf56475dbf587836cdadcdad5aab**. Its diff against the scientific review base e7fe0ee contains only three added review documents. Exact Git object content was fetched; no running source was pulled over.

```bash
git fetch --no-tags https://github.com/hardiksharma6555/CFES-BeatDing.git 76dcb54e240dcf56475dbf587836cdadcdad5aab
git worktree add --detach /home/sachin.chaudhary/ces_fl_sources/cesfl-source-audit-doc-76dcb54-20261109 76dcb54e240dcf56475dbf587836cdadcdad5aab
git worktree add -b results/ff2-source-stratified-audit-20261109-r1 /home/sachin.chaudhary/ces_fl_sources/cesfl-source-audit-results-20261109 76dcb54e240dcf56475dbf587836cdadcdad5aab
```

From the new result directory, the following CPU-only commands were executed in order. The environment assignments below were supplied to each command; CUDA availability or devices were never queried. Checkpoint loading used CPU `map_location` and did not construct a model.

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
CES_AUDIT_PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
CES_AUDIT_SOURCE=/home/sachin.chaudhary/ces_fl_sources/cesfl-source-audit-doc-76dcb54-20261109
CES_AUDIT_PRIVATE=/home/sachin.chaudhary/ces_fl_runs/ff2_source_stratified_audit_20261109_r1

$CES_AUDIT_PY -m unittest test_audit -v
$CES_AUDIT_PY audit.py freeze --source "$CES_AUDIT_SOURCE" --private "$CES_AUDIT_PRIVATE"
$CES_AUDIT_PY audit.py verify --source "$CES_AUDIT_SOURCE" --private "$CES_AUDIT_PRIVATE"
$CES_AUDIT_PY -u audit.py analyze --source "$CES_AUDIT_SOURCE" --private "$CES_AUDIT_PRIVATE"
$CES_AUDIT_PY render_report.py
```

The scripts refuse to overwrite the freeze or silently reopen a completed analysis attempt. These are a record of execution, not instructions to launch another analysis automatically. Rendering the committed aggregate report is independent of private model inference.

Five meaningful synthetic CPU checks passed before source weights were frozen: all donor/master IDs participate in connected components; weighted AUC including ties equals explicit resampling; known source-mixture contrast is recovered; all paired identical-score contrasts are zero; missing required cells stay undefined without renormalizing; documented source mappings reject unsupported identities/mismatches. Several checks share one test method; the unittest runner reports five tests.

Freeze UTC: **2026-11-09T05:49:29.507523+00:00**. First private outcome-parse event UTC: **2026-11-09T05:51:53.130057+00:00**, on the same login host. TRAIN source strata/weights and bootstrap rules were recorded first; known public summaries were already available to the project. This temporal record does not make the exploratory analysis a prospective independent test.

All six checkpoint/config/source/preflight checks and all six cache/manifest/threshold alignments passed. The 2,000 paired draws completed. Native unchanged-image metrics exactly reproduced saved values within absolute1e-12; input SHA256s were rechecked after analysis. Source-specific gender/race/joint support was never filtered to force a result. Primary result remains blocked by empty FFHQ Others cells. Relevant fairness intervals do not satisfy the fixed 1,900-valid-draw minimum.

An additional metadata-only inline check was executed after the analysis; its equivalent preserved implementation is `check_metadata.py`. It verified every CSV field against all six cached row exports (186,642 row checks), one recorded real-master key per real row, recorded identifier namespace counts, and 45 components (30 FFHQ,15 FairFace; zero cross-acquisition components). It does not open raw image pixels, mixed-split source-master manifests, TEST/CAL or a new score generator.

The preserved metadata check was also executed as a file. A final byte-integrity check, initially executed inline and preserved in `check_integrity.py`, rechecked frozen helper/input hashes, the shared initial encoder bytes against all six configuration hashes, and the clean detached source checkout. Both report PASS. Reading the pinned `predict`/`Originals` code confirms that native SELECT exports use original images without the TRAIN JPEG augmentation or procedural pairs; the code was inspected, not executed.

```bash
$CES_AUDIT_PY check_metadata.py
$CES_AUDIT_PY check_integrity.py
```

`render_report.py` independently reconstructed 18 native disparity values and six gender-standardized disparity values from aggregate counts: 24 checks passed. Source tables contain 168 real source/group/model cells; all original empty cells remain explicit. Utility output contains 1,350 aggregate metric/cohort rows.

CPU analysis helpers are separate from the scientific trainer. The pinned source checkout and all historical config/result files remain unchanged. Private prediction rows, faces, weights and bootstrap draws remain off GitHub.
