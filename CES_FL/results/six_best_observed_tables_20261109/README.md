# Six tables requested and approved on 9 November 2026

[Download the combined PDF](SIX_TABLES_BEST_OBSERVED.pdf) · [LaTeX](SIX_TABLES_BEST_OBSERVED.tex) · [Every selected cell and its source](BEST_CELL_PROVENANCE.csv) · [Verification](VERIFICATION.json)

The PDF has nine pages: a reading guide, six numbered tables, and two supporting provenance pages. No training, model inference, checkpoint choice, threshold calibration, generation or private-prediction analysis was performed.

| Requested table | PDF page | Contents |
|---|---:|---|
| 1 | 2 | Ding's original vertical FF++ table:21 published rows, unchanged numeric values/backbones, plus our best observed values in each attribute block |
| 2 | 3 | Ding's original cross-domain Xception table:21 published rows, plus our best observed values |
| 3 | 4 | Ding's original ResNet-50 table:20 published rows, plus our best observed values |
| 4 | 5 | Actual public FairForge-2 TRAIN/validation/TEST counts, real/fake and gender×race breakdown; reserved CAL shown separately |
| 5 | 6 | Eleven-model FairForge-2 validation comparison, best recorded seed per metric, plus a best-own-across-variants row |
| 6 | 7 | Existing FairForge-2 TEST results for six model families/ten checkpoints, same reporting layout; missing evaluations explicitly unavailable |

Table5/6 panelA reports overall AUC, achieved FPR, FNR, gender×race worst-group AUC, and Ding-style gender/race/intersection F_FPR, F_DP and es-AUC. PanelB adds age, skin, age×skin and coarse sex×age×race disparity, es-AUC and worst-group AUC.

## Meaning of “best observed”

This is the user-approved retrospective per-cell minimum/maximum across recorded methods and seeds. It is an exploratory oracle, not one model or an unbiased result of model selection. Every selected cell has a run/seed tag. The CSV also records the contributing run's overall AUC/FPR/FNR, exposing cases where a low disparity comes from weak detection. No utility filter or new threshold is invented.

Tables1–3 use fixed probability0.5 throughout, and keep matching backbone/dataset membership. The candidate pool includes68 existing dataset reports: archived A/B/C and Xception upgrade arms, newer Xception LA seeds7/17/27 and LH seed17, and ResNet LA/LH seed17. This expands the narrower LA/LH-only pool in the older8November bundle; resulting extrema can differ. R0 duplicates historical B and is not independent evidence. CLIP and unrelated datasets are not silently mixed into a backbone-matched row.

FairForge2 tables use existing per-checkpoint validation-real FPR10 calibration. TEST uses its recorded frozen threshold, not a threshold optimized on TEST. The “best own” pool is CES-FL P/A/B, LA/LH-PS, RF-S and RF-SJg where actual measurements exist. The corrected local Ding GDA row is distinct from a published-paper number. Budgets/selection differ across families; these are not claims of a fully matched algorithmic comparison.

All RF downstream seeds share one seed17 pretrained encoder. For validation, RF age×skin/coarse-eight-group metrics are only present for seed17; other seed-axis cells are not invented. Age/skin labels are estimates. No confidence interval is attached to a per-cell oracle.

The underlying public FairForge-2 cell-valid study manifests contain TRAIN218266, SELECT31107 and TEST87564 images. A separate32255-image CAL subset is disclosed and was not used by the completed selection rule. These are not claimed as the full historical/private dataset counts. ResNet LA/LH FF++ records231 shared fit/test identity/source IDs; independence is unverified.

## Actual checks and build

```bash
# On the cluster only; refuses to overwrite an existing snapshot:
python build_six_tables.py --collect

# Rebuild from the committed aggregate snapshot, no cluster access:
python build_six_tables.py
python verify_six_tables.py
pdflatex -interaction=nonstopmode -halt-on-error SIX_TABLES_BEST_OBSERVED.tex
pdflatex -interaction=nonstopmode -halt-on-error SIX_TABLES_BEST_OBSERVED.tex

# Optional official-reference check using a local copy of the paper HTML:
python verify_six_tables.py --paper-html /path/to/2511.10150v5.html
```

Executed with `/usr/bin/python3`, TeX Live2025 and `pdftoppm`. All232 best-observed tagged cells independently matched the correct extremum/source seed, and all494 published numeric values, method identities, backbones and block membership matched the official arXiv v5 HTML. Paper-internal bibliography indices differ between accepted archive/arXiv versions, so they are omitted from the re-typeset tables; the paper's “Ours” is labelled “Ding et al. (published)”. Final build produced nine A4 landscape pages without overfull boxes, undefined references or LaTeX errors. Table1/5/6 pages were rendered and visually inspected.

The independent check caught an initial reporting-only direction error for extended worst-group AUC, which was fixed to maximize AUC before final rendering. It also identified version-specific bibliography index differences. The failed check logs are retained. No saved experiment or numeric input was changed. Verification/current build evidence is in JSON/log files alongside the PDF.

## Sources

- Official paper: https://arxiv.org/html/2511.10150v5 (retrieved9November; HTML SHA256 in `VERIFICATION.json`).
- Reviewed repository base: `76c069bdec9576b64b6543a524ffd6748851fd32`.
- Original published/archived experiment snapshot and CSV: `CES_FL/results/ding_format_tables_s17_20261102/`.
- Saved LA/LH benchmark analysis reports and fixed membership plans under the cluster paths listed in `SNAPSHOT.json`.
- FairForge2 recorded eleven-model VAL aggregates plus the three-seed selected RF receipts, and the already published frozen TEST aggregate CSV.
- Public release group composition and actual TRAIN/VAL/TEST manifest label counts;41 source files hashed in the snapshot.

Only aggregate evidence and artifact paths/hashes are published. Private rows, faces, reviews, scores and weights stay on the cluster. This retrospective packet is separate from the coherent-model mean±SD paper results.
