# Actual reporting commands and outputs — 2 November 2026

This was a static reporting task. No GPU allocation, ML training/inference, model download, checkpoint selection or threshold calibration occurred. Large checkpoint binaries were not reloaded or rehashed. Native training/evaluation PASS receipts were inspected as historical evidence.

## Read and collect existing results

Actual command:

```bash
python /home/sachin.chaudhary/cesel_ma_audit_logs/collect_ding_format_tables_20261102.py
```

Actual output:

```json
{
  "status": "PASS",
  "paper_row_counts": {"1": 21, "2": 21, "3": 20},
  "experiment_count": 11,
  "existing_test_records": 44,
  "input_files_hashed": 74,
  "snapshot_path": "/home/sachin.chaudhary/ces_fl_sources/publish_response_20261101T104925Z/CES_FL/results/ding_format_tables_s17_20261102/source_snapshot.json"
}
```

The private collector contains no training code and reads actual archived paper text, native benchmark configs/results/receipts, manifests and published upgrade aggregates. `source_snapshot.json` includes the input hashes and full numeric provenance required to rebuild the tables offline.

## Formatting, formula checks and PDF

The first attempt used the shell's default `python` and failed importing Pillow `_imaging` through ReportLab. A second attempt under the supported aiface Python imported ReportLab successfully but found that this cluster stores DejaVu in `/usr/share/fonts/dejavu-sans-fonts/`, rather than the initially assumed Debian path. The renderer now discovers either font location. Neither issue changed scientific code, data or results. Partial rendering attempts were preserved in the audit-log directory.

Actual successful final rendering command:

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python - <<'PY'
import importlib.util,pathlib,sys,shutil
r=pathlib.Path('/home/sachin.chaudhary/ces_fl_sources/publish_response_20261101T104925Z/CES_FL/results/ding_format_tables_s17_20261102')
out=pathlib.Path('/home/sachin.chaudhary/cesel_ma_audit_logs/ding_tables_render_20261102_r3')
sys.path.append('/home/sachin.chaudhary/.conda/envs/internvl/lib/jvm/languages/python/lib/python3.10/site-packages')
spec=importlib.util.spec_from_file_location('build',r/'build_tables.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
sys.argv=['build_tables.py','--snapshot',str(r/'source_snapshot.json'),'--out',str(out),'--pdf'];b.main()
for name in ['TABLE_1.md','TABLE_2.md','TABLE_3.md','TABLES_1_2_3.html','VERIFY.json']:
 assert (r/name).read_bytes()==(out/name).read_bytes(),name
shutil.copy2(out/'TABLES_1_2_3.pdf',r/'TABLES_1_2_3.pdf')
print('PUBLISHED_RENDER_BYTE_AGREEMENT_PASS')
PY
```

This command was executed successfully; do not rerun it into its existing output directory. For a new build choose a fresh path. The cross-environment path above supplied the existing pure-Python ReportLab package only; Pillow remained the working aiface installation. No packages were installed and no environment was changed persistently.

Actual output:

```json
{
  "status": "PASS",
  "operation": "Static formatting/numeric verification only",
  "paper_rows": [21, 21, 20],
  "output_rows": [45, 45, 32],
  "existing_test_records_verified": 44,
  "score_reports_checked": 88,
  "ding_style_formula_checks": 792,
  "source_receipt_manifest_checkpoint_agreement": true,
  "all_latest_validation_rows_excluded_from_test_tables": true,
  "training_or_inference_launched": false,
  "calibration_or_selection_performed": false
}
```

Followed by `PUBLISHED_RENDER_BYTE_AGREEMENT_PASS`. Recomputed fairness values agree with the saved values to absolute tolerance 1e-12; percentage conversion, axis order and missing-value formatting checks also passed. These are reporting checks, not new ML smoke tests.

## Layout verification

Actual commands, run from the publishing worktree:

```bash
pdfinfo CES_FL/results/ding_format_tables_s17_20261102/TABLES_1_2_3.pdf
pdftoppm -f 2 -l 2 -scale-to 1600 -singlefile -png CES_FL/results/ding_format_tables_s17_20261102/TABLES_1_2_3.pdf /home/sachin.chaudhary/cesel_ma_audit_logs/ding_tables_page2_20261102
```

Actual final PDF: **3 pages, A4 landscape, 841.89 × 595.276 points**. Page 2 was visually inspected: all three cross-domain blocks, grouped headers, published and measured rows, and the footnote fit within the page. Additional static checks and the Git publication receipt are recorded in the private audit log.
