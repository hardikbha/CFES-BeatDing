# CES-FL paper workspace — 9 November 2026

[Working PDF](main.pdf) · [LaTeX source](main.tex) · [Full TEST tables](RESULTS.md) · [Claims and gaps](CLAIMS_AND_GAPS.md) · [Evidence map](EVIDENCE_MAP.md) · [Writing plan](WRITING_PLAN.md)

This is an actual first draft based on verified saved results, not a submission-ready paper. The working title is **Processing History Confounds Demographic Fairness in Deepfake Detection: A FairForge-2 Audit and Controlled CES-FL Pilot**. Author/affiliation details need confirmation.

The current evidence supports a source-processing diagnosis and a limited intervention effect. It does not establish that the complete proposed counterfactual architecture was trained successfully, that every demographic axis improved, or that CES-FL beats Ding overall.

## Current findings

- RF-S and RF-SJg each have downstream seeds 7/17/27, ten epochs per run, with selected-checkpoint replay records. They all share the same seed17 BCE50 pretrained encoder.
- RF-SJg TEST gender×race `100F_FPR=47.69±3.91` versus RF-S `60.95±1.92`; AUC `98.60±0.06%` versus `98.98±0.01%`. Error bars are sample SD across downstream seeds, not confidence intervals.
- Gender and age disparity worsen relative to RF-S. The local corrected Ding-style GDA achieves lower primary disparity (`9.47`) but lower AUC (`93.08%`). This is a local adapted comparator, not Ding's published benchmark score.
- Ten checkpoint exports exist in the prior frozen TEST list. The audit recomputed all 80 summary rows from 540 aggregate subgroup rows and checked actual checkpoint bytes against receipts. No new inference or training was run to write this draft.
- TEST is now exposed. Do not use these results to tune another candidate on this split.

## Rebuild without the cluster

Python with NumPy/Matplotlib and a standard TeX Live installation are sufficient. The draft uses the pinned official CVPR 2027 author kit; see [template provenance](TEMPLATE_PROVENANCE.md).

```bash
cd CES_FL/paper
python tools/verify_public_snapshot.py
python tools/build_assets.py
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

These commands read committed aggregate evidence only. They do not download a model, access private scores, connect to a GPU, or launch training. CPU/CUDA smoke receipts in historical proposals certify their own exact scientific sources; paper-only edits do not require retraining them.

`tools/collect_paper_evidence.py` is a separate, explicit cluster collection tool. It refuses an existing destination, hashes saved checkpoints, reads aggregate receipts/reports, and never opens private prediction files. It is not needed to rebuild the public draft. Its successful command/output and earlier schema/path failures are preserved in [verification](verification/).

## Publishing boundaries

Code, exact configurations, aggregate metrics, per-group counts/rates, epoch summaries, provenance hashes and PDF/LaTeX belong here. Datasets, face images, private reviews, individual scores and checkpoints remain on the cluster. Historical oracle best-cell tables are development summaries, not one deployable model; they are excluded from the manuscript's main comparison.

This dated branch is the current writing snapshot. Older README/STATUS entries elsewhere in the repository are historical observations, not a live monitor.
