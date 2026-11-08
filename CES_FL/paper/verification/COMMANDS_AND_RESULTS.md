# Executed publication checks — 9 November 2026

No new GPU training, image generation or inference was performed for this paper update. Model sources and frozen protocols are unchanged.

## Aggregate collection and checkpoint byte checks

From `/home/sachin.chaudhary/ces_fl_sources/cesfl-paper-20261109`:

```bash
/usr/bin/python3 -u CES_FL/paper/tools/collect_paper_evidence.py \
  --runs-root /home/sachin.chaudhary/ces_fl_runs \
  --out CES_FL/paper/evidence \
  --repo /home/sachin.chaudhary/ces_fl_sources/cesfl-paper-20261109
```

Actual final output: `checks_passed:17`, `test_checkpoints:10`, `selected_validation_runs:6`, `test_summary_rows:80`, `test_group_rows:540`. Each checkpoint-byte hash matched its receipt. Six selected RF thresholds exactly matched their stored TEST aggregate threshold. Source/configuration and selection checks are recorded in `evidence/PROVENANCE.json`.

Earlier collection attempts failed on differing aggregate schema (P stores threshold under calibration) and the duplicate-run directory requiring `/out`. These collector-only issues were fixed; failed logs remain alongside the successful log. No scientific code or artifact was modified. Failed temporary collection directories remain local and are excluded from Git.

## Independent aggregate arithmetic and source checks

```bash
/usr/bin/python3 CES_FL/paper/tools/verify_public_snapshot.py
/home/sachin.chaudhary/.conda/envs/aiface/bin/python CES_FL/paper/tools/build_assets.py
```

Actual output: `PASS`, all80 summary rows recomputed from540 subgroup rows; six selected runs /60 epoch records; both model-source SHA256s unchanged. Four tables and three numeric figures generated. Paired TEST `100F_FPR` differences (SJg minus S): seed17 `-12.14433145`, seed7 `-11.61917966`, seed27 `-16.01190739`; descriptive mean `-13.25847283`. This is not a significance test.

## Parameter count

```bash
/home/sachin.chaudhary/.conda/envs/aiface/bin/python \
  CES_FL/src/ces_fl_head_ablation.py parameters --head-type linear --encoder-policy exit
```

Actual output: total20,809,001; trainable6,790,433; encoder20,806,952; encoder-trainable6,788,384; linear head2,049. CPU construction/count only; no checkpoint/inference or data loaded.

## PDF build and inspection

From `CES_FL/paper`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdfinfo main.pdf
```

Actual result: seven-page letter-size PDF. Stable pass has no undefined citations/references, LaTeX errors, overfull boxes or package warnings. Pages1,3,4,5 rendered with `pdftoppm` and visually inspected, including both main TEST tables. Exact command logs are preserved here.

## Live-status observation

At login01:43 IST (GPU-node01:32), the following returned no matching RF trainer on either host:

```bash
ssh gpu01 'ps -u sachin.chaudhary -o pid=,comm=,args= | awk '\''$2 ~ /^python/ && /ces_fl_realfpr.py/ {print}'\'''
ssh gpu02 'ps -u sachin.chaudhary -o pid=,comm=,args= | awk '\''$2 ~ /^python/ && /ces_fl_realfpr.py/ {print}'\'''
qstat -u sachin.chaudhary
```

Scheduler IDs45974/45977/45983/45985 were present with generic/pool names. They were not reclassified as CES-FL jobs or modified. A named-process query is evidence about that trainer, not proof that every possible project process is absent.
