You are editing the CLIPFAIR baseline code of a deepfake-fairness project. This folder (R/agents/CLIPFAIR,
R=/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009) is the only place you may write. You may read
anything under R. Do not use the network or ssh, do not touch PBS, do not start GPU work.
Python: /home/sachin.chaudhary/.conda/envs/pulid/bin/python (set CUDA_VISIBLE_DEVICES=""); never the default python3;
no pip installs. Back up every edited file as *.bak_codex_oct26. Log your actions in codex_ops/ACTIONS_01.md.
Read first: R/decisions/2026-10-25-cesel-plan.md section "Amendment 7", README.md in this folder.

TASK (Amendment 7):
1. Find where the training loop computes the Amendment-9 selection value and saves best.pt (the CLIPFAIR train script
   and the train/engine code it uses). Additionally track the best epoch under pooled-only eligibility (val pooled
   TPR >= 80 and FPR <= 20; same selection value as Amendment 9) and save best_pooled.pt plus its epoch. At the end of
   training, if NO trained epoch (epoch >= 0) was eligible under the full Amendment-9 rule, copy best_pooled.pt to
   best.pt and record "selection_fallback": true and the chosen epoch in the TRAINED marker / run summary. If even
   pooled-only has no eligible epoch, keep the current behaviour and record "selection_fallback": "none_eligible".
   Evaluation keeps using best.pt. Note: running training processes already imported the old code; your edit only
   affects future launches — do not rename or restructure modules they import.
2. Add a CPU unit test (tests/test_amend7_fallback.py, stub backbone or a mocked loop) covering: normal eligible path
   (no fallback), all-ineligible path (fallback picks the pooled-best epoch), none-eligible path. Run it and
   tests/clipfair_checks.py; both must print ALL_PASS.
3. Move runs/ffpp_daw_s7 -> runs/ffpp_daw_s7_DEGENERATE_oct26 (and any runs/ffpp_daw_s7.RUNLOCK* entries ->
   append _DEGENERATE_oct26) so the external runner relaunches daw_s7 with the new code. Touch nothing else in runs/.
4. Write codex_ops/REPORT_01.md: changed files with line ranges, tests and output, what was moved. Keep it short.
