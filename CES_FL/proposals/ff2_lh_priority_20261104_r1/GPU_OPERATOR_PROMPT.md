# Priority: our L_H on FairForge-2 public

Fetch the supplied exact commit from https://github.com/hardiksharma6555/CFES-BeatDing.git into a new detached worktree. Read CES_FL/AGENTS.md, WORKFLOW.md and this proposal's README/request. This request prioritizes our L_H over additional unstarted baselines; preserve F3Net, CORE, Ding and other active jobs.

Use the unchanged original L_H: Xception/GAP/linear, exit-block tuning, frozen BN, original hierarchical fairness + procedural sensitivity + sham stability + response alignment. Seed17,batch32,10downstream epochs,head_lr1e-3,encoder_lr1e-5. No lower-LR, SHAM-FIX, NO-ALIGN or new demographic-bin search. Reuse the completed FF2 native Xception BCE50 encoder selected at epoch40, SHA a62af6bc1ff02f4280fcb38e9b9a4b035ccee3427472d160880b4b9e0c31d835. Export only encoder tensors using prepare.py; initialize a fresh seed17head. Document this FF2 pretraining adaptation; it is not the original FF++ encoder or a new pretraining run.

Set:
```bash
export CES_PY=/home/sachin.chaudhary/.conda/envs/aiface/bin/python
export LH_FF2_ROOT=/home/sachin.chaudhary/ces_fl_runs/ff2_lh_priority_20261104_r1
export LH_NATIVE_BCE_RUN=/home/sachin.chaudhary/ces_fl_runs/ff2_baseline_suite_s17_20261103T110324Z/bce/training
export LH_COMMANDS=CES_FL/proposals/ff2_lh_priority_20261104_r1/commands.sh
bash "$LH_COMMANDS" prepare
```
Verify actual paths before running; do not repeat prepare on an existing root. It binds the exact encoder export, checks both manifests, hashes/decodes inputs and generates private procedural previews. L_H/L_A configs are identical; native CLI arm D versus A controls the objectives. The common preflight therefore applies to both.

CRITICAL KNOWN DEPENDENCY: the earlier FF2 review had72approved,8rejected,8uncertain among88cases. Inspect the previous evidence and the new previews. Record a substantive named review and reasons; preserve reviewer type (AI/human). Do not copy FF++ approvals, flip old rejection flags automatically, drop difficult sources or silently change the intervention recipe. If the same content remains invalid, report BLOCKED_INTERVENTION_VALIDITY with exact cases and a concrete TRAIN-only remedy proposal. A padding/mask/occlusion recipe change requires separately named source/config, controls and checks; do not disguise it as unchanged L_H. This is the first priority dependency, not a reason to launch more unrelated baselines.

When actual review supports qualification:
```bash
export LH_REVIEW=/actual/private/completed_review.json
bash "$LH_COMMANDS" qualify
```

Refresh existing ledger(s) and allocations. This request grants no additional GPU-hours and does not permit double-counting reservations. Latest original-study observation21.7714/24used; latest FF2 suite snapshot28.010872charged+42reserved of72. Verify current unreserved allowance and record which authorized ledger funds this run. Do not cancel existing jobs or transfer their reservations. Obtain one properly bound GPU; validate CUDA visibility/UUID. Run `bash "$LH_COMMANDS" cuda-checks`. On a discarded model, time20actual TRAIN updates using the exact native loader/auxiliary schedule, including input loading, and estimate all10epochs plus validation/replay. Publish the full projection before training. If it cannot fit remaining allowance, report BUDGET_PENDING with measured hours needed; do not truncate epochs or silently extend budgets.

Set LH_CUMULATIVE_GPU_HOURS to the verified allowance, wrap execution in the established ledger/allocation guard and scheduler limit, then run `bash "$LH_COMMANDS" train-h` as the first training priority. Publish source/config/checkpoint hashes and genuine progress after each validation epoch. Preserve incomplete last.pt and failures; any resume must retain the same exact config/source and be explicitly accounted. Train matched L_A with `train-a` afterward only if remaining budget permits. The already completed fully trained BCE50 is an external reference, not a substitute for this matched downstream control.

Require DONE, RUN_VERIFIED and selected-checkpoint replay before calling either arm complete. Select earliest maximum validation AUC and preserve the global VAL-real10%threshold. Run `report-h`/`report-a` on CPU. Report gender,race,gender×race,age and skin groups plus all missing/unknown support. Age bins are <30,30–44,45–59,60+; skin bins0–3 are proxies with unknowns, not human/Fitzpatrick truth. Additional intersections may be evaluated from saved predictions with explicit definitions/support, without retraining or changing native bins.

After completion and before any TEST scores, record selected checkpoint/threshold hashes. Use the native model's `evaluate` command on the fixed request.json TEST manifest once, with no new overlap exemptions or TEST calibration; keep0.5 and stored VAL threshold views. It uses the original head-ablation checkpoint format, not ff2_expanded/evaluate_locked.py. Then run report.py with --split test on the resulting private scores cache. Keep final TEST metadata/manifests unchanged. No tuning after TEST observations.

Publish aggregate results and the ledger; keep weights, faces and row scores private. Include F_FPR,F_DP,es-AUC,worst AUC, per-group FPR/FNR and extended MAX-EO,sum-EO,accuracy-gap metrics. Our F_MEO implementation is max(FPRgap,FNRgap); do not label it identical to FairForensics' mean convention without verifying the formula. Existing1000draw bootstrap covers native Ding metrics, with900usable minimum; extended metrics currently have point estimates. Return exact commits, completed-stage evidence, our FF2 L_H row versus matched L_A and completed external baselines, plus any remaining blockers. No paper score may replace an unrun FF2 result.
