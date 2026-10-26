# CESel (Counterfactual Evidence Selectivity) screen + FF2 public release fixes — pre-registration, 25 Oct 2026
User approved (25 Oct, AskUserQuestion "Yes, execute all"; Q11b threshold = validation-chosen A1). Fixed BEFORE any CESel
number exists. Code: R/agents/CESEL (CPU checks ALL_PASS 15/15). Name: CESel everywhere (not CES/CFES).

## Data (R/release_ff2_public)
- Fix generator->family map (23/35 bug in ff2_tableB.py): all 35 labels -> 7 families; recompute per-family tables.
- Drop InSwapper from the public image release (54,441 imgs; expected ~414,638 before opt-outs). Optional local-regen
  recipe for users who obtain inswapper_128.onnx themselves.
- Keep FairFace (gated, CC BY-NC-SA 4.0 + DUA + takedown).
- FFHQ Flickr opt-out check on the public FFHQ reals; drop deleted-source reals and all fakes derived from them; then
  FREEZE the public split.
- IRB + 10-annotator human validation: user's side (HUMAN_VALIDATION_PROTOCOL.md).
- All FF2 training (B0, CFES-D, CESel; Arm E if run) uses ONLY the frozen public TRAIN; CESel FF2 pairs filtered to it.

## CESel design (as coded; user-confirmed)
CLIP ViT-L/14 + rank-1 SVD adapters only (no CNN runs); additive head over 4 support-defined cue branches; cue removal =
head-level replacement (paired source cue; FF++ unpaired fakes -> detached same-group real-cue mean); classifier-verified
feature-space demographic swap (no pixel editing); L_floor over group x class (x family on FF2, batch 304, n>=4/cell);
validated cue = lower bootstrap 2.5% bound > 0 on validation, target = median clamped [0.1, 2.0] (FF2: T2I/GAN may skip
boundary); loss weights sens 0.5 / inv 0.5 / floor 1.0 fixed; demographic labels in training only, never at test.

## Screen (seed 7), FF++ (Ding/Lin protocol) first, then FF2-public
1. GPU smoke: 20 bf16 steps at batch 176 (FF++) and 304 (FF2-public): peak VRAM + s/step.
2. B0 (architecture + detection only) from runs/clip_s7_p1/last.pt, CFES-D phase-2 budget; validation-only selection
   (Amendment 9 rule); validated-cue JSON from B0's best checkpoint on validation.
3. From B0 best, equal extra budget for 9 arms: B0-continued, +sens, +inv, +floor, Full CESel, C_randcue, C_shufgroup,
   C_novalid, B0+CFES-D.
4. Evaluation: full test sets (FF++, DFDC, Celeb-DF, DFD), threshold 0.5 and A1 (fair_oppoint rule) per model.

## Decision rules (validation picks; test only reported)
- Keep a CESel term only if the arm with it beats step-matched B0-continued on the validation selection value; drop any
  term whose negative control (random cue, shuffled groups, no validity check) matches the Full arm within the seed
  spread (TTD/full-data CFES-D 3-seed spread used until CESel's own exists).
- Promote to 3 seeds (7, 17, 27): B0, B0+CFES-D, and the winning CESel arm only.
- Success (paper claim) requires ALL of: (a) beats B0, CFES-D and Arm E on validation selection value; (b) at the A1
  threshold, intersectional F_FPR below Ding's published value on >= 3 of 4 test sets (3-seed mean); (c) margins over
  our baselines exceed the inter-seed spread. Otherwise CESel is reported as it is, no claim.
- Interpretability (per-cue faithfulness, deletion curves, per-group cue contributions) is part of the claim; reported
  for every promoted arm.
- After the seed-7 screen only: optional CESel + Arm E fpr_gap combination.
GPU rules: gpu01 inside PBS, allowed GPUs {0,1,2,3,4,6,7}, free = >=60 GB free and <10% util, <=4 concurrent; never
other users' processes; keep ICLR_Misinfo alive.

## Amendment 1 (before launch, 25 Oct; written before any CESel training number exists)
Budget and selection
- B0: CFES-D phase-2 budget = 10 epochs x 1976 steps (configs/rev4/clip_D.yaml chain: study/clip_A epochs 10,
  clip_p2 steps_per_epoch 1976), lr 1e-4 adapters / 1e-3 new modules, warmup 200, cosine, bf16; backbone from
  CFES/runs/clip_s7_p1/last.pt; selection = Amendment-9 val_oppoint rule, start checkpoint eligible.
- DECLARED EXTRA BUDGET for all 9 arms: 5 epochs x 1976 steps (9,880 steps) each, from B0 best.pt (+ its CES state)
  with B0's validated-cue JSON; fresh optimiser and schedule (warmup 200, cosine over the 5 epochs); same selection
  rule with B0 best as the eligible start checkpoint and reference. Same seed (7) and data stream for every arm.
- Spread for the control rule = RANGE (max - min) of the full-data CFES-D 3-seed validation selection values
  (runs_rev4/clip_D_w0.3_s{7,17,27}: 87.62, 89.68, 89.28 -> 2.06); a control "matches Full" iff |sel - sel_Full| <= 2.06.
  Control -> term: C_randcue -> the counterfactual removal (sens and floor), C_shufgroup -> floor, C_novalid -> inv.
  Winner = highest selection value among {+sens, +inv, +floor, Full} that beat B0-continued and contain no dropped term.
Implementation choices
- Removal baseline of UNPAIRED fakes = detached EMA real-cue mean of the SAME group (user decision), falling back to
  the real class mean while that group has no estimate (ces.fake_baseline: group; reals keep the fake class mean,
  paired fakes their source). Used identically in training, in ces_validate_cues.py and in the post-hoc analyses.
  Unit check added (tests/ces_checks.py, ALL_PASS 17/17 incl. Amendment 2's check).
- eval.rank_metrics: true (as CFES-D): test metrics from rank_prob(logit), identical 0.5 decisions.
- Evaluation = CFES evaluate.py logic ported (scripts/evaluate_ces.py: leak-free variants, summary.csv, *_scores.npz
  incl. per-cue logits s_k) + unchanged calibrate.py + scripts/ces_posteval.py: A1 by the verbatim CFES fair_oppoint
  rule (scripts/fair_a1.py; reproduces CFES-D s7 A1 t = -46.946, DFDC i-F_FPR 16.30) and per-cue analyses
  (per-group utilities, per-cue deletion curves alpha in {0,.25,.5,.75,1} with the neutral EMA baseline, cumulative
  deletion by validation importance, solo-cue AUC). None of this selects anything.
- User request 25 Oct: per-group x class x cue starvation table of B0 best (FF++ val + 4 test sets; mean utility,
  bootstrap 95% CI, deficit vs pooled, cue share) -> results/b0_cue_starvation/ (scripts/ces_cue_starvation.py);
  the same table later for every promoted arm.
- Orchestration: ONE task list, two PBS runners sharing it through RUNLOCK dirs (gpu01 GPUs {0,1,2,3,4,6,7}, <= 4
  runs; gpu02 GPUs {6,7} only with the MPS-off env, <= 2 runs); free GPU = >= 60 GB free and < 10% util at launch.
- GPU smoke (PBS 44066, gpu01, 20 bf16 steps): Full at 176 images/step: peak 47.0 GB allocated (48.1 reserved),
  0.515 s/step median; B0+CFES-D (176 + 64 nuisance views): 63.2 GB (64.6 reserved), 0.656 s/step. FF2-public smoke
  not run (configs/ces_ff2_public.yaml did not exist yet).

## Amendment 2 (before any FF2 run, 25 Oct; user request)
- Per-family cue validation, FairForge-2 ONLY (FF++ stays global, as pre-registered and launched): flag
  ces.validation.per_family (true in configs/ces_ff2.yaml and therefore in configs inheriting it; must be on in
  ces_ff2_public*.yaml; false for FF++). For each manipulation family with >= family_min_n = 200 validation fakes:
  valid = bootstrap (B 1000, seed 0) lower 2.5% bound of the family's mean fake utility > 0, target = median clamped
  [0.1, 2.0]; families below 200 fall back to the global decision. L_sens uses each fake's family mask / target
  (reals and unmapped fakes: global); L_floor keeps the global validated mask. Unit check in tests/ces_checks.py.

## Amendment 3 (25 Oct ~19:35, user: "yes add the four CLIP baselines"; before any baseline run or CESel result)
Matched-backbone fairness baselines on the SAME CLIP ViT-L/14 + rank-1 SVD adapters, same FF++ (Lin) data, samplers,
start checkpoint (runs/clip_s{seed}_p1/last.pt), budget (10 x 1976 steps = B0's), Amendment-9 validation selection,
evaluation (threshold 0.5 + A1) and seeds 7/17/27: (1) DAW-FDD and (2) DAG-FDD (Ju et al. WACV'24; losses per the
official code in agents/CFES/third_party/ff2id/AI-Face-FairnessBench), (3) PG-FDD (Lin et al. CVPR'24; official
Fairness-Generalization code: its losses + SAM), (4) Ding et al. CVPR'26 SFD + GDA ported to the ViT (official code in
agents/DINGREPRO_LIN/code; adaptation documented: Xception channels -> ViT feature channels). Hyper-parameters = the
papers' defaults; nothing tuned on test. Implemented in a SEPARATE code copy (agents/CLIPFAIR) so the running CESel
screen's imports are untouched; queued after the CESel FF++ screen (PBS dependency), never competing with it.
Ding's published Xception rows stay as the reference; the CLIP rows answer "is the gain the method or the backbone".

## Amendment 4 (Codex review fixes, before any arm; 25 Oct ~20:40-21:30, B0 training)
Source: agents/CESEL/CODEX_REVIEW_20261025.md (F1-F11; 1 blocker, 7 major, 3 minor). Made while PBS 44067 trained B0
(runs/ffpp_B0_s7, started 19:10, epochs 0-5 done); NO arm, validate or evaluation had started. Running processes keep the
code they imported; every later launch (B0's own evaluation/calibration/A1, validate, starvation, the 9 arms, reports,
PBS 44068 and the CLIPFAIR jobs 44074/44075) uses the fixed code. .sh files were replaced by `mv` (the running B0 bash
keeps the old inode, checked). Every edited file has a *.bak_codexfix copy. A over B = agents/CESEL.
- F1 blocker + F11 (run/cesel_run.sh): new RUNLOCK protocol. Ownership token written once into <out>.RUNLOCK.new.<tok>/
  and published by ONE atomic rename (fails if the lock exists); retire (release or stale takeover) = atomic rename of
  the lock to <out>.RUNLOCK.gone.<observed token> (compare-and-swap: a generation can be retired once; tombstones kept
  7 days); stale = heartbeat COUNTER unchanged over a 240 s observation AND clock age > 1800 s lease, or same host and
  owner pid dead; fencing (a runner whose generation was retired kills its run); EXIT trap retires only its own
  generation; all paths quoted, override lists are bash arrays. Measured clock skew: gpu01 and gpu02 run ~642 s behind
  mgmt01/login (Lustre mtimes carry the writer's clock), so the old 900 s clock-only rule was 4.5 min from fencing a live
  run when read from mgmt01. B0's pre-fix lock is read as a legacy generation. scripts/cesel_pipeline.py held_elsewhere
  lease 900 -> 1800 s (future driver launches only; the running 44067 driver keeps 900 s, harmless: a false "not held"
  only yields exit 4 = HELD). Test tests/test_runlock_f1.sh: Codex interleaving (barrier right before the CAS) with 2
  and 8 stale-heartbeat contenders and 6 dead-pid contenders -> exactly ONE ACQUIRED each; fresh lock -> exit 4; live
  owner with an arbitrarily old clock not fenced (counter advanced); taken-over owner leaves the successor's lock;
  fencing; paths with spaces. ALL_PASS (3 runs). Same protocol ported to agents/CLIPFAIR/run/clipfair_run.sh.
- F2 (scripts/cesel_report.py): excluded terms = terms dropped by a matching control UNION terms whose single-term arm
  does not beat B0-continued (an arm containing an excluded term cannot win). Fail closed: no winner / no promotion while
  any keep or drop gate is unresolved or the reference spread is unavailable. The CFES-D 3-seed selection values are
  recomputed and must match Amendment 1 (87.62 / 89.68 / 89.28, tol 0.005; they do: range 2.0552); the declared spread
  2.06 is then used (before: the recomputed 2.0552; differs only for |diff| in (2.055, 2.06]).
- F3 (scripts/train.py validate, used by train_ces.py): with eval.rank_metrics the validation AUC, group AUCs and the
  ding@0.5 block come from rank_prob(logit) (no sigmoid ties); 0.5 decisions unchanged; the validation-TPR threshold is
  stored in logit units (threshold_tpr_units); the operating point (common intercept, pooled / group FPR, TPR) was and
  is on logits. All 9 arms (B0-continued included, start checkpoint re-validated at epoch -1) use the fixed rule, so
  the arm comparison is internally consistent.
  B0 SELECTION CAVEAT: B0's checkpoint choice (best_epoch among -1..9) was made by the OLD code, i.e. with
  probability-based validation AUC inside the Amendment-9 value. B0's validation logits are large (epoch 3: fake mean
  22.9, SD 17.2; float64 sigmoid saturates above ~37), so ties among saturated images are possible; reals sit far
  below (mean -7), so cross-class ties should be rare, but this is unverified per epoch. Per-epoch validation logits are
  NOT saved (train_ces.py keeps only train_log.jsonl summaries; debug.save_val_probs off), so the per-epoch choice
  cannot be recomputed exactly. B0 is NOT retrained. After B0's evaluation, scripts/b0_selection_audit.py re-scores
  best.pt from eval/val_scores.npz (logit AUC, group es-AUC, selection value under the fixed rule, saturation counts)
  -> results/b0_selection_audit_F3.{json,md}.
- F4 (scripts/evaluate_ces.py, cfes/metrics.evaluate_scores(thr_scores=)): validation-TPR threshold chosen AND applied
  on raw logits on every test set; ranks for AUC only. acc@val / fpr@val / tpr@val / fair@val change meaning (now one
  fixed logit threshold); AUC, 0.5 metrics, A1 and calibration unaffected.
- F5 / F6 (FF2 configs; not launched): configs/cfesd_ff2_public.yaml rebuilt = the resolved rev4 CFES-D run
  (CFES runs_rev4/clip_D_w0.3_s7/config.yaml: local_norm, risk 0.3 / tau 1 / include_views / n_min 2, Amendment-9
  val_oppoint with the start eligible) with only the dataset keys switched to FF2-public (train/SELECT/ff2_test,
  pairing ff2, FF2 face template, 3732 steps/epoch, public CFES-format pairs). New configs/ff2_public_p1.yaml = the
  resolved FF2 phase-1 recipe (runs_ff2full/clip_s7_p1) on public TRAIN/SELECT. New cfes/public_guard.py, called by
  scripts/train.py and scripts/train_ces.py: every FF2-public config (12: ces_ff2_public.yaml, ff2_public/*,
  cfesd_ff2_public.yaml, ff2_public_p1.yaml) REFUSES to start unless data.train.csv is the frozen public TRAIN (sha256
  e93a5f3a...bc36) and (phase 2) its init checkpoint's run dir holds PUBLIC_P1 (written by the finished public phase-1
  run, with the checkpoint sha256) or PUBLIC_LINEAGE (written by a guarded public run, e.g. B0 for the arms); phase 1
  refuses any init checkpoint. Run them with the CESEL copy of the CFES trainer (identical engine/model/losses).
- F7 (scripts/fair_a1.py, ces_posteval.py, cesel_report.py): a1_for(val, test, schemes) scores EVERY supplied test set
  with its configured scheme (incl. ff2_test); a set without a scheme raises instead of disappearing. The A1 rule is
  unchanged: CFES-D s7 reference still t = -46.946319580078125, DFDC i-F_FPR 16.301856900099107.
- F8 (cfes/data.cluster_key): strips the frame index only for Lin/Ding video-frame names (<0|1>_<video>_<frame>);
  anything else = full image identity (person_id overrides where the meta has it, as before). FF++ protocol unchanged:
  every train/val/test cluster key identical; leak-free-video drops ffpp_test 2,571 (same), DFDC / Celeb-DF / DFD 0
  (same). FF2-public ff2_test: 1,274 PixArt rows no longer dropped (now 0 drops).
- F9 (ces_posteval.py intervene()): deletion and cumulative curves rebuild the intervened score from retained cue
  terms + replacement constants + head bias; full deletion = the exact constant b + sum beta0 (AUC 50); alpha = 0 is
  the saved model score.
- F10 (cesel_report.py; also CLIPFAIR clipfair_report.py): md and json via unique temp files + fsync + atomic replace
  under a mkdir publish lock; a report never replaces one further along (complete beats partial; then more finished
  screen runs / A1 / interp files); --force overrides.
- FLOOR CELLS (documentation only; user: "keep the implemented group x class (and x family on FF2 with batch size
  304)"; behaviour unchanged): FF++ L_floor cells are (class, condition, group) with condition = manipulation family
  for fakes (data.families [swap, reenact]; Deepfakes / FaceSwap / FaceShifter -> swap, Face2Face / NeuralTextures ->
  reenact) and 0 for reals: (real, -, g) for the 8 Ding/Lin intersectional groups g, (fake, swap, g) and
  (fake, reenact, g) -> at most 24 cells per step; a cell enters when it holds >= 4 images of the step (176 images =
  64 balanced + 96 stratified [4 per (label, group, condition): 8x4 reals + 8x2x4 fakes] + 8 source pairs x 2); its
  reference = sg(mean utility over all groups of the same (class, condition)); delta 0.1; validated cues only; sum.
  So on FF++ the floor is family-conditioned too (group x class x family), as in README / DESIGN_NOTES.
- CLIPFAIR (separate copy; jobs 44074/44075 not started): metrics.py, data.py, train.py, evaluate_ces.py, fair_a1.py,
  ces_posteval.py were byte-identical to the pre-fix CESEL files (F3, F4, F7, F8, F9 shared) -> replaced by the fixed
  ones (+ cfes/public_guard.py, imported by train.py); evaluate_clipfair.py (F4) and clipfair_posteval.py (F7) patched;
  RUNLOCK (F1) and report publication (F10) fixed as above. tests/clipfair_checks.py ALL_PASS 18/18.
- Tests (CPU): tests/ces_checks.py ALL_PASS 17/17; new tests/test_runlock_f1.sh, test_report_f2_f10.py,
  test_codex_eval_fixes.py (F3/F4/F7/F9), test_cluster_key_f8.py, test_public_guard_f5_f6.py: all ALL_PASS;
  tests/rerun_codex_repros.py re-runs the Codex REPRO blocks: every defect gone, formula/A1 oracle still passes.
- End-to-end CPU dry run of the whole screen with the fixed code (CESEL_DRY=1, scratch root, stub backbone): driver ->
  new cesel_run.sh -> B0, validate, starvation, all 9 arms, evaluation, calibration, A1, interp, final report: all 12
  tasks rc 0, every RUNLOCK retired to its own tombstone, final report complete with no unresolved gate; cumulative
  full deletion AUC exactly 50.0; acc@val in logit units. STOP_CESEL was NOT needed (all fixes in place ~21:15, B0 at
  epoch 6 of 10); no PBS job was resubmitted or deleted.

## Amendment 5 (26 Oct ~00:55, user approved "Redesign: arms like B0" + "Retry with backoff"; before any redesigned run)
Why: in the warm-start screen (Amendment 1) every arm started from B0 best.pt, whose operating point became the
Amendment-9 reference (worst-group val FPR 17.2, group TPRs); every later epoch of every arm (B0-continued, +sens, +inv,
+floor, Full; 5 epochs each) failed the start-referenced checks, so all arms kept B0's checkpoint and the pre-registered
comparison is degenerate. DISCLOSED: before this amendment we saw those arms' per-epoch VALIDATION trajectories
(B0-continued FPR gap 29-41; Full 19.6-25; +sens 19-36; +inv 26-46; +floor 21-27; no epoch above B0's selection value
85.62). No test metric of any arm epoch was computed or seen (only the start checkpoint was evaluated). The warm-start
runs are kept as an exploratory record, not used for decisions.
Redesign (decisions unchanged otherwise): every arm (+sens, +inv, +floor, Full, C_randcue, C_shufgroup, C_novalid,
B0+CFES-D) trains exactly like B0: from runs/clip_s7_p1/last.pt (CFES phase 1), fresh heads/branches/CES state, 10 x 1976
steps, same seed/data order, same Amendment-9 selection with the start checkpoint as reference (as B0 had). The existing
B0 (runs/ffpp_B0_s7) is the comparison baseline (no B0-continued needed). The validated-cue JSON stays the one computed
from B0 best on validation (fixed before these runs). Keep/kill/success rules as before, with B0 in place of
B0-continued. Retry: a run that fails (e.g. CUDA OOM from another process taking the GPU after the free check) is
retried up to 4 attempts with a 10-20 min randomised wait, resuming from last.pt. Order after this screen: the 4 CLIP
baselines (Amendment 3), then FF2-public.
Implementation + launch (26 Oct ~00:45-01:10, before any redesigned number):
- Code (backups *.bak_amend5): scripts/cesel_pipeline.py CESEL_DESIGN=p1 (default; x5 = the exploratory warm-start
  graph): B0 / validate / starve_B0 (done) -> 8 tasks p1_<arm> (run/cesel_run.sh arm_p1: configs/ffpp/<arm>.yaml +
  model.init_backbone_ckpt=CFES clip_s7_p1/last.pt + ces.validated_json=B0's JSON; epochs = config 10, as B0) ->
  runs/ffpp_p1_<arm>_s7; each arm -> CPU starvation table results/p1_cue_starvation/. Retry: <= 4 failed attempts per
  run in total across runners (shared record runs/attempts/p1_<task>.jsonl), uniform 600-1200 s wait, resume from
  last.pt. STOP grace (CESEL_STOP_GRACE_S=7200 in the new PBS scripts): STOP_CESEL during a driver's first 2 h = wait.
  cesel_run.sh (mv-swapped) honours CESEL_STOP_FILE. scripts/cesel_report.py --design p1 -> results/cesel_ffpp_screen_p1.*
  (B0 is the reference; same decide() rules; per-arm starvation table for promoted arms); the x5 report is marked
  EXPLORATORY. New tests/test_amend5_p1.py; all CPU tests ALL_PASS (runlock test needs CESEL_STOP_FILE unset STOP).
- Resolved-config diff (results/amend5_config_diff.txt): vs B0 only ces.validated_json + the arm's CES / CFES-D terms
  differ; init, budget, lr, schedule, batch, seed, selection identical; launched configs == pre-resolved ones.
- CPU dry run (stub, two runners, injected failures, STOP grace): all 19 tasks rc 0, retries logged, final p1 report.
- Launch: queue limit max_user_run = 2 (gpu and workq) made "new runner RUNNING before the old one ends" impossible;
  order used: new 44104 (gpu02) + 44105 (gpu01, user-held) submitted; 44074 -> afterany:44105, 44075 -> afterany:44104;
  old 44067 children killed -> 44067 ended (gpu01 protected by workq jobs) -> 44104 RUNNING on gpu02 -> old 44068
  child killed -> 44068 ended (ICLR_Misinfo colour_autodiscard_val.sh on gpu02 survived) -> STOP_CESEL removed ->
  44105 released, RUNNING. Arms start from the same epoch -1 state as B0 (val AUC 56.759).

## Amendment 6 (26 Oct ~04:45; user: "go ahead, use all GPUs and train it fast", final-model spec agreed; written before any CESel-MA, Ding-GDA-connected or new FairReL number exists, and before the Amendment-5 screen report)
Why: reports/cvpr_gap_20261026 (00_SUMMARY.md: "FairReL (BMVC'26, same CLIP rank-1 setup) is missing" and "Ding's
released GDA loss is inert (inside no_grad), so a 'GDA connected' variant is needed to avoid a 'broken baseline'
critique"; 03_experiments_evidence.md M9 "Ding-CLIP with GDA connected (gda_grad: true) x 3 seeds"; 01_novelty_
positioning.md G2 "Blocker": FairReL, official code, 3 seeds, Lin's split, reusing FAIRREL_REPRO). Code: agents/CESEL_MA
(copy of CESEL; CESEL and CLIPFAIR code untouched).
Matched baselines added (both enter the primary-endpoint comparator pool of decisions/2026-10-26-final-model-spec.md):
- **Ding SFD + GDA, GDA connected**: the Amendment-3 CLIP port with ONLY clipfair.ding.gda_grad=true (the Sinkhorn
  fairness term back-propagates; everything else = the as-released row: same SFD channels runs/ffpp_ding_sfd_s<seed>,
  lambda 0.005, blur 5e-4, budget 10 x 1976, Amendment-9 selection, evaluation, A1). CLIPFAIR scripts called unchanged
  by CESEL_MA/run/ma_run.sh ding_gdagrad (sets CLIPFAIR_ALLOW_GDA_GRAD=1, which this amendment authorises) ->
  CLIPFAIR/runs/ffpp_ding_gdagrad_s<seed>; seeds 7/17/27.
- **FairReL**: the OFFICIAL code and config (agents/FAIRREL_REPRO/code/run_task.sh FAIRREL <seed>, unchanged; same CLIP
  ViT-L/14 rank-1 backbone; its own Lin-split extraction, 10 epochs, its own best.pt rule); seed 7 exists (DONE),
  seeds 17/27 run now. DISCLOSED: not budget / selection-matched to Amendment 3 (no matched-recipe port exists); its
  FF++-validation Amendment-9 value (for "best matched-CLIP baseline by its own validation selection value") is
  computed from its best.pt validation scores before any comparison (open item; no test number is read for it).
CESel-MA implementation choices (fixed now; the spec leaves them open):
- Labels = agents/DEMO_AXES/labels/labels_ffpp_{train,val}.csv (training and validation only): gender and gender x race
  = AI-Face cleaner v2 (cl_male, cl_race; Xu fallback flagged in cl_src; FF++ val has Xu only); age = age4_if
  (InsightFace per-video median age, bins <30 / 30-44 / 45-59 / 60+) because AI-Face's categorical predicted age cannot
  be cut at the spec's edges (age4_aif = sensitivity only); ITA = per-video median ita_gw (ita_vid), bins = the
  declared ita4 (FF++ TRAIN quartiles 19.17 / 35.22 / 53.37); kernel anchors = 8 points spanning the TRAIN ita_vid range
  (-85.0 .. 89.6, h = range / 8 = 21.8).
- L_floor-MA: gender / age / skin cells per class (spec: sg(ubar_{k,c}) pooled over manipulation conditions); the
  gender x race part = the CESel cells (class x condition x group, n >= 4, delta 0.1) unchanged except the cleaner
  group labels; reals weighted 2 on every part; L_inv subspaces and the stratified sampler keep the Lin groups.
- Frequency warm-up: own linear head on c_freq, balanced BCE over the CES pool, weight 0.2 for the first
  1 x steps_per_epoch steps, 0 afterwards (the head never enters the score). Drop rule: ces_validate_cues on MA best ->
  validated_cues_post.json; frequency not valid (bootstrap rule) -> reported as three-cue (no retraining).
- Selection gap: groups with >= 2 distinct REAL validation videos (the FPR denominator); eligibility box and
  start-referenced checks unchanged (Lin groups). All MA seeds use B0 s7's validated-cue JSON (as the screen arms) and
  start from CFES runs/clip_s<seed>_p1.
- Gate (scripts/gate_check.py, validation only): PASS iff Full or +floor has sel - sel(B0) > 2.06 and no negative
  control (C_randcue, C_shufgroup, C_novalid) within 2.06 of it; report must be complete with the spread verified;
  PENDING after 2 Nov = FAIL. MA base = the passing arm with the higher value (Full -> configs/ma_ffpp.yaml, +floor
  only -> configs/ma_floor_ffpp.yaml).
- "Not worse" (seed 7): val_oppoint_ma recomputed from saved validation logits for MA s7 and for the screen's winner
  (gate.json single_axis_arm; the passing arm if the report has no winner), same code and labels; not worse iff MA >=
  winner -> MA seeds 17 / 27. FF2-public final model: MA if not worse, else the single-axis winner; gate FAIL -> CFES-D.
Order (user 26 Oct: "first beat Ding's numbers, thereafter our own dataset"): BEAT_DING tasks (their own decision file
decisions/2026-10-26-beat-ding-plan.md) -> CESel-MA FF++ (gate PASS only) -> matched-CLIP baselines s7 (PG-FDD, DAW,
DAG, FairReL, Ding as released, Ding GDA connected) -> the same s17 / s27 -> Ding official repro s17 / s27
(DINGREPRO_LIN run_task_full.sh) and Arm E w10 s17 / s27 (CFES rev4) -> all FairForge-2 (public phase 1, CFES-D, B0,
validated JSON, final model). Runner: CESEL_MA/scripts/priority_runner.py + run/tasks.yaml (two PBS runners, gpu01
GPUs {0,1,2,3,4,6,7} <= 4, gpu02 GPUs {6,7} <= 2, same free rule / RUNLOCK / retry / STOP design); PBS 44074 / 44075
(never ran) replaced.

## Amendment 7 (26 Oct ~12:45; matched-CLIP baselines; selection fallback for degenerate runs)
Problem (validation logs only): the Amendment-9 eligibility compares every epoch with the START checkpoint's group
operating point. The baselines start from a FRESH (random) head, so that reference is arbitrary: a random head can
give some group a TPR near 100%, which no trained epoch can match within 2 points.
- DAW-FDD s7: all 10 trained epochs ineligible (val AUC 92.8–93.2, pooled TPR ~84%, FPR ~14%). The run therefore
  kept its random start head; its test AUC (44.5) measures that start, not DAW-FDD.
- PG-FDD s7: epochs 0–3 ineligible, epoch 4 onward eligible.

Rule, applied uniformly to every matched-CLIP baseline run:
- The pre-registered Amendment-9 rule stands.
- If NO trained epoch is eligible, the run falls back to the pooled-only eligibility (val TPR ≥ 80%, FPR ≤ 20%) with
  the same selection value, as in TTD Amendment 4.
- Implementation: each run also saves `best_pooled.pt`; the fallback uses it.
- Degenerate runs are re-run with this code: DAW-FDD s7 (archived as `ffpp_daw_s7_DEGENERATE_oct26`), plus DAW-FDD
  s17 and PG-FDD s17, which were stopped early and restarted with the new code.
- Disclosed: this was decided after seeing DAW-FDD s7's (meaningless) start-head test numbers.
