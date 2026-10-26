# C-FER pilot — binding spec (user pseudocode, 24 Oct 2026) + resolutions of what it leaves open
User request: implement the pasted C-FER pseudocode, verify it in a reviewer loop, run it on Tiny Tiny Ding (TTD) in one
PBS job (4 GPUs, 8 CPUs; fewer GPUs if needed), report results. Supersedes decisions/2026-10-23-1700-cfer-pilot.md
where they differ. The TTD design + Amendment 1 (decisions/2026-10-23-1600-tiny-tiny-ding.md) stays.

## Order of work inside the job (each step gated on the previous one)
1. Arm E weight: full-data arm E w=10 seed 7 (rev4_arm2.sh E 7 10), because the spec says wait for it. Weight =
   argmax over {1,3,10} of the Amendment-9 validation selection value of the full-data seed-7 runs (w1 87.77, w3 88.01).
2. TTD acceptance runs (TTD-train/val, full test sets): control, A, B, C, D, E(w=3) at seed 7; D at seeds 17, 27.
   Acceptance: Spearman(TTD vs full mean cross-domain AUC, 6 seed-7 arms) >= 0.80 AND mean |TTD AUC - full AUC| <= 2
   points per test set. If it fails: STOP, report, do not run or interpret steps 3-5.
3. R1 = TTD D (seeds 7, 17) — reuse the acceptance runs, no retraining. R2 = TTD D + arm E at the selected weight,
   seeds 7 and 17, existing arm-E code (sum_g |FPR_g - FPR_pooled|, running 90th-pct real threshold, tau 0.5).
4. R3 = R2 + reliability-aware fusion, no reliability supervision. Gate bounds: chosen on TTD-val at seed 7 from
   {[0.10,0.90], [0.25,0.75]} by the selection value; the winner is used for R3 seed 17 and for R4.
5. R4 = R3 + cross-fitted reliability supervision. K=3 folds per seed, grouped by source-video family, R2 recipe on
   each fold's train part, out-of-fold logits -> target t = 1-|sigmoid(z_oof)-y| (detached). Patch targets only if the
   registered-pair fraction >= 0.30 (TTD build measured 1023/9033 = 0.11 -> image-level targets only; declared).

## Resolved details
- Exact reuse: R1/R2 run the existing CFES-D / arm-E code paths unchanged (bit-identical loss+grad check with all new
  options off). Pair loss = the existing CFES sensitivity (softplus margin) with ONE shared sampled intervention per
  (fake, source) pair — the reviewer must confirm the existing code shares parameters; if not, that is a bug to report.
- Reliability head inputs: pooled feature, patch features, instability |sg(z_global) - sg(z_view)| where z_view is the
  model's own logit on a nuisance view, plus z_global and z_patches. Training: the view is the stability view already
  drawn for that image. Test/val: one DETERMINISTIC view (background blur, rho = 4, face kept) so evaluation is
  reproducible; demographic/domain labels are never inputs. This doubles inference cost (disclosed).
- Fusion: z_fused = alpha*z_global + (1-alpha)*patch_score; alpha = lo + (hi-lo)*sigmoid(a(r_global));
  patch_score = top-8 cells of z_patches weighted by softmax of r_patches over those 8, with CFES-D's local
  normalisation applied to the pooled patch score.
- Selection and calibration: the same Amendment-9 validation rule and intercept fitting as all other runs, on TTD-val.
- Decision (validation only): keep R3 if its TTD-val selection value exceeds R2's by more than the seed spread
  (|s7 - s17| of R2) on both seeds; keep R4 only if it further exceeds R3 the same way. Test metrics reported, never
  used to decide. TTD results alone establish neither novelty nor cross-domain generalisation.
- Seeds 7 and 17. num_workers = 1 per run (8 CPUs for up to 4 concurrent runs).
- Compute: allowed GPUs only (gpu01:{0,1,3,4}); never other users' processes; one PBS job on gpu01, ngpus=4, ncpus=8.

## Amendment 1 (after review round 1, before any run; review A/results/ttd/2026-10-24-cfer-review.md)
- B1 fix (a): each R4 fold model gets its OWN phase 1 trained only on that fold's TTD train part (phase-1 recipe, steps
  scaled to the fold size as TTD scales phase 2), then the R2 phase-2 recipe; out-of-fold logits come from it. Reason:
  the shared phase-1 checkpoint was trained on all FF++ train (100% TTD-train accuracy, targets ~1.0), which violates
  "a sample's target must come from a model that did not train on that sample". Declared check: if > 90% of OOF
  targets exceed 0.9, R4 is reported as "uninformative" and excluded from the keep decision.
- B2 fix: R3/R4 start exactly at R2's starting score: gate initialised at 2/3 (gate_b = logit((2/3-lo)/(hi-lo))),
  cls_head and the last patch-head layer scaled by 1.5; cfer_checks asserts |start(R3) - start(R2)| < 1e-4.
- R4 reliability-supervision weight lambda_rel = 0.5 (declared; the pseudocode leaves it open).
- Keep rule: runs whose selection value is -1e9 (no eligible checkpoint) count as "no improvement". R3 at seed 7 is the
  better of two gate runs picked on the same validation set (optimistic); the seed-17 R3-R2 margin is reported
  separately and the decision still requires both seeds.
- Pipeline: a GPU counts as free with >= 60 GB free; PBS mem 200gb; a <= 10-min GPU smoke of R3/R4 (bf16) at job
  start; cfer_oof asserts each fold run used its own fold's csv. Expected wall time 20-30 h on 4 GPUs / 8 CPUs.
- Acceptance compares TTD with full-data TEST AUCs by definition (proxy fidelity); it chooses no model.

## Amendment 2 (23 Oct 22:58, user approval; no result seen yet)
- User approved using the free gpu01 GPUs 2, 6, 7 in addition to {0,1,3,4} (never other users' processes; a GPU is
  used only if >= 60 GB free and util < 10%), and resubmitting with 32 CPUs. Job 43892 is replaced by a job with
  ncpus=32, ngpus=7, mem=300gb, ALLOWED_GPUS=0,1,2,3,4,6,7, MAX_CONC=7, data.num_workers=3 per run (the data stream does
  not depend on the worker count: no worker-side randomness). Finished runs are kept; interrupted runs resume from
  their last epoch checkpoint. Nothing else changes.

## Amendment 3 (24 Oct, user-approved "option 1"; second acceptance attempt, disclosed)
Attempt 1 failed acceptance (rho -0.03; mean |dAUC| 2.5-4.2). Diagnosis (no pilot run existed): on TTD-val (some groups
~30 reals) the Amendment-9 eligibility checks rejected every trained checkpoint of A, B, E (s7) and D (s17), so those runs
kept their phase-1 start and the ranking was meaningless. Fix, fixed before any new run: TTD keeps its 20% training
subset, but ALL checkpoint selection and intercept calibration (acceptance runs, R2-R4, fold models) use the FULL FF++
validation set (DINGREPRO_LIN/data/ff++_ding/{realval,fakeval}.csv, 25,386 images), as the full-data runs do. Same
acceptance rule (rho >= 0.80 AND mean |dAUC| <= 2 per test set); if it fails again: stop, report, no third attempt
without the user. Attempt-1 runs moved to A/runs_ttd_attempt1_tinyval/ (kept, not reused). Full-data E grid unchanged
(chosen weight 10).

## Amendment 4 (24 Oct ~13:00, user approved "Relax rule on TTD"; third acceptance attempt, disclosed)
Attempt 2 (job 43906) was stopped before any acceptance or test number was seen; only VALIDATION logs were read.
Diagnosis: with the full FF++ val, every trained epoch of TTD arms A (epochs 0-8) and B (0-6) was still ineligible. The
binding check is the start-referenced group-TPR check (each group's TPR >= start - 2): male,asian val TPR 68.5-74.3 vs
>= 74.7 required (start 76.7 from the full-data phase-1 checkpoint); B epoch 2 missed by 0.4 with val AUC 92.99 > start
92.33. Training phase 2 on 20% of the data costs the largest group 1-5 TPR points, so the full-data rule pins TTD runs
to their start and the proxy cannot rank arms.
Change, for ALL TTD-trained runs only (acceptance arms A-E, R2, R3, R4, fold phase-2 runs; configs/ttd/clip_{A..E}.yaml
`select: {ref_checks: false}`, inherited by configs/cfer/*): the eligibility keeps pooled val TPR >= 80% and pooled val
FPR <= 20% at the fitted intercept, and DROPS the two start-referenced checks (worst-group FPR <= start; every group TPR
>= start - 2). Selection value unchanged (val AUC - 0.5 x val FPR gap). Full-data runs (acceptance reference, step-1 E
grid) keep the full rule. The control arm uses its own metric (unaffected; its finished run is kept, eval re-run).
Disclosed consequence: TTD selection is less conservative than full-data selection, so acceptance now also tests
whether a proxy with this looser rule ranks the arms like the full-data runs do. Same acceptance rule (rho >= 0.80 AND
mean |dAUC| <= 2 per test set). If it fails: stop and report; no fourth attempt without the user. Attempt-2 A/B runs
moved to A/runs_ttd_attempt2_strict/ (kept, not reused); log -> logs/cfer_pipeline_attempt2.log.
