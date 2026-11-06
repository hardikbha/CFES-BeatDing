## Amendment 4 (06 Nov ~03:45 IST): ResNet priority, Codex (gpt-6-luna xhigh) strategy adopted
- Source: agents/RESNET_STRAT/2026-11-06-codex-resnet-strategy.md. Codex forecast: 25/40 on Table 3 is not credible;
  realistic 10-16/40, optimistic 18.
- **Arms (ResNet-50, Ding recipe):** SGD 1e-3, mom .9, wd 5e-3, 64+64 natural sampler, no aug, 50 ep.
  - **R0-SWA:** Ding's CE path unchanged, plus SWA over epochs 40-49 with BN recomputed on FF++ train.
  - **R3-SWA:** R0-SWA plus, every 4th update, a separate fair batch (8 real + 8 fake per gender x race cell,
    independent RNG). Its loss is 4 x [0.5 L_C + 0.02 L_R]:
    - L_C = class-conditional group CORAL (mean + population covariance) on L2-normalised pooled 2048-d features;
    - L_R = mean of the two largest per-cell real means of softplus(s + 1), with s = z_fake - z_real.
  - No balanced CE sampler, no W1, no full-strength tail.
- **Seeds:** 37, 47, 57, all fresh confirmatory. Primary metric: op05 vs Ding published T3. Secondary: one global
  FF++-val-real 90th-percentile threshold.
- **Criteria:**
  - target >= 25/40 mean wins AND mean AUC >= Ding on all 4 sets;
  - method evidence: >= 3/40 more mean wins than paired R0-SWA, and <= 0.5 AUC loss vs R0-SWA on any set.
- **Kill rule:** kill an R3 seed if its FF++ val AUC is > 2 pt below its paired R0 at both epochs 10 and 15, or if
  the aux gradient norm exceeds CE on > 20% of fair steps. Non-finite values = immediate fail. Never substitute a
  best-val epoch.
- **Execution:** direct SSH on gpu01/gpu02 (user, 06 Nov):
  - per-epoch checkpoints, plus a login-node watchdog that resumes after PBS kills ssh processes on a node;
  - allowed GPUs only.

## Amendment 5 (06 Nov ~10:45 IST): user goal "beat Ding on ResNet, esp. Celeb-DF, very soon"
- **Amendment-4 results** (seeds 37/47/57, op05, vs Ding T3 published):
  - R0-SWA: 14.0 ± 1.0 /40 per seed; mean-row 15/40 (FF++ 5, DFDC 9, Celeb-DF 0, DFD 2). AUC below Ding on all
    4 sets (Celeb-DF 75.72 vs 77.35).
  - R3-SWA: all 3 seeds KILLED in epoch 1 by the aux > CE gradient rule. Design flaw: the CE gradient shrinks as
    training fits, so the ratio rule fires even at lambda 0.02. No R3 result exists.
- **Celeb-DF diagnosis** (reported, not selected on):
  - Ding's Celeb-DF intersection scheme has 3 cells: male-black n=1922, nonmale-white n=25432, nonmale-others
    n=120 (90 reals). Our nonmale-others FPR is 0.51-0.67 vs overall 0.34.
  - es-AUC loses on the group-AUC gap: Black ~0.66, Others ~0.59-0.68, vs overall ~0.757.
  - The F_DP columns are label-blind and favour flat predictors (oracle CDF i-F_DP 59.6 vs Ding 9.11).
  - Threshold choice cannot change AUC or es-AUC (4 of the 10 columns).
- **New arms (ResNet-50, Ding recipe + SWA 40-49), seeds 37/47/57:**
  - **R0-SBI-SWA:** Self-Blended Images (Shiohara & Yamasaki CVPR'22) generated online from FF++ TRAIN reals only.
    They replace exactly 32 of the 64 fake slots per batch (label fake). This is the recipe-level generalisation
    lever, a matched control.
  - **R3-SBI-SWA:** R0-SBI-SWA plus Amendment-4's R3 aux loss, unchanged (4 x [0.5 L_C + 0.02 L_R], every 4th
    update).
- **Kill rule (amended before any R3-SBI run):** the aux > CE rule is DROPPED (shown to be degenerate); aux/CE stays
  logged as a diagnostic. Kept: the val-AUC rule (> 2 pt below paired R0 at both ep10 and ep15), and non-finite =
  fail.
- **Reporting:** op05 (primary) and the val-FPR10 threshold (secondary), vs Ding T3 published, per-seed plus mean.
  No test-based selection of seeds, epochs or thresholds; all three seeds are reported.
- **Comparisons:** R0-SWA (no SBI) vs R0-SBI-SWA = SBI effect; R0-SBI-SWA vs R3-SBI-SWA = our fairness term.
