# CES-FL audit and actual execution report

Observation: 2026-11-03T06:21:11.119000+00:00 management UTC. Refresh the separately timestamped LIVE_STATE.json before relying on a process state.

The bounded pilot is RUNNING. Software and input gates pass. Neither candidate has yet established an improvement over the existing D control. No new Ding-model computation, new FairForge-2 experiment, TEST inference, demographic image generation, or extra training seed was launched.

## Source and protocol

- Starting results commit: `1f3098ea66f5537bf0c947957d0f7078fe85efcd`.
- Starting scientific commit: `c332fd34fb603b8919b15fee08e8d022fb9f57c8`; SHA256 `89530025a00b9b0405fba7026d3eed5f2749d86620174ad3ffde93a44f8b770b`.
- New immutable scientific commit: `8d5f0996a3f1cc02fbeaff9d25c1d95f714e8e77`; SHA256 `0eeafd838da3ca158336dcfe3b8195842dd15bc9fecd67ebbaf2ea03ba3c2680`.
- Current remote main was independently observed as `67d2308225e422a91c715c40427f40c3df6812cf`, dated 2 November. It lacks the newer result snapshot. No running checkout was pulled or rewritten.
- New branch: `experiment/cesfl-support-warmup-s17-20261103`.
- Exact frozen configurations: `E_support_bound.json`, `F_support_ramp3_bound.json`; file and semantic hashes: `evidence/BOUND_CONFIG_HASHES.json`.

Architecture remains one Xception encoder, global/local predictions, learned image/patch gates and fusion: 21,594,159 total parameters, 7,575,591 trainable. Exit-block tuning and frozen BatchNorm buffers remain unchanged. Gates are learned weights; local maps are signed logit contributions. The independently measured initial encoder/head/gate/fusion hashes match across candidates and applicable old controls.

Both candidates use seed 17, ten epochs, 2,380 original minibatches per epoch, batch 32, fixed P50 initialization, AdamW (heads 0.001, encoder exit 0.00001, weight decay 0.0001), fixed preprocessing, auxiliary sampling and all-original BCE coverage. F differs from E in only alignment_warmup_epochs (0 versus 3). Native --arm D denotes our CES-FL all-objective configuration. No Ding algorithm is executed.

## Focused hypothesis and concrete limitations

E replaces the two-sided soft error disparity with a one-sided penalty relative to a detached TRAIN-prior-weighted reference, multiplied by sampled recorded-contributor diversity U/n. This removes the direct error-term incentive to increase an already-better cell’s error. It does not prevent shared parameters or other objectives from harming that group. The factor-two normalization, detached reference and diversity discount form one package; E versus D cannot attribute an effect to each component separately.

F adds only a three-epoch alignment ramp (1/3, 2/3, 1 thereafter). The full equations and unchanged total-loss weights are fixed in README.md and decision_protocol.json, committed before training. No weight, threshold or group-specific validation rule will be changed after seeing outcomes.

Fresh TRAIN-only diagnostics show alignment/classification gradient conflict on four of five axes at both fresh heads and the historical selected D checkpoint. At that checkpoint the age cosine is -0.4164, skin -0.5919 and intersection -0.5751. Revised error-control gradient norms (about 1.1–1.8) are much larger than BCE (about 0.00308) there. This is a risk that the new error term could dominate, not proof of what caused historical regressions. Diagnostics made no optimizer update and read no TEST data.

Auxiliary cells still contain four samples per class/group; response quantiles and soft-error estimates are noisy. Repeated contributors are not independent observations. Discounts can reduce already-sparse minority influence. A fixed-temperature soft error around an epoch-held TRAIN threshold is still a surrogate for hard FPR/FNR at the selected validation threshold. Response alignment still cancels a shared group-specific score offset; it does not establish error parity. These limitations were not silently solved by the revision.

Old C/D selected epoch 1 after ten complete epochs. Training BCE and alignment/sensitivity losses decrease while validation AUC drops, consistent with possible overfitting/objective competition. These observations do not isolate a cause. Keep the fixed AUC-only selection rather than selecting a convenient fairness epoch.

## Actual gates

| Check | Actual result and evidence |
|---|---|
| Legacy/new CPU unit tests | 13 + 9 pass; evidence/cpu_*_units_final.log |
| Resource guards | 5 pass; evidence/operator_guards_v2.log |
| Full CPU / CUDA synthetic smoke | PASS; evidence/CPU_SMOKE_PASS.json and CUDA_SMOKE_PASS.json |
| Architecture, gradients, frozen BN, decomposition, checkpoint/resume | Included in the source-bound full smoke; fixture-only, not performance evidence |
| Both full real-data preflights | PASS; 101,525 decoded TRAIN/VAL images each; E/F_PREFLIGHT_PASS.json |
| TRAIN/VAL byte overlap | Zero; 163 recorded identity/source exceptions checked against the unchanged six-key audit; biological separation not established |
| New representative intervention previews/review | 88 cases per config, 22 originals / 18 recorded IDs; E freshly visually inspected, F actual pixels byte-identical |
| New native qualifications | PASS, separately bound to each new source/config; no old receipt transplanted |
| Real CUDA backward | E 5.8390 seconds, F 0.7777 seconds; discarded model, finite gradients; not a throughput comparison |
| TRAIN gradient diagnostics | Ten snapshots, zero optimizer updates; evidence/TRAIN_GRADIENTS.json |
| Frozen old CES-FL A/C/D validation replay | Exact native metric replay and unchanged checkpoint hashes; CONTROL_VALIDATION_REPLAY.json |

Review was performed by the named AI visual reviewer, not a fabricated human reviewer. It qualifies a limited representative procedural recipe; it does not establish identity preservation, human demographic-label truth, causal cue specificity, real-forgery validity or per-image validity for every epoch. Glasses, blur, pose, nuisance processing and perceptual-difficulty confounds remain. No Qwen edit is represented as a validated manipulation quartet or demographic counterfactual.

All original 76,139 TRAIN rows retain BCE each epoch. The first completed epochs record exactly 12,926 original-real procedural pairs, including the 250 unknown-skin rows. All group/epoch original coverage, auxiliary draws, repeated contributors and ramp coefficients are published as aggregate exposure, not broad independent demographic-edit coverage.

## Validation comparison

A/B/C/D below are completed applicable CES-FL controls, not Ding. New values are selected-so-far by AUC and may change until all ten epochs finish. F_FPR uses fractions; 0.379345 corresponds to 37.9345% in a paper-percent display. AUC, achieved FPR and FNR columns use percent.

| Arm | State | Completed epochs | Selected epoch | AUC % | Intersection F_FPR | Gender F_FPR | Age F_FPR | Worst intersection AUC % | Achieved FPR % | Fake FNR % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|A|COMPLETED|10/10|4|98.426411|0.505265|0.043699|0.228592|93.959518|9.995112|4.146708|
|B|COMPLETED|10/10|2|98.406797|0.468638|0.041739|0.230098|93.967743|9.995112|4.235935|
|C|COMPLETED|10/10|1|98.307659|0.480534|0.043699|0.276733|93.620499|9.995112|3.709965|
|D|COMPLETED|10/10|1|98.252564|0.379345|0.053498|0.303616|93.851876|9.995112|3.963558|
|E_support|RUNNING; interim|1/10|1|98.210202|0.410801|0.057418|0.306131|93.831134|9.995112|4.062177|
|F_support_ramp3|RUNNING; interim|3/10|1|98.319331|0.495578|0.048598|0.296048|93.770339|9.995112|3.897812|

All supported axes, subgroup n_real/n_fake, FPR/FNR/gaps, family metrics and checkpoint hashes are in the public LIVE_STATE.json. The 10% calibration request achieved 409/4,092 = 9.9951124% on validation real images in the observed selected checkpoints. This is not a TEST FPR guarantee. No TEST scores participate in calibration or selection.

The historical D-over-C intersection gain is descriptive: 65.8834% comes from reducing two false positives in Female-Black, a subgroup of only 30 real validation images / two recorded components. D worsens gender and age fairness. Current candidates must satisfy the additional prospectively fixed guards; an improved intersection number alone is insufficient.

## Uncertainty and progression decision

The exact decision_protocol.json was fixed before candidate updates: intersection F_FPR strictly below old D, AUC no more than 0.5 percentage points below old A, gender/age/FNR/worst-group guards relative to old C, and at least 90% usable paired primary bootstrap replicates with the 95% interval entirely below zero against D. No guard will be relaxed for sparse support. Reject if primary/AUC fail; revise if other guards or uncertainty fail; continue only if every guard passes. No third configuration.

Both arms must finish and pass native selected-checkpoint replay before the finite execution controller submits one CPU-only analysis job for 1,000 paired connected-recorded-contributor replicates per prescribed comparison. Thresholds and eligibility remain fixed; fixed-0.5 diagnostics and selected effective ramp are also reported. Validation has 209 components, largest 360 rows; recorded IDs do not prove biological identity. Bootstrap uncertainty is conditional on one seed and cannot estimate seed variability or provide independent TEST confirmation.

A two-replicate CPU timing probe is explicitly not a confidence interval or decision. It estimated approximately 2.7 CPU-hours for the four full comparisons; this is a coarse timing estimate, and the analysis requests four CPU-hours. Its first attempt failed before bootstrap in the unrelated default internvl Python (missing NumPy extension); aiface retry succeeded. Actual training/analysis commands already use explicit aiface. Failure evidence is preserved; no GPU time or training was affected.

## Resource state and continuation

Actual controllers/children: E gpu01 3372062/3372134 on physical GPU4; F gpu02 347471/347559 on physical GPU5. CUDA UUID bindings are recorded in launch logs and the ledger. Route is the explicitly user-authorized shared pool-v2 lease, not a PBS or exclusive GPU allocation. No existing job was preempted or modified. Old idle FF2-dependent coordinator was deferred; existing Ding training remains untouched.

The original 24 TOTAL GPU-hour ledger is authoritative. It included 10.058886 hours before this revision; new smoke/checks/diagnostics used 0.083559 hours before training. Both active cards are charged by full task-held walltime. Each new arm reserves up to 4.5 hours conservatively; actual time is continuously updated. Native two-hour epoch-boundary stops can resume the identical ten-epoch run to cumulative three/four-hour allowances within that shared ledger, never under a different source/config/review.

Measured first epoch: E 453.0 seconds, F 296.9 seconds; F second epoch 277.2 seconds. At those speeds the ten-epoch task-held cost is approximately 1.26 and 0.80 GPU-hours respectively, plus replay/startup. Contention can change this. This is more specific than historical 1.2–3.2-hour estimates, not an authorized budget expansion or an ETA guarantee.

Publisher PID3543851 observes actual processes and updates the dedicated branch every ten minutes, for at most 24 hours. It cannot train, submit jobs or create approvals. CPU-analysis execution controller PID3543499 waits genuine completions before the one CPU analysis submission and cannot train, promote or read TEST. Frozen helper hashes and launch records are retained. Both PIDs were checked live in this observation period; future states require refresh.

The first launch attempt failed before GPU reservation because node git is absent. The preserved failure was corrected only in the separately committed resource executor, using detached-HEAD and source-byte verification plus management checkout evidence; scientific source and hashes were unchanged.

## Ding reference and novelty boundaries

See DING_COMPARISON_AUDIT.md for pinned primary paper, archived source/result hashes, dataset membership limits, compression/preprocessing, initialization, staged budgets, optimizer/BN, selection, thresholds, formulas and published versus historical results. No Ding model or checkpoint was executed or loaded. Published TEST numbers and historical detached-GDA results remain separately labeled. Our VAL values are not compared with paper TEST values as evidence of a win. The archived released fairness loss is detached; its existing records do not certify a corrected differentiable method.

The public-release counts alone do not establish exact author membership outside the historically verified DFDC manifest. Known contributor overlap and duplicate handling, ambiguous published staging/initialization/threshold details, inherited age/skin label quality, and missing current-method ResNet-50 Table3 remain limitations. Prior broader-project TEST exposure is disclosed. A later final model/config/evaluation plan must be frozen before any separately authorized TEST inference; no TEST command or automatic promotion is attached.

One-sided error control, diversity weighting and ramping are optimization ideas, not a stand-alone novelty claim. The scoped research claim remains whether conditional, matched procedural response training adds fairness beyond applicable error/BCE/control baselines without unacceptable detection or other-axis regressions. Current visual checks and software tests do not validate causal forgery cues, generalization or SOTA. Outperforming Ding would require a defensibly matched, frozen held-out evaluation showing detection and fairness benefit with uncertainty and disclosed protocol limitations. This seed-17 validation pilot cannot establish that claim.

**Current decision: continue the two already-authorized fixed-budget pilots; do not promote either candidate or claim a win.** Final continue/revise/reject decisions remain pending native completion and the prespecified paired analysis.
