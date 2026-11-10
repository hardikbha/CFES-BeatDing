# Full L_H plus response-informed channel selection — 10 November 2026

A proposed feature-selection extension, with software checks rather than new benchmark results. The standalone implementation is [ces_fl_lh_channel_selection.py](../../src/ces_fl_lh_channel_selection.py). It embeds the existing Xception/L_H implementation and requires no other project Python file. Dependencies are Python3.9+, torch, numpy, Pillow and scikit-learn; local data, checkpoint and receipts are external inputs. No model/package downloads occur automatically.

The final figure is [L_H_Channel_Selection_Full.png](figures/L_H_Channel_Selection_Full.png). The earlier isolated frozen-encoder/BCE drawing is superseded for this request: the user explicitly requested the full L_H hierarchy and response objectives. This proposal keeps those objectives and the exit-layer adaptation policy. The mask-selection reference itself is frozen.

## Architecture and retained objectives

All views use Xception → fixed channel mask → global average pooling → linear2048→1. Early encoder layers and BatchNorm running buffers remain frozen. Exit layers and the classifier learn. Group labels enter the auxiliary losses and TRAIN-only channel audit, never detector inference. One selected checkpoint supplies one global validation-calibrated threshold.

All original L_H terms remain: inverse(label,ids8)-cell-weighted original BCE; hierarchical real-error control on the joint gender×race axis; the other original soft-error terms; control stability; local procedural sensitivity; and matched response alignment. In the inherited implementation:

- Control: mean absolute difference between asinh(z_control) and asinh(z_original).
- Response: asinh(z_probe − z_control), with a positive margin0.5 sensitivity penalty.
- Alignment: mean absolute deviation of sorted equal-count group responses from their mean quantiles, within a matched operator/strength event.
- Hierarchy: joint real risks and TRAIN-proportion-weighted parent risks above a detached pooled real-risk reference; fake-error and other-axis terms remain.

Loss weights fair/sensitivity/alignment are0.1, control weight0.5; auxiliary events occur every4steps with factor4. `evidence/LH_OBJECTIVES_UNCHANGED.json` checks the original objective/preprocessing functions structurally against the archived functional core. The mask does not add a new loss or learned reliability gate. It adds2048buffer elements and **zero parameters**:20,809,001total /6,790,433requiring gradients under exit adaptation. Structural parameter reduction is zero.

## New selection rule, fixed before running

Use the completed FF2 L_A checkpoint, including its trained linear head, as the common warm start. Its SHA256 is `cce4476491829702d60a74ed7b2044aa8436768d6e08e0129da5a1fd84b0cb6d`. This differs from fresh heads on a P50 encoder in other studies; all three NEW arms share this warm start. Existing results cannot substitute for the new matched control.

For a pooled channel h_c, reference contribution a_c=w_c h_c. Removing only that channel gives the exact reference logit z−a_c. On64deterministically selected TRAIN originals per joint-group/class cell (1024total), compute the positive BCE increase in each cell. The maximum cell increase defines channel utility. Protect the highest-utility half, resolving ties by channel index.

Use every case in the new reviewed bank. The bank contains at least8TRAIN-real sources per joint group, plus every source selected by the original L_H stratified preview across all axes. Each source has both operators at both strengths; none of the earlier difficult sources is deliberately excluded. Selection may block if supplied labels/support do not meet the specification. The new review draws use identical nominal JPEG/geometry randomness for each operator/strength across sources. Padding protection can still make realized changes differ.

On the frozen reference, compute per-channel control drift |asinh(a_control)−asinh(a_original)| and local response asinh(a_probe−a_control). Within each operator/strength, interpolate each observed group's response quantiles onto32points; divide mean group deviation from the mean quantile curve by mean absolute response plus1e−6. All reviewed cases remain in these estimates, including unequal group counts. Average strata/groups equally. A large response is **not** rewarded: magnitude only normalizes dispersion.

Among the low-utility half, remove the41channels with highest sum of midranks of control drift and relative response spread. The random control removes41channels from the SAME utility-eligible pool. Thus it controls for count and utility protection. The no-mask control retains every channel.41/2048is approximately2%; no rate/weight sweep is authorized. Save the mask once with teacher/TRAIN/config/source/review bindings, before any new validation scores are computed.

This rule is a hypothesis, not established novelty or causal disentanglement. Source composition may explain group response differences. Procedural warp/resampling is not validated removal of real forgery evidence. Adaptive exit layers may reroute information around a fixed mask; even a benefit would not prove that protected information has disappeared. Small reviewed coverage and inferred demographic labels limit interpretation.

## Bounded matched study

Three arms: `none`, `random`, `informed`. All run full L_H (native armD), seed17,3epochs, batch32, original224px preprocessing, identical initialization/order, AdamW exit1e−5/head1e−3/weight decay1e−4, clip5, five-axis schedule and unchanged objectives. Select earliest maximum validation AUC, then use that checkpoint's global10%-FPR empirical validation-real threshold. This is a mechanism pilot, not a50+50paper reproduction or a new Table1–3 claim.

Primary exploratory endpoint: validation gender×race F_FPR, alongside every axis/group, AUC, overall/worst-group FPR/FNR and source support. Compare informed−none and informed−random at their stored global thresholds. The `compare` command uses identical recorded-contributor bootstrap draws; intervals with fewer than95%valid replicates or fewer than1000requested replicates are marked INCONCLUSIVE and suppressed. These are conditional on already selected checkpoints, not selection-adjusted or unseen-data intervals. Connected metadata components are not verified persons.

Retain historical absolute utility guards AUC≥0.995 and FNR≤0.005651; never lower them after results. Do not promote a mask unless its primary point estimate improves against both controls while those guards hold. Report both paired intervals and all regressions; even a positive pilot is not confirmatory evidence. Preserve failed/inconclusive outcomes. No automatic next experiment, seed extension, TEST/CAL access or Ding computation.

## Resources, dependencies and readiness

Maximum one allocated GPU and two aggregate GPU-hours **inside a reconciled remaining existing allowance**, not an additional grant. Selector≤15minutes; each training arm≤30minutes; remaining time covers checks/validation/overhead. A bound cohort deadline and per-arm step deadline stop overruns, preserving partial outputs. Operator must assess conservative real-input cost before training; lack of time is BLOCKED, not permission to reduce epochs or drop controls. Existing jobs remain untouched.

CPU software evidence is in [CHECKS.md](CHECKS.md). Actual teacher/data hashes, cluster CUDA checks, current allocation/budget, and substantive named human review are not verified locally. No production mask or benchmark run exists from this package. `audit_config.json` binds the documented public artifact locations/hashes; `bind` rechecks actual cluster bytes. Read [GPU_OPERATOR_PROMPT.md](GPU_OPERATOR_PROMPT.md) before executing. A push is not a launch.
