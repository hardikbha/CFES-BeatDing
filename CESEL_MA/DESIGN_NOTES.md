# CES design notes (25 Oct 2026)

These notes cover what was decided for the code, why, and what is still open. Nothing here is a result: no CES model
has been trained. The prior design this builds on is `CFES/results/cfea/2026-10-24-cfea-design.md` (CFEA); its
measurements are reused below, and each one says so.

## Decisions

### D1. Cues are defined by support, not by name
- Texture, boundary and semantic read disjoint inputs: core cells, ring cells, and the CLS token.
- Frequency reads pixels through a separate network.
- The cell masks come from the CFES face template. For FF++ they reproduce the CFEA counts: 46 core cells and 44 ring
  cells.
- Caveats:
  - CLIP attends globally, so "texture" tokens also carry context from outside the core.
  - "Semantic" names the input (CLS), not a proven content type.
  - Frequency and texture overlap in pixel support (the face core). They differ in representation (spectral amplitude
    vs ViT tokens), not in pixels.
  - The paper must not claim the four cues are disentangled without a leakage measurement.

### D2. Interventions happen in cue space, at head level
- Removing a cue and swapping demographics both replace rows of C (N×K×d) and re-evaluate H.
- No pixels are generated, and there is no second backbone pass.
- With the additive head, the removed score is closed-form and exact: s − w_kᵀc_k + w_kᵀR_k.
- With the mlp head, the head is re-evaluated K times.

### D3. Removal baselines
- A fake's own paired source is the best "evidence absent" reference. CFES pairs give 8 pairs per step.
- Every other image uses the opposite-class EMA mean, as specified.
- Replacements are detached. L_sens therefore trains the image's own cues against a fixed reference, and does not
  move the source's cues: the source real has its own utility term as a real.
- Flag needed later: gradient through the paired source (`replace_grad`) is not implemented.

### D4. Margin uses asinh
- CFES-D logits reach |s| ~ 50–400 (CFEA Sec. 1, measured on `runs_rev4/clip_D_w0.3_s7`).
- m = y'·asinh(s) keeps utilities O(1–6) and keeps their sign.
- The linear margin is available and is the one used for the exact additivity test: with asinh, the per-cue logit
  drops add up but the utilities do not.

### D5. L_det is the CFES-D detection exposure
- L_det is BCE on the ordinary batch, plus the pair images, plus the stratified batch.
- Every arm forwards the same 176 images per step. Pairs are forwarded even in B0, so exposure is matched.
- The stratified batch is stratified by (label, group, condition) with 4 per cell. This gives the L_floor cells
  n ≥ n_min = 4 from the stratified batch alone.
- For FF++ this is 96 images, so the step total of 176 equals CFES-D's.

### D6. B0 still maintains the CES state
- The state (EMA means, subspaces, probe) is updated with no gradient. Later arms start from B0's checkpoint with warm
  baselines and a trained probe.

### D7. Orthogonality term kept on by default
- The FairReL orthogonality term (`loss.orth` 0.1) stays on for CLIP.
- It is part of the CFES backbone recipe (adapter regulariser), not a CFES-D objective. This deviates from "OFF except
  detection"; set `loss.orth.weight=0` for a strict reading.

### D8. Demographic subspace
- The subspace is per cue, with rank 2.
- It comes from SVD of the class-wise centred EMA group means: 2 classes × up to 8 groups rows, d = 128.
- The donor projection is detached.
- The swap applies to all four cues jointly: one counterfactual image per sample.
- Validity uses a linear probe on the concatenated detached cues, trained online with its own Adam. A sample is valid
  if the probe prefers the target group over the source group on C'.
- The per-group validity rate is logged every step. If it is low for some group, the subspace does not capture that
  group's demographic signal, and L_inv says little about that group.

### D9. Validity of the interventions themselves
- The synthetic test shows the swap works when demographics live in a low-rank mean-difference subspace.
- Whether CLIP cues behave this way is an empirical question. Read the logged `probe_acc` (clean) and
  `inv_valid_rate` / `inv_valid_g*` before interpreting any +inv result.

### D10. FF2 pairs come from the generation provenance
- Source: `FAKE_MANIFEST.csv real_path`.
- Synth and gan fakes are excluded, because their real_path is an assigned cell real and not a pixel source (checked
  in gen_params: e.g. flux has a text prompt only).
- Result: 464,517 TRAIN pairs. The 26,897 + 10,499 synth/gan pairs are dropped at load. 4,526 train fakes have no
  manifest entry (pilot methods). All of these use the EMA baseline.

### D11. Separate random streams for CES draws
- Every CES random draw (random-cue control, floor shuffle, inv partners) uses its own generator, drawn in every arm.
- Each control therefore changes only its own term, and the remaining terms are identical draw for draw. This is
  verified in the tests.

### D12. Loss weights are declared, not tuned
- Weights: λ_sens = 0.5 (the CFES sens weight), λ_inv = 0.5, λ_floor = 1.0.
- A failed screen is evidence against CES at these weights only.

## Open questions

### Q1. How is "validated useful" decided? Current rule (`ces_validation.py`), on the starting model's validation pass:
- A cue is valid if the bootstrap 2.5% bound of its class-balanced mean utility is above 0.
- Its target is τ_k = clamp(median utility, 0.1, 2.0).

Alternatives to decide:
- (a) Validate per class. A cue may help only fakes; on reals the "removal" is a substitution by the fake mean, which is
  not evidence removal.
- (b) Require utility on held-out generators (FF2 unseen methods). This is closer to "useful", but uses data that is
  more test-like.
- (c) Use per-cue targets equal to a fixed quantile vs one shared τ.
- (d) Re-validate mid-training. Currently it happens once, on B0.

Also: with τ = median, half the images start below target by construction, so L_sens starts at about half its
weighted maximum.

### Q2. Starting point of the arms
- The arms are designed to start from B0's best checkpoint, so that utilities and validated cues are meaningful. This
  makes a two-stage protocol.
- The comparison must be with "B0 continued" for the same budget, not B0 itself.
- Alternative: train all arms from scratch, with L_sens/L_floor switched on after a warm-up and the default all-valid
  JSON. That is simpler but has no validation gate.

### Q3. L_floor on reals
- On reals, utility means "how much cue k makes this real look real, relative to the fake mean".
- A group deficit there is close to an FPR-side evidence gap, which is plausible.
- The floor pushes deficient groups' utilities up and never pulls others down (pooled is stop-gradient). The model
  could also satisfy it by inflating a deficient group's cue scale. The cue LayerNorm bounds the cue norm but not
  w_kᵀc_k: watch `s_*_absmean` by group.

### Q4. Comparable manipulation condition
- FF++ uses 2 families: swap (DF, FS, FSh) and reenact (F2F, NT).
- FF2 uses the 7 FAKE_MANIFEST families.
- Should FF++ use the 4 methods instead? That doubles the fake cells and the stratified batch (8×4×4 = 128).
- FF2 at 4 per cell needs 256 stratified images per step (304 total, about 1.7× CFES-D). This is a compute question
  for the user. Alternatives: per_cell 2 with n_min 2, or coarser families.

### Q5. Paired real source in L_sens
- A source real uses the fake EMA mean as its baseline. A symmetric option would replace it with its own fake's cue
  (not implemented).

### Q6. Gradient through the replacement or the donor
- Both are detached by default (`detach_donor`; replacements are always detached).
- Letting gradient through the source cue in a pair turns L_sens into a per-cue CFES sensitivity term. Possibly
  useful, but a different method.

### Q7. Matched-set L_inv
- `inv.source: matched` reuses the MFER MatchedSetTable. No matched-set file has passed its audit (`mfer.validated`
  false), so it is unusable for now.

### Q8. Leakage audit before any claim
- Under head-level replacement, cross-cue leakage is 0 by construction for the additive head. Leakage must therefore
  be measured at the backbone level: perturb one support in pixel space (e.g. the CFEA interventions) and measure how
  all four cues move.
- Also measure the per-group probe validity.
- Neither is implemented. Both need a trained B0.

### Q9. Additive vs mlp head
- Additive gives exact contributions but no cue interactions. Mlp gives interactions but no exact decomposition.
- Which head is the paper's claim? The default is additive.
