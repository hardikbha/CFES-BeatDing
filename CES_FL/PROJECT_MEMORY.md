# CES-FL / CFES project memory and research blueprint

Updated 3 November 2026 for the plain-head ablation and FairForge-2 baseline suite. This is a maintained context index, not a live cluster monitor. Native code, immutable configurations, checkpoint hashes and timestamped result artifacts take precedence over prose. Preserve old observations when adding new ones. Never treat a proposal, generated diagram, synthetic smoke score or an interim checkpoint as a completed scientific result.

## 1. Recover context immediately

Repository: https://github.com/hardikbha/CFES-BeatDing. Read `CES_FL/AGENTS.md`, `README.md`, `STATUS.md`, `WORKFLOW.md`, then this file and the active proposal. `main` has sometimes lagged experimental branches; resolve the exact requested branch/commit, not only `main`.

Latest authorized work: [FairForge-2 baseline suite](proposals/ff2_baseline_suite_s17_20261103T110324Z/README.md), [source](src/ff2_fairness_baselines.py), [GPU prompt](proposals/ff2_baseline_suite_s17_20261103T110324Z/GPU_OPERATOR_PROMPT.md). The user requested external methods compared in Ding's paper, publishing everything required and a2–3GPU operator handoff, then explicitly approved **up to72 additional GPU-hours**. This ledger is separate from prior studies; at most3 GPUs may be allocated concurrently for these new baselines. No actual benchmark training, GPU allocation or TEST inference was launched by this preparation.

The native BCE/DAG-FDD/DAW-FDD first wave is implemented with a matched50epoch seed17 common recipe, source/data hashes, full Xception/BN training, official objective/gradient parity, exact checkpoint replay and deterministic mid-epoch resume. It has20,809,001 trainable parameters. CPU native/resume checks pass; actual cluster inputs/CUDA/cost remain pending. PG-FDD and FairAdapter sources are pinned/fetched but their method/data adapters need the explicit work orders. RSEF official source/data remain unavailable. These are disclosed adaptations, not completed FF2 results or a fully matched comparison to Ding's512-adapter50+50 run. The handoff reserves at most18h per initial job (54h total), with18h initially uncommitted. A locked72h ledger, explicit allocation record, real backward profiling and own-child wall limits constrain execution. No new Ding-on-FF++ runs or TEST tuning are authorized.

FairForge-2 suite branch: `experiment/ff2-baseline-suite-s17-20261103T110324Z`, based on plain-head publication `0d699d241e27f758ec0990fbe58493a15bdb0ff9`. Verified pinned sources: AI-Face-FairnessBench `588f80c60a3f6c50253da37cc57104e05e99508d`, Fairness-Generalization `e552a5422987897ab747e3b91b4b1b38d6531446`, FairnessDetection `b81e1cc0205e79a0393513e72aa0208021452704`. A filtered Git source fetch transferred excluded checkpoint history, so the final fetch retrieves individual pinned non-checkpoint files and verifies Git blobs. An initial synthetic integration checkpoint write failed under local disk pressure; preserved failure evidence is followed by a passing fresh retry. Neither failure is a benchmark result. Publication remains on the writable `hardiksharma6555` fork because upstream push permission is still false.

Earlier authorized implementation, preserved: [plain-head proposal](proposals/plain_head_s17_20261103T101418Z/README.md). Standalone code: [ces_fl_head_ablation.py](src/ces_fl_head_ablation.py). Operator handoff: [GPU_OPERATOR_PROMPT.md](proposals/plain_head_s17_20261103T101418Z/GPU_OPERATOR_PROMPT.md). The user requested plain Xception plus a linear classifier to determine whether Part C/global-plus-local gating is necessary, a parameter count, tested code, a GitHub push and an executable GPU handoff. The later `/compact` request adds this comprehensive memory; it does not cancel that implementation.

Worktree base: `58ed0cb64341ec1413000eb2266a5846215582b1`, from `experiment/cesfl-hier-real-s17-20261103`. New branch: `experiment/cesfl-plain-head-s17-20261103T101418Z`. An additional read-only remote check found H update `438291e0ca7d5bee73cff737bbebab80e63b23a0`, observed at 2026-11-03T10:19:59.565126Z: H9/10, selected epoch1, not complete. That later status does not change this proposal's science source or authorize a new run. Original `main` was `67d2308225e422a91c715c40427f40c3df6812cf` during that check.

New model source SHA256: `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`. Its format is `CESFL_HEAD_ABLATION_v1`. No real benchmark training, generation or TEST inference has been launched for the new head. CPU synthetic smoke passed; GPU/input checks remain pending. Code/test/config commit is `97ecb7df3330a250d73d44fcc76f8df5aa74aef2`. Publishing to upstream returned403: neither configured account has push access and there is no pending invitation. The handoff uses the existing writable fork `https://github.com/hardiksharma6555/CFES-BeatDing`; do not assume upstream contains this proposal. A later documentation-only commit records that destination. Final publication/launch SHA belongs in the handoff and actual launch receipt; a push is not a job submission.

## 2. Vision, motivation and research claim boundaries

The intended detector should classify manipulation accurately while avoiding demographic shortcuts. A useful forensic response should change when manipulation-related evidence changes; an irrelevant demographic or processing change should not dominate its decision. The paper goal is a defensible fairness contribution, competitive with Ding on the same benchmark conditions, with transfer to FairForge-2. The user values fairness metrics, not AUC alone, and wants a CVPR-quality method, clear diagrams, reproducible code and tables.

The original aspirational target was to beat Ding with a novel approach. No mechanism guarantees that outcome. An arbitrary novelty rating, attractive architecture figure, added loss or higher AUC does not establish novelty, fairness or superiority. Claims need matched controls, real data, disclosure of failures, valid interventions and uncertainty. A focused negative ablation that removes unnecessary complexity is useful research progress.

Resources: user initially indicated approximately24 aggregate GPU-hours and access to two80GB H100s, with possible modest extension if evidence justifies it. On3November the user separately approved72 **additional** GPU-hours for the new FairForge-2 baselines, using at most3 allocated GPUs. Two GPUs for one hour consume two GPU-hours. Earlier bounded proposals retain their own ledger; the new allowance is not unlimited compute for all work. Avoid expensive diffusion at full-data scale until a small validated pilot demonstrates value.

## 3. User preferences and standing constraints

- Work autonomously on authorized coding/review/testing/publishing; provide meaningful updates. Never describe a push as a training launch or an old RUNNING artifact as live status.
- User explicitly prohibited additional Ding-model experiments on Ding's data; subsequent work there runs CES-FL only. Preserve any separately existing Ding jobs/checkpoints. No job termination, scheduler changes or GPU borrowing follows from a code request.
- Preserve historical results, preregistration, split roles, thresholds, named intervention decisions and protocol exceptions. New proposals get unique names and exact configurations.
- Use a separate branch/worktree and immutable commit on the GPU node. Do not pull into source read by an active run, blanket-stage unrelated files, reset shared work or overwrite outputs.
- Keep private faces, original/edit contact sheets, row-level predictions, labels with paths, checkpoints, datasets and credentials outside the public repo. Publish aggregate metrics, hashes, commands, logs without private data, and limitations.
- No TEST-driven tuning, fairness-selected epoch substitution, silently reduced subgroup coverage, fabricated demographic labels or fabricated review approvals. No different seed as an unannounced retry.
- A human saying some older examples look good does not approve a new bank or prove identity/authenticity preservation. Named AI visual review must remain labeled AI.
- Table requests: use Ding's row/column structure where applicable, show one genuinely selected CES-FL model row last, lightly highlight it without bolding all scores. Mark hypothetical targets with `*`/distinct styling and keep them out of measured-results tables. Never mix per-metric best checkpoints into a fictional model row.
- The earlier downstream head ablation remains unchanged. The subsequent explicit FF2 baseline request authorizes the new baseline suite and conditional method work orders within72 additional GPU-hours; it does not authorize new ResNet, Qwen, TEST or Ding-on-FF++ arms.

## 4. Architecture lineage and why it changed

### Early CESel-MA / one-head work

The conversation began with an older CESel-MA status file and weak age-editing results. The user wanted older/younger identity-preserving variants, scaling to100,000–200,000 images, and a clearer CVPR diagram. Early one-head/v0.1 and the later diagram implementation are distinct architectures. The old CESel-MA gate remained FAIL; later exploratory success must not relabel it PASS. Historical one-head references are not a controlled ablation of today's head.

### CES-FL diagram v0.2 and corrected v0.2.1

The diagram model uses a2048-channel Xception feature map F, global average pooling g, global MLP2048→128→1, shared patch1×1 MLP2048→128→1, global gate2048→64→1 and patch gate2048→64→1. Both gates use sigmoid; two fusion weights use softplus:

```
zg = global_head(mean_hw(F)); zp = patch_head(F)
rg = sigmoid(global_gate(g)); rp = sigmoid(patch_gate(F))
z = ag * rg * zg + mean_hw(ap * rp * zp) + bias
M = ap * rp * zp
```

M explains only the local contribution; global contribution and fusion bias must be exported separately. Gates are learned weights, not proven reliability estimates. Maps are not forgery masks. Patch predictions use the same feature map, not independently encoded image crops. Global/local attention is not itself a demonstrated novel contribution.

The unused random512-channel Ding adaptation was removed rather than frozen. Total21,594,159 parameters; frozen encoder trains787,207. With the current exit-block policy,7,575,591 train. v0.2.1 added explicit gradient/BN policies, separate optimizer groups, controlled checkpoint transfer and identical downstream head seeds. It enabled forensic encoder pretraining to be tested separately from demographic invariance.

### Counterfactual invariance objectives

The original score objective used BCE on originals plus absolute `asinh(z)-asinh(z_demographic)` invariance. Optional sensitivity used a signed margin on `asinh(z)-asinh(z_forensic)` with default fake-only scope. A coupled controlled ablation required at most one demographic and one forensic pair per original. Optional map consistency was disabled by default and required explicit spatial validation. These are not automatically active in every later experiment.

Qwen runs offline to generate candidates; inference needs only the detector. Original, no-change editor reconstruction and matched geometric resize controls distinguish demographic effects from editor/preprocessing shifts. Every edit should derive independently from the original, not a chain of prior edits.

### Manipulation-response quartets and procedural-response replacement

The full response idea needs matched conditions: no-change/original-style image, demographic edit, and the same manipulation applied to both, with reviewed provenance. Demographic Qwen edits alone cannot supply those quartets. The bank was absent (`bank: null`) in the actual FF++ configuration. Earlier descriptions of a complete method were therefore too broad.

The lower-cost implemented response study uses deterministic procedural local warp/local resampling and matched sham processing on original real images. These are engineering probes inspired by self-blending ideas, not validated removal of an existing deepfake or a faithful implementation of an external synthesizer. Synthetic views receive response/control losses; they are not assigned invented benchmark real/fake labels.

Current procedural A/B/C/D objectives:

- A: weighted BCE on every original.
- B: A plus soft class-conditional error disparity across demographic groups.
- C: B plus procedural sensitivity and sham-control stability.
- D: C plus conditional cross-group response alignment, using transformed score differences `asinh(z_treated-z_sham)`.

The D-versus-C comparison isolates alignment within this recipe. A-versus-D changes both objectives and auxiliary forwards. These A/B/C labels differ from the earlier ImageNet/pretrained/invariance pilot: always name the proposal and recipe.

### E/F support/warmup and H hierarchy

E/F were bounded attempts to improve error control and alignment scheduling; both completed and failed their fixed primary/age safeguards. They are not retry candidates to relaunch automatically.

H retains the gated architecture and the D procedural recipe. It changes only the REAL summand of the gender×race auxiliary error loss. For group j and parent gender s:

```
rj = mean_real,j sigmoid((z - t_TRAIN)/T)
pi_j = N_train_real,j / N_train_real
q_j|s = N_train_real,j / N_train_real,parent_s
Rs = sum_j:parent(j)=s q_j|s * rj
b = stopgrad(sum_j pi_j*rj)
hierarchy = mean_j relu(rj-b) + mean_s relu(Rs-b)
joint_error = 0.5*hierarchy + 0.5*legacy_fake_error
```

The fake summand, including attached-reference gradients, is unchanged. Other axes retain their legacy losses. Complete TRAIN joint coverage and consistent parent labels are required. The hypothesis is that response differences can cancel a group-specific additive score offset, whereas original-real false-alarm risks need direct control. Soft-risk control does not guarantee hard FPR-tail parity; other/shared gradients can oppose it.

### Current plain-head experiment

```
F = Xception(x)
z = Linear2048to1(mean_hw(F))
M[u,v] = w^T F[u,v]
mean_hw(M) + b = z
```

Only2,049 head parameters. Total20,809,001; exit-policy trainable6,790,433. Removing all nonlinear heads/gates/fusion saves785,158 parameters (3.64% full-model reduction). It is standard GAP+linear classification, not claimed as architectural novelty. The objective investigation is independent of whether this head is novel.

New L_A versus existing gated A tests the head under BCE. New L_H versus completed gated H tests the head under the unchanged H objective. Both reuse the same P50 encoder, so this is conditional on gated-head pretraining. It does not isolate which subcomponent of the old head mattered, or establish the best separately tuned hyperparameters for either architecture. No speed improvement is measured yet.

## 5. Data, fairness definitions and protocol

Ding comparison dataset1 is FF++/FaceForensics++; use the adopted public benchmark manifests and state compression/crop/provenance limits. Current FF++ TRAIN76,139 (12,926 real /63,213 fake), VAL25,386 (4,092 real /21,294 fake). Exact manifest, audit and P50 hashes are pinned in current configs. There are163 declared TRAIN/VAL contributor overlaps; identity/source strings are not independently verified biological identities. Do not claim identity-disjoint generalization.

FairForge-2 is a separate dataset/protocol. Public split: TRAIN218,266 / VAL31,107 / TEST87,564. Older internal splits had TRAIN502,035 / VAL74,974 / TEST182,884 and are not directly comparable. User's earlier “RFAIR42”/“Flowforge2” wording was clarified as Ding's dataset and FairForge-2. Do not confuse FairForge-2 with FairForensics/GenFace.

Metadata axes include inherited gender, race, gender×race/ids8, age-bin and skin-bin. Age/skin may be proxies; do not infer or relabel sensitive attributes from faces. “Gender×race” forms joint cells (for example Female–Black), not a multiplication of gender FPR and race FPR.

For a group g:

```
FPR_g = false positives among real images in g / real images in g
FNR_g = false negatives among fake images in g / fake images in g
F_FPR = sum_g |FPR_g - overall_FPR|
FPR_gap = max_g FPR_g - min_g FPR_g
```

Use the actual implementation's eligible-group support rules. F_FPR is a sum and need not be bounded by1;100×display is a scaled value, not necessarily a probability. Group-count differences matter when comparing axes. Also report F_DP, es_AUC, group AUC/FNR, worst-group AUC and group denominators. Marginal gender improvement does not mathematically guarantee intersectional improvement because composition and cell-specific error rates differ.

Current selection: earliest strict maximum VAL AUC after the prescribed epochs. Calibration: one global threshold per selected model using validation-real empirical FPR10% with conservative finite-sample tie handling. FF++ permits409false positives among4,092real images; this is not a guarantee of TEST FPR. Fixed0.5 results are separate diagnostics. TRAIN-only threshold estimation for fairness gradients is distinct from VAL threshold calibration and never reads VAL for training gradients.

Inference/selection comparisons must disclose architecture, pretraining, tuning policy, batch size, epochs/updates, sample exposure, seed, optimizer, preprocessing, threshold and data membership. Same backbone or epoch count alone is not apples-to-apples. Fairness-selected checkpoints or group-specific thresholds would change the comparison.

Uncertainty: use paired connected-recorded-contributor bootstrap with fixed eligible groups/thresholds, not independent frames. Current sparse FF++ support yields746/1000 usable primary draws under the inherited rule, below900 required by H's fixed decision gate. Do not loosen that requirement after results. Female–Black VAL has30real images and only2recorded components: one false positive changes group FPR by3.33percentage points. This is a major limitation. A single training seed does not estimate training-seed variability.

## 6. Completed results, with protocols kept separate

All tables below are historical snapshots. AUC uses percent; F_FPR uses native fraction unless stated otherwise. Do not compare numbers across panels as one controlled ranking.

### Original frozen encoder/invariance pilot

Source `0e001dd1251570ce43fb82bde307c519dea4045d9734506e1ce90d2ed014f8de`; see historical `STATUS.md`, `configs/completed_seed17/`, `results/completed_seed17/` and `docs/ces_fl/2026-10-31/HANDOFF_2026-10-31.md`.

| Arm | Selected epoch | VAL AUC % |
|---|---:|---:|
| P detection pretraining |9|98.1711|
| A frozen ImageNet |8|67.9134|
| B frozen detection-pretrained |7|98.1949|
| C B plus limited invariance |3|98.0661|

C used36reviewed edits from6originals; it is invariance-only. The large encoder improvement is evident; demographic invariance benefit is not established. These were not full forensic-selectivity experiments.

### Ten-plus-ten upgrade TEST results

Historical FF++ TEST, seed17, batch16; see `results/upgrade_tests_s17_20261101T040407Z/REPORT.md` and `results/ding_format_tables_s17_20261102/`. Fairness below is100×F_FPR for readability.

| Variant | TEST AUC % | Gender100×F_FPR | Race100×F_FPR | Intersection100×F_FPR |
|---|---:|---:|---:|---:|
| R1 partial fine-tuning+BCE |98.401257|0.174265|11.50706|29.91822|
| R2 group DRO |98.396217|0.881363|15.63496|30.231677|
| RN no-change consistency |97.847191|0.264698|8.54540|22.930385|
| RD age consistency |98.196471|0.751709|11.82940|26.273831|

R1 lacks counterfactual consistency, forensic sensitivity and DRO. R2 did not improve the primary fairness metric. RD did not beat RN on these fairness measures, so editor/control effects prevent an age-specific benefit claim. All used the gated head; this is not evidence for gating over a linear head. Historical test data must not become the tuning set for today's proposal.

### P50 plus50 original-only downstream validation

See `results/manip_response_originals_s17_20261102/REPORT.md`. Batch16, no matched quartet bank. Fairness is100×F_FPR.

| Variant | VAL AUC % | Gender | Race | Intersection | Age |
|---|---:|---:|---:|---:|---:|
| BCE |98.448762|6.231712|14.990947|65.781925|24.694304|
| Fairness baseline |98.487114|4.761855|10.339325|68.420860|16.296698|

Longer training slightly improved AUC and some marginal disparities but worsened the primary intersectional metric. No no-change/score/response quartet arms ran. More baseline epochs cannot replace the missing intervention bank.

### Procedural response: P50 plus10, batch32, VAL only

Source `c332fd34fb603b8919b15fee08e8d022fb9f57c8`, SHA `89530025a00b9b0405fba7026d3eed5f2749d86620174ad3ffde93a44f8b770b`. Six completed runs independently replayed; see `results/forensic_response_recovery_c332fd34_20261103/milestone_six_completed_20261103/REPORT.md`.

| Dataset/arm | VAL AUC % | Intersection F_FPR | FPR gap pp | Worst joint AUC % | Fake FNR % |
|---|---:|---:|---:|---:|---:|
| FF++ A |98.426411|0.505265|33.839133|93.959518|4.146708|
| FF++ B |98.406797|0.468638|30.416667|93.967743|4.235935|
| FF++ C |98.307659|0.480534|30.984848|93.620499|3.709965|
| FF++ D |98.252564|0.379345|23.839133|93.851876|3.963558|
| FF2 A |99.808536|0.290986|11.660852|99.600348|0.331368|
| FF2 B |99.807457|0.295459|11.028403|99.586397|0.369737|

D versus C reduces primary F_FPR by21.0576% relatively, at0.055096percentage-point AUC cost. But gender F_FPR worsens0.043699→0.053498 and age0.276733→0.303616. Race improves0.077373→0.072905 and skin0.073475→0.051027. This is cross-axis tradeoff evidence, not general fairness superiority. The selected C/D checkpoints are both epoch1 after all10epochs. Do not substitute a later fairness-favorable checkpoint.

FF2 B slightly worsens primary F_FPR despite lowering the gap; AUC is nearly unchanged. FF2 C/D are blocked by actual intervention review, not a trainer problem. Of88cases across22sources,72approved,8rejected,8uncertain: padding/rotation borders, hat, costume mask and hand/soft-face occlusion caused failures. Qualification correctly failed. No sources were silently dropped and no approvals were invented. The next FF2 dependency is a reviewed, versioned intervention-support/eligibility change while retaining every original BCE row.

### E/F and H

See `proposals/support_warmup_s17_20261103_r1/` and `proposals/hierarchical_real_risk_s17_20261103_r1/`. E/F completed10epochs and failed fixed primary/age guards. E−D primary delta+0.088936,95%CI[−0.011646,0.154790]; F−D+0.116232,[−0.004303,0.144386]. Only746usable primary draws; report this explicitly.

Latest read-only H publication `438291e...`, observed2026-11-03T10:19:59Z:9/10complete, selected epoch1. H VAL AUC98.197763; gender F_FPR0.059377, intersection0.421326, age0.291921. D baseline:98.252564,0.053498,0.379345,0.303616. H at that point is worse on gender/intersection and better on age. It is not a final result or verified current process state. Shared ledger13.4994/24GPUh,10.5006remaining at that timestamp.

H's fixed gates require gender≤0.042798380714166834; intersection<D0.3793453731153111; AUC/age/FNR/worst-group safeguards; primary paired95%CI below0 and≥900usable draws. The last requirement is structurally unmet under existing support. All-point numerical success would still be INCONCLUSIVE; safeguard failure is FAIL. Preserve that protocol. The new head ablation has descriptive comparisons, not a retroactively revised H success gate.

## 7. Ding and FairForensics: what comparisons support

Ding paper: https://arxiv.org/abs/2511.10150 (earlier review used v5). Table1 is FF++ in-domain TEST; Table2 covers transfer to DFDC/Celeb-DF/DFD; Table3 compares backbones including ResNet-50. A current VAL-only Xception score cannot fill all three tables. See `results/ding_format_tables_s17_20261102/` and `results/ding_epochs_wins_s17_20261102/` for existing measured tables, source provenance and caveats.

Previously transcribed published Xception Table1 values: AUC97.71%; gender/race/intersection100×F_FPR0.53/9.29/20.18. Use pinned paper/table artifacts for final typesetting and verify version. Some historical CES-FL cells beat paper numbers while others do not. No aggregate SOTA claim follows from cherry-picked cells, mixed thresholds or validation-versus-test comparisons.

Archived Ding stages differ: initial classifier training updates the detector; SFD ranks/masks channels without optimizer updates; GDA trains the final classifier while BN buffers may still update. Detector21,858,090total; classifier1,026trainable in the archived adaptation stage. Structural channel masks do not necessarily reduce the stored parameter count. Comparing frozen CES787,207trainable versus Ding1,026adaptation parameters is not a same-stage efficiency comparison.

Archived GDA fairness-gradient issue: the fairness term was detached under no-grad; prior dependency-specific probes found BCE could still backpropagate and changing fairness coefficient0/0.005/1000 produced identical classifier updates. Document published method, archived behavior and any explicitly corrected variant separately. This is not proof the paper's reported scores are invalid. The user subsequently excluded new Ding-model runs.

Epoch concern: archived schedule includes50classification+50GDA with an SFD stage; it is not100identical end-to-end epochs. Earlier CES runs used10+10; later original-only50+50 and current P50+10 are distinct. Batch64 in the referenced Ding setup differs from our16/32. Matching wall epochs alone does not match updates or exposure.

Actual archived full Ding seed7 AUCs: FF++97.4443%, DFDC56.0468%, Celeb-DF71.2196%, DFD72.5374%; classification seed17 FF++96.3210%. See `results/ding_reproduced_vs_cesfl_20261102/REPORT.md`. Membership can match while seed, preprocessing, optimization, calibration and archived GDA gradients differ. That is a disclosed reproduction comparison, not a pure controlled causal comparison. Do not invent an actual matched Ding ResNet reproduction or substitute internal FF2 scores for the public split.

FairForensics paper discussed: https://arxiv.org/abs/2608.01661. Previous paper-level analysis described expression-prior/vision features, identity-related perturbations, demographic text prompts and class-conditional prototype alignment. Author code availability and exact implementation need renewed verification before using it as a baseline. Its GenFace setting is not automatically FairForge-2 or FF++. Borrowed prototype/contrastive alignment is prior art, not a new CES invention. Lesson: connect a measured failure mechanism to one targeted intervention and a controlled ablation.

Global/local attention and CAM are established ideas (for example Multi-Attentional Deepfake Detection, CVPR2021; CAM, CVPR2016). Novelty should reside in a supported forensic/fairness mechanism and rigorous evidence rather than renaming ordinary gating. Historical ratings out of10 were subjective brainstorming, not review evidence.

## 8. Counterfactual generation and why scaling is not the next step

Early small invariance experiments used36reviewed edits from6originals. A later224-candidate Qwen bank had no recorded approvals at its audit; earlier approvals belonged to other images. A large bank was never validated as matched quartets. Per-output generation estimates around69.5seconds made a full proposed bank roughly10,292GPU-hours, far outside budget. These are historical estimates, not an up-to-date benchmark of every editor.

One random demographic edit per original could reduce generation cost, but uniform randomness does not guarantee balanced attribute coverage, appropriate direction/strength, identity preservation, authenticity preservation or absence of editor shortcuts. A reviewer must be able to distinguish no-change reconstruction, resize-only effects, intended demographic change and unintended evidence changes. Original-only BCE coverage must remain complete even when interventions are invalid or unavailable.

An age-only bounded quartet pilot was a validity/effect-size test, not a claim that age is the only final fairness axis. Expanding to gender/skin/all demographics should follow evidence and a separately reviewed bank. Demographic appearance edits do not establish a person's gender/race, and prompt assertions do not prove causal validity.

Current procedural H/L_H use inherited demographic metadata to align responses; they do NOT generate age/gender/skin counterfactual versions of every original. Say this explicitly in diagrams, paper and tables. Full demographic/forensic quartet evidence remains unfinished.

## 9. Failure analysis and useful interpretations

Earlier validation failure sheets inspected Male–White real193/1264FP; Female–Black real9/30FP; age60+real24/63FP; Male–Others fake112/1170FN; age45–59fake220/2398FN. Review found crop/blur/compression/cosmetic/pose differences on both correct and error examples, sometimes within the same component. No single verified visual cause or gross empty/black crop was established. PNG container format does not establish upstream compression. Metadata was not relabeled by appearance.

Likely mechanisms worth testing, not established explanations: original score offsets surviving response alignment; noisy few-example auxiliary risk estimates; mismatch between smooth risk penalties and extreme hard-FPR tails; scarce contributor support; competing marginal/intersection gradients; quality/occlusion shortcuts in procedural interventions; head complexity that contributes no benefit. H tests the first narrow mechanism; the plain-head ablation tests the last. Do not change all of them together.

“Close gender FPR and intersection will follow” is not guaranteed. Parent rates average differently weighted child cells. Report joint cells and marginal rates together. Raising errors in an advantaged group can reduce two-sided disparity; lower disparity alone is not adequate without global error/worst-group safeguards.

## 10. Implementation details and verified software behavior

New head source is standalone: embedded Xception, NumPy/Pillow/PyTorch/sklearn; no project Python import, model download or Qwen dependency. New version/head_type prevents accidental interchange with old checkpoints. Core Xception, data exposure, sampler, intervention, loss, optimizer, calibration and metric functions are AST-identical to H. Retained gated initialization, fixture outputs/losses/gradients are bitwise equal to H in focused tests.

The local Python3.9/PyTorch2.8.0 CPU environment has no CUDA. Production dependency doc recommends Python≥3.10; cluster operator must record its environment. Eleven new focused tests passed; existing forensic13tests passed; hierarchy13tests passed with one skip because a pinned older source was unavailable; support9tests passed. Full unittest discovery initially ran64tests with3errors/1skip: two operator modules require a cluster-only `allocated_stage.py`; an older upgrade integration hit a checkpoint writer failure during local disk pressure. It is not an all-suite pass.

Plain-head synthetic end-to-end CPU smoke passed: production-width224forward/gradients/CAM/frozen BN; tiny A/H native training; checkpoint selection/reload; bitwise stop/resume; original/pair exposure; synthetic test-label flip leaves calibration unchanged; paired bootstrap and head-aware map export. Synthetic TEST fixtures are software-only and not benchmark TEST access.

First attempt stopped on a9.3e−10 repeat-forward numerical difference, fixed by a normal floating tolerance. Second attempt hit checkpoint-write failure while concurrent tests stressed nearly full disk. Failures/logs were retained; only this task's failed synthetic checkpoint binaries were removed after recording paths/sizes/hashes. Third isolated smoke passed. Do not present failed attempts as benchmark runs or discard their diagnostic provenance.

New helper `reuse_review.py` checks exact pinned H qualification and identical source/sheet bytes before copying prior named decisions. New `replay_reference.py` accepts only completed pinned A/H and fixed VAL, verifies hashes/earliest maximum-AUC selection, replays strict gated weights and demands exact stored metrics. Helpers are guard-tested; cluster-level use remains pending. New CUDA smoke and real-input dry runs must match new source/head/config/device before `train` accepts them.

## 11. Current blockers, immediate next work and roadmap

Immediate, authorized:

1. COMPLETED: code, exact configs, CPU evidence, proposal, project memory and GPU prompt were pushed to the existing writable fork after upstream denied access. Remote snapshot `293bcd9f965e065e226b3a692c8a40eada393842` was verified by git ls-remote; a subsequent documentation-only receipt records this. Scientific code/config remain commit `97ecb7df3330a250d73d44fcc76f8df5aa74aef2`. No benchmark training launched.
2. GPU operator checks active jobs, remaining ledger and actual allocated device; creates detached checkout and new private output root. Verify cluster input/pretraining hashes and decode TRAIN/VAL.
3. Run fresh CUDA synthetic smoke and discarded production TRAIN-threshold backward for L_A/L_H. Preserve failures; no protocol workaround.
4. L_A may train once its checks pass. L_H additionally needs valid source-bound procedural qualification, possibly byte-identical reuse of existing H review. Preserve active gated H and wait for its actual completion before final paired reporting.
5. Complete exactly10epochs each, replay selected checkpoints, compare only fixed validation predictions to gated A/H, report all axes/support/uncertainty/compute and final hashes. Do not tune in response to the result. Maximum6additional aggregate GPU-hours and remaining original ledger both apply.

Subsequent work requires a new explicit proposal:

- If the linear head is comparable within meaningful uncertainty and guardrails, simplify the architecture/diagram; if it loses materially, investigate the smallest useful head component. This experiment removes a bundle, not gates alone.
- Address FF2 intervention support/occlusion/padding failures without silently dropping original BCE rows, then re-review the changed recipe. No FF2 C/D result exists until that gate passes and training completes.
- Validate a bounded matched manipulation/demographic quartet bank before claiming demographic counterfactual forensic selectivity. Measure editor no-change and resize effects.
- Resolve sparse-group uncertainty through better data/support or a prospectively revised justified estimation protocol, not post-result cutoff changes.
- Consider matched architecture/init comparisons, independent seeds and frozen TEST/transfer/ResNet experiments only after validation design choices are frozen and authorized. Tables2/3 need the corresponding real experiments.
- Measure measured throughput/memory under matched hardware before claiming computational benefit. Consider frozen-feature caching only as a separately declared workflow change if appropriate.
- Literature audit: define closest prior work, mathematical distinction, claim scope and required ablations before asserting7–8/10novelty or a CVPR contribution.

Open hypotheses: does head simplification change bias/error tails? Does hierarchy improve original-real risk independent of response alignment? Is a monotone/tail-aware group loss needed? Can realistic manipulation probes preserve demographics and transfer to unseen synthesis? How much of observed invariance is editor reconstruction regularization? Are improvements stable across subgroups/contributors/seeds and operating points? Which results survive a fixed TEST protocol? None is answered by a diagram.

## 12. Artifact map and continuity procedure

Principal code lineage: `src/ces_fl_diagram_model_v021.py`; controlled-upgrade and manipulation-response sources under `src/`; current preserved H `src/ces_fl_forensic_response.py` SHA `2549cb1345d1e2ee617e8a77ca288a0acbbfd05cf480f7b91a46b8627fcb74b2`; new plain-head `src/ces_fl_head_ablation.py`. Use actual configs/source hashes, not a remembered filename alone.

Current local task worktree: `/Users/hardiksharma/Downloads/fairness_cvpr/CFES-BeatDing-plain-head-20261103T101418Z`. Prior worktree remains untouched: `/Users/hardiksharma/Downloads/fairness_cvpr/CFES-BeatDing-forensic-response-overlapfix-20261103`. Cluster interpreter in configs/handoffs: `/home/sachin.chaudhary/.conda/envs/aiface/bin/python`. P50 checkpoint is `/home/sachin.chaudhary/ces_fl_runs/manip_response_ssh_20261101T115720Z/P/best.pt`, hash pinned above. Cluster file existence and scheduler state are not established by a local path string.

Existing local visuals: `output/imagegen/cesfl_h_interim_20261103/cesfl_h_interim.png` and raster-embeddedPDF, with prompt/edit/evidence sidecars. They describe H, not the new plain head. `output/cesfl_ablation_evidence_20261103/fairness_ablation_evidence.png/.pdf` separates oldTEST/newVAL/longVAL panels; compare within panels. `output/cesfl_failure_diagnostics_20261103` holds earlier diagnostic artifacts. These may be outside the repository; verify existence before linking or editing. Historical local face/diagram paths in the conversation are not evidence that their contents were inspected in this task.

Paper tables and TeX history live under dated `results/ding_format_tables_s17_20261102/` and related report directories. Before sharing “best” rows, inspect actual provenance/threshold/model selection; preserve measured-versus-hypothetical distinction. Never write unavailable new head scores into a paper table.

For every continuation: resolve exact source/remote commit; read actual latest evidence and its timestamp; inspect relevant code/config; identify authorized scope; preserve private data and existing jobs; update this file with dated facts, evidence paths and unresolved questions. Maintain explicit states: PROPOSED, SOFTWARE_CHECKED, INPUTS_VERIFIED, QUALIFIED, RUNNING_OBSERVED, COMPLETED_VERIFIED, BLOCKED or FAILED. Never compress away failures, protocol changes or uncertainty. If a statement here conflicts with a native artifact, record the correction and its provenance instead of silently rewriting history.

## Verified cluster launch — FF2 native baselines,3November2026,12:46UTC

User explicitly pinned `815b4d4bc679c55d4ee6323cc7284335ce210568` and authorized72additionalGPUh/max3concurrent for the FF2 wave. Detached science checkout `/home/sachin.chaudhary/ces_fl_sources/cesfl-ff2-suite-815b4d4`, sourceSHA `f14b818cad66c92a07f09e0c06767a2980a0b8644e8bb74b33710e142b971904`, configs unchanged. Private root `/home/sachin.chaudhary/ces_fl_runs/ff2_baseline_suite_s17_20261103T110324Z`. See public `results/ff2_baseline_suite_s17_20261103T110324Z_cluster/` for evidence, exact launch commands, allocation/PID/session metadata, costs and conditional blockers.

Native7tests/budget6tests passed with no skips. Full independent preflights passed for all249,373TRAIN/VAL images; exact218,266TRAIN(16,777real/201,489fake),31,107VAL(2,438real/28,669fake). Native mapping and TRAIN/VAL provenance/byte checks passed. All source-only dependency blobs verified; no weights downloaded. Each fresh CUDA integration and20production TRAIN backward steps passed. Total/trainable20,809,001,full Xception/BN,float32SGD,50epochs,batch128,seed17,natural all-original exposure,earliest highestVALAUC and globalVAL10%FPR calibration unchanged. No benchmarkTEST read.

Initial18h sessions returned BUDGET_PENDING before any training because40%-margin projections were24.80/18.61/20.52h. All evidence preserved; initial actual held charge including bookends0.1070343257GPUh. The prospective resource continuation reserves24hBCE/21hDAG/23hDAW (68h), leaving3.8929656743h uncommitted inside72h. Resource wrapperv1 preserved; new separatev2 changes only resource allowance/verifiedUUID targeting. Pinned worker/scientific configuration untouched; it reran all fresh checks and projections, which now fit8.86/9.08/9.35h including40%margin under different observed contention/cache conditions. No first-run profile state was transferred; training starts fresh from ImageNet.

Actual RUNNING native training: BCEsession`ff2_bce_gpu01_20261103T124210Z`,gpu01GPU6,controller3683232/worker3683278; DAGsession`ff2_dag_gpu01_20261103T124210Z`,GPU4,controller3683248/worker3683309; DAWsession`ff2_daw_gpu01_20261103T124210Z`,GPU5,controller3683267/worker3683293. These are actual shared pool-v2 UUID leases under prior directSSH authorization, not claimedPBS/exclusive assignments. Source checks use real sharedGit2.45.2. Three new methods/GPUs maximum, ledger locked. Old Ding and FF++plain-head jobs left untouched. No full epoch or new validation score existed at this observation.

PG-FDD stays BLOCKED_ADAPTER: official6specific classes/threeencoders/SAM require a reviewed26method target mapping andTRAINcounts; supplied nativeCSV has sevencolumns, so official column7specific-label read would fail. Canonical column7is skin_bin and must not be used as forgery target. FairAdapter stays BLOCKED_ENTRYPOINT/DATA: differing native trainers,hardcoded20VOCobject category views andexternal bicycle-length CSV do not matchFF2face data. Compatible localViT-L/14file exists(hash in conditional report), but method/view/loss parity is uncertified. RSEF remains BLOCKED_SOURCE/DATA. No conditionalGPU method started.

Publication now uses verified write access upstream `hardikbha/CFES-BeatDing`,branch`results/ff2-baselines-815b4d4-20261103`,separate publication worktree. Prior preparation's fork-only permission statement was stale. Check processes/receipts anew on continuation, preserve initial budget deferrals and old experiments, reconcile new allocation bookends after release, verify50epochDONE/selectedreplay before final CPU1000draw paired report. No automatic monitoring, retries or TEST evaluation is installed.
