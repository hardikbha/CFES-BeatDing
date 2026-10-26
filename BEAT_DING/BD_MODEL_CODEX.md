**1. Summary**

BD is a configurable extension of CFES/Arm E for binary deepfake detection, with fake labeled 1. Its CLIP implementation combines a global CLS-token logit with a learned, signed, normalized top-8 patch-evidence contribution. All BD arms retain detection supervision, counterfactual stability, fake–source sensitivity, real-group risk, soft intersectional FPR-gap regularization and adapter orthogonality. The experimental components are **A:** processing augmentation; **B:** class-conditional, multi-quantile score alignment across gender, race and intersectional groups, with larger stratified batches; **C:** an additional evidence map from four intermediate transformer blocks, controlled by a zero-initialized signed gate; and **D:** a soft group-AUC floor. BD0–BD4 select different combinations. A separate ResNet-50 implementation uses the same component scheme but different features, trainable scope and base objective. These are training recipes and architectures; the code alone does not establish that they outperform Ding. (`cfes/model.py:168–267,334–413`; `cfes/engine.py:21–84`; `configs/bd/BD0.yaml:3–5`; `configs/bd/BD1.yaml:2–3`; `configs/bd/BD2.yaml:2–4`; `configs/bd/BD3.yaml:2–3`; `configs/bd/BD4.yaml:2–3`.)

Citations below are relative to `/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING`, except **FairReL**, which means `/home/sachin.chaudhary/ICLR_2027/related_work/FairReL`, and **plan**, which means the supplied pre-registration file.

**2. Architecture**

**CLIP backbone and trainable scope.** The configuration selects OpenAI-pretrained `ViT-L-14`, rank 1, CLS pooling and 224×224 input. The external FairReL wrapper replaces every attention Q/K/V/output projection with
\[
W=W_p+U_r\operatorname{diag}(S_r)V_r^\top.
\]
For a 1024×1024 projection, the principal rank-1023 part is frozen; only the rank-1 factors \(U_r,S_r,V_r\) train. All other backbone parameters are frozen. The wrapper returns the last block’s patch tokens and CLS token after `ln_post`; it does not apply CLIP’s final projection in this forward path. (`configs/base.yaml:5,10–27`; FairReL `fairrel/models/backbone.py:38–47,88–103,105–142`; FairReL `fairrel/models/svd_linear.py:36–54,61–76`.)

Inputs are normalized inside the detector with mean `(0.48145466, 0.4578275, 0.40821073)` and standard deviation `(0.26862954, 0.26130258, 0.27577711)`. The trainable heads are:

- Global: `Linear(1024,1)` on the CLS feature.
- Existing patch head: `LayerNorm(1024) → Linear(1024,256) → GELU → Linear(256,1)`, applied to the 16×16 patch grid.
- Existing fusion gate: one trainable scalar, with \(\alpha=\tanh(a)\); `alpha_max=1`. `patch_detach=false`, so patch losses can update the backbone adapters. (`cfes/model.py:32–33,117–125,179–205,342–345`; `configs/base.yaml:15–20`; `configs/study/clip_A.yaml:10`; `configs/abl/clip_Pdet.yaml:5`.)

Let \(P\) denote the existing patch-logit map and
\[
T_k(P)=\frac1k\sum_{j\in\operatorname{TopK}(P)}P_j,\qquad k=8.
\]
The implemented base score is
\[
s=s_{\mathrm{cls}}+\alpha\,\widetilde t,\qquad
t=T_8(P),\qquad
\widetilde t=t\min(1,c_0/r).
\]
At each training forward, \(r_b=\sqrt{\operatorname{mean}_{\text{forward batch}}t^2}\) is computed without gradients. Initially \(c_0=r=r_b\); subsequently \(r\leftarrow0.99r+0.01r_b\). Thus this is an EMA of **batch RMS values**, not the square root of an EMA of squares. The forward batch includes ordinary images, nuisance views, pairs and stratified images. Evaluation uses saved buffers; before initialization, normalization is skipped. There is no denominator epsilon in this normalization. (`cfes/model.py:44–46,334–364`; `cfes/engine.py:322–366`; `configs/rev4/clip_B.yaml:3`.)

`engine.predict` converts the final logit to an uncalibrated \(\sigma(s)\); matched-threshold evaluation instead operates directly on logits. (`cfes/engine.py:564–581`; `cfes/bd_metrics.py:61–84`.)

**Component C.** In BD3/BD4, hooks capture outputs of CLIP blocks **8, 12, 16 and 20**, numbered from 1. Each drops its CLS token and applies its own
`LayerNorm(1024) → Linear(1024,256) → GELU → Linear(256,1)`.
The four **logit maps are averaged before top-k pooling**:
\[
P_2=\tfrac14\sum_{b\in\{8,12,16,20\}}h_b(X_b^{\text{patch}}),\quad
t_2=T_8(P_2),
\]
\[
s=s_{\mathrm{cls}}+\alpha\widetilde t+\beta\widetilde t_2,\qquad
\beta=\tanh(b),\quad b_{\mathrm{init}}=0.
\]
It has its own RMS buffer \(r_2\), updated with momentum 0.01. Under `c0: base`, its reference \(c_{0,2}\) is copied from the existing branch’s \(c_0\) at initialization:
\[
\widetilde t_2=t_2\min(1,c_{0,2}/r_2).
\]
The gate is a **global signed scalar**, not a per-image reliability gate or convex mixture. Heads initialize in a private RNG context with seed 0. Initially C contributes exactly zero. Its heads receive no separate map loss: stability accesses `patch`, not `patch_ml`; C is trained through the final-score losses. (`configs/bd/base_E.yaml:24–26`; `cfes/model.py:240–267,365–413`; `cfes/losses.py:35–45`; `cfes/engine.py:378–406`.)

**Initialization and counts.** The YAML chain does not itself supply `model.init_ckpt`; the runner supplies the appropriate CFES phase-1 `last.pt`. Loading a raw phase-1 \(\alpha_1\) into the tanh gate sets \(\alpha=\operatorname{sign}(\alpha_1)/2\) and scales the existing patch head’s final weight and bias by \(2|\alpha_1|\), preserving scores. C’s new heads/gate remain initialized. (`run/bd_run.sh:157–167,250–255`; `cfes/model.py:134–154`; `cfes/engine.py:101–118`.)

The code computes parameter reports at runtime rather than stating a fixed full-model total. From the declared dimensions, the global head has **1,025** parameters, the existing patch head **264,705**, and C adds **1,058,821**, including its gate. Frozen SVD matrices/bases are registered as buffers, so the reported `total=sum(parameters)` is not a count of every stored weight/buffer. (`cfes/model.py:179–185,255–265,439–449`; FairReL `fairrel/models/svd_linear.py:42–52`.)

**3. Training objective**

Configuration inheritance is recursive dictionary merging, with child values overriding parent values and command-line overrides applied last. (`cfes/config.py:7–30`.)

The shared CLIP chain is:

`configs/bd/base_E.yaml` → `configs/rev4/clip_E.yaml` → `clip_D.yaml` → `clip_B.yaml` → `clip_A.yaml` → `configs/abl/clip_Pboth.yaml` → `clip_Pdet.yaml` → `configs/study/clip_A.yaml` → `configs/clip_p2.yaml` → `configs/ffpp_ding.yaml` → `configs/base.yaml`.

The `_base_` references occur respectively at lines **6, 5, 2, 2, 3, 2, 4, 9, 4, 3** of those files.

For CLIP, the complete active objective is
\[
\begin{aligned}
L={}&L_{\rm det}+L_{\rm pairdet}+L_{\rm stratdet}
+0.5L_{\rm stab}+0.5L_{\rm sens}\\
&+0.1L_{\rm orth}+0.3L_{\rm risk}
+L_{\rm fprgap}+w_B L_{\rm ESA}+w_D L_{\rm GAUC}.
\end{aligned}
\]
Here \(w_B=1\) in BD0/BD2/BD3/BD4 and 0 in BD1; \(w_D=0.5\) only in BD4. The engine sums these weighted terms directly; it does not normalize by the sum of weights. (`cfes/engine.py:559–560`; `configs/study/clip_A.yaml:14–21`; `configs/abl/clip_Pboth.yaml:4`; `configs/base.yaml:57`; `configs/rev4/clip_E.yaml:7–8`; BD YAML citations in §5.)

Define \(\operatorname{sg}\) as stop-gradient, \(\operatorname{sp}(x)=\log(1+e^x)\), and
\[
H_1(d)=
\begin{cases}
d^2/2,&|d|\le1\\
|d|-1/2,&|d|>1.
\end{cases}
\]

| Active term | Formula and actual inputs | Weight | Definition / call |
|---|---|---:|---|
| Ordinary detection | \(B(S)=\frac12\operatorname{mean}_{y=0}\operatorname{sp}(s)+\frac12\operatorname{mean}_{y=1}\operatorname{sp}(-s)\), on the ordinary batch. If a class is absent, averages only present classes. | 1 | `cfes/losses.py:detection_loss:16–24`; `cfes/engine.py:compute_losses:371–376` |
| Pair detection | Same balanced BCE, separately reduced over fake/source-real images **after shared intervention**. | 1 | `cfes/losses.py:detection_loss:16–24`; `cfes/engine.py:compute_losses:336–342,396–399` |
| Stratified detection | Same balanced BCE, separately reduced over the auxiliary stratified batch. | 1 | `cfes/losses.py:detection_loss:16–24`; `cfes/engine.py:compute_losses:401–406` |
| Stability | \(\operatorname{mean}_i H_1(s'_i-\operatorname{sg}s_i)+\operatorname{mean}_{i\in BG,j\in F}[\sigma(P'_{ij})-\operatorname{sg}\sigma(P_{ij})]^2\). \(BG\): background-intervention views; \(F\): cells entirely inside the face core. Map coefficient is 1. Background-evidence BCE coefficient is **0**. | 0.5 overall | `cfes/losses.py:stability_loss:27–47`; `cfes/engine.py:compute_losses:378–383`; `configs/base.yaml:51`; `configs/abl/clip_Pboth.yaml:4` |
| Sensitivity | \(\operatorname{mean}_{\rm pairs}\operatorname{sp}(2-(s_f-s_r))\). Applies to all supplied pairs; `map_terms=false`, so no changed/unchanged patch penalties. | 0.5 | `cfes/losses.py:sensitivity_loss:50–79`; `cfes/engine.py:compute_losses:386–392`; `configs/base.yaml:52`; `configs/abl/clip_Pboth.yaml:4` |
| Orthogonality | Sum over SVD projections and \(X\in\{U,V\}\) of \(2\|X_p^\top X_r\|_F^2+\|X_r^\top X_r-I\|_F^2\). | 0.1 | FairReL `fairrel/models/svd_linear.py:SVDResidualLinear.orthogonality_loss:88–101`, `total_orthogonality_loss:134–138`; `cfes/model.py:orthogonality_loss:427–428`; `cfes/engine.py:compute_losses:492–493` |
| Real-group risk | For each intersectional group with ≥2 reals, \(R_g=\operatorname{mean}_g\operatorname{sp}(s)\), \(w_g=\operatorname{softmax}_g[\log(\operatorname{sg}R_g+10^{-30})/1]\). Loss \(=\sum_gw_gR_g/\max(\operatorname{sg}\operatorname{mean}_gR_g,10^{-30})\). Requires ≥2 eligible groups. No size-aware weighting. | 0.3 | `cfes/losses.py:real_group_risk_loss:174–203`; `cfes/engine.py:compute_losses:438–444`; `configs/rev4/clip_E.yaml:7` |
| Soft FPR gap | Append current real scores to a detached rolling window, retain last 4096, then \(h=Q_{0.9}(\text{window})\). Let \(p_i=\sigma((s_i-h)/0.5)\). Loss \(=\sum_g|\operatorname{mean}_{i\in g}p_i-\operatorname{mean}_{\rm all\ reals}p_i|\), groups with ≥2 reals; requires ≥2 eligible groups. The pooled rate retains gradients. | 1 | `cfes/losses.py:RunningQuantile.update:214–219`, `fpr_gap_loss:228–251`; `cfes/engine.py:compute_losses:445–452`; `configs/rev4/clip_E.yaml:8` |
| B: ESA | On \(u=\operatorname{asinh}(s)\), mean over eligible \((c,\text{level},g)\) cells of \(\sum_{q\in Q_c}|Q_{u,c,g}(q)-\operatorname{sg}Q_{u,c,\rm pool}(q)|\). Reals: \(Q_0=\{.5,.75,.9,.95\}\); fakes: \(Q_1=\{.05,.1,.25,.5\}\). Each cell needs ≥4 images of that class. Levels: gender, race, intersection. | 1 when B is on | `cfes/losses.py:esa_loss:284–313`; `cfes/engine.py:compute_losses:453–473`; `configs/bd/base_E.yaml:12–13` |
| D: group-AUC floor | \(A(S)=\operatorname{mean}_{i:y_i=1,j:y_j=0}\sigma(u_i-u_j)\), \(u=\operatorname{asinh}(s)\), temperature 1. Loss \(=\sum_{\text{level},g}\max(0,\operatorname{sg}A(\rm pool)-A(g))\). Each group needs ≥4 fakes and ≥4 reals. | 0.5 in BD4 | `cfes/losses.py:soft_auc:316–318`, `group_auc_floor_loss:321–345`; `cfes/engine.py:compute_losses:474–475`; `configs/bd/BD4.yaml:3` |

ESA uses selected empirical order statistics, **not interpolated quantiles**: zero-based index is
\[
\min(n-1,\max(0,\lceil qn-10^{-9}\rceil-1)).
\]
The four quantile deviations are summed within a cell, not averaged. (`cfes/losses.py:empirical_quantiles:274–281,307–311`.)

The **risk/FPR-gap pool** contains ordinary images, stratified images and ordinary nuisance views; it excludes pairs. The **ESA/AUC-floor pool** additionally includes both members of every intervened pair, each with its own manifest group label. Under A, the “clean” ordinary input is already processing-augmented; an additional unprocessed copy is not forwarded. (`cfes/engine.py:438–477`; `cfes/data.py:147–153,270–277`.)

Direct localization, sigmoid-rate alignment, older Wasserstein alignment, view-detection BCE, scale penalties, MFER and reliability supervision are inactive. Sensitivity’s map terms and stability’s background suppression are also inactive. (`configs/base.yaml:55–66,68–75`; `configs/study/clip_A.yaml:18–21`; `configs/abl/clip_Pboth.yaml:4`; `cfes/engine.py:21–84`.)

CLIP training uses 10 epochs × 1976 ordinary batches, AdamW with learning rate \(10^{-4}\), weight decay \(0.004\), 200 warmup steps, cosine decay, gradient clipping at 1 and CUDA bfloat16 autocast. (`configs/clip_p2.yaml:5–9`; `configs/study/clip_A.yaml:22`; `configs/base.yaml:84–85`; `scripts/train.py:204–227,297–304`; `cfes/engine.py:362`.)

**4. Data, samplers, augmentation and pairing**

Training uses the configured Lin/Ding FF++ training CSV; validation uses its real/fake validation CSVs. FF++ and DFDC use eight intersectional groups; Celeb-DF and DFD use six. Gender/race groups are derived by combining those intersectional IDs. (`configs/ffpp_ding.yaml:5–25`; `cfes/data.py:17–20,62–93`; `cfes/metrics.py:51–56`.)

Each ordinary batch contains **32 real + 32 fake** images. Each class is independently shuffled and recycled as necessary; each resulting batch is shuffled. This balances labels, not demographic groups or videos. The auxiliary sampler draws \(k\) images from every **present** label×intersection cell, shuffling and cycling each cell independently: \(k=2\) in BD1, \(k=4\) in BD0/BD2–BD4. With all 16 cells present, these are 32 and 64 auxiliary images respectively. Eight fake–real pairs are additionally sampled per step, shuffled and cycled. (`cfes/data.py:183–247,311–313`; `cfes/engine.py:194–239,274–287`; `configs/clip_p2.yaml:5–7`; `configs/study/clip_A.yaml:12–13`.)

Consequently, with all cells present, a forward contains **176 images in BD1** and **208 in BD0/BD2–BD4**: 64 ordinary + 64 ordinary nuisance views + 16 pair images + 32/64 stratified images. (`cfes/engine.py:322–358`.)

**Component A** runs on both classes, after resizing to model input size and before float conversion. With probability 0.6 it selects exactly one of six operations uniformly: **0.1 unconditional probability per operation**, and 0.4 probability of no processing operation. An independent horizontal flip follows with probability 0.5, including when no processing operation was selected. (`cfes/bd_aug.py:26–32,53–77`; `cfes/data.py:99–109,147–153`.)

| Operation | Implemented parameters/behavior |
|---|---|
| JPEG | Integer quality uniformly from 30 through 90 inclusive; OpenCV encode/decode. |
| Bilateral smoothing | Diameter uniformly from `{5,7,9}`; independently sampled color/space sigmas in `[20,80)`. |
| Gaussian blur | Sigma in `[0.4,1.6)` pixels; OpenCV kernel size `(0,0)`. |
| Down/up resize | Scale in `[0.4,0.9)`; rounded dimensions, minimum 1; bicubic down and up. |
| Photometric | Gamma `[0.6,1.6)`, multiplicative brightness `[0.75,1.25)`, contrast `[0.75,1.25)` around the resulting global luminance mean, saturation `[0.7,1.3)` around per-pixel luminance; in that order. Luminance coefficients `(0.299,0.587,0.114)`. |
| Unsharp mask | Gaussian sigma `[0.5,1.5)`, amount `[0.5,1.5)`; \(x+a(x-G_\sigma(x))\), rounded/clipped to uint8. |

Sources: `cfes/bd_aug.py:53–118`. JPEG chroma subsampling is not explicitly set by the implementation.

Augmentation randomness is keyed by `(20261026, seed, epoch, stream, item_index)`. Thus repeated appearances of an item in the same stream and epoch reuse its augmentation; ordinary, stratified and pair streams differ. A pair receives one shared draw, including its flip. (`cfes/bd_aug.py:27–28,49–50`; `cfes/data.py:147–150,299–305`.)

**Existing nuisance views remain separate from A.** One view is created per ordinary input: background blur with probability 0.6, mild photometric intervention with probability 0.4, background swap with probability 0. Background-blur sigma is `[2,6)`; the face core is preserved with a feathered ellipse. Mild photometric parameters are additive brightness ±0.08, contrast/saturation ±0.10 and gamma `[0.9,1.1)`. Pair members also receive the same nuisance operation and random parameters after A. (`configs/base.yaml:33–39`; `configs/study/clip_A.yaml:11–12`; `cfes/counterfactuals.py:72–101,105–160`; `cfes/engine.py:331–340`.)

**Pairing rule.** BD consumes an existing `fake_path,real_path` CSV rather than discovering pairs itself. The configured file is the CFES training-pair manifest. The external builder matches a fake named `<label>_<Method>_<target>_<source>_<frame>.png` to the **target video’s** real frame `<label>_youtube_<target>_<frame>.png`, restricted to the training manifest. Thus “own source real” here means the original target-video frame, not the donor identity named by `<source>`. (`configs/ffpp_ding.yaml:17–18`; `cfes/data.py:260–267`; external `agents/CFES/scripts/build_pairs.py:58–79`.)

`PairTable` marks alignment using stored `bg_diff <= 0.015`; it computes patch-change fractions from the unprocessed resized pair with mean absolute RGB difference `>0.0314` inside the face core, then flips the fraction map if A flips the pair. These fractions and alignment flags do **not** affect the active BD sensitivity loss because `map_terms=false` and localization is off. (`configs/base.yaml:46–47`; `cfes/data.py:263–264,287–305`; `cfes/losses.py:63–68`.)

**5. Differences between arms and ResNet-50**

| Arm | Resolved arm inheritance | A | B: ESA weight | C | D: floor weight | Stratified images/cell |
|---|---|---:|---:|---:|---:|---:|
| BD0 | `BD0 → base_E` | Off | 1 | Off | 0 | 4 |
| BD1 | `BD1 → base_E` | On | 0 | Off | 0 | 2 |
| BD2 | `BD2 → BD1 → base_E` | On | 1 | Off | 0 | 4 |
| BD3 | `BD3 → BD2 → BD1 → base_E` | On | 1 | On | 0 | 4 |
| BD4 | `BD4 → BD3 → BD2 → BD1 → base_E` | On | 1 | On | 0.5 | 4 |

Sources: `configs/bd/BD0.yaml:3–5`, `BD1.yaml:2–3`, `BD2.yaml:2–4`, `BD3.yaml:2–3`, `BD4.yaml:2–3`; the corresponding `configs/bd_resnet/BD*.yaml` files use the same switches. B therefore changes **both** the objective and auxiliary sampling exposure.

The ResNet base chain is:

`configs/bd_resnet/base_E.yaml` → `configs/rev4/resnet_A.yaml` → `configs/study/resnet_B.yaml` → `resnet_A.yaml` → `configs/resnet_p2.yaml` → `configs/ffpp_ding_resnet.yaml` → `configs/ffpp_ding.yaml` → `configs/base.yaml`.

The references occur at lines **8, 4, 3, 9, 6, 4, 3**.

| Property | ResNet-50 implementation |
|---|---|
| Features | Torchvision ResNet-50 trunk initialized from the configured ImageNet V1 checkpoint; then `Conv2d(2048,512,1) → BatchNorm → ReLU`. |
| Global score | Adaptive average pooling → `Linear(512,2)`; \(s_{\rm cls}=\ell_{\rm fake}-\ell_{\rm real}\). |
| Existing patch head | `LayerNorm(512) → Linear(512,128) → GELU → Linear(128,1)` on the 7×7 feature map. |
| Trainable scope | Only the network’s final `fc`, patch head, fusion gate and enabled C heads/gate. Trunk and `adjust_channel` are frozen; all BN statistics are frozen. |
| Fusion | \(s=s_{\rm cls}+\tanh(a)T_4(P)+\tanh(b)T_4(P_2)\), omitting C when disabled. **No local RMS normalization.** |
| C | Layer2/3/4 outputs, with 512/1024/2048 channels; separate `Conv2d(C,1,1)` heads; each map adaptively average-pooled to 7×7, then maps averaged. Gate starts at zero. |
| Input | Bilinear resize to 224; normalization `(x−0.5)/0.5`. |
| Objective | Same applicable terms/weights as the CLIP arm, **excluding orthogonality and real-group risk**; FPR-gap weight remains 1. |
| Optimization | 10×1976 steps; SGD, LR 0.001, momentum 0.9, weight decay 0.005, constant schedule, one warmup step, clip norm 5, fp32. |

Sources: `cfes/model_resnet.py:21–40,44–98,107–148`; `configs/resnet_p2.yaml:7–11`; `configs/ffpp_ding_resnet.yaml:6–38`; `configs/study/resnet_A.yaml:10–25`; `configs/bd_resnet/base_E.yaml:9–24`; `cfes/engine.py:82–83`.

Derived from these declarations, ResNet has **67,844 trainable parameters without C**, and **71,432 with C**; C adds 3,588. The runtime count implementation is `cfes/model_resnet.py:163–170`.

**6. Checkpoint selection and operating points**

Within a run, maximize
\[
100\,\mathrm{AUC}_{val}
+
100\frac{\mathrm{AUC}_{val}}
 {1+\sum_{g\in\text{intersection}}|\mathrm{AUC}_{val,g}-\mathrm{AUC}_{val}|}.
\]
Groups lacking either class are skipped. Validation AUCs use `rank_prob(logit)`, the monotone map \(0.5+z/[2(\max|z|+1)]\), avoiding sigmoid saturation. Although the result is stored under `ding@0.5`, its es-AUC term is threshold-independent. (`scripts/train.py:102–112,149–152`; `cfes/metrics.py:42–48,59–78,140–146`.)

The starting model is eligible as epoch −1. Subsequent checkpoints replace `best.pt` only on **strict improvement**, so ties retain the earlier checkpoint. `last.pt` additionally stores optimizer/scheduler state and the FPR-gap rolling window. (`configs/bd/base_E.yaml:27`; `scripts/train.py:249–258,316–328`.)

The matched operating point is recomputed from saved validation logits:

1. Read Ding reproduction `full_s7_ep49_ffval.npz`.
2. Compute \(r=\operatorname{mean}_{y=1}[z_{\rm Ding}\ge0]\).
3. Set the BD model’s threshold \(t=\operatorname{np.quantile}(z_{val,y=1},1-r)\).
4. Predict fake iff \(z\ge t\), using the same threshold on every test set.

The target is read from the reference file, not hard-coded to rounded 94.18%. Finite samples/ties mean the quantile rule need not achieve exactly that empirical TPR. (`cfes/bd_metrics.py:25–28,94–114`; `scripts/bd_eval.py:66–96`.)

At each demographic level, the diagnostic uses
\[
F_{\rm FPR}=100\sum_g|\mathrm{FPR}_g-\mathrm{FPR}_{all}|,
\quad
F_{\rm DP}=100\max(\operatorname{range}_g PR_g,\operatorname{range}_g(1-PR_g)),
\]
\[
es\text{-}\mathrm{AUC}
=100\,\mathrm{AUC}_{all}/(1+\sum_g|\mathrm{AUC}_g-\mathrm{AUC}_{all}|).
\]
It counts strict wins across the nine gender/race/intersection fairness columns plus overall AUC, comparing the model at its matched threshold with Ding validation at logit 0. Lower FPR-gap/DP and higher es-AUC/AUC win. This is logged during training but **does not select epochs**. Arm comparison maximizes `(val Ding-score, val AUC)`; promotion requires the winning score to be at least the best reference’s score. (`cfes/bd_metrics.py:61–119`; `scripts/train.py:120–128`; `scripts/bd_report.py:winner_rule:85–94`.)

Additional operating points are:

- **Native:** \(t=-b\), where \(b\) minimizes class-balanced validation BCE for logits \(z+b\).
- **A1:** minimum validation intersectional \(F_{\rm FPR}\), subject to TPR≥0.80 and FPR≤0.20; ties prefer higher TPR, then higher threshold. No feasible threshold yields `None`.

(`scripts/calibrate.py:35–44`; `scripts/bd_eval.py:49–56,77–78`; `scripts/fair_a1.py:16,26–50`.)

ResNet uses the same Xception reproduction’s matched-TPR target and validation reference, while its published-test comparisons use Ding Table 3. (`cfes/bd_metrics.py:39–51,94–114`; `scripts/bd_eval.py:69–78`.)

**7. Absolute Python source paths**

The BD implementation, training, data and evaluation path is defined by:

```text
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/cfes/model.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/cfes/model_resnet.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/cfes/losses.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/cfes/bd_aug.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/cfes/counterfactuals.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/cfes/data.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/cfes/engine.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/cfes/config.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/cfes/utils.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/cfes/metrics.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/cfes/bd_metrics.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/cfes/matchcheck.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/scripts/train.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/scripts/evaluate.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/scripts/calibrate.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/scripts/fair_a1.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/scripts/bd_eval.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/scripts/bd_report.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/scripts/bd_wstar.py
/home/sachin.chaudhary/ICLR_2027/related_work/FairReL/fairrel/models/backbone.py
/home/sachin.chaudhary/ICLR_2027/related_work/FairReL/fairrel/models/svd_linear.py
```

`matchcheck.py` supplies provenance hashing to training; the configured pairing manifest’s external builder is:

```text
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/CFES/scripts/build_pairs.py
```

Other evaluation utilities present in the tree are:

```text
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/scripts/protocol_eval.py
/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/BEAT_DING/scripts/ding_tables.py
```

The active BD runner invokes `evaluate.py`, `calibrate.py`, `bd_eval.py` and `bd_report.py`; the alternate Xception/FSVFM model implementations and MFER objective are not selected by these BD configurations. (`scripts/train.py:33–38`; `cfes/engine.py:87–96`; `run/bd_run.sh:170–176,258–263`.)

**8. Plan–code discrepancies and implementation qualifications**

- **The plan’s statement that no photometric augmentation existed anywhere in training is too broad.** The inherited CFES code already generates photometric nuisance views and shared pair interventions with probability 0.4. A adds a new, stronger processing-augmentation pipeline, but it does not introduce training-time photometric transformations for the first time. (Plan:47–48; `configs/study/clip_A.yaml:11–12`; `cfes/counterfactuals.py:72–83,105–160`.)

- **The promised matched-calibrated probability is not implemented as an output probability.** The plan says calibrated \(P(\mathrm{fake})=0.5\) sits at the matched threshold. Code implements the equivalent decision directly as `logit >= t`; saved `prob` remains sigmoid of the raw logit, while `calibrate.py` fits the separate native intercept. There is no matched \(\sigma(s-t)\) output in these paths. The matched decisions themselves follow the plan. (Plan:99–101; `cfes/engine.py:574–580`; `scripts/evaluate.py:78–80,99–101`; `scripts/bd_eval.py:77–96`; `scripts/calibrate.py:35–44,67–69`.)

- **“Clean, augmented and nuisance views” is implemented as coverage of all forwarded images, not simultaneous clean-plus-augmented copies.** A replaces the ordinary/stratified/pair input with its processed version; ESA then covers that input, ordinary nuisance views, intervened pairs and stratified images. If the plan intended an additional unprocessed anchor for every processed image, that is absent. (Plan:67; `cfes/data.py:147–153,299–305`; `cfes/engine.py:322–358,453–469`.)

- **Several numerical choices were unspecified by the plan.** Code fixes uniform operation selection, integer JPEG quality, unsharp sigma/amount `[0.5,1.5)`, empirical non-interpolated ESA quantiles, and C’s signed tanh gate bounded by ±1 with base-branch RMS reference. These are concrete implementation choices rather than contradictions of specified values. (Plan:52–77; `cfes/bd_aug.py:53–75`; `cfes/losses.py:274–281`; `configs/bd/base_E.yaml:26`.)

- **Initialization depends on the runner.** Directly invoking `scripts/train.py` with only a BD YAML does not enforce the plan’s phase-1 start; `run/bd_run.sh` supplies it. The resolved 10×1976 budget is present in the YAML chain. (Plan:83; `cfes/engine.py:101–118`; `run/bd_run.sh:157–167,255`; `configs/study/clip_A.yaml:22`; `configs/clip_p2.yaml:5`.)

- **ResNet fallback is more specific than Amendment 2.** Amendment 2 says to transfer the winning CLIP component combination; the runner transfers it only if promoted, otherwise choosing the ResNet `base_E` recipe with all BD components off. (Plan:153–160; `run/bd_run.sh:307–316`.)

No further discrepancy was found in the specified A/B/C/D switches, weights, quantile sets, CLIP block choices, within-run selection formula or validation-based arm-ranking rule. This is a source-code comparison, not verification that particular runs followed the plan or achieved its success criteria.