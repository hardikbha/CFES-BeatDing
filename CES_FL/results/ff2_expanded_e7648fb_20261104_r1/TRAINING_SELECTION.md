# Frozen next training wave: seed 17, FairForge-2 public TRAIN/VAL

This selection was made from source/config checks, local weight loading, CUDA smoke, 20-step real-TRAIN timing profiles and the refreshed72-hour ledger. It uses no TEST scores.

| Method | Adapter scope | Config digest | Measured 50-epoch training projection | +40% and0.25h overhead | Session cap |
|---|---|---|---:|---:|---:|
| F3Net | AI-Face FAD-only implementation; DCT256→224 adaptation | `ae529c3af7dc58f55bc1b27064ca866a5eb00013a33e2cb4cd548d74741b2bcf` | 15.861h | 22.456h | 22.75h |
| CORE | Same-source original/flip paired-view adaptation | `b861f3e0aa608430cbcf53848f446be93364abd342b486aa377ba5946bacf389` | 13.358h | 18.951h | 19.25h |
| **Total** | Two new methods, one GPU each | — | 29.219h | **41.407h** | **42.00h** |

At selection, the shared FF2 ledger showed43.989GPU-hours uncommitted. The planned caps leave1.989GPU-hours for cleanup/reconciliation. Launch the two runs concurrently on separate pool-v2 UUID leases only if each still passes the current memory-headroom check. No other run is authorized by this selection.

The exact50epoch configs remain unchanged: seed17, batch128 original images, SGD0.001/momentum0.9/weight decay0.005, float32, same public split, architecture-appropriate normalization, earliest maximum-validation-AUC checkpoint and validation-real empirical10%FPR threshold. The validation thresholds also retain fixed logit0. Preserve the manifest membership and no TEST access.

EfficientNet-B4's projected54.798h with margin does not fit. UnivFD would use38.058h alone, but cannot be combined with another new run under this remaining cap. SPSL projects31.430h alone, also leaving insufficient budget for a second adapter. SRM's 20-step profile hit CUDA OOM under the active shared GPU load. ViT-B/16 lacks a compatible local checkpoint. UCF and PG-FDD lack verified TRAIN Specific labels and required native paired adapters. These are explicit deferred blockers, not omitted rows with scores.

The new methods are research adaptations, not verbatim paper reproductions: F3Net is FAD-only; CORE uses aligned same-source flips. Report per-method validation metrics and subgroup support; no performance claim is made until actual50epoch completion and replay. Uncertainty is contributor bootstrap on fixed validation predictions and does not estimate training-seed variation.
