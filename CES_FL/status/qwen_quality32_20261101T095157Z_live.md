# Qwen 32-source quality pilot: actual execution snapshot

Observed management UTC: **2026-11-01T11:42:30.693619+00:00**.

**State: RUNNING**. Worker liveness: LIVE.

| Item | Actual status |
|---|---|
| Completed originals | 14/32 |
| Generated diffusion outputs | 100/224 |
| Host / worker PID | gpu01 / 3806205 |
| Bound physical GPU | 3 |
| Source-selection CPU checks | 6 passed; synthetic stubs do not validate edits |
| Human approval / forensic preservation | Not established for new candidates |
| Training / full bank / test evaluation | Not started |
| FairForge-2 | DO NOT RUN |

Expected generation: approximately **4–6 hours once suitable GPU headroom is secured**, plus review/retries. GPU waiting is additional; the worker gives up after two hours without eligible headroom. No completion time is guaranteed.

This is the explicitly approved bounded **quality-only** diagnostic: 32 originals, 16 real and 16 fake, two per label × ids8 cell. It retains 40 steps, BF16, the pinned Qwen model, seven independent conditions and resize controls. It is not a reduced-coverage execution of the full 76,139-original study. The full-training coverage guard rejects this subset. Within-sample contributor proxies are disjoint, but eight proxies overlap VAL under the preserved benchmark split; no identity-disjoint study is claimed.

The parent 14-output pilot was reportedly reviewed by the user; individual decisions and substantive forensic-validity evidence remain unrecorded. “Go ahead” was recorded as permission for this bounded diagnostic, not fabricated per-pair approvals.

Generation progress and source labels are not classification results. No fairness improvement or win over Ding is established. New candidates cannot enter training without actual reviews and a separately justified training protocol.

Direct SSH uses the existing shared dispatcher exclusion plus a project lease, at least 53,248 MiB free before loading, and a 55% allocator cap. Existing processes continue; no PBS allocation or exclusive ownership is claimed.

Run root: `/home/sachin.chaudhary/ces_fl_runs/qwen_quality32_20261101T095157Z`. Worker log: `logs/pilot_worker.log`; generation log: `logs/qwen_32_source_quality.log`.

[Protocol, operational code and authorization](../proposals/qwen_bounded_quality_20261101_r2/README.md) · [Machine-readable actual status](../results/qwen_quality32_20261101T095157Z/status.json).

This finite observer updates only these status files, normally no more than every ten minutes and on terminal state. It cannot advance to training, retry failures or validate images. Read the observation timestamp.
