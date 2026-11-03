# Execution update: direct SSH GPU continuation

Date: 4 November 2026. User authorized a budget increase and direct SSH execution on gpu01/gpu02, with FairForge-2 kept separate.

## Budget

The shared study ledger now records the original 24 GPU-hour allowance plus 24 additional GPU-hours, for a 48 GPU-hour ceiling. At reservation time it showed 17.286613 GPU-hours consumed, no active reservations and 30.713387 remaining. The canceled PBS check `45387.mgmt01` never started and consumed zero GPU-hours.

Measured cost anchors:

- ResNet P10 on the matching FF++ split, batch 16, one H100: 6,226.747 seconds (1.72965 GPU-hours). Linear 50-epoch estimate: 8.6483 GPU-hours.
- Existing ResNet 10-epoch A/B/C arms: 0.632, 0.884 and 0.838 GPU-hours respectively under the older full-network protocol. New layer4-only L_A/L_H use a different trainable scope and batch, so plan 1.0–1.5 GPU-hours per arm until measured.
- Existing Xception L_A/L_H four-dataset evaluation: 0.167 GPU-hours for both models. Plan up to 0.5 GPU-hours for ResNet evaluation pending measured throughput.
- Ding seed-17 native GDA checkpoint intervals: mean 1,827.35 seconds at 1,976 updates/epoch. Scaling to the requested 2,380 updates/epoch projects 6.114 GPU-hours for ten epochs alone. This does not include the corrected adapter's checks and independent validation replay.

The planning total for P50, both ResNet downstream arms, one locked evaluation pass, and Ding-10 is about 19–22 GPU-hours including checks. This fits the newly recorded allowance with room for measured deviations; all sessions still reserve against the same shared ledger. A stage stops if the projected remaining work no longer fits.

## Direct SSH device selection

At the latest node inspection, gpu01 GPU1 (`GPU-6cc0a5b3-99ca-13e3-5508-a201a06e82d6`) had 81,083 MiB free, no compute process and no shared-pool task lease. PBS lists the same host CPU slot under a one-CPU service job with no GPU resource; gpu01's other physical cards are allocated or leased. GPU02 has no compatible free card with sufficient unreserved memory. P50 therefore starts on the verified gpu01 device; downstream L_A and L_H may use both hosts only after a fresh resource inspection.

The direct runner checks the host, UUID, compute applications, shared-pool lease, CUDA binding, source/config/input hashes and ledger reservation. It runs P50 and downstream synthetic CUDA smoke plus a real-input backward dry-run with zero optimizer updates before starting P50. The 50-epoch training then uses the exact frozen configuration and new output directory, with native validation-AUC selection and independent replay. No TEST data is read.
