# Verification scope

Actual CPU checks passed on synthetic data: source/fake gradient directions and detached teacher; production/fixture separation; coefficient-dependent learned updates; both complete three-epoch arms; exact interrupted/resumed head and history; selected-score replay; unchanged teacher/encoder feature hashes; source-cache tamper rejection; full-width224px Xception feature/head versus complete-forward equality with frozen encoder. See evidence/checks.log and SOFTWARE_RECEIPT.json for source hashes.

Production teacher checkpoint bytes, source sidecar/provenance, real cache extraction, exact stored native teacher replay, CUDA/resource profiling and full exported-student real validation replay remain cluster gates. No real training was run here. Synthetic fixture utility numbers are software outputs and must never appear in benchmark tables.
