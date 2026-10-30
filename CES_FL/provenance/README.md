# Provenance

source_snapshot.json pins the exact code of the completed seed17 pilot. The repository's initial src/ces_fl_diagram_model_v021.py is byte-identical to that cluster source. Xception classes are embedded with the source attribution preserved; external model weights/data are not included and have separate provenance/usage conditions.

historical_smoke_cpu.json and historical_smoke_cuda.json are retained actual cluster test receipts, not newly executed tests and not evidence for later source modifications. environment_versions.json records the existing aiface environment; no packages/models were installed or downloaded for this publication. The model's source, per-run configs and provenance are the sources of truth for executable behavior.

New status-helper tests and publication checks are recorded separately in the setup receipt. Their passing does not establish detector performance or counterfactual validity.
