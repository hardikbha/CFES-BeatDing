# Cluster execution record — plain-head A/H

Scientific checkout: `/home/sachin.chaudhary/ces_fl_sources/cesfl-plain-head-0d699d2`, detached at `0d699d241e27f758ec0990fbe58493a15bdb0ff9`. Source SHA256 `08a9c5e5ca0f6a6f1fb19d86c25ea3e62fcabe45302f5d388252de61a1c4a177`. No scientific source or configuration edits.

Private run root: `/home/sachin.chaudhary/ces_fl_runs/plain_head_s17_20261103T101418Z`. Stage logs, private inventories, preview/review, checkpoints and row-level scores remain there.

Actual CPU checks (aiface Python, OMP/MKL2, OpenBLAS1): `python -u -m unittest discover -s CES_FL/tests -p PATTERN -v`, for `test_head_ablation.py`, `test_forensic_response.py`, `test_hierarchical_real.py`, `test_support_warmup.py`: respectively11/13/13/9 passing tests, zero skips. Parameter CLI, source/helper compilation and `bash -n commands.sh` passed. Exact output logs are under `checks/`.

Each fixed configuration is independently passed to `ces_fl_head_ablation.py preflight --config CONFIG --out ROOT/preflight_A` or `preflight_H`. These run on CPU without holding a GPU. No inventory/cache from the old scientific source is relabeled as a new receipt.

`INITIALIZATION.json` verifies full initial A/H state equality and exact P50 encoder tensor equality. The linear classifier is freshly initialized with seed17. The old gated models are references, not retrained arms.

PBS job `45295.mgmt01` was submitted on2026-11-03 with an initial user hold, queue `workq`, `host=gpu02:ncpus=6:ngpus=1:mem=32gb`, walltime05:50:00. It must stay held until both preflights and the fresh H review qualification pass. Submission is not evidence of GPU execution.

The existing24-total-GPU-hour ledger was13.525502385622305h used, zero active reservations before this request. One conservative5.75h reservation bounds this job, within the new6h study cap. All allocated time including checks and idle overhead is charged. No budget reset. The native per-arm allowance remains2h at epoch boundaries. If incomplete, stop and inspect before any compatible same-run continuation.

Cluster policy uses the user's previously explicit shared pool-v2 UUID route. The job additionally requests a real PBS `ngpus=1` allocation. If PBS provides a CUDA assignment it is respected; otherwise a compatible UUID lease is recorded under that prior authorization. Neither an idle card nor a PBS GPU count is represented as exclusive physical ownership. Existing Ding devices and other tasks are preserved.

After release, the explicit command sequence is fresh linear CUDA smoke, production-input A backward check, production-input H backward check, `commands.sh train-a`, and `commands.sh train-h`. Any failed command stops the sequence; a missing native DONE also stops it. No automatic retry, new candidate, model download, TEST inference or new Ding experiment.

Reuse of old A/H validation scores requires strict native run/checkpoint/config/selection verification, current validation manifest and image hashes, exact row/metadata equality, exact stored-threshold metric replay and byte-identical copying of the original cache. No threshold is recalibrated for a reference model. If this fails, preserve the failure and investigate before claiming a paired comparison.

Publication: an attempted push to `hardiksharma6555/CFES-BeatDing` was rejected with permission denied. Current authenticated account `hardikbha` has verified push/admin on `hardikbha/CFES-BeatDing` and no push on the fork. The new aggregate-only branch `results/cesfl-plain-head-0d699d2-20261103` was successfully pushed to `hardikbha/CFES-BeatDing`. The pinned scientific commit fetched from the fork is preserved.
