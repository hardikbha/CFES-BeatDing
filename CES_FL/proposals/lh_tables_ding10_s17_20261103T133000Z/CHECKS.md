# Verification

Four focused CPU checks PASS on Python3.9/PyTorch2.8.0: reject incomplete/tampered selected runs and wrong thresholds, preserve earliest maximum-AUC selection, reject TEST/unmarked threshold caches, show unchanged AUC/es-AUC across threshold choices, and reject stale source plans. Evidence: `evidence/cpu_tests.log`. Source compiles.

These tests use synthetic caches/receipts and mock private-input validation; they do not certify real cluster checkpoints or execute benchmark inference. Native scientific source is unchanged/hash-pinned. Actual completion binding, input overlap inventories, CUDA inference,1000draw CPU analysis and Tables1/2 values remain pending. ResNet L_H, corrected Ding10 and cosine-schedule trainers remain work orders requiring code and smoke tests. No job or TEST inference was launched; SSH access failed with connection reset.
