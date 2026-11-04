# Checks and actual execution scope

The original scientific trainer is unchanged at SHA256 `1997eebf093aa3e37de1cbbe58302a1539ac2e517f966a4607aae89dc695a8e0`. C1.json is byte-identical to the published H1 configuration, SHA256 `93c1c71b9688edfa0a4d33dd5978f4e43ce25e7ec49d823468762955a4b328ac`. Native CLI armC, rather than armD, is the only training-objective change.

Executed locally on CPU:

```bash
python -m unittest discover -s CES_FL/tests -p test_lh_alignment_followup.py -v
python -m py_compile CES_FL/scripts/lh_alignment_followup.py
python CES_FL/scripts/lh_alignment_followup.py --help
```

Nine focused tests passed. Actual output is in `evidence/CPU_TESTS.log`.

1. Exact scientific-source/config binding and fixed20eventplan.
2. Native C removes only alignment: every other loss value matches D, and the total-loss gradient difference equals the weighted alignment gradient.
3. Deterministic32image uniform TRAIN plan without replacement across events, independent of labels/groups, while retaining native BCE weights.
4. Identity leaves tensors unchanged; common resize preserves identical sham/treated controls.
5. The prospective diagnostic trigger requires sufficient batches, repeated negative cosines and non-negligible gradient magnitude; unsupported/nonfinite values cannot supply the count.
6. Empty, duplicate and unresolved review inventories are rejected without creating approvals.
7. Diagnostic component gradients are measured with zero model/buffer updates.
8. Full-width224px native ResNet armC backward:14,966,785trainable parameters receive finite gradients, frozen encoder parameters do not, BN buffers remain unchanged and alignment loss is zero.
9. Tiny **software fixture** using the native C training/data/checkpoint paths: two-epoch continuous versus interrupted/resumed training produces identical weights/calibration, all-original and real-pair coverage is preserved, selected-checkpoint replay completes, and no alignment loss appears.

The lifecycle fixture uses a tiny encoder to avoid large synthetic checkpoint storage. Full-width native ResNet backward is checked separately. These are software tests, not real-data model results or CUDA qualification. Temporary images/checkpoints are synthetic test inputs only and are not published as scientific evidence.

Unavailable locally: actual private reference weights/caches and352reviewed cases, source-bound production binding, repeated H0/H1 TRAIN diagnostics, CUDA C backward and C1training. Those checks are explicit operator stages. No new training, benchmark TEST inference, real intervention images or review approvals were produced here.

The helper records but does not itself establish GPU ownership. Actual scheduler/authorized-lease and process-to-UUID verification plus the total one-GPU-hour deadline must be provided by the cluster execution controller. No automatic scheduler submission, extension or retry is implemented.
