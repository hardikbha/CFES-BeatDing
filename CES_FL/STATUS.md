# Current CES-FL status

Last observed (UTC): **2026-10-31T05:31:39.146000+00:00**.

Point-in-time observation. No monitoring daemon or automatic GitHub publication.

Experiment: `seed17_encoder_controls_20261030`; seed 17.

| Arm | Observed state | Selected epoch | Validation AUC (%) | Real test evaluated |
|---|---|---:|---:|---|
| P_pretrain | COMPLETED_VERIFIED | 9 | 98.1711 | False |
| A_imagenet_frozen | COMPLETED_VERIFIED | 8 | 67.9134 | False |
| B_detection_frozen | COMPLETED_VERIFIED | 7 | 98.1949 | False |
| C_detection_invariance | COMPLETED_VERIFIED | 3 | 98.0661 | False |

Review-source SHA256: `0e001dd1251570ce43fb82bde307c519dea4045d9734506e1ce90d2ed014f8de`.

Review source matches the completed experiment source: **True**.
If this becomes False, historical CPU/CUDA receipts and results do not verify the newly edited code.

## Interpretation and pending work

Read [the dated handoff](../docs/ces_fl/2026-10-31/HANDOFF_2026-10-31.md) for the historical fairness conclusion. This status page does not infer fairness from overall AUC.

The current experiment is invariance-only. Full forensic selectivity is not validated. The historical CESel-MA gate remains FAIL.

Table1 requires real FF++ test evaluation; Table2 requires DFDC/Celeb-DF/DFD adapters and evaluation; Table3 requires a separately prepared current-method ResNet-50 experiment. FairForge-2 evaluation/training is a separate phase. No training is launched by refreshing this page.

Explicit evaluation records tracked: **0**. Their detailed evidence/status appears in [current.json](status/current.json). Presence of files is not verification of protocol compliance.

Checkpoint hashes above are from existing verification receipts; checkpoint bytes were not rehashed during this refresh. Recorded progress without verified completion is not proof a process is still alive.

## Resource observation

```json
{
  "scheduler": {
    "exit_code": 0,
    "stdout": "mgmt01: \n                                                            Req'd  Req'd   Elap\nJob ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time\n--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----\n43937.mgmt01    sachin.* workq    v3_sched2  13813*   1   8    --  240:0 R 162:2\n44835.mgmt01    sachin.* gpu      symtl2     344307   1   8  200gb 10:00 R 06:09\n44836.mgmt01    sachin.* gpu      poolg      21698*   1   8  200gb 12:00 R 03:02\n44842.mgmt01    sachin.* workq    symtl      38286*   1   8  200gb 12:00 R 00:08",
    "stderr": ""
  },
  "gpu01": {
    "exit_code": 0,
    "stdout": "0, NVIDIA H100 80GB HBM3, 81559 MiB, 55463 MiB, 100 %\n1, NVIDIA H100 80GB HBM3, 81559 MiB, 24205 MiB, 61 %\n2, NVIDIA H100 80GB HBM3, 81559 MiB, 49867 MiB, 100 %\n3, NVIDIA H100 80GB HBM3, 81559 MiB, 55816 MiB, 89 %\n4, NVIDIA H100 80GB HBM3, 81559 MiB, 66597 MiB, 99 %\n5, NVIDIA H100 80GB HBM3, 81559 MiB, 48401 MiB, 99 %\n6, NVIDIA H100 80GB HBM3, 81559 MiB, 49062 MiB, 0 %\n7, NVIDIA H100 80GB HBM3, 81559 MiB, 76137 MiB, 0 %",
    "stderr": ""
  },
  "gpu02": {
    "exit_code": 0,
    "stdout": "0, NVIDIA H100 80GB HBM3, 81559 MiB, 77585 MiB, 0 %\n1, NVIDIA H100 80GB HBM3, 81559 MiB, 76577 MiB, 0 %\n2, NVIDIA H100 80GB HBM3, 81559 MiB, 76577 MiB, 0 %\n3, NVIDIA H100 80GB HBM3, 81559 MiB, 76575 MiB, 0 %\n4, NVIDIA H100 80GB HBM3, 81559 MiB, 54729 MiB, 0 %\n5, NVIDIA H100 80GB HBM3, 81559 MiB, 54711 MiB, 0 %\n6, NVIDIA H100 80GB HBM3, 81559 MiB, 30449 MiB, 65 %\n7, NVIDIA H100 80GB HBM3, 81559 MiB, 56632 MiB, 65 %",
    "stderr": ""
  }
}
```

Read [WORKFLOW.md](WORKFLOW.md) before proposing or launching a run. See [README.md](README.md) for code, configuration and result paths.
