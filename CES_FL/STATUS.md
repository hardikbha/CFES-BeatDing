## ResNet-50 nuisance pilot — observed 2026-11-04T15:24:25Z

[Full report and aggregate artifacts](results/lh_nuisance_pilot_20261104_r1/REPORT.md) · [Proposal](proposals/lh_nuisance_pilot_20261104_r1/README.md). Exact source commit `84ad9838ac14f9e252d292146df5b1e1b1121d7c`; four seed-17 arms completed 3/3 epochs with validation replay. No TEST read.

| Arm | Selected epoch | FF++ validation AUC | FNR | Gender F_FPR | Race F_FPR | Gender×race F_FPR | Age F_FPR | Skin F_FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 3 | 96.6217% | 9.6036% | 6.0357% | 12.3531% | 40.5914% | 12.9181% | 7.7712% |
| H0 | 1 | 96.4214% | 9.9934% | 5.6438% | 9.9922% | 33.3735% | 12.2680% | 5.0148% |
| A1 | 1 | 96.4903% | 9.3407% | 6.6237% | 11.0651% | 37.6674% | 6.7056% | 8.6143% |
| H1 | 1 | 96.6281% | 9.3547% | 5.6438% | 11.4295% | 35.6454% | 4.6997% | 7.3456% |

Mixed/inconclusive: H1 reduced resize nuisance score/response drift and improved several point metrics over A1; race F_FPR worsened slightly, FNR did not improve, and the gender×race bootstrap intervals are below the 900/1000 support floor. AI-only preview review did not establish real-forgery validity. The pilot used 1.1951/3.0000 GPU-hours on one shared gpu02 H100; no existing job was preempted and the FairForge-2 budget was untouched.

## FairForge-2 baseline preparation — 2026-11-03T11:33:27.003004+00:00

New [baseline suite and operator prompt](proposals/ff2_baseline_suite_s17_20261103T110324Z/GPU_OPERATOR_PROMPT.md): user approved72 additional GPU-hours, maximum3 allocated GPUs. Native BCE/DAG/DAW CPU checks pass; CUDA/input checks and training are pending. PG-FDD/FairAdapter adaptations remain conditional; RSEF source unavailable. No jobs launched here; existing jobs/results preserved. This is a preparation observation, not live cluster status. Published to the writable fork because upstream push permission remains unavailable.

# Current CES-FL status

## Plain Xception head ablation — 3 November 2026

[Project memory and research blueprint](PROJECT_MEMORY.md) · [Two-arm proposal](proposals/plain_head_s17_20261103T101418Z/README.md) · [GPU operator prompt](proposals/plain_head_s17_20261103T101418Z/GPU_OPERATOR_PROMPT.md). New standalone Xception+GAP+linear source:20,809,001 total /6,790,433 exit-policy trainable parameters. CPU synthetic smoke and targeted checks pass; CUDA/input checks and both real-data runs are pending. Existing H and completed results are preserved. This publication launches no jobs.

## One hierarchical original-real risk pilot — 3 November 2026

[Current H process/results](results/hierarchical_real_risk_s17_20261103_r1/STATUS.md) · [Failure diagnosis](proposals/hierarchical_real_risk_s17_20261103_r1/FAILURE_DIAGNOSIS.md) · [Frozen single-candidate proposal](proposals/hierarchical_real_risk_s17_20261103_r1/README.md). E/F finished and rejected their fixed guards. H is seed17/FF++ only and changes only the joint-event real fairness summand; no new Ding, FairForge-2 or TEST computation. The retained bootstrap support requirement makes numerical success at best INCONCLUSIVE. Historical snapshots below remain preserved.


## Seed-17 support/error-control pilots — 3 November 2026

Two CES-FL-only candidates are actually training on FF++ TRAIN/validation after fresh CPU/CUDA, real-input and recipe-review gates. [Live process, metric, exposure and budget evidence](results/support_warmup_s17_20261103_r1/STATUS.md) · [Audit and actual execution report](proposals/support_warmup_s17_20261103_r1/AUDIT_AND_RUN_REPORT.md) · [Frozen proposal and decision rules](proposals/support_warmup_s17_20261103_r1/README.md). No new Ding-model computation, new FairForge-2 experiment or TEST inference. This is a separate branch; historical snapshots below remain preserved.


## Latest verified milestone — 1 November 2026

All five controlled-upgrade arms completed ten epochs. All 25 authorized frozen-checkpoint test evaluations completed, with unchanged validation thresholds and exact validation-AUC replay. [Results and limitations](results/upgrade_tests_s17_20261101T040407Z/REPORT.md) · [Completed test observations](status/upgrade_tests_s17_20261101T040407Z_live.md).

A separate [FairForge-2 public in-domain plan](proposals/ff2_indomain_s17_20261101T042828Z/README.md) is prepared with verified TRAIN/VAL/TEST manifests. No training for that new plan has started. FairForge-2 scores in the completed batch are FF++-trained transfer results. The older snapshot below is preserved and does not represent the current five-arm batch.

## Preserved original pilot snapshot — 31 October 2026

Last observed (UTC): **2026-10-31T06:11:36.029000+00:00**.

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
    "stdout": "mgmt01: \n                                                            Req'd  Req'd   Elap\nJob ID          Username Queue    Jobname    SessID NDS TSK Memory Time  S Time\n--------------- -------- -------- ---------- ------ --- --- ------ ----- - -----\n43937.mgmt01    sachin.* workq    v3_sched2  13813*   1   8    --  240:0 R 163:0\n44835.mgmt01    sachin.* gpu      symtl2     344307   1   8  200gb 10:00 R 06:49\n44836.mgmt01    sachin.* gpu      poolg      21698*   1   8  200gb 12:00 R 03:42\n44842.mgmt01    sachin.* workq    symtl      38286*   1   8  200gb 12:00 R 00:48\n44847.mgmt01    sachin.* gpu      cesX_s17      --    1   8   32gb 12:00 Q   -- \n44848.mgmt01    sachin.* gpu      cesP_s17      --    1   8   32gb 12:00 Q   -- \n44849.mgmt01    sachin.* gpu      cesB_s17      --    1   8   32gb 12:00 H   -- \n44850.mgmt01    sachin.* gpu      cesC_s17      --    1   8   32gb 12:00 H   -- \n44851.mgmt01    sachin.* gpu      cesA_s17      --    1   8   32gb 12:00 Q   --",
    "stderr": ""
  },
  "gpu01": {
    "exit_code": 0,
    "stdout": "0, NVIDIA H100 80GB HBM3, 81559 MiB, 33375 MiB, 100 %\n1, NVIDIA H100 80GB HBM3, 81559 MiB, 49629 MiB, 98 %\n2, NVIDIA H100 80GB HBM3, 81559 MiB, 49443 MiB, 89 %\n3, NVIDIA H100 80GB HBM3, 81559 MiB, 26904 MiB, 71 %\n4, NVIDIA H100 80GB HBM3, 81559 MiB, 64777 MiB, 100 %\n5, NVIDIA H100 80GB HBM3, 81559 MiB, 49763 MiB, 100 %\n6, NVIDIA H100 80GB HBM3, 81559 MiB, 49064 MiB, 92 %\n7, NVIDIA H100 80GB HBM3, 81559 MiB, 76137 MiB, 0 %",
    "stderr": ""
  },
  "gpu02": {
    "exit_code": 0,
    "stdout": "0, NVIDIA H100 80GB HBM3, 81559 MiB, 77585 MiB, 0 %\n1, NVIDIA H100 80GB HBM3, 81559 MiB, 76577 MiB, 0 %\n2, NVIDIA H100 80GB HBM3, 81559 MiB, 76577 MiB, 0 %\n3, NVIDIA H100 80GB HBM3, 81559 MiB, 76575 MiB, 0 %\n4, NVIDIA H100 80GB HBM3, 81559 MiB, 54729 MiB, 0 %\n5, NVIDIA H100 80GB HBM3, 81559 MiB, 54711 MiB, 0 %\n6, NVIDIA H100 80GB HBM3, 81559 MiB, 77028 MiB, 100 %\n7, NVIDIA H100 80GB HBM3, 81559 MiB, 71352 MiB, 100 %",
    "stderr": ""
  }
}
```

Read [WORKFLOW.md](WORKFLOW.md) before proposing or launching a run. See [README.md](README.md) for code, configuration and result paths.

## Additional authorized benchmark batch

The stage table above describes the original pilot. Current batch: [benchmark_s17_20261031T112401](https://github.com/hardikbha/CFES-BeatDing/blob/main/CES_FL/status/benchmark_s17_20261031T112401_live.md). Read that separately timestamped page for its submitted jobs and actual progress.
