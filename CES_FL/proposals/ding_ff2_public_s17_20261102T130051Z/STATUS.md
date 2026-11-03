# Actual execution status

Observed management UTC: **2026-11-03T23:56:43.789200+00:00**.

State: **RUNNING**. Source commit `8487d98303fd5d7766fef2fa3dd32c37f38e5f4b`.

Host `gpu01`, physical GPUs0/2 in shared pool-v2 mode; no PBS allocation or exclusive CPU/GPU ownership claimed.
Controller PID `2754578`, training torchrun PID `2754977`.

Latest recorded event: `TRAIN_STEP`, stage `classification`, epoch `41`, batch `450/3149`, loss `0.04330404847860336`.

Eight CPU tests PASS; real-TRAIN full-native-batch two-GPU CUDA smoke PASS. 128classification images/update; 64main+64auxiliary GDA images/update; 224px; seed17;50classification→TRAIN-SFD→50corrected-GDAepochs.
Corrected differentiable class-conditioned GDA with frozen encoder BN is explicitly distinguished from the detached archived GDA and paper results. Values below are completed validation epochs, not TEST scores. No TEST inference is attached.

Continuous private log: `/home/sachin.chaudhary/ces_fl_runs/ding_ff2_public_s17_20261102T130051Z/logs/training.log`.

Initial launch failed before GPU claim because compute nodes lack Git; a private byte-identical copy of management /usr/bin/git fixed source verification. Its hash and command are in execution_receipt.json. Model code/settings were unchanged.

Compute-node wall clock is approximately649s behind management at launch; preserve both clocks. Elapsed timings are node-local.

Finite observation/publication only, every15min for at most72h; no training retries or new jobs.

See live_status.json for verified process observation, aggregate validation/group metrics, checkpoint hashes and completion/failure state. Review historical PROJECT_BLUEPRINT.md as a dated snapshot; this page is the execution update.

## classification: 40 completed validation epochs

Best validation AUC: 99.8104% at epoch21 (fixed AUC-only selection).

| Epoch | Train CE | VAL AUC (%) | Worst gender×race AUC (%) | FPR gap (pp) | F_FPR (Ding scale) |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.260279 | 97.0067 | 96.4288 | 9.3780 | 21.1840 |
| 2 | 0.121754 | 97.8417 | 96.8216 | 8.6683 | 23.6116 |
| 3 | 0.064228 | 97.9288 | 96.5497 | 10.8264 | 32.6147 |
| 4 | 0.035999 | 98.1493 | 97.0077 | 11.2456 | 24.6681 |
| 5 | 0.022846 | 97.7694 | 95.7756 | 10.5395 | 27.6514 |
| 6 | 0.016415 | 98.4244 | 97.2242 | 9.7682 | 25.2413 |
| 7 | 0.013938 | 97.5974 | 95.6088 | 8.3786 | 21.4778 |
| 8 | 0.017006 | 98.4500 | 96.9578 | 12.6407 | 35.1372 |
| 9 | 0.022334 | 98.8073 | 97.9807 | 10.1730 | 29.2438 |
| 10 | 0.027830 | 99.3516 | 98.7454 | 11.8379 | 31.3901 |
| 11 | 0.034788 | 99.1404 | 98.0091 | 10.0892 | 21.5891 |
| 12 | 0.038114 | 99.4541 | 98.7340 | 18.9120 | 46.7261 |
| 13 | 0.042096 | 99.1756 | 98.6686 | 10.3785 | 22.2637 |
| 14 | 0.042325 | 99.7849 | 99.5677 | 12.8235 | 34.3318 |
| 15 | 0.042294 | 99.4696 | 98.6994 | 16.8177 | 43.3956 |
| 16 | 0.042409 | 99.2098 | 98.5513 | 6.6031 | 14.2527 |
| 17 | 0.043008 | 99.4953 | 98.8544 | 17.3656 | 48.6868 |
| 18 | 0.042745 | 99.6012 | 99.2505 | 9.0346 | 23.0586 |
| 19 | 0.043186 | 99.3558 | 97.3877 | 13.6230 | 38.1573 |
| 20 | 0.042979 | 97.9774 | 95.7355 | 17.2685 | 50.4534 |
| 21 | 0.044197 | 99.8104 | 99.6370 | 17.2045 | 41.7077 |
| 22 | 0.041974 | 99.7198 | 99.2634 | 19.8134 | 53.7759 |
| 23 | 0.042463 | 99.4571 | 98.7770 | 13.1456 | 39.9643 |
| 24 | 0.042602 | 99.7378 | 99.4621 | 18.9120 | 44.7962 |
| 25 | 0.040675 | 99.6674 | 99.1869 | 14.4662 | 39.8137 |
| 26 | 0.040882 | 99.7579 | 99.5615 | 10.8588 | 26.1557 |
| 27 | 0.040649 | 99.2810 | 97.9046 | 12.9845 | 33.8810 |
| 28 | 0.039299 | 99.4791 | 99.1789 | 9.7992 | 20.8278 |
| 29 | 0.040045 | 99.7454 | 99.3289 | 19.5561 | 44.8606 |
| 30 | 0.038367 | 97.8300 | 96.6695 | 7.0111 | 18.6196 |
| 31 | 0.038389 | 99.4317 | 98.9418 | 9.9731 | 23.7596 |
| 32 | 0.038906 | 99.4426 | 98.3751 | 19.2017 | 45.8635 |
| 33 | 0.037822 | 99.3986 | 98.7932 | 18.3641 | 48.6033 |
| 34 | 0.037752 | 98.6103 | 96.9409 | 9.1539 | 17.8953 |
| 35 | 0.037903 | 99.5731 | 98.2499 | 18.9759 | 51.6822 |
| 36 | 0.037047 | 99.1881 | 98.0593 | 20.3289 | 57.2541 |
| 37 | 0.036303 | 99.7608 | 99.4450 | 13.5324 | 29.9988 |
| 38 | 0.036537 | 99.7240 | 99.0528 | 17.8810 | 45.4003 |
| 39 | 0.036842 | 99.6299 | 99.1347 | 13.9590 | 27.7184 |
| 40 | 0.036584 | 99.7729 | 99.5691 | 13.1140 | 28.2633 |

## gda_corrected: 0 completed validation epochs

No completed validation epoch in this stage yet.
