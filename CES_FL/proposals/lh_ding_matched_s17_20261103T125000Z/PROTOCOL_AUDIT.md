# Protocol audit: published prose, archived code and our new recipe

Reviewed repository base commit `815b4d4bc679c55d4ee6323cc7284335ce210568`. Source hashes: [evidence/source_manifest.json](evidence/source_manifest.json). Primary paper: [Ding et al., v5, implementation details](https://arxiv.org/html/2511.10150v5#S4.SS1). This is read-only inspection of archived files, not new Ding execution or verification of private cluster checkpoints.

| Evidence | Finding |
|---|---|
| `DINGREPRO_LIN/code/Ding_mod/train_ori_model.py:24` | Imports `src.transform.resnet_default_data_transforms`, whose train transform at `src/transform.py:49` directly resizes224×224. |
| `train_ori_model.py:35`,`:52`,`:57`,`:106`,`:122`,`:136` | CombinedDataset length=fake rows; choose random real per fake;50epochs; DataLoader64pairs; concatenate real/fake into128images; SGD.001/.9/.005; final checkpoint each epoch. |
| `DINGREPRO_LIN/results/DING_base_s17/args_train.txt` | Recorded seed17,50epochs,batch_size64,lr.001 for classification. |
| `train_ori_model.py:84` | A256px tensor is used for FLOPs tracing; this does not set the Dataset image size. |
| `DINGREPRO_LIN/code/Ding_mod/GDA.py:82`,`:88`,`:122`,`:188`,`:198`,`:202` |50epochs,32pairs=64main images, last_linear-only gradients, SGD, model.train leaves BN buffers updating; additional group batch. |
| `GDA.py:232`; `DINGREPRO_LIN/results/GDA_GRAD_CHECK.json` | Group prediction/fairness computed under no_grad; measured fairness gradient zero and identical updates at different fairness weights. Connected probe has nonzero gradients. Dependency-sensitive total-loss grad behavior is documented in that receipt. |
| `DINGREPRO_LIN/code/run_task_full.sh`; archived reproduction report |50classification+TRAIN-only SFD2%+50GDA; GDA code resets RNG121. Full seed7 has exported four-dataset TEST metrics. Seed17 full checkpoints are reported complete, but corresponding verified four-dataset exports were not found in the inspected archive. |
| `CES_FL/results/ding_reproduced_vs_cesfl_20261102/REPORT.md` | Records exact split membership matches and fixed.5 evaluation, aspect-preserving224+center crop for Ding. Its statement that classification training used256 conflicts with current imported source; preserve it as historical evidence and resolve exact executed source on cluster. |
| `DINGREPRO_LIN/code/Ding_mod/ding_table_from_raw.py:18`,`:27` | Fixed.5; F_FPR=sum absolute group-minus-pooled FPR; FDP=max-minus-min positive rate; es-AUC=overallAUC/(1+sum group AUC deviations). Gender×race means joint groups, not multiplication of metrics. |

The paper describes batch64,50epochs and SGD.001. Archived source clarifies the pair units used in this reproduction. Neither paper prose nor the stored execution logs inspected here establish bitwise equivalence with the author's published run.

The new CES recipe matches the archived main-stream epochs, full image batches, sampling distribution, optimizer settings and final-epoch policy. It also records both fixed.5 and validation10%FPR metrics instead of comparing thresholds silently. It retains distinct model/head capacity, exit tuning, frozen downstream BN, seed17 and extra H forwards. Thus **partial protocol matching, not full apple-to-apple evidence yet**.

No raw published paper number is a result produced by this new experiment. Do not compare L_H validation AUC or calibrated F_FPR directly to Table1 TEST rows as evidence of a win. A paired local TEST comparison will require aligned row-level outputs and metrics after a locked evaluation handoff. If new Ding work stays disallowed, use available archived results with these limitations clearly labeled.
