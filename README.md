# CFES / Beat-Ding: fairness-aware deepfake detection vs Ding et al. (CVPR'26)

## Current supervisor entry point — CES-FL

**[Start with CES_FL/README.md](CES_FL/README.md)** · [Current observed status](CES_FL/STATUS.md) · [Supervisor prompt](CES_FL/SUPERVISOR_PROMPT.md) · [Review → push → cluster training workflow](CES_FL/WORKFLOW.md)

The current standalone CES-FL model, actual completed seed17 configurations, validation histories and verification receipts are now versioned under `CES_FL/`. The status page is timestamped; it is not an unattended live monitor. B/C training is completed, fairness findings are mixed, and current Ding-style real-test Tables1–3 remain incomplete. Read that section before using older archived methods or launching anything.

Historical comparison code plus selected aggregate progress/results; no binary checkpoints or raw images.
The archived agent paths mirror parts of the cluster. The current CES-FL working directory is outside this checkout;
new CES-FL runs should use the pinned repository source described below.

| Dir | What |
|---|---|
| `BEAT_DING/` | The "BD" recipe (Arm E + augmentation + score alignment + multi-layer evidence + group-AUC floor) aimed at beating Ding's published Tables 1-3; matched-operating-point eval, failure-case analysis, claim-fragility report. |
| `CESEL/`, `CESEL_MA/` | Counterfactual Evidence Selectivity model + multi-axis (gender/age/skin-tone) variant, and the priority task-queue runner that drives all GPU work across both nodes. |
| `CLIPFAIR/` | Matched-backbone baselines (DAW-FDD, DAG-FDD, PG-FDD, Ding SFD+GDA port, FairReL) trained on the same CLIP backbone as ours, for apples-to-apples comparison. |
| `DEMO_AXES/` | Demographic label pipeline: cleaner (AI-Face) vs noisy (Xu) labels, FairFace age/race/sex annotation, ITA skin-tone estimation, multi-axis re-scoring. |
| `DINGREPRO_LIN/` | Archived reproduction of Ding et al.'s official code (SFD+GDA); distinguish these local results from the paper's published numbers. |
| `FAIRREL_REPRO/` | Reproduction of FairReL (BMVC'26), a same-backbone baseline. |

See `decisions/*.md` for every pre-registered rule, and `STATUS_SNAPSHOT_27oct.md` for the project log as of this push.

## Running on the cluster
Compute nodes (gpu01/gpu02) have no internet; only the login node does. Workflow:
```
ssh <you>@mgmt01
cd ~/ICLR_2027/research/cvpr_readiness/r20261009/agents/<BEAT_DING|CESEL_MA|CLIPFAIR|...>
git pull origin main
qsub run/pbs_*.sh      # same launchers as before
```
**Caution:** several `run/*.sh` files are read live by running PBS jobs. Never let `git pull` edit one of those in
place while its job is running — check `qstat` first, or ask whoever manages the queue.
