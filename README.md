# CFES / Beat-Ding: fairness-aware deepfake detection vs Ding et al. (CVPR'26)

Code only — no checkpoints, run outputs, logs, or raw images (see .gitignore). Mirrors the live cluster paths under
`~/ICLR_2027/research/cvpr_readiness/r20261009/agents/` exactly, so `git pull` here updates the code PBS jobs read
in place.

| Dir | What |
|---|---|
| `BEAT_DING/` | The "BD" recipe (Arm E + augmentation + score alignment + multi-layer evidence + group-AUC floor) aimed at beating Ding's published Tables 1-3; matched-operating-point eval, failure-case analysis, claim-fragility report. |
| `CESEL/`, `CESEL_MA/` | Counterfactual Evidence Selectivity model + multi-axis (gender/age/skin-tone) variant, and the priority task-queue runner that drives all GPU work across both nodes. |
| `CLIPFAIR/` | Matched-backbone baselines (DAW-FDD, DAG-FDD, PG-FDD, Ding SFD+GDA port, FairReL) trained on the same CLIP backbone as ours, for apples-to-apples comparison. |
| `DEMO_AXES/` | Demographic label pipeline: cleaner (AI-Face) vs noisy (Xu) labels, FairFace age/race/sex annotation, ITA skin-tone estimation, multi-axis re-scoring. |
| `DINGREPRO_LIN/` | Our reproduction of Ding et al.'s official code (SFD+GDA), used as the ground-truth comparison target. |
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
