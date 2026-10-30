# Prompt to share with the supervisor or their LLM

Repository: https://github.com/hardikbha/CFES-BeatDing
Start: https://github.com/hardikbha/CFES-BeatDing/tree/main/CES_FL

Copy the following prompt:

> Take over the research review of CES-FL in this repository. First read CES_FL/README.md, CES_FL/STATUS.md and its observation timestamp, CES_FL/WORKFLOW.md, CES_FL/AGENTS.md, and the linked dated handoff. Inspect the actual code in CES_FL/src/ces_fl_diagram_model_v021.py, the saved configs under CES_FL/configs/completed_seed17/, the verification/provenance/epoch histories under CES_FL/results/completed_seed17/, and the historical software receipts under CES_FL/provenance/.
>
> The seed17 Xception P/A/B/C pilot is completed. B is the detection-pretrained frozen-encoder BCE baseline; C adds demographic invariance using 36 reviewed Qwen variants of six TRAIN originals. B/C is a comparison, not a third model. The current scores are validation scores, fairness gains are mixed, full forensic selectivity is not tested, Ding-style real-test Tables1–3 are incomplete, and the earlier CESel-MA gate remains FAIL. Verify those claims from artifacts and do not relabel older CLIP/CFES results as current CES-FL.
>
> Review implementation and methodology, then propose changes through a branch and a uniquely named request.json plus exact config.json under CES_FL/proposals/<request_id>/. Explain the hypothesis, matched baseline, changed factors, data/split/encoder hashes, seed, update budget, validation selection/calibration policy, intervention approvals, dependencies, software checks and new output path. Do not tune against test or overwrite completed results. Preserve source images and human approvals; do not fabricate labels or treat realistic edits as proof of forensic preservation.
>
> Push or merge reviewed code/config changes so the cluster operator can fetch the exact commit and launch from an immutable detached worktree using the documented workflow. A push must not auto-launch training. GPU jobs require explicit ngpus and verified allocation/binding. Do not rerun completed Ding reproduction, start large generation, terminate jobs or alter preregistration casually.
>
> After meaningful work, update the request/result evidence, refresh CES_FL/STATUS.md and CES_FL/status/ from real cluster artifacts when available, preserve dated snapshots, and commit/push only intended changes. If you lack cluster access, report the status as the last observed snapshot and request actual artifacts rather than claiming live checks or training completion. Keep large/private artifacts outside the public repository. Tell the operator exactly which commit, request/config, command and prerequisites to use next.

Account hardikbha already has admin permission on the repository. GitHub access does not itself provide cluster/GPU access. Do not share credentials.
