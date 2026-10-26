"""Validate a staged audit-score npz against the DING_full_s7 reference of the same set (same rows, same order).
Used by run_task_full_fast.sh before an npz is moved into results/audit_scores (a half-written npz never counts as done).
  python check_audit_npz.py <candidate.npz> <set>      exit 0 = valid, 1 = invalid (reason printed)
"""
import sys

import numpy as np

REF = ("/home/sachin.chaudhary/ICLR_2027/research/cvpr_readiness/r20261009/agents/DINGREPRO_LIN/results/audit_scores/"
       "full_s7_ep49_{}.npz")


def main():
    cand, s = sys.argv[1], sys.argv[2]
    try:
        z, r = np.load(cand, allow_pickle=True), np.load(REF.format(s), allow_pickle=True)   # img_path is an object array (own files)
        keys = ("logit", "prob", "label", "group", "img_path")
        miss = [k for k in keys if k not in z.files]
        if miss:
            raise ValueError(f"missing arrays {miss}")
        n = len(r["label"])
        if any(len(z[k]) != n for k in keys):
            raise ValueError(f"row count {[len(z[k]) for k in keys]} != {n}")
        if not (np.array_equal(z["label"], r["label"]) and np.array_equal(z["group"], r["group"])
                and np.array_equal(z["img_path"], r["img_path"])):
            raise ValueError("rows differ from the s7 reference (label / group / img_path)")
        if not (np.isfinite(z["logit"]).all() and np.isfinite(z["prob"]).all()):
            raise ValueError("non-finite scores")
    except Exception as e:      # noqa: BLE001
        print(f"INVALID {cand}: {e}")
        sys.exit(1)
    print(f"valid {cand}: {n} rows")


if __name__ == "__main__":
    main()
