"""Compare tests/identity_rev4_EF.py outputs: loss values (hex), total, gradient sha256 and batch sha256 must be
identical across code trees. Prints IDENTICAL or the differences.
  python tests/identity_compare.py <ref.json> <other.json> [...]"""
import json
import sys

KEYS = ("terms", "vals", "total", "grad_sha256", "grad_n", "batch_sha256")
recs = [(p, json.load(open(p))) for p in sys.argv[1:]]
ref_p, ref = recs[0]
bad = []
for p, r in recs[1:]:
    for k in KEYS:
        if r.get(k) != ref.get(k):
            bad.append(f"{p}: {k} differs: {r.get(k)} vs {ref.get(k)}")
for p, r in recs:
    print(f"{p}: terms {r['terms']} total {r['total']} grad {r['grad_sha256'][:16]} batch {r['batch_sha256'][:16]}")
print("\n".join(bad) if bad else f"IDENTICAL ({len(recs)} records)")
sys.exit(1 if bad else 0)
