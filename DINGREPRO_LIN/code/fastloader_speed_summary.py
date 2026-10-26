import json, statistics as st, sys
d = sys.argv[1]; res = {}
for tag in ("orig", "fast"):
    r = json.load(open(f"{d}/rec_{tag}.json")); s = r["steps"]
    new = [x for x in s if x.get("group_iter_new")]; cont = [x for x in s if not x.get("group_iter_new")]
    walls = [x["t_wall"] for x in s[1:]]
    res[tag] = {"steps": len(s), "group_iterators_created": r["group_iter_created"],
        "wall_s_per_step_mean_excl_step0": st.mean(walls), "wall_s_per_step_median": st.median(walls),
        "group_fetch_mean_all": st.mean(x["t_group_fetch"] for x in s),
        "group_fetch_mean_fresh_iterator_steps": st.mean(x["t_group_fetch"] for x in new) if new else None,
        "group_fetch_mean_reuse_steps": st.mean(x["t_group_fetch"] for x in cont) if cont else None,
        "n_fresh": len(new), "n_reuse": len(cont),
        "combined_fetch_mean_excl_step0": st.mean(x["t_combined_fetch"] for x in s[1:]),
        "total_s": r["total_s"], "workers": r["workers"], "host": r["host"]}
same = [a["combined"] == b["combined"] for a, b in zip(*(json.load(open(f"{d}/rec_{t}.json"))["steps"] for t in ("orig", "fast")))]
res["combined_batches_identical"] = f"{sum(same)}/{len(same)}"
json.dump(res, open(f"{d}/speed_summary.json", "w"), indent=1); print(json.dumps(res, indent=1))
