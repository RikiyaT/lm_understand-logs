The second HOLDOUT pass also favors the edit: +0.88 percentage points, with a 0.64-point standard error. Both confirmation passes therefore point in the same direction, but I’ll report the pooled estimate rather than emphasize either pass individually. I’ll verify that the frozen files are unchanged, record the interim aggregate, and prepare the final predeclared order. No further weight choices will be made from these results.

```python
assert edit_checksums() == frozen_checksums
stats9, orig9, edit9, diff9 = paired_stats(9)
stats9["cluster_se_pp"] = cluster_se_pp(orig9, diff9)
(wdir / "stats_pass9.json").write_text(json.dumps(stats9, indent=2))

def pooled_confirmation(pass_numbers):
    all_o, all_e, all_d = [], [], []
    pass_summaries = []
    for pn in pass_numbers:
        st, oo, ee, dd = paired_stats(pn)
        st["cluster_se_pp"] = cluster_se_pp(oo, dd)
        all_o.extend(oo)
        all_e.extend(ee)
        all_d.extend(dd.tolist())
        pass_summaries.append(st)
    da = np.array(all_d, dtype=np.float64)
    result = {
        "passes": pass_numbers, "n_per_model": len(da),
        "distinct_problems": len({s["problem_id"] for s in all_o}),
        "original_correct": sum(bool(s["correct"]) for s in all_o),
        "edited_correct": sum(bool(s["correct"]) for s in all_e),
        "original_accuracy_pct": 100*np.mean([s["correct"] for s in all_o]),
        "edited_accuracy_pct": 100*np.mean([s["correct"] for s in all_e]),
        "gain_pp": float(100*da.mean()),
        "cluster_se_pp": cluster_se_pp(all_o, da),
        "sample_pair_se_pp": float(100*da.std(ddof=1)/np.sqrt(len(da))),
        "wrong_to_right": int((da == 1).sum()),
        "right_to_wrong": int((da == -1).sum()),
    }
    for label, ss in [("original", all_o), ("edited", all_e)]:
        result[label + "_mean_tokens"] = float(np.mean([s["tokens"] for s in ss]))
        result[label + "_cut_pct"] = float(100*np.mean([s["cut"] for s in ss]))
        result[label + "_repeated_conclusion_pct"] = float(100*np.mean([
            len(re.findall(r"final answer", s["text"], flags=re.I)) >= 3 for s in ss]))
    return result, pass_summaries

interim_pool, _ = pooled_confirmation([8, 9])
(wdir / "holdout_interim.json").write_text(json.dumps(interim_pool, indent=2))
with open(wdir / "notes.md", "a") as f:
    f.write("\n## Pass 9 confirmation result\n" + json.dumps(stats9) + "\n"
            "Interim pooled confirmation: " + json.dumps(interim_pool) + "\n"
            "No changes to candidate; checksum verification passed.\n"
            "\n## Pass 10 planned: final confirmation\n"
            "Exact edit unchanged: down_proj columns 8:[699],14:[483,984],17:[1179] x1.5, hash599aba54ec.\n"
            "Set: same256 HOLDOUT, third predeclared order (rotation85); both, n4, max8192.\n"
            "Final report pools all three HOLDOUT passes (3072 answers/model) and uses "
            "problem-cluster standard error over the256 problems.\n")
print("Interim HOLDOUT:", {k: interim_pool[k] for k in
      ["n_per_model", "original_accuracy_pct", "edited_accuracy_pct", "gain_pp", "cluster_se_pp"]})
print("Final predeclared HOLDOUT order:")
print(json.dumps(holdout_orders[2]))
```
