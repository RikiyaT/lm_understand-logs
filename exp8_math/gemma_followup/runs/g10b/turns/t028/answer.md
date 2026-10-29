The first HOLDOUT pass favors the frozen edit: accuracy rose from 7.03% to 8.20%, a gain of 1.17 percentage points with a 0.74-point standard error. This is encouraging but not conclusive, so I’ll continue the two predeclared confirmation passes and pool all three. The experimenters also confirmed that earlier “no answer” counts were inflated; I will not use those earlier counts as evidence. The weights remain frozen, and the next pass changes only the predeclared problem order.

```python
assert edit_checksums() == frozen_checksums
stats8, orig8, edit8, diff8 = paired_stats(8)
def cluster_se_pp(original_samples, differences):
    by_problem = collections.defaultdict(list)
    for sample, difference in zip(original_samples, differences):
        by_problem[sample["problem_id"]].append(float(difference))
    means = np.array([np.mean(v) for v in by_problem.values()])
    return float(100 * means.std(ddof=1) / np.sqrt(len(means)))
stats8["cluster_se_pp"] = cluster_se_pp(orig8, diff8)
(wdir / "stats_pass8.json").write_text(json.dumps(stats8, indent=2))
with open(wdir / "notes.md", "a") as f:
    f.write("\n## Failure-summary correction from experimenters\n"
            "Earlier no_answer_finished counts also included wrong stated final answers. "
            "Do not interpret them as missing-answer rates. Accuracy grades and our repetition "
            "metrics are unchanged; no selection relied on the faulty category.\n"
            "\n## Pass 8 confirmation result\n" + json.dumps(stats8) + "\n"
            "No tuning: weights remain frozen1.5x, hash599aba54ec.\n"
            "\n## Pass 9 planned\n"
            "Exact edit unchanged: down_proj columns 8:[699],14:[483,984],17:[1179] x1.5.\n"
            "Set: same256 HOLDOUT, second predeclared order (reverse); both, n4, max8192.\n"
            "Pool with passes8 and10; no candidate selection from HOLDOUT.\n")
print("HOLDOUT1:", {k: stats8[k] for k in ["gain_pp", "cluster_se_pp", "wrong_to_right", "right_to_wrong"]})
print("Repetition rates:", {m: stats8[m]["three_or_more_final_answers"] for m in ["original", "edited"]})
print("Next predeclared HOLDOUT order:")
print(json.dumps(holdout_orders[1]))
```
