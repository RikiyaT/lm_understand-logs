HOLDOUT block 2 lost 1.37 percentage points, offsetting the first block’s small gain and weakening the generalization hypothesis. It also showed more cutoffs, so the edit has a recurring cost on these harder problems. I will continue the fixed schedule without changing weights, while recovering the earlier DEV dose-response results for an accurate final explanation. Those earlier results can clarify what was localized, but cannot override the independent HOLDOUT evidence.

```python
prepare_frozen_holdout_pass(37)

print("DEV_CAUSAL_TESTS_RECAP")
for k in [12, 13, 14, 15, 16, 20, 21, 24, 25, 26, 27, 28, 31, 32]:
    s = pooled_pairs([k])
    print({
        "pass": k,
        "gain_pp": round(s["gain_pp"], 3),
        "problem_SE_pp": round(s["problem_cluster_SE_pp"], 3),
        "original_accuracy": round(s["original_accuracy"], 4),
        "edited_accuracy": round(s["edited_accuracy"], 4)
    })
with (work / "gen_1.jsonl").open() as f:
    diagnostic_example = json.loads(next(f))
print("DIAGNOSTIC_RECORD_KEYS", list(diagnostic_example))
"Pass37 logged; retrospective DEV summaries do not change the frozen candidate."
```