Block 1 again showed a small positive effect, but its uncertainty still includes no benefit and its cutoff rate increased. Because the second round is incomplete, the current cumulative sample average also gives block 1 extra weight. I will log block 2 and add a compact audit of the completed first round’s accuracy and generation costs. The final comparison will use equal sampling across all 128 HOLDOUT problems.

```python
def holdout_behavior_summary(pass_numbers):
    result = {}
    for model in ["original", "edited"]:
        samples = []
        for k in pass_numbers:
            expected = next(e for e in holdout_schedule if e["pass"] == k)
            records = [
                json.loads(line)
                for line in (work / f"gen_{k}_{model}.jsonl").read_text().splitlines()
                if line.strip()
            ]
            assert len(records) == 32
            assert {r["id"] for r in records} == set(expected["problems"])
            assert all(len(r["samples"]) == 32 for r in records)
            samples.extend(s for r in records for s in r["samples"])
        n = len(samples)
        result[model] = {
            "answers": n,
            "correct": sum(bool(s["correct"]) for s in samples),
            "accuracy_percent": 100 * np.mean([bool(s["correct"]) for s in samples]),
            "mean_tokens": np.mean([s["tokens"] for s in samples]),
            "cutoff_percent": 100 * np.mean([bool(s["cut"]) for s in samples]),
            "incorrect_completed": sum(not s["correct"] and not s["cut"] for s in samples),
            "incorrect_cutoff": sum(not s["correct"] and s["cut"] for s in samples)
        }
    return result

prepare_frozen_holdout_pass(40)
round1_behavior = holdout_behavior_summary(list(range(35, 39)))
print("BALANCED_ROUND1_BEHAVIOR", {
    m: {k: round(float(v), 3) if isinstance(v, (float, np.floating)) else v
        for k, v in s.items()}
    for m, s in round1_behavior.items()
})
"Pass40 logged; all completed round1 records match the locked problem/sample schedule."
```