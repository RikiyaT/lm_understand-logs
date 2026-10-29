The fourth block was slightly negative, leaving the first complete HOLDOUT round essentially flat. The edit’s increased length and cutoff rate are more consistent than its accuracy gains, which limits the proposed repair mechanism. I will begin the precommitted second round and summarize the original diagnostic failures without equating an absent extraction field with an absent answer. Repeated measurements will test whether the near-zero accuracy effect persists.

```python
prepare_frozen_holdout_pass(39)

diagnostic_records = [
    json.loads(line) for line in (work / "gen_1.jsonl").read_text().splitlines()
    if line.strip()
]
diagnostic_samples = [s for r in diagnostic_records for s in r["samples"]]
print("ORIGINAL_DIAGNOSTIC_SUMMARY", {
    "problems": len(diagnostic_records),
    "answers": len(diagnostic_samples),
    "correct": sum(bool(s["correct"]) for s in diagnostic_samples),
    "incorrect_completed": sum(not s["correct"] and not s["cut"]
                               for s in diagnostic_samples),
    "incorrect_cutoff": sum(not s["correct"] and s["cut"]
                            for s in diagnostic_samples),
    "all_cutoffs": sum(bool(s["cut"]) for s in diagnostic_samples),
    "mean_tokens": round(float(np.mean([s["tokens"] for s in diagnostic_samples])), 1)
})
failure_example = next(s for s in diagnostic_samples
                       if not s["correct"] and not s["cut"])
print("ONE_DIAGNOSTIC_FAILURE_EXCERPT", ascii(failure_example["text"][:260]))
"Pass39 logged with unchanged weights and the precommitted eight-position rotation."
```