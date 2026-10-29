The generation file is grouped by problem, rather than containing one record per answer; the activation index supplies the individual-sample mapping. My first inspection therefore displayed reference answers, not generated failures, and stopped before any weights changed. I’ll flatten the records and verify them against the activation index. Then I can reliably inspect failures and select DEV using mixed baseline success.

```python
baseline_groups = baseline
print("Problems:", len(baseline_groups))
print("Sample schema:")
for k, v in baseline_groups[0]["samples"][0].items():
    if isinstance(v, str):
        print(k, "str", ascii(v[:100]))
    elif isinstance(v, (int, float, bool)) or v is None:
        print(k, type(v).__name__, v)
    else:
        print(k, type(v).__name__)

records = []
for group in baseline_groups:
    for j, sample in enumerate(group["samples"]):
        records.append({
            **sample,
            "problem_id": group["id"],
            "sample_index": j,
            "reference": group["answer"],
            "source": group["source"],
        })
with open("/work/gen_1_acts/index.json") as f:
    act_index = json.load(f)
assert len(records) == len(act_index["samples"]) == 2048
assert all(
    r["problem_id"] == a["item"] and r["sample_index"] == a["sample"]
    for r, a in zip(records, act_index["samples"])
)
print("Verified activation/sample alignment:", len(records))

sample_text_key = next(
    (k for k in ["text", "response", "completion", "generated_text", "answer"]
     if isinstance(records[0].get(k), str)),
    None
)
print("Generated text field:", sample_text_key)
counts = {
    g["id"]: sum(bool(s["correct"]) for s in g["samples"])
    for g in baseline_groups
}
print("Success-count histogram:", dict(sorted(collections.Counter(counts.values()).items())))
print("Mixed-success MATH problems:",
      sum(0 < counts[g["id"]] < 16 for g in baseline_groups if g["source"] == "math"))

for i in [0, 2, 16, 32, 128, 256, 512, 1024, 1536, 1792]:
    r = records[i]
    s = r[sample_text_key]
    excerpt = s[:160] + (" ... " + s[-120:] if len(s) > 160 else "")
    print("Example", i, r["problem_id"], "correct:", r["correct"],
          "tokens:", r.get("tokens"), ascii(excerpt))
```
