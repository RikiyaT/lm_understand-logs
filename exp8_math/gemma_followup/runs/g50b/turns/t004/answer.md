The unedited model scored 16.75% on this deliberately easier pool. Failures were not just cut-offs: 763 answers were wrong, 790 finished without an extractable answer, and 152 hit the limit. I’ll inspect examples and the activation metadata before locking DEV, because “no answer” could mean missing formatting, abandoned reasoning, or degeneration—each calls for a different repair.

```python
with (work / "gen_1.jsonl").open() as f:
    baseline = [json.loads(line) for line in f]

print("Record schema:", {k: type(v).__name__ for k, v in baseline[0].items()})
print("Scalar fields:", {k: v for k, v in baseline[0].items()
                         if not isinstance(v, (dict, list))
                         and len(str(v)) < 180})
act_dir = work / "gen_1_acts"
print("Activation files:", [p.name for p in act_dir.iterdir()])
for p in act_dir.glob("*.json"):
    print(p.name, p.read_text()[:2500])

text_keys = [k for k in baseline[0]
             if isinstance(baseline[0][k], str)
             and len(baseline[0][k]) > 300]
print("Long-text fields:", text_keys)
for i in [0, 1, 16, 17, 64, 65, 128, 129]:
    r = baseline[i]
    snippets = {k: r[k][:140] + " ... " + r[k][-140:]
                for k in text_keys}
    print(json.dumps({"index": i, "brief": {
        k: v for k, v in r.items()
        if isinstance(v, (bool, int, float)) or k in ("id", "problem_id", "failure")
    }, "snippets": snippets}, ensure_ascii=True))
"Inspected baseline schema and short excerpts; no edits."
```