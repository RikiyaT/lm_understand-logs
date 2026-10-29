The interface treated the filename as individual one-character texts, so pass 2 produced no usable mathematical evidence. That was an input-format mistake, and I’ll exclude it from all scientific comparisons. With eight passes remaining, I’ll use an explicit raw-text batch for baseline and edited teacher forcing, then reserve paired generation passes for strength testing and HOLDOUT confirmation. I’ll first size the DEV reference solutions so the next request can be submitted correctly.

```python
reference_only = [problem_by_id[pid]["solution"] for pid in dev_ids]
reference_tokens = [len(encode_ids(s)) for s in reference_only]
print("All DEV reference solutions:", len(reference_only),
      "tokens:", sum(reference_tokens),
      "characters:", sum(map(len, reference_only)),
      "max tokens:", max(reference_tokens))
print("Solution token quantiles:", np.quantile(reference_tokens, [0, .25, .5, .75, 1]).tolist())
print("DEV ids:", json.dumps(dev_ids))
print("Reference previews:")
for pid in dev_ids[:6]:
    print(pid, problem_by_id[pid]["solution"][:220].encode("ascii", "backslashreplace").decode())
with open(wdir / "notes.md", "a") as f:
    f.write("\n## Pass 2 INVALID EXPERIMENT (budget consumed)\n"
            "A string filename supplied as texts was interpreted as 23 single-character texts.\n"
            "No mathematical scoring data; exclude all pass-2 results from analysis.\n"
            "No edits. Eight passes remain. Subsequent requests must include literal raw-text lists.\n")
```
