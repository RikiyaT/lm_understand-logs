```python
# Hypothesis: the single-layer loop-direction removal has a modest causal effect, so a multi-layer intervention should show a stronger, testable dose response.
print("Recent files:", sorted(os.listdir("/work"))[-25:])
print("Notes:", Path("/work/notes.md").read_text()[-4000:])
print("Current edits:",edited())
print("Available state:",[(k,k in globals()) for k in ["diffH","downweights","mlp_direction_edit","diagnostic_ids","g1","M4","H4"]])
print("Generation result files:", sorted(str(p) for p in Path("/work").glob("gen*.jsonl")))
```
