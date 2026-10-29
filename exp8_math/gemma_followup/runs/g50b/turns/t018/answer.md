Doubling the selected outputs increased answer length and cut-offs, but its accuracy change was essentially zero: +0.20 points with a 2.63-point standard error. That supports an effect on generation behavior, not yet a clear mathematical failure mechanism. I’ll narrow the intervention to the four strongest neuron associations and test whether the modest accuracy gains are concentrated there.

```python
candidate = set_unit_edit(8, units8[:4], 1.0)
experiments[7] = dict(candidate)
with (work / "notes.md").open("a") as f:
    f.write("\nPass 6 amplification: +0.20 pp, reported SE 2.63 pp; "
            "longer answers but no established accuracy harm. "
            "Upcoming pass 7: BOTH, DEV_ids.json, n=32, max_tokens=8192. "
            "Exact edit: "+json.dumps(candidate)+". All other tensors original.\n")
print(candidate)
"Top-four neuron ablation ready."
```