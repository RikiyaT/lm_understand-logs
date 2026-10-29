Half suppression gained 1.76 points with a 1.75-point standard error, similar to full suppression rather than clearly better. I’ll now double the same neuron outputs as an opposite-direction control. If these components contribute to the observed failures, amplification should worsen accuracy or increase operation drift; otherwise, the suppression gains may be sampling noise or an indirect effect.

```python
checkpoint("candidate_half", {"candidate":candidate,"result":summarize_pair(5)})
candidate = set_unit_edit(8, units8, -1.0)
experiments[6] = dict(candidate)
with (work / "notes.md").open("a") as f:
    f.write("\nPass 5: +1.76 pp, reported SE 1.75 pp; half-suppression saved. "
            "Upcoming pass 6: opposite-direction control, BOTH on DEV_ids.json, "
            "n=32, max_tokens=8192. Exact edit: "+json.dumps(candidate)+
            ". Selected columns doubled; all other tensors original.\n")
print(candidate)
"Opposite-direction control ready."
```