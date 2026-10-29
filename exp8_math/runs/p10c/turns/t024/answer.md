```python
# Hypothesis: the initial four-neuron ablation was not sufficient; recover its measured mechanism and dose before choosing the next causal intervention.
print(Path("/work/notes.md").read_text())
print("Available analysis state:", [k for k in globals() if any(s in k.lower() for s in ["candidate","target","strength","trial","score","circuit","gen_4","repeat","direction"])])
for key in ["targets","candidates","trial_ids"]:
    if key in globals():
        print(key, str(globals()[key])[:1800])
print("Current edits:", edited())
```
