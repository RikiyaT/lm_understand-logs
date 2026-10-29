```python
# Hypothesis: the four-neuron ablation did not causally repair repetition; recover experiment state before selecting a more structural intervention.
print("Notes:")
p=Path("/work/notes.md")
print(p.read_text()[-7500:] if p.exists() else "No notes file.")
print("Current edits:", edited())
print("Work files:", sorted(os.listdir("/work")))
print("Relevant state variables:", [k for k in list(globals()) if any(s in k.lower() for s in ["edit","ablat","test","candidate","repeat","score","embed","norm"]) and not k.startswith("_")])
```
