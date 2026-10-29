```python
# Hypothesis: the four-neuron ablation did not causally repair repetition; recover experiment records before selecting a different circuit.
print("Work entries:", sorted(os.listdir("/work")))
if os.path.exists("/work/notes.md"):
    print(Path("/work/notes.md").read_text()[:8500])
print("Current edits:", edited())
print("Relevant state variables:", [k for k in globals() if any(s in k.lower() for s in ["candidate","strength","paired","pass","spec","repeat","probe","score","head","dense"])][:100])
```
