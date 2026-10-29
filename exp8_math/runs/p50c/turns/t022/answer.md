```python
# Hypothesis: the modest L19-only effect means repetition is distributed; recover the experiment log before selecting the next localized intervention.
print(Path("/work/notes.md").read_text())
print("Work files:", sorted(os.listdir("/work")))
print("Current edits:", edited())
print("Available analysis variables:", [k for k in ["diffH","downweights","H4","M4","meansM","meansH","probe_ids","mlp_direction_edit"] if k in globals()])
```
