```python
# Hypothesis: the layer-19 direction has only a weak effect alone; inspect the recorded state before testing stronger, component-localized interventions.
print(Path("/work/notes.md").read_text()[-6000:])
print("Edits:", edited())
print("Forward result directories:", sorted(x for x in os.listdir("/work") if x.startswith(("gen_","fwd_"))))
print("Current layer19 relative change:",float((load(target19)-load_original(target19)).norm()/load_original(target19).norm()))
```
