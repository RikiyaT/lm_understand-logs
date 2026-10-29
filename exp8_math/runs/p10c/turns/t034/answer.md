```python
# Hypothesis: the first causal intervention did not repair repetition; recover its exact targets and the remaining budget before choosing a stronger or different circuit.
print("Work files:", sorted(os.listdir("/work")))
print("Current edits:", edited())
notes_path=Path("/work/notes.md")
if notes_path.exists():
    print("NOTES:\n"+notes_path.read_text()[-14000:])
for var in ["targets","candidates","factors","FROZEN"]:
    obj=globals().get(var)
    if obj is not None:
        print(var, str(obj)[:1800])
if "apply_circuit_strength" in globals():
    import inspect
    try:
        print(inspect.getsource(apply_circuit_strength))
    except Exception:
        print("Function constants:",apply_circuit_strength.__code__.co_consts)
```
