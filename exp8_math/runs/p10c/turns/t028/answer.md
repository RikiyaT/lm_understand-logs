```python
# Hypothesis: the four-neuron ablation did not repair the loop, so we must recover its measured effects before choosing a more causally specific intervention.
print(Path("/work/notes.md").read_text())
print("Forward metadata files:", sorted(str(p) for p in Path("/work").glob("fwd_*/meta.json")))
print("Available analysis variables:", sorted(k for k in globals() if not k.startswith("_") and any(s in k.lower() for s in ["target","candidate","score","delta","stat","head","gen4","span","probe","edit","dose"])))
for p in ["/work/gen_4_original_acts/index.json","/work/gen_4_edited_acts/index.json"]:
    if os.path.exists(p):
        d=json.load(open(p))
        print(p, "layers",d.get("layers"),"mlp_layers",d.get("mlp_layers"),"n",len(d.get("samples",[])))
```
