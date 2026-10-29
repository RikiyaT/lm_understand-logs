```python
# Hypothesis: the first causal intervention did not improve accuracy; recover its exact weights and records before interpreting or revising it.
print("Current edits:", edited())
print("Work files:", sorted(os.listdir("/work")))
notes_path=Path("/work/notes.md")
if notes_path.exists():
    print("Notes:", notes_path.read_text()[-7000:])
for name in edited():
    w=load(name); w0=load_original(name)
    d=w-w0
    print("EDIT", name, "relative norm",float(d.norm()/w0.norm()),"changed entries",int((d!=0).sum()))
    if w.ndim==2:
        cols=torch.nonzero((d!=0).any(0)).flatten()
        print("Changed columns:",cols.tolist()[:100])
        print("Column norm ratios:",[(int(j),round(float(w[:,j].norm()/w0[:,j].norm()),4)) for j in cols[:30]])
print("Kernel variables present:", {k:k in globals() for k in ["flat1","TFM","repeat_deltas","dense_rows","meta3","diag_ids"]})
for p in ["/work/gen_4_original_acts/index.json","/work/gen_4_edited_acts/index.json"]:
    ix=json.load(open(p))
    print(p, "layers",ix.get("layers"), "mlp_layers",ix.get("mlp_layers"),
          "first samples",ix["samples"][:3])
```
