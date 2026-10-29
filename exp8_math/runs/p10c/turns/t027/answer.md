```python
# Hypothesis: the repetition-circuit intervention must be assessed against its paired causal test before any further edit is chosen.
print("Work files:", sorted(os.listdir("/work")))
print("Current edits:", edited())
for key in ["candidates", "targets", "factors"]:
    if key in globals():
        print(key, globals()[key])
for name in edited():
    w0 = load_original(name)
    w = load(name)
    changed_cols = torch.where((w != w0).any(dim=0))[0]
    print(name, "changed columns:", changed_cols.tolist(),
          "relative change:", float((w-w0).norm()/w0.norm()))
    for j in changed_cols[:15]:
        ratio = float(torch.dot(w[:,j],w0[:,j])/torch.dot(w0[:,j],w0[:,j]))
        print(" column", int(j), "retained coefficient", round(ratio,5))
for path in sorted(Path("/work").glob("gen_*.jsonl")):
    data = [json.loads(s) for s in open(path)]
    flat = [dict(id=r["id"], source=r.get("source"), **s) for r in data for s in r["samples"]]
    print(path.name, "n",len(flat),"correct",sum(bool(r["correct"]) for r in flat),
          "cuts",sum(bool(r["cut"]) for r in flat),"mean tokens",round(np.mean([r["tokens"] for r in flat])))
```
