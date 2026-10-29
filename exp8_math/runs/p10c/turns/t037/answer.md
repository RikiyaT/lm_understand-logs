```python
# Hypothesis: the current three-layer edit is a causal test of the repetition-associated neurons; recover its exact strength and paired outcome before further changes.
print("Current edits:", edited())
for name in edited():
    w, w0 = load(name), load_original(name)
    changed = (w != w0).any(dim=0) if w.ndim == 2 else (w != w0)
    js = torch.where(changed)[0]
    print(name, "changed columns:", js.tolist())
    if w.ndim == 2:
        print("Column scales:", [(int(j), round(float(torch.dot(w[:,j],w0[:,j]) / w0[:,j].square().sum()),4)) for j in js[:30]])
for key in ["candidates", "notes", "configuration"]:
    if key in globals():
        print(key, str(globals()[key])[:1600])
def read_generations(path):
    rows=[json.loads(s) for s in open(path)]
    return [dict(id=r["id"], source=r["source"], answer=r["answer"], **s) for r in rows for s in r["samples"]]
for file in sorted(Path("/work").glob("gen_*.jsonl")):
    rs=read_generations(file)
    print(file.name, "n",len(rs), "correct",sum(r["correct"] for r in rs),
          "cut",sum(r["cut"] for r in rs), "mean_tokens",round(np.mean([r["tokens"] for r in rs])))
if Path("/work/gen_4_edited.jsonl").exists():
    go4=read_generations("/work/gen_4_original.jsonl")
    ge4=read_generations("/work/gen_4_edited.jsonl")
    delta=np.array([int(e["correct"])-int(o["correct"]) for o,e in zip(go4,ge4)])
    print("Pass gen_4 paired gain",delta.mean(),"SE",delta.std(ddof=1)/np.sqrt(len(delta)),
          "W->R",int((delta==1).sum()),"R->W",int((delta==-1).sum()))
    print("First 12 item ids:",[r["id"] for r in go4[:12]])
```
