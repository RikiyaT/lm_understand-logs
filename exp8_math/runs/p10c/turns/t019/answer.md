```python
# Hypothesis: the targeted repetition-neuron ablation has produced a causal test; recover its exact strength and paired outcomes before choosing a dose.
print("Edited tensors:", edited())
for name in edited():
    w=load(name); w0=load_original(name)
    d=w-w0
    cols=torch.nonzero(d.abs().sum(0)>0).flatten() if w.ndim==2 else torch.nonzero(d!=0).flatten()
    print(name, "changed columns:",cols.tolist(),
          "relative norm change:",round(float(d.norm()/w0.norm()),6))
    if w.ndim==2:
        print("column multipliers:",[(int(j),round(float((w[:,j]*w0[:,j]).sum()/w0[:,j].square().sum()),4)) for j in cols[:40]])
for fn in ["gen_4_original.jsonl","gen_4_edited.jsonl"]:
    p=Path("/work")/fn
    if p.exists():
        rows=[json.loads(s) for s in open(p)]
        flat=[dict(id=r["id"],source=r["source"],**s) for r in rows for s in r["samples"]]
        print(fn,"n",len(flat),"accuracy",np.mean([r["correct"] for r in flat]),
              "cut",np.mean([r["cut"] for r in flat]),"mean tokens",np.mean([r["tokens"] for r in flat]))
        print("by source",[(src,sum(r["correct"] for r in flat if r["source"]==src),sum(r["source"]==src for r in flat)) for src in ["gsm8k","math"]])
        globals()[fn.split(".")[0]]=flat
if "gen_4_original" in globals() and "gen_4_edited" in globals():
    dd=np.array([int(e["correct"])-int(o["correct"]) for o,e in zip(gen_4_original,gen_4_edited)])
    print("Paired gain",float(dd.mean()),"SE",float(dd.std(ddof=1)/np.sqrt(len(dd))),"rescues",int((dd>0).sum()),"regressions",int((dd<0).sum()))
print("State variable names:",[k for k in list(globals()) if not k.startswith("_")][-65:])
```
