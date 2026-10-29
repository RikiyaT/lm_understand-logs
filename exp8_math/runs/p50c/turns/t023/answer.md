```python
# Hypothesis: selective removal of the layer-19 repetition-writing direction improves completion without suppressing useful reasoning; inspect its paired causal result.
def readgen(path):
    return [json.loads(s) for s in open(path)]
def gensummary(rows):
    ss=[s for r in rows for s in r["samples"]]
    return {"n":len(ss),"correct":sum(s["correct"] for s in ss),"cut":sum(s["cut"] for s in ss),"mean_tokens":round(np.mean([s["tokens"] for s in ss]))}
for k in [5,7]:
    a=readgen(f"/work/gen_{k}_original.jsonl")
    b=readgen(f"/work/gen_{k}_edited.jsonl")
    delta=np.array([float(y["correct"])-float(x["correct"]) for ra,rb in zip(a,b) for x,y in zip(ra["samples"],rb["samples"])])
    print(k,"original",gensummary(a),"edited",gensummary(b),"difference",round(delta.mean(),4),"SE",round(delta.std(ddof=1)/np.sqrt(len(delta)),4))
probe_ids=[r["id"] for r in readgen("/work/gen_7_original.jsonl")]
print("Probe size:",len(probe_ids))
print("Current edit relative change:",float((load("model.language_model.layers.19.mlp.down_proj.weight")-load_original("model.language_model.layers.19.mlp.down_proj.weight")).norm()/load_original("model.language_model.layers.19.mlp.down_proj.weight").norm()))
print("Available useful variables:",[n for n in ["practice","byid","rng","diagnostic_ids","diffH","downweights","mlp_direction_edit","M4","H4","meansM","meansH"] if n in globals()])
```
