```python
# Hypothesis: removing the layer-19 loop-writing direction should reduce repetition without changing the thinking protocol; inspect the paired causal test.
def read_gen(path):
    return [json.loads(s) for s in open(path)]
def summarize_pair(path_o,path_e):
    a=read_gen(path_o); b=read_gen(path_e)
    xo=np.array([int(s["correct"]) for r in a for s in r["samples"]])
    xe=np.array([int(s["correct"]) for r in b for s in r["samples"]])
    d=xe-xo
    out={"n":len(d),"original":float(xo.mean()),"edited":float(xe.mean()),
         "difference":float(d.mean()),"se":float(d.std(ddof=1)/np.sqrt(len(d))),
         "wrong_to_right":int((d==1).sum()),"right_to_wrong":int((d==-1).sum())}
    for label,rows in [("original",a),("edited",b)]:
        ss=[s for r in rows for s in r["samples"]]
        out[label+"_tokens"]=float(np.mean([s["tokens"] for s in ss]))
        out[label+"_cut"]=float(np.mean([s["cut"] for s in ss]))
    return out
print(summarize_pair("/work/gen_7_original.jsonl","/work/gen_7_edited.jsonl"))
print("Current edits:",edited())
print("Probe ids:",json.dumps([r["id"] for r in read_gen("/work/gen_7_original.jsonl")]))
```
