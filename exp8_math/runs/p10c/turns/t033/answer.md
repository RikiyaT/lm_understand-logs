```python
# Hypothesis: ablating the four repetition-associated neurons should reduce copying while preserving substantive mathematical reasoning; recover the paired causal result.
import os, json, collections, math
from pathlib import Path
import numpy as np
import torch
def read_generations(path):
    rows=[json.loads(x) for x in open(path)]
    return [dict(id=r["id"],source=r.get("source"),answer=r.get("answer"),**s) for r in rows for s in r["samples"]]
g4o=read_generations("/work/gen_4_original.jsonl")
g4e=read_generations("/work/gen_4_edited.jsonl")
def paired_summary(a,b):
    assert [(x["id"],x.get("sample")) for x in a]==[(x["id"],x.get("sample")) for x in b]
    ya=np.array([x["correct"] for x in a],float); yb=np.array([x["correct"] for x in b],float)
    d=yb-ya
    return {"n":len(a),"original_correct":int(ya.sum()),"edited_correct":int(yb.sum()),
            "gain":float(d.mean()),"paired_se":float(d.std(ddof=1)/np.sqrt(len(d))),
            "rescued":int((d>0).sum()),"regressed":int((d<0).sum()),
            "original_cut":float(np.mean([x["cut"] for x in a])),
            "edited_cut":float(np.mean([x["cut"] for x in b])),
            "original_mean_tokens":float(np.mean([x["tokens"] for x in a])),
            "edited_mean_tokens":float(np.mean([x["tokens"] for x in b]))}
print("Recovered causal result:",paired_summary(g4o,g4e))
for src in sorted(set(r["source"] for r in g4o)):
    print(src,paired_summary([r for r in g4o if r["source"]==src],[r for r in g4e if r["source"]==src]))
print("Available variable names:", sorted(k for k in globals() if not k.startswith("_"))[:150])
print("Edited excerpts:")
for j in [0,1,2,3]:
    r=g4e[j]
    print(r["id"],"correct",r["correct"],"tokens",r["tokens"],"think-close",r["text"].find("</think>"))
    print(repr(r["text"][:240]),repr(r["text"][-240:]))
```
