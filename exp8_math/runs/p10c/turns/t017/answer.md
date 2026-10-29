```python
# Hypothesis: ablating the four repetition-associated MLP outputs should reduce repetitive cutoffs and improve paired accuracy, not merely shorten answers.
import os, json, collections, math
from pathlib import Path
import numpy as np
import torch
def read_generations(path):
    rows=[json.loads(s) for s in open(path)]
    return [dict(id=r["id"],source=r["source"],answer=r["answer"],sample=j,**s)
            for r in rows for j,s in enumerate(r["samples"])]
g4o=read_generations("/work/gen_4_original.jsonl")
g4e=read_generations("/work/gen_4_edited.jsonl")
for label,rs in [("original",g4o),("four-neuron ablation",g4e)]:
    print(label,"N",len(rs),"correct",sum(r["correct"] for r in rs),
          "cut",sum(r["cut"] for r in rs),"mean tokens",round(np.mean([r["tokens"] for r in rs])))
    print("by source",[(s, sum(r["correct"] for r in rs if r["source"]==s),
                       sum(r["source"]==s for r in rs)) for s in ["gsm8k","math"]])
d=np.array([int(b["correct"])-int(a["correct"]) for a,b in zip(g4o,g4e)])
grouped=collections.defaultdict(list)
for r,v in zip(g4o,d): grouped[r["id"]].append(v)
di=np.array([np.mean(v) for v in grouped.values()])
print("Paired gain:",round(100*d.mean(),2),"pp; item-cluster SE",round(100*di.std(ddof=1)/np.sqrt(len(di)),2),
      "pp; wrong-to-right",int((d==1).sum()),"right-to-wrong",int((d==-1).sum()))
for i in [0,1,2,len(g4e)//2]:
    r=g4e[i]
    print("Edited excerpt",r["id"],"sample",r["sample"],"correct",r["correct"],"tokens",r["tokens"],
          "tail",repr(r["text"][-240:]))
print("Test item IDs:",json.dumps(list(grouped)))
```
