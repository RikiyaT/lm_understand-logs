```python
# Hypothesis: the first ablation removed repetition-correlated detectors rather than its causal driver; inspect its behavioral effect before choosing another intervention.
import os, json, math, collections
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(8)
print("Work entries:",sorted(os.listdir("/work")))
print("Notes:\n",Path("/work/notes.md").read_text() if Path("/work/notes.md").exists() else "none")
print("Available diagnostic variables:", [k for k in ["flat1","idx1","M1","TFM","spans3","repeat_deltas","down_cache","by_id","diag_ids"] if k in globals()])
def ids_of(s):
    t=tok.encode(s)
    return list(t.ids) if hasattr(t,"ids") else list(t)
def repeated_fraction(ids,n=12):
    if len(ids)<n: return 0.
    grams=[tuple(ids[i:i+n]) for i in range(len(ids)-n+1)]
    return 1-len(set(grams))/len(grams)
def read_generations(path):
    return [dict(id=r["id"],source=r.get("source"),answer=r.get("answer"),**s)
            for line in open(path) for r in [json.loads(line)] for s in r["samples"]]
for path in sorted(Path("/work").glob("gen_*.jsonl")):
    data=read_generations(path)
    print(path.name,"n",len(data),"acc",np.mean([r["correct"] for r in data]),
          "cut",np.mean([r["cut"] for r in data]),"mean length",round(np.mean([r["tokens"] for r in data])),
          "tail-repeat",round(np.mean([repeated_fraction(ids_of(r["text"][-12000:])[-2048:]) for r in data]),3))
print("Current edits:",edited())
```
