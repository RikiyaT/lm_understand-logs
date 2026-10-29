```python
# Hypothesis: the current edit and any intervening paired results can be reconstructed from saved artifacts before choosing the next causal dose.
import os, json, collections, numpy as np, torch
from pathlib import Path
print("Work files:", sorted(os.listdir("/work")))
print("Current edits:", edited())
for key in ["targets", "candidates", "scores"]:
    if key in globals():
        value=globals()[key]
        print(key, str(value)[:1600])
def read_generations(path):
    rows=[json.loads(line) for line in open(path)]
    return [dict(id=r["id"],source=r.get("source"),**s) for r in rows for s in r["samples"]]
for path in sorted(Path("/work").glob("gen_*.jsonl")):
    rows=read_generations(path)
    print(path.name, "n",len(rows),"correct",sum(r["correct"] for r in rows),
          "cut",sum(r["cut"] for r in rows),"mean tokens",round(np.mean([r["tokens"] for r in rows])))
for name in edited():
    w=load(name); o=load_original(name)
    changed=(w-o).abs()
    print(name,"relative norm",float(changed.norm()/o.norm()),
          "changed columns",torch.where(changed.max(0).values>0)[0].tolist()[:100] if w.ndim==2 else "vector")
```
