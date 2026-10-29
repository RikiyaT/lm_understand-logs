```python
# Hypothesis: the current four-neuron ablation should reduce exact repetition and improve paired accuracy; recover its measured results before choosing the next dose.
import os, json, collections, numpy as np, torch
from pathlib import Path
print(Path("/work/notes.md").read_text())
def flatten_generations(path):
    rows=[json.loads(s) for s in open(path)]
    return [dict(id=r["id"],source=r.get("source"),answer=r.get("answer"),**s) for r in rows for s in r["samples"]]
for tag in ["original","edited"]:
    path=f"/work/gen_4_{tag}.jsonl"
    if os.path.exists(path):
        records=flatten_generations(path)
        print(tag,"n",len(records),"correct",sum(r["correct"] for r in records),
              "cut",sum(r["cut"] for r in records),"mean tokens",round(np.mean([r["tokens"] for r in records])))
        print("By source:",{s:(sum(r["correct"] for r in records if r["source"]==s),
                              sum(r["source"]==s for r in records)) for s in ["gsm8k","math"]})
if os.path.exists("/work/gen_4_edited.jsonl"):
    g4o=flatten_generations("/work/gen_4_original.jsonl")
    g4e=flatten_generations("/work/gen_4_edited.jsonl")
    d=np.array([int(e["correct"])-int(o["correct"]) for o,e in zip(g4o,g4e)])
    print("Paired gain / SE:",d.mean(),d.std(ddof=1)/np.sqrt(len(d)),
          "rescues",sum(d==1),"regressions",sum(d==-1))
print("Actual nonzero weight changes:")
for name in edited():
    old=load_original(name); new=load(name)
    changed=torch.where((new!=old).any(dim=0))[0]
    print(name,"columns",changed.tolist(),"factors",
          [(int(j),round(float(torch.sum(new[:,j]*old[:,j])/torch.sum(old[:,j]**2)),4)) for j in changed])
print("Available generation files:",sorted(p.name for p in Path("/work").glob("gen*.jsonl")))
```
