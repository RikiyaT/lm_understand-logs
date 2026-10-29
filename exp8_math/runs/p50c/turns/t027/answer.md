```python
# Hypothesis: the current best intervention must be reconstructed from saved evidence before any further causal tests.
import os, json, math, random, collections, numpy as np, torch
from pathlib import Path
print("Notes:", Path("/work/notes.md").read_text()[-4500:])
print("Work artifacts:", sorted(os.listdir("/work")))
print("Current edited tensors:", edited())
for n in edited():
    a=load(n); b=load_original(n); delta=a-b
    print(n, "relative_change", round(float(delta.norm()/b.norm()),5),
          "changed_columns",int((delta.abs().sum(0)>0).sum()) if delta.ndim==2 else None)
print("Relevant state variables:", [k for k in globals() if k in ["diffH","downweights","H4","M4","diagnostic_ids","practice","byid","mlp_direction_edit","clear_edits","candidates4"]])
def gen_summary(path):
    rr=[json.loads(x) for x in open(path)]
    ss=[s for r in rr for s in r["samples"]]
    return {"n":len(ss),"correct":sum(bool(s["correct"]) for s in ss),"cut":sum(bool(s["cut"]) for s in ss),
            "mean_tokens":round(np.mean([s["tokens"] for s in ss]))}
for fn in sorted(Path("/work").glob("gen_*.jsonl"),key=lambda p:(int(p.name.split("_")[1].split(".")[0]),p.name)):
    print(fn.name,gen_summary(fn))
```
