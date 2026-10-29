```python
# Hypothesis: the four-neuron intervention has already received its first causal test; recover exact strengths and paired results before further experiments.
import os, json, collections, inspect
import numpy as np
import torch
from pathlib import Path
print("Work files:", sorted(os.listdir("/work")))
for key in ["targets","candidates","direction","scores"]:
    value=globals().get(key)
    print(key, str(value)[:900])
if "apply_circuit_strength" in globals():
    try:
        print(inspect.getsource(apply_circuit_strength))
    except Exception:
        print("Function constants:", apply_circuit_strength.__code__.co_consts)
for p in sorted(Path("/work").glob("gen_*.jsonl")):
    rows=[json.loads(line) for line in open(p)]
    ss=[s for r in rows for s in r.get("samples",[])]
    print(p.name, "n",len(ss),"accuracy",np.mean([s["correct"] for s in ss]),
          "cut",np.mean([s["cut"] for s in ss]),"tokens",np.mean([s["tokens"] for s in ss]))
for name in edited():
    w=load(name); w0=load_original(name)
    cols=torch.where((w-w0).abs().sum(0)>0)[0]
    print(name,"changed columns",cols.tolist(),
          "ratios",[(int(j),round(float(torch.dot(w[:,j],w0[:,j])/w0[:,j].square().sum()),4)) for j in cols])
```
