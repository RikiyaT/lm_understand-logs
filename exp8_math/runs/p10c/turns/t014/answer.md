```python
# Hypothesis: suppressing four repetition-amplifying MLP neurons should break the feedback loop while preserving the mathematical reasoning circuit.
import os, json, collections, math
from pathlib import Path
import numpy as np
import torch
print("Current edits:", edited())
print("Work files:", sorted(os.listdir("/work")))
if Path("/work/notes.md").exists():
    print(Path("/work/notes.md").read_text()[-3500:])
gen1 = [json.loads(line) for line in open("/work/gen_1.jsonl")]
flat1 = [dict(id=r["id"],source=r["source"],answer=r["answer"],**s) for r in gen1 for s in r["samples"]]
diag_ids = [r["id"] for r in gen1]
practice = {}
for file in ["gsm8k_train.jsonl","math_train.jsonl","aime.jsonl"]:
    practice[file] = [json.loads(line) for line in open("/practice/"+file)]
by_id = {r["id"]:r for rows in practice.values() for r in rows}
for name in edited():
    a,b=load(name),load_original(name)
    changed=(a!=b).any(0) if a.ndim==2 else (a!=b)
    print(name,"changed columns/entries:",torch.where(changed)[0].tolist()[:30],
          "relative change:",float((a-b).norm()/b.norm()))
print("First 64 diagnostic IDs:",json.dumps(diag_ids[:32]+diag_ids[96:128]))
```
