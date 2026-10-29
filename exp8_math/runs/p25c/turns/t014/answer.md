```python
# Hypothesis: the single-neuron ablation is ready for a paired causal test; restore diagnostic state after the kernel change without changing the edit.
import os, json, math, random, re, collections
import numpy as np
import torch
torch.set_num_threads(8)
with open("/work/notes.md") as f:
    print(f.read())
with open("/work/gen_1.jsonl") as f:
    gen1=[json.loads(line) for line in f]
samples1=[dict(s,id=r["id"],source=r["source"],reference_answer=r["answer"]) for r in gen1 for s in r["samples"]]
practice={}
for source,path in [("gsm8k","/practice/gsm8k_train.jsonl"),("math","/practice/math_train.jsonl"),("aime","/practice/aime.jsonl")]:
    with open(path) as f:
        practice[source]=[json.loads(line) for line in f]
by_id={r["id"]:r for rows in practice.values() for r in rows}
good=np.array([s["correct"] for s in samples1])
cut=np.array([s["cut"] for s in samples1])
M=np.load("/work/gen_1_acts/mlp_mean_answer.npy").astype(np.float32)
H=np.load("/work/gen_1_acts/hidden_mean_answer.npy").astype(np.float32)
mixed=[(i if cut[i] else i+1,i+1 if cut[i] else i) for i in range(0,len(samples1),2) if good[i]!=good[i+1]]
candidate_name="model.language_model.layers.23.mlp.down_proj.weight"
candidate_original=load_original(candidate_name)
print("Present edit:",edited())
print("Work artifacts:", sorted(os.listdir("/work")))
print("Baseline sample count:",len(samples1))
```
