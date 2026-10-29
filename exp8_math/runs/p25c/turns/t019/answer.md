The single-neuron ablation did not help: −0.8 percentage points, with a 3.4-point standard error. I’ll reject it and test whether repetition is sustained by a distributed circuit rather than one neuron.

```python
# Hypothesis: repetition is maintained by several coordinated MLP components; a single-neuron ablation leaves that circuit intact.
import os, json, math, random, collections, re
import numpy as np
import torch
torch.set_num_threads(8)
print("Work files:", sorted(os.listdir("/work")))
if os.path.exists("/work/notes.md"):
    print("Notes:", open("/work/notes.md").read()[-4500:])
print("Current edits:", edited())
for name in list(edited()):
    drop_edit(name)

practice={}
for source,path in [("gsm8k","/practice/gsm8k_train.jsonl"),("math","/practice/math_train.jsonl"),("aime","/practice/aime.jsonl")]:
    with open(path) as f:
        practice[source]=[json.loads(line) for line in f]
by_id={r["id"]:r for rows in practice.values() for r in rows}
with open("/work/gen_1.jsonl") as f:
    gen1=[json.loads(line) for line in f]
samples1=[dict(s,id=r["id"],source=r["source"]) for r in gen1 for s in r["samples"]]
good=np.array([s["correct"] for s in samples1])
cut=np.array([s["cut"] for s in samples1])
M=np.load("/work/gen_1_acts/mlp_mean_answer.npy").astype(np.float32)
H=np.load("/work/gen_1_acts/hidden_mean_answer.npy").astype(np.float32)
mixed=[(i if cut[i] else i+1,i+1 if cut[i] else i) for i in range(0,len(good),2) if good[i]!=good[i+1]]
pair_delta=np.stack([M[b]-M[g] for b,g in mixed])
delta=pair_delta.mean(0)
consistency=(pair_delta>0).mean(0)
weights={}
down_norm=[]
for l in range(24):
    name=f"model.language_model.layers.{l}.mlp.down_proj.weight"
    weights[name]=load_original(name)
    down_norm.append(weights[name].norm(dim=0).numpy())
down_norm=np.stack(down_norm)
tfM={l:np.load(f"/work/fwd_3/mlp_act_{l}.npy").astype(np.float32) for l in range(14,24)}
rep_delta=np.zeros_like(delta)
for l,a in tfM.items():
    rep_delta[l]=a[1,330:450].mean(0)-a[0,80:180].mean(0)
score=delta*down_norm
eligible=(consistency>=.9)&(delta>.1)&(rep_delta>.25)&(M[cut].mean(0)>2*np.maximum(M[good].mean(0),.05))
score[~eligible]=-np.inf
circuit=[]
for k in np.argsort(score.ravel())[::-1][:16]:
    l,j=np.unravel_index(k,score.shape)
    if np.isfinite(score[l,j]):
        circuit.append((int(l),int(j)))
print("Distributed candidates:",circuit)
print("Candidates' paired contribution scores:",[round(float(score[l,j]),3) for l,j in circuit])
for filename in sorted(os.listdir("/work")):
    if filename.endswith(".jsonl") and filename.startswith("gen_") and filename!="gen_1.jsonl":
        with open("/work/"+filename) as f:
            rows=[json.loads(line) for line in f]
        ss=[s for r in rows for s in r.get("samples",[])]
        if ss:
            print(filename,"n",len(ss),"accuracy",round(np.mean([s["correct"] for s in ss]),4),
                  "cut",round(np.mean([s["cut"] for s in ss]),4),
                  "mean_tokens",round(np.mean([s["tokens"] for s in ss])))
```
