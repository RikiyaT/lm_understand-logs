```python
# Hypothesis: neuron 23:22 is only one contributor to a distributed repetition circuit; its isolated ablation is insufficient.
import os, json, math, collections, numpy as np, torch
print("Kernel variables available:",[k for k in ["weights","M","H","tfM","samples1","mixed","delta","score","tfmeta"] if k in globals()])
def read_generations(path):
    with open(path) as f:
        rows=[json.loads(line) for line in f]
    return [dict(s,id=r["id"],source=r["source"],reference_answer=r.get("answer")) for r in rows for s in r["samples"]]
g5o=read_generations("/work/gen_5_original.jsonl")
g5e=read_generations("/work/gen_5_edited.jsonl")
d=np.array([int(e["correct"])-int(o["correct"]) for o,e in zip(g5o,g5e)])
print("Trial A paired accuracy delta and naive sample SE:",d.mean(),d.std(ddof=1)/np.sqrt(len(d)))
print("Wrong->right:",int((d==1).sum()),"right->wrong:",int((d==-1).sum()))
for i in np.flatnonzero(d)[:6]:
    print("Changed outcome",i,g5o[i]["id"],"delta",int(d[i]),
          "original/edited tokens",g5o[i]["tokens"],g5e[i]["tokens"])
    print("Edited ending:",repr(g5e[i]["text"][-220:]))
if "weights" not in globals():
    torch.set_num_threads(8)
    weights={n:load_original(n) for n in NAMES if ".layers." in n}
if "M" not in globals():
    M=np.load("/work/gen_1_acts/mlp_mean_answer.npy").astype(np.float32)
    H=np.load("/work/gen_1_acts/hidden_mean_answer.npy").astype(np.float32)
    samples1=read_generations("/work/gen_1.jsonl")
good=np.array([s["correct"] for s in samples1])
mixed=[(i if not good[i] else i+1,i+1 if not good[i] else i)
       for i in range(0,len(samples1),2) if good[i]!=good[i+1]]
pair_delta=np.stack([M[b].astype(np.float32)-M[g].astype(np.float32) for b,g in mixed])
delta=pair_delta.mean(0)
down_norm=np.stack([weights[f"model.language_model.layers.{l}.mlp.down_proj.weight"].norm(dim=0).numpy() for l in range(24)])
tfM={l:np.load(f"/work/fwd_3/mlp_act_{l}.npy").astype(np.float32) for l in range(14,24)}
print("Cross-probe replication of matched-loop candidates:")
candidates=[]
for l in range(14,24):
    probe_delta=tfM[l][1,330:450].mean(0)-tfM[l][0,80:180].mean(0)
    consistency=(pair_delta[:,l]>0).mean(0)
    effect=delta[l]*down_norm[l]
    eligible=(consistency>=.9)&(probe_delta>.2)&(delta[l]>.1)
    for j in np.flatnonzero(eligible):
        candidates.append((float(effect[j]),l,int(j),float(delta[l,j]),float(probe_delta[j]),float(consistency[j])))
candidates.sort(reverse=True)
print(candidates[:20])
with open("/work/notes.md","a") as f:
    f.write("\nTrial A paired generation (gen_5): original 15/128 correct; edited 14/128. No accuracy benefit; not accepted. Lower cutoffs alone do not justify this edit.\n")
```
