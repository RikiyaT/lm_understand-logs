```python
# Hypothesis: removing the layer-19 loop-writing direction changes repetition causally; recover the paired result and experiment state after the kernel reset.
import os, json, math, random, collections
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(8)
print("Result files:", sorted(os.listdir("/work")))
def readgen(path):
    return [json.loads(s) for s in open(path)]
def gensummary(rows):
    ss=[s for r in rows for s in r["samples"]]
    return {"n":len(ss),"correct":sum(s["correct"] for s in ss),"cut":sum(s["cut"] for s in ss),"mean_tokens":round(np.mean([s["tokens"] for s in ss]))}
def paired_summary(stem):
    a=readgen(stem+"_original.jsonl");b=readgen(stem+"_edited.jsonl")
    aa=np.array([s["correct"] for r in a for s in r["samples"]],dtype=float)
    bb=np.array([s["correct"] for r in b for s in r["samples"]],dtype=float)
    dd=bb-aa
    print("Original",gensummary(a),"Edited",gensummary(b),"Delta",round(dd.mean(),4),"SE",round(dd.std(ddof=1)/np.sqrt(len(dd)),4))
    return a,b
g7o,g7e=paired_summary("/work/gen_7")
g1=readgen("/work/gen_1.jsonl")
diagnostic_ids=[r["id"] for r in g1]
probe_ids=[r["id"] for r in g7o]
meta4=json.load(open("/work/fwd_4/meta.json"));lens4=meta4["lengths"]
M4={l:np.load(f"/work/fwd_4/mlp_act_{l}.npy").astype(np.float32) for l in range(24)}
H4={l:np.load(f"/work/fwd_4/hidden_{l}.npy").astype(np.float32) for l in range(25)}
meansM={l:np.stack([a[i,lens4[i]-36:lens4[i]].mean(0) for i in range(8)]) for l,a in M4.items()}
meansH={l:np.stack([a[i,lens4[i]-36:lens4[i]].mean(0) for i in range(8)]) for l,a in H4.items()}
diffH={l:(a[1::2]-a[::2]).mean(0) for l,a in meansH.items()}
downweights={l:load_original(f"model.language_model.layers.{l}.mlp.down_proj.weight") for l in range(24)}
def clear_edits():
    for n in list(edited()): drop_edit(n)
def mlp_direction_edit(layers,strength=1.0):
    clear_edits()
    for l in layers:
        d=torch.from_numpy(diffH[min(l+1,23)].copy());d/=d.norm()
        w=downweights[l]
        save_edit(f"model.language_model.layers.{l}.mlp.down_proj.weight",w-strength*torch.outer(d,d@w))
print("Current edits:",edited())
print("Direction verification relative error:",float((load("model.language_model.layers.19.mlp.down_proj.weight")-(downweights[19]-torch.outer(torch.from_numpy(diffH[20])/np.linalg.norm(diffH[20]),(torch.from_numpy(diffH[20])/np.linalg.norm(diffH[20]))@downweights[19]))).norm()))
```
