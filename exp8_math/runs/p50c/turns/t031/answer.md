```python
# The restart preserved edits and results: single-site loop-direction removal had a weak +6.25pp gain; next inspect the two-site test and reconstruct analysis before further causal probes.
import os,json,random,math,collections,shutil
from pathlib import Path
import numpy as np,torch
torch.set_num_threads(8)
def readgen(path):
    return [json.loads(s) for s in open(path)]
def statsgen(path):
    rows=readgen(path)
    ss=[s for r in rows for s in r["samples"]]
    return {"n":len(ss),"correct":sum(s["correct"] for s in ss),"cut":sum(s["cut"] for s in ss),"mean_tokens":round(np.mean([s["tokens"] for s in ss]))}
for k in [5,7,8]:
    paths=[f"/work/gen_{k}_{v}.jsonl" for v in ["original","edited"]]
    if all(os.path.exists(p) for p in paths):
        a,b=[readgen(p) for p in paths]
        ds=np.array([float(t["correct"])-float(s["correct"]) for r,q in zip(a,b) for s,t in zip(r["samples"],q["samples"])])
        print(k,[statsgen(p) for p in paths],"paired diff/SE",round(float(ds.mean()),4),round(float(ds.std(ddof=1)/np.sqrt(len(ds))),4))
practice={s:[json.loads(x) for x in open("/practice/"+f)] for s,f in [("gsm8k","gsm8k_train.jsonl"),("math","math_train.jsonl"),("aime","aime.jsonl")]}
byid={r["id"]:r for rows in practice.values() for r in rows}
g1=readgen("/work/gen_1.jsonl")
diagnostic_ids=[r["id"] for r in g1]
meta4=json.load(open("/work/fwd_4/meta.json"));lens4=meta4["lengths"]
H4={l:np.load(f"/work/fwd_4/hidden_{l}.npy").astype(np.float32) for l in range(25)}
M4={l:np.load(f"/work/fwd_4/mlp_act_{l}.npy").astype(np.float32) for l in range(24)}
meansH={l:np.stack([a[i,lens4[i]-36:lens4[i]].mean(0) for i in range(8)]) for l,a in H4.items()}
meansM={l:np.stack([a[i,lens4[i]-36:lens4[i]].mean(0) for i in range(8)]) for l,a in M4.items()}
diffH={l:(a[1::2]-a[::2]).mean(0) for l,a in meansH.items()}
downweights={l:load_original(f"model.language_model.layers.{l}.mlp.down_proj.weight") for l in range(24)}
def clear_edits():
    for n in list(edited()): drop_edit(n)
def direction_edit(layers,strength=1.0,kind="mlp"):
    clear_edits()
    changes=[]
    for l in layers:
        d=torch.from_numpy(diffH[min(l+1,23)].copy()).float();d/=d.norm()
        suffix="mlp.down_proj.weight" if kind=="mlp" else ("self_attn.o_proj.weight" if l%4==3 else "linear_attn.out_proj.weight")
        name=f"model.language_model.layers.{l}.{suffix}"
        w=load_original(name)
        new=w-strength*torch.outer(d,d@w)
        save_edit(name,new)
        changes.append((l,round(float((new-w).norm()/w.norm()),4)))
    return changes
print("Current edits",edited())
```
