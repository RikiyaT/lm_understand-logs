```python
# Hypothesis: removing the distributed repetition direction from one MLP gives a stronger causal effect than ablating a single neuron.
import os, json, random, math, collections
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(8)
def readgen(k, model=None):
    p=f"/work/gen_{k}"+(f"_{model}" if model else "")+".jsonl"
    return [json.loads(s) for s in open(p)]
def summarize_pair(k):
    a,b=readgen(k,"original"),readgen(k,"edited")
    aa=np.array([s["correct"] for r in a for s in r["samples"]],float)
    bb=np.array([s["correct"] for r in b for s in r["samples"]],float)
    d=bb-aa
    print("Pair",k,"N",len(d),"accuracies",aa.mean(),bb.mean(),"gain",d.mean(),"SE",d.std(ddof=1)/np.sqrt(len(d)),
          "W->R",int((d>0).sum()),"R->W",int((d<0).sum()))
    for label,rs in [("original",a),("edited",b)]:
        print(label,"mean tokens",round(np.mean([s["tokens"] for r in rs for s in r["samples"]])),
              "cuts",sum(s["cut"] for r in rs for s in r["samples"]))
summarize_pair(7)
practice={s:[json.loads(x) for x in open("/practice/"+fn)] for s,fn in
          [("gsm8k","gsm8k_train.jsonl"),("math","math_train.jsonl"),("aime","aime.jsonl")]}
byid={r["id"]:r for rows in practice.values() for r in rows}
g1=readgen(1); diagnostic_ids=[r["id"] for r in g1]
probe_ids=[r["id"] for r in readgen(5,"original")]
meta4=json.load(open("/work/fwd_4/meta.json")); lens4=meta4["lengths"]
H4={l:np.load(f"/work/fwd_4/hidden_{l}.npy").astype(np.float32) for l in range(25)}
M4={l:np.load(f"/work/fwd_4/mlp_act_{l}.npy").astype(np.float32) for l in range(24)}
meansH={l:np.stack([a[i,lens4[i]-36:lens4[i]].mean(0) for i in range(8)]) for l,a in H4.items()}
meansM={l:np.stack([a[i,lens4[i]-36:lens4[i]].mean(0) for i in range(8)]) for l,a in M4.items()}
diffH={l:(a[1::2]-a[::2]).mean(0) for l,a in meansH.items()}
downweights={l:load_original(f"model.language_model.layers.{l}.mlp.down_proj.weight") for l in range(24)}
def clear_edits():
    for n in list(edited()): drop_edit(n)
def mlp_direction_edit(layers, strength=1.0):
    clear_edits()
    changes=[]
    for l in layers:
        d=torch.tensor(diffH[min(l+1,23)]); d/=d.norm()
        w=downweights[l]
        new=w-strength*torch.outer(d,d@w)
        save_edit(f"model.language_model.layers.{l}.mlp.down_proj.weight",new)
        changes.append((l,round(float((new-w).norm()/w.norm()),4)))
    return changes
print("Current edit relative change:",float((load("model.language_model.layers.19.mlp.down_proj.weight")-downweights[19]).norm()/downweights[19].norm()))
```
