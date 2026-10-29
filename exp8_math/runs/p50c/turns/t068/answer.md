Strength one shows the intended pattern across the new contexts: productive-text likelihood is mostly preserved or slightly improved, while repetitive-text likelihood decreases in every pair. I’ll quantify the token-aligned changes and then test strength two on the same probe. I’ll also check whether a small set of MLP neurons accounts for most of the repetition-directed output, which could suggest a more selective repair. Any such narrower edit will remain exploratory until generation accuracy supports it.

```python
# Hypothesis: the repetition effect has a dose-response and may be concentrated in a small, transferable set of MLP writers.
import json
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr

bounds=json.loads(Path("/work/transfer_probe_bounds.json").read_text())
base=np.load("/work/fwd_24/token_logprobs.npy")
cur=np.load("/work/fwd_25/token_logprobs.npy")
metrics={}
for condition in ("productive","repetitive"):
    changes=[]
    tails=[]
    for i,b in enumerate(bounds):
        if b["condition"]!=condition: continue
        changes.append(float((cur-base)[i,b["start"]-1:b["length"]-1].mean()))
        tails.append(float((cur-base)[i,b["length"]-25:b["length"]-1].mean()))
    metrics[condition]={"branch_delta":float(np.mean(changes)),"tail24_delta":float(np.mean(tails)),
                        "per_case_branch":changes}
print("Alpha1 transfer:",json.dumps(metrics))
close_changes=[]
for i,b in enumerate(bounds):
    if b["condition"]=="premature_close":
        n=b["length"]
        values=[float(x[i,n-2]) for x in (base,cur)]
        close_changes.append(values)
        print("Premature close",b["case"],values,"probability",np.exp(values).tolist())

directions=torch.load("/work/repetition_directions.pt",weights_only=True)
neuron_subsets={}
for layer,u0 in directions.items():
    u=u0.clone()
    h=np.load(f"/work/fwd_24/hidden_{layer+1}.npy",mmap_mode="r")
    tails=np.stack([np.asarray(h[i,n-24:n],dtype=np.float32).mean(0)
                    for i,n in enumerate(json.loads(Path("/work/fwd_24/meta.json").read_text())["lengths"][:12])])
    if float(torch.tensor((tails[1::2]-tails[::2]).mean(0))@u)<0: u=-u
    w=load_original(f"model.language_model.layers.{layer}.mlp.down_proj.weight")
    proj=u@w
    scores=[]
    for folder,count in (("fwd_4",8),("fwd_24",12)):
        meta=json.loads(Path("/work",folder,"meta.json").read_text())
        a=np.load(Path("/work",folder,f"mlp_act_{layer}.npy"),mmap_mode="r")
        means=np.stack([np.asarray(a[i,n-24:n],dtype=np.float32).mean(0)
                        for i,n in enumerate(meta["lengths"][:count])])
        scores.append(proj*torch.tensor((means[1::2]-means[::2]).mean(0)))
    score=scores[0]
    top=torch.argsort(score,descending=True)
    neuron_subsets[layer]=top[:64].clone()
    positive=score.clamp_min(0).sum()
    print("Writer layer",layer,"top12",top[:12].tolist(),
          "positive mass top8/32/64",[round(float(score[top[:k]].clamp_min(0).sum()/positive),3) for k in (8,32,64)],
          "transfer rank correlation",round(float(spearmanr(scores[0].numpy(),scores[1].numpy()).statistic),3),
          "top64 overlap",len(set(top[:64].tolist())&set(torch.argsort(scores[1],descending=True)[:64].tolist())))
torch.save(neuron_subsets,"/work/repetition_writer_top64.pt")

for layer,u in directions.items():
    name=f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    w=load_original(name)
    save_edit(name,w-2*torch.outer(u,u@w))

plan=json.loads(Path("/work/current_plan.json").read_text())
plan["next_forward_pass"]=18
plan["edit"]["alpha"]=2
plan["edit"]["formula"]="W0 - 2*outer(u,u@W0)"
Path("/work/current_plan.json").write_text(json.dumps(plan,indent=2))
with open("/work/notes.md","a") as f:
    f.write("\nPass17/fwd25 alpha1 transfer metrics: "+json.dumps(metrics)+
            "; premature-close baseline/edited logps "+json.dumps(close_changes)+
            ". Top64 writer subsets derived ONLY from fwd4 saved repetition_writer_top64.pt; not applied.\n"
            "NEXT PASS EXACT PLAN:\n"+json.dumps(plan)+"\n")
```
