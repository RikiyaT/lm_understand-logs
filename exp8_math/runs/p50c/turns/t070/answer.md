Strength two again lowers likelihood on every repetitive transfer text while largely preserving the productive continuations. This supports the proposed mechanism, but it does not establish a better final model than the independently confirmed strength-one edit. I’ll now attempt a fresh held-out generation test of strength two, using 128 new problems and two samples each. If the generation service works, this will provide 256 paired answers per model without reusing the tuning problems.

```python
# Hypothesis: alpha2's selective transfer effect yields an independent accuracy gain on 128 new problems with two samples each.
import json
from pathlib import Path
import numpy as np
import torch

bounds=json.loads(Path("/work/transfer_probe_bounds.json").read_text())
base=np.load("/work/fwd_24/token_logprobs.npy")
cur=np.load("/work/fwd_26/token_logprobs.npy")
metrics={}
for condition in ("productive","repetitive"):
    ii=[i for i,b in enumerate(bounds) if b["condition"]==condition]
    metrics[condition]={
        "branch_delta":float(np.mean([(cur-base)[i,bounds[i]["start"]-1:bounds[i]["length"]-1].mean() for i in ii])),
        "tail24_delta":float(np.mean([(cur-base)[i,bounds[i]["length"]-25:bounds[i]["length"]-1].mean() for i in ii]))}
print("Alpha2 transfer",metrics)
print("Alpha2 premature-close logp:",
      [(b["case"],float(cur[i,b["length"]-2])) for i,b in enumerate(bounds) if b["condition"]=="premature_close"])

practice={}
for filename in ("gsm8k_train.jsonl","math_train.jsonl"):
    with open("/practice/"+filename) as f:
        for line in f:
            if line.strip():
                r=json.loads(line)
                practice[r["id"]]=r
used=set()
for p in Path("/work").glob("gen_*.jsonl"):
    with open(p) as f:
        for line in f:
            if line.strip(): used.add(json.loads(line)["id"])
rng=np.random.default_rng(389014)
validation2=[]
for source in ("gsm8k","math"):
    pool=sorted(k for k,r in practice.items() if r["source"]==source and k not in used)
    validation2+=list(rng.choice(pool,64,replace=False))
splits=json.loads(Path("/work/splits.json").read_text())
splits["validation_2"]=validation2
Path("/work/splits.json").write_text(json.dumps(splits,indent=2))
request={"type":"generate","model":"both","problems":validation2,"n":2,"max_tokens":32768}
plan={"next_forward_pass":19,
      "edit":{"tensors":[f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in (15,19)],
              "formula":"W0-2*outer(u,u@W0)","alpha":2,"directions":"/work/repetition_directions.pt"},
      "request":request,"purpose":"independent confirmation of alpha2, never-used problems"}
Path("/work/current_plan.json").write_text(json.dumps(plan,indent=2))
with open("/work/notes.md","a") as f:
    f.write("\nPass18/fwd26 alpha2 transfer: "+json.dumps(metrics)+
            ". Next attempt full-budget independent generation on new validation_2 set,128 problems x2.\n"
            "NEXT PASS EXACT PLAN:\n"+json.dumps(plan)+"\n")
print(json.dumps(request))
```
