The strength-three teacher-forced run completed successfully and slightly improved overall likelihood on the productive text while lowering it on the repetitive text. I’ll separate the repeated tail from the shared setup and check whether the attention pattern changed. Then I’ll retry the unresolved generation comparison, with the exact candidate and problem set recorded again. The dense probe is encouraging, but only a paired accuracy gain can justify retaining this strength.

```python
# Hypothesis: strength three selectively disrupts the repetitive continuation while preserving productive mathematics.
import json
from pathlib import Path
import numpy as np
import torch

bounds = json.loads(Path("/work/probe_bounds.json").read_text())
close_ids = tok.encode("</think>").ids
for tag in ("fwd_19", "fwd_20"):
    p = Path("/work")/tag
    lp = np.load(p/"token_logprobs.npy")
    top_ids = np.load(p/"topk_ids.npy")
    top_lp = np.load(p/"topk_logprobs.npy")
    print("\n",tag)
    for i,b in enumerate(bounds):
        n=b["length"]
        print("text",i,
              "branch_logp",round(float(lp[i,b["branch_start"]-1:n-1].mean()),5),
              "tail48_logp",round(float(lp[i,n-49:n-1].mean()),5))
        if len(close_ids)==1:
            mask=top_ids[i,:n]==close_ids[0]
            probs=np.where(mask,np.exp(top_lp[i,:n]),0).sum(-1)
            print("close token top20 appearances",int(mask.sum()),
                  "max observed close probability",round(float(probs.max()),6))
    period=16
    n=bounds[1]["length"]
    q=np.arange(n-64,n)
    for layer,head in ((15,7),(19,5)):
        a=np.load(p/f"attn_{layer}.npy",mmap_mode="r")
        print("induction",layer,head,round(float(np.asarray(a[1,head])[q,q-period+1].mean()),5))

dose_ids=json.loads(Path("/work/splits.json").read_text())["dose_response_64"]
plan={"next_forward_pass":15,
      "edit":{"tensors":[f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in (15,19)],
              "formula":"W0 - 3*outer(u,u@W0)","alpha":3,
              "directions":"/work/repetition_directions.pt"},
      "request":{"type":"generate","model":"both","problems":dose_ids,"n":4,"max_tokens":32768}}
Path("/work/current_plan.json").write_text(json.dumps(plan,indent=2))
with open("/work/notes.md","a") as f:
    f.write("\nPass14/fwd20 alpha3 edited paired TF to fwd19 original: overall productive logp "
            "-1.5776->-1.5559; repetitive -1.1642->-1.1941. "
            "Original attention induction scores L15h7=.3039, L19h5=.3082; observational candidates only. "
            "Retry alpha3 generation (previous attempts failed startup, no accuracy result yet).\n"
            "NEXT PASS EXACT PLAN:\n"+json.dumps(plan)+"\n")
```
