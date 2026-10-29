Halving the two copying heads changed likelihood on both texts, so the aggregate result is less selective than the MLP-direction probe. I’ll inspect the repetitive tail separately before drawing a conclusion. The head intervention still needs a causal generation test: weakening a copying head could interrupt loops, but it could also impair useful arithmetic or reference tracking. I’ll try a full paired comparison rather than retaining it based on attention patterns alone.

```python
# Hypothesis: attenuating the two observed copying heads can improve generated answers, despite possible costs to useful copying.
import json
from pathlib import Path
import numpy as np

bounds=json.loads(Path("/work/probe_bounds.json").read_text())
for tag in ("fwd_19","fwd_20","fwd_22"):
    lp=np.load(Path("/work")/tag/"token_logprobs.npy")
    print(tag,[(i,round(float(lp[i,b["length"]-49:b["length"]-1].mean()),5))
               for i,b in enumerate(bounds)])

dose_ids=json.loads(Path("/work/splits.json").read_text())["dose_response_64"]
plan={"next_forward_pass":16,
      "edit":{"tensors":[f"model.language_model.layers.{l}.self_attn.o_proj.weight" for l in (15,19)],
              "head_sites":[[15,7],[19,5]],"head_output_gain":0.5,
              "formula":"W0 with only head*256:(head+1)*256 output columns multiplied by 0.5",
              "mlp_edits":"none"},
      "request":{"type":"generate","model":"both","problems":dose_ids,"n":4,"max_tokens":32768}}
Path("/work/current_plan.json").write_text(json.dumps(plan,indent=2))
with open("/work/notes.md","a") as f:
    f.write("\nPass15/fwd22: isolated L15h7/L19h5 head outputs at .5, no MLP edits. "
            "Mean total productive logp -1.57757->-1.59451; repetitive -1.16423->-1.18896. "
            "Not yet evidence of improved accuracy; next paired generation tests it.\n"
            "NEXT PASS EXACT PLAN:\n"+json.dumps(plan)+"\n")
```
