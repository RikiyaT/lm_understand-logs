The cleanup interval did not resolve startup, so strength four remains unmeasured on accuracy. I’ll stop retrying that generation request for now and use the working teacher-forced tool to assess its selectivity. The same transfer set will reveal whether stronger suppression starts harming productive mathematics or encouraging unfinished thinking to close. The final choice will still require independent generation evidence; strength two remains the confirmed fallback.

```python
# Hypothesis: alpha4 retains repetition selectivity, but dense transfer probes can expose overcorrection before further generation tests.
import json, shutil
from pathlib import Path
import torch

shutil.copytree("/work/edited","/work/candidate_L15_L19_a4",dirs_exist_ok=True)
for name in edited():
    w=load(name)
    print(name,"finite",bool(torch.isfinite(w).all()))
request=json.loads(Path("/work/transfer_probe_request.json").read_text())
request["model"]="edited"
plan={"next_forward_pass":21,
      "edit":{"tensors":[f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in (15,19)],
              "formula":"W0-4*outer(u,u@W0)","alpha":4,"directions":"/work/repetition_directions.pt"},
      "request_file":"/work/transfer_probe_request.json","request_overrides":{"model":"edited"},
      "problem_set":[request["texts"][i].split("\nPlease reason")[0].split("user\n",1)[1] for i in range(0,12,2)],
      "conditions":"productive/repetitive for all six, premature-close for cases1,3",
      "return":request["return"]}
Path("/work/current_plan.json").write_text(json.dumps(plan,indent=2))
with open("/work/notes.md","a") as f:
    f.write("\nAlpha4 generation failed again after90s wait; waiting is not a proven workaround. "
            "No alpha4 accuracy evidence. Saved candidate_L15_L19_a4. Next alpha4 transfer TF, "
            "then random-direction specificity control. BEST CONFIRMED alpha2 remains backed up.\n"
            "NEXT PASS EXACT PLAN:\n"+json.dumps(plan)+"\n")
```
