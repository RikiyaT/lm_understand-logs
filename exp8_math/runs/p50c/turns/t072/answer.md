Generation is working again, and strength two passed independent validation decisively: accuracy rose from 8.2% to 43.0%, a paired gain of 34.8 ± 3.5 percentage points. It rescued 91 samples and lost two, with gains on both GSM8K and MATH. I’ll preserve this as the new confirmed candidate and resume the planned strength-three test on the diagnostic set. The remaining failures and increased wrong completions make it important to test stronger suppression rather than assume more is always better.

```python
# Hypothesis: alpha3 may improve on the confirmed alpha2 repair, but residual repetition and finished errors must both be measured.
import json, shutil, re
from pathlib import Path
import numpy as np
import torch

shutil.copytree("/work/edited","/work/best_confirmed_a2",dirs_exist_ok=True)
with open("/work/gen_27_edited.jsonl") as f:
    rows=[json.loads(line) for line in f if line.strip()]
samples=[s for r in rows for s in r["samples"]]
correct=[s for s in samples if s["correct"]]
print("Alpha2 held-out correct-token median:",float(np.median([s["tokens"] for s in correct])))
print("Finished wrong excerpts:")
shown=0
for r in rows:
    for s in r["samples"]:
        if not s["correct"] and not s["cut"] and shown<4:
            print(r["id"],"tokens",s["tokens"],"pred",str(s["pred"])[:60],
                  "tail",s["text"][-230:].replace("\n"," "))
            shown+=1

directions=torch.load("/work/repetition_directions.pt",weights_only=True)
for layer,u in directions.items():
    name=f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    w=load_original(name)
    save_edit(name,w-3*torch.outer(u,u@w))
dose_ids=json.loads(Path("/work/splits.json").read_text())["dose_response_64"]
plan={"next_forward_pass":20,
      "edit":{"tensors":[f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in (15,19)],
              "formula":"W0-3*outer(u,u@W0)","alpha":3,"directions":"/work/repetition_directions.pt"},
      "request":{"type":"generate","model":"both","problems":dose_ids,"n":4,"max_tokens":32768}}
Path("/work/current_plan.json").write_text(json.dumps(plan,indent=2))
with open("/work/notes.md","a") as f:
    f.write("\nGENERATION SERVICE RECOVERED. Pass19/gen27 INDEPENDENT alpha2 validation_2,128problems x2: "
            "21/256->110/256 (8.20->42.97%), gain34.77pp SE3.53pp,91rescues2losses. "
            "GSM13.28->60.94%,MATH3.12->25.00%; cutoff92.58->51.17%; mean30644->21742. "
            "New best confirmed: alpha2, backed up best_confirmed_a2. No head edits. "
            "Next alpha3 on original diagnostic dose64x4; prior alpha3 generation requests all failed and did not count.\n"
            "NEXT PASS EXACT PLAN:\n"+json.dumps(plan)+"\n")
```
