The two-sample layout succeeded. Strength three improved accuracy from 8.2% to 48.0%, a paired gain of 39.8 ± 3.8 percentage points, with 102 rescues and no regressions on the diagnostic set. Cutoffs fell substantially, but 42 samples finished incorrectly or without an answer, so the tradeoff remains important. I’ll test strength four next, then use a matched-size random-direction control to check whether the benefit depends on the identified repetition directions.

```python
# Hypothesis: alpha4 may further interrupt repetitive loops, while the accuracy/error tradeoff will reveal overcorrection.
import json, shutil
from pathlib import Path
import numpy as np
import torch

shutil.copytree("/work/edited","/work/candidate_L15_L19_a3",dirs_exist_ok=True)
with open("/work/gen_29_edited.jsonl") as f:
    rows=[json.loads(line) for line in f if line.strip()]
ss=[s for r in rows for s in r["samples"]]
print("Alpha3 correct length median:",float(np.median([s["tokens"] for s in ss if s["correct"]])))
print("Finished without extracted answer:")
shown=0
for r in rows:
    for s in r["samples"]:
        if not s["correct"] and not s["cut"] and s["pred"] is None and shown<3:
            print(r["id"],s["tokens"],s["text"][-250:].replace("\n"," "))
            shown+=1
directions=torch.load("/work/repetition_directions.pt",weights_only=True)
for layer,u in directions.items():
    name=f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    w=load_original(name)
    candidate=w-4*torch.outer(u,u@w)
    save_edit(name,candidate)
    print(layer,"alpha4 relative change",round(float((candidate-w).norm()/w.norm()),5))

dose=json.loads(Path("/work/splits.json").read_text())["dose_response_128"]
plan={"next_forward_pass":21,
      "edit":{"tensors":[f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in (15,19)],
              "formula":"W0-4*outer(u,u@W0)","alpha":4,"directions":"/work/repetition_directions.pt"},
      "request":{"type":"generate","model":"both","problems":dose,"n":2,"max_tokens":32768}}
Path("/work/current_plan.json").write_text(json.dumps(plan,indent=2))
with open("/work/notes.md","a") as f:
    f.write("\nPass20/gen29 alpha3 dose128x2 SUCCEEDED: original21/256->edited123/256, "
            "+39.84pp SE3.82pp,102rescues0losses; GSM7.81->53.91%,MATH8.59->42.19%; "
            "cut92.97->35.55%; wrong finished0->36,noanswer0->6, mean30864->18914. "
            "Candidate alpha3 not independently confirmed. Use128x2 layout;64x4 repeatedly fails startup. "
            "BEST INDEPENDENTLY CONFIRMED remains alpha2 (gen27 +34.77pp SE3.53), backup best_confirmed_a2.\n"
            "NEXT PASS EXACT PLAN:\n"+json.dumps(plan)+"\n")
```
