The teacher-forced probe succeeded, including attention outputs, while the generation failures were limited to engine startup. The original assigns higher average likelihood to the repetitive text, though that alone is unsurprising once a phrase repeats. I’ll measure where its heads attend and then run the identical texts through the strength-three candidate. The key test is whether the edit disrupts repeated continuations more than productive calculations, without simply making the thinking-close token universally likely.

```python
# Hypothesis: the targeted correction selectively weakens repeated continuations rather than generally favoring early termination.
import json
from pathlib import Path
import numpy as np
import torch

folder = Path("/work/fwd_19")
meta = json.loads((folder/"meta.json").read_text())
texts = meta["texts"]
ids = np.load(folder/"input_ids.npy")
lp = np.load(folder/"token_logprobs.npy")
probe_bounds = []
for i, text in enumerate(texts):
    enc = tok.encode(text)
    start_char = text.index("<think>\n") + len("<think>\n")
    branch_char = text.index("His pay") if i == 0 else text.index("But is")
    start = next(j for j, (a,b) in enumerate(enc.offsets) if b > start_char)
    branch = next(j for j, (a,b) in enumerate(enc.offsets) if b > branch_char)
    length = meta["lengths"][i]
    probe_bounds.append({"answer_start":start,"branch_start":branch,"length":length})
    print("text", i, probe_bounds[-1],
          "answer_mean_logp", round(float(lp[i,start-1:length-1].mean()),4),
          "last48_mean_logp", round(float(lp[i,length-49:length-1].mean()),4))

period = len(tok.encode(" But is 20 the total? No, it is the remaining time.").ids)
print("Repeated phrase token period:", period)
for layer in (15,19):
    attn = np.load(folder/f"attn_{layer}.npy")
    n = meta["lengths"][1]
    q = np.arange(max(probe_bounds[1]["branch_start"]+2*period,n-64), n)
    rows = []
    for head in range(8):
        a = np.asarray(attn[1,head],dtype=np.float32)
        row = {"head":head, "previous_token":float(a[q,q-1].mean()),
               "previous_repeat_same_phase":float(a[q,q-period].mean()),
               "previous_repeat_next_phase":float(a[q,q-period+1].mean()),
               "prompt_mass":float(a[q,:probe_bounds[1]["answer_start"]].sum(-1).mean())}
        rows.append({k:round(v,4) if isinstance(v,float) else v for k,v in row.items()})
    print("attention layer", layer, rows)

request = json.loads(Path("/work/attention_probe_request.json").read_text())
request["model"] = "edited"
Path("/work/probe_bounds.json").write_text(json.dumps(probe_bounds))
plan = {"next_forward_pass":14,
        "edit":{"tensors":[f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in (15,19)],
                "formula":"W0 - 3*outer(u,u@W0)","alpha":3,
                "directions":"/work/repetition_directions.pt"},
        "request":request}
Path("/work/current_plan.json").write_text(json.dumps(plan,indent=2))
with open("/work/notes.md","a") as f:
    f.write("\nPass13/fwd19 original: exact-template productive/repetitive pay-problem pair, "
            "lengths198/203; attention15/19 outputs available. Paired edited TF next, "
            "not a generation accuracy result.\nNEXT PASS EXACT PLAN:\n"+json.dumps(plan)+"\n")
print("Next pass recorded: identical two texts, edited alpha3, same returns.")
```
