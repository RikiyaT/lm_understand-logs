```python
# Hypothesis: the kernel state was reset during the environment change; recover experiment state before making any further edits.
import os, json, math, random, collections, re
import numpy as np
import torch
torch.set_num_threads(8)

print("Current edits:", edited())
print("Work files:", sorted(os.listdir("/work")))
if os.path.exists("/work/notes.md"):
    with open("/work/notes.md") as f:
        print("Experiment notes:\n", f.read()[-6500:])
candidate_name = "model.language_model.layers.23.mlp.down_proj.weight"
candidate_original = load_original(candidate_name)
current = load(candidate_name)
changed = ((current-candidate_original).abs().sum(dim=0)>0).nonzero().flatten()
print("Changed L23 MLP columns:", changed.tolist()[:80])
print("Relative change:", float((current-candidate_original).norm()/candidate_original.norm()))
for name in sorted(os.listdir("/work")):
    if name.startswith("gen_") and name.endswith(".jsonl"):
        with open("/work/"+name) as f:
            rows = [json.loads(line) for line in f]
        ss = [s for r in rows for s in r.get("samples",[])]
        if ss:
            print(name, "samples", len(ss), "correct", sum(s["correct"] for s in ss),
                  "cut", sum(s["cut"] for s in ss),
                  "mean_tokens", round(np.mean([s["tokens"] for s in ss])))
```
