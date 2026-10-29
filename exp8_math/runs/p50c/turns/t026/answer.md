```python
# Hypothesis: repetition is maintained by amplification at both layers 15 and 19, so removing the shared direction at those two sites should have a larger causal effect.
import json, os, numpy as np, torch
from pathlib import Path
print(Path("/work/notes.md").read_text()[-2200:])
print("Two-site edit:", mlp_direction_edit([15,19],1.0))
def generation_summary(path):
    rows=[json.loads(s) for s in open(path)]
    ss=[s for r in rows for s in r["samples"]]
    return dict(n=len(ss),correct=sum(s["correct"] for s in ss),cut=sum(s["cut"] for s in ss),
                mean_tokens=round(np.mean([s["tokens"] for s in ss])))
print("Previous original:",generation_summary("/work/gen_7_original.jsonl"))
print("Previous edited:",generation_summary("/work/gen_7_edited.jsonl"))
with open("/work/notes.md","a") as f:
    f.write("Pass5 gen_7 paired32: L19 MLP repetition-direction removal alpha1, +6.25pp +/-4.35pp SE (4->6 correct), cut29->26, mean30123->27698. Directional and neuron probes modest. Next two-site L15+L19 alpha1.\n")
```
