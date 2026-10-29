```python
# Hypothesis: the four-neuron ablation did not repair the loop, so the next causal test should follow the latest measured state rather than activation correlations alone.
import os, json, collections
from pathlib import Path
import numpy as np
import torch
print("Work files:", sorted(os.listdir("/work")))
print("Notes:\n", Path("/work/notes.md").read_text() if Path("/work/notes.md").exists() else "none")
print("Edits:", edited())
for p in sorted(Path("/work").glob("fwd_*/meta.json")):
    m=json.load(open(p))
    print(p.parent.name, "keys",list(m), "lengths",m.get("lengths"), "model",m.get("model"))
print("Available analysis variables:", sorted(k for k in globals() if not k.startswith("_") and k not in ["NAMES","tok"])[:180])
```
