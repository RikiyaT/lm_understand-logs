Reducing all of layer 11 head 2 suppressed repetition but badly hurt accuracy, so that head also carries useful information. The saved candidate instead removes one repetition-associated output direction while preserving the rest of the head. This association is only a hypothesis; generation must establish whether removing it helps. I will verify the saved edit and DEV identifiers before testing it, without inspecting HOLDOUT outcomes.

```python
import json, os, numpy as np, torch
from pathlib import Path

work = Path("/work")
dev = json.loads((work / "DEV.json").read_text())
print("DEV:", dev)
print("Current edit:", (work / "current_edit.json").read_text())
print("Edited tensors:", edited())
delta = np.load(work / "l11h2_repetition_rank1_delta.npy")
print("Rank-one delta:", {"shape": delta.shape, "norm": float(np.linalg.norm(delta))})
print("Earlier experiment notes:\n", (work / "notes.md").read_text()[:6500])
```
