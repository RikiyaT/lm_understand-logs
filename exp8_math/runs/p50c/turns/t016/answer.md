```python
# Hypothesis: the execution environment was restarted; recover saved evidence and edits before further causal tests.
import os,json,random,collections,math
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(8)
print("Work files:", sorted(os.listdir("/work"))[:60])
print("Current edits:", edited())
print(Path("/work/notes.md").read_text() if Path("/work/notes.md").exists() else "No notes found")
print("Available cached names:", [n for n in ["g1","H4","M4","diffH","downweights","mlp_direction_edit"] if n in globals()])
```
