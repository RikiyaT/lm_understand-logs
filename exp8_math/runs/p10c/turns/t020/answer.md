```python
# Hypothesis: the four repetition-correlated neurons are downstream markers rather than the causal feedback source; recover the experiment record before choosing a stronger localization test.
import os, json, math, collections
from pathlib import Path
import numpy as np
import torch
print(Path("/work/notes.md").read_text() if Path("/work/notes.md").exists() else "No notes file.")
print("Work outputs:", sorted(os.listdir("/work")))
print("Circuit targets:", targets if "targets" in globals() else "not retained")
print("Candidate summary:", str(candidates)[:1200] if "candidates" in globals() else "not retained")
print("Current edits:", edited())
print("Model config:", Path("/model/config.json").read_text()[:5000])
```
