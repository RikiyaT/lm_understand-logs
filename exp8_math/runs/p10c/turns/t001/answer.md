```python
# Hypothesis: baseline math failures can be localized by combining failure types with layerwise and MLP activation differences.
import os, json, math, time, collections
import numpy as np
import torch
from pathlib import Path
print("Model files:", os.listdir("/model"))
print("Practice files:", os.listdir("/practice"))
print("Tensor count:", len(NAMES))
print("Representative tensors:")
for name in NAMES:
    if ".layers.0." in name or ".layers.3." in name:
        print(name, tuple(load_original(name).shape))
practice = {}
for file in ["gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"]:
    with open("/practice/" + file) as f:
        rows = [json.loads(line) for line in f]
    practice[file] = rows
    print(file, len(rows), "fields:", list(rows[0]), "first ids:", [r["id"] for r in rows[:3]])
print("Existing edits:", edited())
```
