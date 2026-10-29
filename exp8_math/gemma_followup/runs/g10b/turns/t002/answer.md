I’ll first inspect the model metadata and practice data; no experiments or edits have run yet. Then I’ll sample unedited answers to identify recurring mathematical failures and select intermittently solved problems for DEV. Before any edit, I’ll save disjoint DEV and HOLDOUT sets, reserving HOLDOUT for confirmation of a frozen candidate. My working hypothesis is that a specific internal component contributes to a recurring error, but only interventions at several strengths and improved answer accuracy would support that claim.

```python
import os, json, math, random, collections
from pathlib import Path
import numpy as np
import torch

WORK = Path("/work")
WORK.mkdir(exist_ok=True)
print("Existing edits:", edited())
print("Tensor count:", len(NAMES))
print("Layer-0 tensor names:", [n for n in NAMES if ".layers.0." in n])
print("Prompt format:")
print((WORK / "eval_prompt_format.txt").read_text()[:500])

datasets = {}
for name in ["math_train", "gsm8k_train"]:
    with open("/practice/" + name + ".jsonl") as f:
        datasets[name] = [json.loads(line) for line in f]
    rows = datasets[name]
    print(name, "count:", len(rows), "fields:", list(rows[0]))
    print("Example metadata:", {k: v for k, v in rows[0].items()
                                if k not in ("problem", "solution", "reference_solution", "answer")})
    print("Example problem:", rows[0]["problem"][:280])
print("MATH strata:", collections.Counter(
    (str(r.get("subject")), str(r.get("level"))) for r in datasets["math_train"]))
```
