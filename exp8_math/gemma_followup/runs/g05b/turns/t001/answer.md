I’ll first inspect the available weights, practice data, and prompt format without changing anything. Then I’ll sample unedited answers to identify recurring failures and components associated with them within the same problems. Before editing, I’ll save disjoint development and holdout sets; only development results will guide the edits. The five-pass plan is baseline diagnosis, teacher-forced localization, two paired development tests at different strengths, and one frozen holdout test. Improved answer accuracy—not merely shorter responses—will determine whether a repair helps.

```python
import os, json, glob, math, collections
import numpy as np
import torch

print("Model tensors:", len(NAMES))
print("Representative names:", [
    n for n in NAMES
    if ("layers.0." in n or "layers.17." in n)
][:24])
print("Work files:", sorted(os.listdir("/work"))[:30])
with open("/work/eval_prompt_format.txt") as f:
    prompt_format = f.read()
print("Prompt format:", repr(prompt_format))

practice = {}
practice_by_source = {}
for source in ["gsm8k_train", "math_train"]:
    with open("/practice/" + source + ".jsonl") as f:
        rows = [json.loads(line) for line in f]
    practice_by_source[source] = rows
    for row in rows:
        practice[row["id"]] = row
    print(source, "count:", len(rows), "fields:", sorted(rows[0].keys()))
    if source == "math_train":
        print("Levels:", dict(collections.Counter(str(r.get("level")) for r in rows)))
        print("Subjects:", dict(collections.Counter(str(r.get("subject")) for r in rows)))

print("Tokenizer interface:", type(tok).__name__)
print("Existing edits:", edited())
print("No weights changed.")
```
