No experiment has run yet; the previous block was rejected because its explanation was not included in the same reply. I’ll now inspect tensor names, tokenizer support, and the practice-file schema without changing any weights. This will establish how to prepare valid diagnostic requests and reproducible problem splits. I have no failure-mechanism hypothesis yet; the first generations will supply that evidence.

```python
import os, json, math, random, glob
import numpy as np
import torch
from collections import Counter, defaultdict

print("Tensor count:", len(NAMES))
print("First tensor names:", list(NAMES)[:24])
print("Work files:", sorted(os.listdir("/work"))[:30])
print("Model files:", sorted(os.listdir("/model"))[:20])
for path in ["/practice/math_train.jsonl", "/practice/gsm8k_train.jsonl"]:
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    print(os.path.basename(path), "count", len(rows), "fields", list(rows[0]))
    print("First example:", {k: str(v)[:180] for k,v in rows[0].items() if k not in ("solution", "reference_solution")})
print("Prompt format:")
print(open("/work/eval_prompt_format.txt").read())
print("Tokenizer class:", type(tok).__name__)
print("Chat template available:", hasattr(tok, "apply_chat_template"))
```
