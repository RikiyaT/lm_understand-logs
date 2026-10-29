```python
# Hypothesis: the unrecorded paired trials can distinguish failed repetition interventions from promising ones.
import os, json, re, math, pathlib
from collections import Counter
import numpy as np
import torch

W = pathlib.Path("/work")
print("NOTES:\n", (W / "notes.md").read_text()[-12000:])
print("Edited tensors:", edited())
print("Tensor count:", len(NAMES))
print("Representative tensor names:", [
    n for n in NAMES if re.search(r"\.layers\.(0|3|23)\.", n)
][:65])

gen_data = {}
for p in sorted(W.glob("gen_*.jsonl")):
    rows = [json.loads(line) for line in p.open() if line.strip()]
    gen_data[p.stem] = rows
    print("\nFILE", p.name, "rows", len(rows))
    if rows:
        print("Fields:", list(rows[0].keys()))
        for key in rows[0]:
            if any(s in key.lower() for s in ("correct", "grade", "finish", "length", "tokens", "cutoff")):
                vals = [r.get(key) for r in rows]
                if all(isinstance(v, (int, float, bool)) for v in vals):
                    print(key, {"mean": float(np.mean(vals)), "sum": float(np.sum(vals))})
                elif all(isinstance(v, (str, type(None), bool)) for v in vals):
                    print(key, dict(Counter(vals)))
        small = {k: v for k, v in rows[0].items()
                 if not isinstance(v, (list, dict)) and len(str(v)) < 160}
        print("First-row scalar metadata:", small)

for d in sorted(W.glob("*acts*")) + sorted(W.glob("fwd_*")):
    print("\nSAVED", d.name, "files:", [p.name for p in list(d.iterdir())[:12]])
    for name in ("index.json", "meta.json"):
        p = d / name
        if p.exists():
            obj = json.loads(p.read_text())
            print(name, "type", type(obj).__name__)
            if isinstance(obj, dict):
                print("keys:", list(obj))
                for k, v in obj.items():
                    if isinstance(v, (str, int, float, bool)):
                        print(k, str(v)[:200])
                    elif isinstance(v, list):
                        print(k, "length", len(v), "first item:", str(v[:1])[:900])
            else:
                print("length", len(obj), "first item:", str(obj[:1])[:900])
```
