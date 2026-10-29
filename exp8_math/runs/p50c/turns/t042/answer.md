Both current edits change every MLP output column, but only by about 3.4–3.5% in total magnitude; I’ll check whether each is the intended rank-one removal rather than a broader transformation. The generation files store results inside nested samples, so the first inspection did not recover their scores. I’ll now extract the paired outcomes, identify reused problems, and inspect the saved diagnostic metadata. This will determine which evidence is genuinely independent and which comparisons still need replication.

```python
# Hypothesis: current edits remove a single repetition direction, and nested paired outcomes reveal their actual benefit.
import json
from pathlib import Path
import numpy as np
import torch

gens = {}
for p in sorted(Path("/work").glob("gen_*.jsonl")):
    gens[p.stem] = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

for tag in ("gen_1", "gen_11_original", "gen_11_edited"):
    row = gens[tag][0]
    print("\nSCHEMA", tag, list(row))
    sample = row["samples"][0]
    print("sample keys:", list(sample))
    print("sample excerpt:", {k: (v[:260] if isinstance(v, str) else v)
                              for k, v in sample.items()
                              if not isinstance(v, (dict, list))})

for k in (5, 7, 8, 9, 10, 11):
    a = gens[f"gen_{k}_original"]
    b = gens[f"gen_{k}_edited"]
    print("\nPAIR", k, "ids identical:", [r["id"] for r in a] == [r["id"] for r in b])
    for label, rows in (("original", a), ("edited", b)):
        samples = [s for r in rows for s in r["samples"]]
        numeric = {key: round(float(np.mean([s[key] for s in samples])), 5)
                   for key, val in samples[0].items()
                   if isinstance(val, (bool, int, float))
                   and all(isinstance(s.get(key), (bool, int, float)) for s in samples)}
        print(label, "n =", len(samples), numeric)
    print("overlap with diagnostic:", len({r["id"] for r in a} & {r["id"] for r in gens["gen_1"]}))
    print("overlaps earlier paired sets:",
          {j: len({r["id"] for r in a} & {r["id"] for r in gens[f"gen_{j}_original"]})
           for j in (5, 7, 8, 9, 10, 11) if j < k})

edit_directions = {}
for name in edited():
    w = load_original(name)
    d = load(name) - w
    u = d[:, int(d.square().sum(0).argmax())].clone()
    u /= u.norm()
    projection = torch.outer(u, u @ w)
    alpha = -float((d * projection).sum() / projection.square().sum())
    rank1_error = float((d - torch.outer(u, u @ d)).norm() / d.norm())
    edit_directions[name] = u
    print("\nEDIT", name, "removal_strength =", alpha, "rank1_error =", rank1_error)

for folder in ("fwd_2", "fwd_4", "gen_1_acts"):
    p = Path("/work") / folder
    print("\nFILES", folder, [x.name for x in p.iterdir()][:25])
    for filename in ("meta.json", "index.json"):
        q = p / filename
        if q.exists():
            obj = json.loads(q.read_text())
            print(filename, "type =", type(obj).__name__)
            if isinstance(obj, dict):
                print("keys:", list(obj))
                for key, value in obj.items():
                    if isinstance(value, list):
                        print(key, "length =", len(value), "first =", str(value[0])[:500] if value else None)
                    elif isinstance(value, dict):
                        print(key, "keys =", list(value)[:20])
                    else:
                        print(key, str(value)[:200])
            elif isinstance(obj, list):
                print("length =", len(obj), "first =", str(obj[0])[:700] if obj else None)
```
