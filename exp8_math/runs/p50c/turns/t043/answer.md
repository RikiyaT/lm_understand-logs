The current edits are indeed rank-one removals in layers 15 and 19. The latest 32-problem comparison improved accuracy from 12.5% to 28.1%, but all six comparisons reused the same problems, so that gain is exploratory rather than confirmed. I’ll preserve this candidate and select an independent, balanced validation set using verified IDs. I’ll also recover the diagnostic contrasts so subsequent strength tests remain tied to repetition, not merely to faster stopping.

```python
# Hypothesis: removing the localized repetition direction improves accuracy on unseen problems, not just the diagnostic set.
import json, shutil
from pathlib import Path
import numpy as np
import torch

gens = {p.stem: [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        for p in Path("/work").glob("gen_*.jsonl")}
used_ids = {r["id"] for rows in gens.values() for r in rows}

for k in (5, 7, 8, 9, 10, 11):
    aa = gens[f"gen_{k}_original"]
    bb = gens[f"gen_{k}_edited"]
    av = np.array([np.mean([s["correct"] for s in r["samples"]]) for r in aa])
    bv = np.array([np.mean([s["correct"] for s in r["samples"]]) for r in bb])
    diff = bv - av
    print("pair", k, "correct", int(av.sum()), "->", int(bv.sum()),
          "gain_pp", round(100*diff.mean(), 2),
          "SE_pp", round(100*diff.std(ddof=1)/np.sqrt(len(diff)), 2),
          "rescued", int((diff > 0).sum()), "lost", int((diff < 0).sum()))

practice = {}
for filename in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
    for line in Path("/practice", filename).read_text().splitlines():
        r = json.loads(line)
        practice[r["id"]] = r

rng = np.random.default_rng(621403)
validation_ids = []
for source in ("gsm8k", "math"):
    pool = sorted(k for k, r in practice.items() if r["source"].lower() == source and k not in used_ids)
    if len(pool) < 64:
        print("Available source labels:", sorted({r["source"] for r in practice.values()}))
        raise ValueError("Unexpected source label")
    validation_ids += list(rng.choice(pool, size=64, replace=False))

Path("/work/splits.json").write_text(json.dumps({
    "diagnostic_ids": sorted(used_ids), "validation_1": validation_ids
}, indent=2))
shutil.copytree("/work/edited", "/work/candidate_L15_L19_a1", dirs_exist_ok=True)
directions = {}
for layer in (15, 19):
    name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    d = load(name) - load_original(name)
    u = d[:, int(d.square().sum(0).argmax())].clone()
    directions[layer] = u / u.norm()
torch.save(directions, "/work/repetition_directions.pt")

meta = json.loads(Path("/work/fwd_4/meta.json").read_text())
print("\nDIAGNOSTIC TEXT EXCERPTS")
for i, text in enumerate(meta["texts"]):
    assistant = text.split("<think>")[-1]
    print(i, "tokens", meta["lengths"][i], assistant[:240].replace("\n", " "))
print("\nSAVED ACTIVATION SHAPES")
for filename in ("hidden_15.npy", "hidden_19.npy", "mlp_act_15.npy"):
    x = np.load(Path("/work/fwd_4") / filename, mmap_mode="r")
    print(filename, x.shape, x.dtype)

with open("/work/notes.md", "a") as f:
    f.write("\nRecovered passes 6-9: gen8/9 gave 7/32 vs 4/32; gen10/11 gave 9/32 vs 4/32. "
            "All paired tests reuse the same 32 diagnostic problems; duplicate outputs are not independent evidence. "
            "Current edits verified rank-one projections W <- W-u(u^T W), alpha=1, L15/L19; "
            "relative Frobenius changes 3.526%/3.375%. Saved candidate_L15_L19_a1 and repetition_directions.pt. "
            "Independent 64 GSM8K +64 MATH validation IDs stored in splits.json; not yet evaluated.\n")

print("\nNEXT VALIDATION REQUEST")
print(json.dumps({"type": "generate", "model": "both", "problems": validation_ids,
                  "n": 1, "max_tokens": 32768}))
```
