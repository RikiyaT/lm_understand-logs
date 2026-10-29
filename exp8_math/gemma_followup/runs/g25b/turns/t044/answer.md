Removing the proposed repetition direction reduced DEV accuracy by 4.96 percentage points, with a reported standard error of 2.15 points, and increased cutoffs. That contradicts the hypothesis that this direction simply drives harmful repetition. It may instead help stabilize useful computation, so I will test the opposite intervention: amplifying only this direction by 50%. Recovery or improvement in accuracy would support that revised interpretation; another decline would argue against this direction as a repair target.

```python
import shutil, json, numpy as np, torch
from pathlib import Path

work = Path("/work")
name = "model.layers.11.self_attn.o_proj.weight"
shutil.copytree(work / "edited", work / "candidate_D_remove_full", dirs_exist_ok=True)
with (work / "notes.md").open("a") as f:
    f.write("\nPass14 rank-one removal1.0: DEV accuracy36.31%->31.35%, gain-4.96pp, reported SE2.15pp; cutoffs3.77%->5.95%. Harmful-repetition interpretation contradicted.\n")

delta = torch.from_numpy(np.load(work / "l11h2_repetition_rank1_delta.npy"))
for tensor_name in list(edited()):
    drop_edit(tensor_name)
w = load_original(name).clone()
w[:, 512:768] += 0.5 * delta
save_edit(name, w)
current = {name: {"head": 2, "columns": [512,768],
                 "operation": "subtract strength * fixed repetition rank-one delta",
                 "delta_file": "/work/l11h2_repetition_rank1_delta.npy", "strength": -0.5}}
(work / "current_edit.json").write_text(json.dumps(current))
shutil.copytree(work / "edited", work / "candidate_D_amplify_half", dirs_exist_ok=True)
with (work / "notes.md").open("a") as f:
    f.write("\n## Pass15 planned\nReverse causal test: only L11H2 rank-one direction amplified50%, removal strength=-0.5; all other weights original. DEV63,both,n16,max_tokens8192. Accuracy primary.\n" + json.dumps(current) + "\n")

row = json.loads((work / "gen_14_edited.jsonl").read_text().splitlines()[0])
print("Generation record keys:", list(row))
print("Available analysis functions:",
      [k for k,v in list(globals().items()) if callable(v) and getattr(v, "__module__", None) == "__main__"])
print("Pass15 edit saved:", current)
```
