The kernel has the required helpers, and no edits are present. I’ll reserve HOLDOUT now, then sample the unedited model repeatedly on an easier, separate diagnostic pool to find problems it solves inconsistently. DEV will be fixed from that pool before any edit; HOLDOUT will remain unused until the candidate is frozen. Recording MLP activations will let me compare correct and failing answers within the same problem, reducing—but not eliminating—difficulty confounds.

```python
import numpy as np
import torch

rng = np.random.default_rng(20261025)
math_rows = practice["math_train.jsonl"]
by_id = {r["id"]: r for rows in practice.values() for r in rows}

holdout_idx = rng.choice(len(math_rows), size=128, replace=False)
holdout_ids = [math_rows[int(i)]["id"] for i in holdout_idx]
holdout_set = set(holdout_ids)

diagnostic_ids = []
for level, count in [("Level 1", 48), ("Level 2", 48), ("Level 3", 32)]:
    eligible = [r["id"] for r in math_rows
                if r["level"] == level and r["id"] not in holdout_set]
    diagnostic_ids.extend(rng.choice(eligible, size=count, replace=False).tolist())

work = Path("/work")
(work / "HOLDOUT_ids.json").write_text(json.dumps(holdout_ids))
(work / "diagnostic_ids.json").write_text(json.dumps(diagnostic_ids))
notes = (
    "# Experiment log\n"
    "No weight edits. HOLDOUT: 128 randomly selected MATH training problems, "
    "reserved before diagnosis; do not inspect outcomes until candidate freeze.\n"
    "Upcoming pass 1: original, diagnostic_ids.json, 16 samples/problem, "
    "8192 new tokens; residual and MLP activations for layers 1 through 18.\n"
    "DEV will be locked to at most 32 inconsistently solved diagnostic problems "
    "before any edit. Primary endpoint: paired answer accuracy.\n"
)
(work / "notes.md").write_text(notes)
print(json.dumps({"diagnostic_ids": diagnostic_ids,
                  "holdout_count": len(holdout_ids)}, ensure_ascii=True))
"Split reservation and pass-1 notes saved."
```