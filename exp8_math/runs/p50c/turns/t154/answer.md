Pass 49 gained 46.9 percentage points, with 30 rescues and no regressions; the repeated summaries describe the same run and must be counted only once. The kernel restart erased variables, so I will reconstruct the evaluation state from the saved files and verify the frozen weight hashes. No weights will be changed. Matching hashes and problem IDs will confirm that the last scheduled comparison can proceed consistently.

```python
# Hypothesis: Saved artifacts preserve the exact frozen intervention and unduplicated evaluation after the kernel restart.
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch

frozen_eval_plan = json.loads(Path("/work/frozen_eval_plan.json").read_text())
progress = json.loads(Path("/work/frozen_eval_progress.json").read_text())
confirmed_runs = progress["confirmed_runs"]
frozen_eval_completed = progress["completed"]
frozen_audit = json.loads(Path("/work/frozen_edit_audit.json").read_text())

plan_hash = hashlib.sha256(json.dumps(frozen_eval_plan, sort_keys=True).encode()).hexdigest()
assert plan_hash == "385ede3be4fafc80475c38786a2bb05340fb79913a95444488afd2a8b4eee60f"
expected_names = {f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in [15, 19]}
assert set(edited()) == expected_names
for layer in [15, 19]:
    name = f"model.language_model.layers.{layer}.mlp.down_proj.weight"
    w = load(name)
    digest = hashlib.sha256(w.contiguous().numpy().tobytes()).hexdigest()
    assert digest == frozen_audit[str(layer)]["sha256"]
    print("FROZEN HASH VERIFIED", layer, digest)
del w

def read_paired_records(run_ids):
    assert len(run_ids) == len(set(run_ids))
    records = []
    for run_id in run_ids:
        with open(f"/work/gen_{run_id}_original.jsonl") as f:
            original = {r["id"]: r for r in map(json.loads, f)}
        with open(f"/work/gen_{run_id}_edited.jsonl") as f:
            changed = {r["id"]: r for r in map(json.loads, f)}
        assert original.keys() == changed.keys()
        for pid, r in original.items():
            assert len(r["samples"]) == len(changed[pid]["samples"]) == 1
            a, b = r["samples"][0], changed[pid]["samples"][0]
            assert r["source"] == changed[pid]["source"]
            records.append({
                "id": pid, "run": run_id, "source": r["source"],
                "original": int(a["correct"]), "edited": int(b["correct"]),
                "delta": int(b["correct"])-int(a["correct"]),
                "original_cut": int(a["cut"]), "edited_cut": int(b["cut"]),
                "original_tokens": a["tokens"], "edited_tokens": b["tokens"]
            })
    frame = pd.DataFrame(records)
    assert frame.id.is_unique
    return frame

def summarize_paired_frame(frame):
    return {
        "items": len(frame),
        "original_accuracy": float(frame.original.mean()),
        "edited_accuracy": float(frame.edited.mean()),
        "gain": float(frame.delta.mean()),
        "se": float(frame.delta.std(ddof=1)/np.sqrt(len(frame))),
        "rescues": int((frame.delta == 1).sum()),
        "losses": int((frame.delta == -1).sum()),
        "by_source": {
            str(source): {
                "original": float(g.original.mean()),
                "edited": float(g.edited.mean()),
                "difference": float(g.delta.mean())
            } for source, g in frame.groupby("source")
        }
    }

def paired_run_summary(run_ids):
    return summarize_paired_frame(read_paired_records(run_ids))

def register_frozen_run(run_id):
    if run_id not in confirmed_runs:
        planned = frozen_eval_plan[len(frozen_eval_completed)]
        single_frame = read_paired_records([run_id])
        assert set(single_frame.id) == set(planned["request"]["problems"])
        summary = summarize_paired_frame(single_frame)
        confirmed_runs.append(run_id)
        frozen_eval_completed.append({
            "label": planned["label"], "run": run_id, "summary": summary
        })
    pooled = paired_run_summary(confirmed_runs)
    Path("/work/frozen_eval_progress.json").write_text(json.dumps({
        "confirmed_runs": confirmed_runs, "completed": frozen_eval_completed,
        "pooled": pooled
    }, indent=2))
    print("POOLED", pooled)
    with open("/work/notes.md", "a") as f:
        f.write(f"\nRegistered gen{run_id} once; pooled: {json.dumps(pooled)}. Weights unchanged.\n")
    if len(frozen_eval_completed) < len(frozen_eval_plan):
        print("NEXT REQUEST")
        print(json.dumps(frozen_eval_plan[len(frozen_eval_completed)]["request"]))

register_frozen_run(68)
assert len(frozen_eval_completed) == 15
assert len(confirmed_runs) == 17
with open("/work/notes.md", "a") as f:
    f.write("\nKernel restarted after pass49; recovered state from persisted JSON files. "
            "Repeated gen68 summaries counted once. Plan and both frozen weight hashes verified. "
            "One pass remains: final_3_batch4. No changes to submitted weights.\n")
```
