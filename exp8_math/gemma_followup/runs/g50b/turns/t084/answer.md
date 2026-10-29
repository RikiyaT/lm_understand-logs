The last DEV replication gained 0.98 percentage points, with a reported standard error of 2.14 points, so it is weak evidence on its own. It also increased answer length and cutoffs, ruling out a simple “shorter answers” explanation for this result. I will freeze the unchanged candidate now and lock the remaining sixteen passes to the reserved HOLDOUT problems. A positive pooled HOLDOUT gain would support generalization; a null or negative result would weaken that claim, without prompting further tuning.

```python
import hashlib
import shutil
import numpy as np
import torch
from pathlib import Path
import json

work = Path("/work")
final_units = [971, 180, 149, 1680, 232, 1724, 312, 1096]
final_head_name = "model.layers.11.self_attn.o_proj.weight"
final_mlp_name = "model.layers.11.mlp.down_proj.weight"

assert set(edited()) == {final_head_name, final_mlp_name}
expected_head = load_original(final_head_name).clone()
expected_head[:, 512:768] *= 1.1
expected_mlp = load_original(final_mlp_name).clone()
expected_mlp[:, final_units] *= 0.5
assert torch.allclose(load(final_head_name), expected_head, rtol=1e-6, atol=1e-8)
assert torch.equal(load(final_mlp_name), expected_mlp)

final_spec = {
    "layer_numbering": "zero-based",
    "components": [
        {"tensor": final_head_name, "columns_half_open": [512, 768],
         "query_head": 2, "multiplier": 1.1},
        {"tensor": final_mlp_name, "columns": final_units, "multiplier": 0.5}
    ],
    "other_tensors": "original",
    "frozen_before_pass": 35,
    "generation_edit_hash": "399af7b267"
}
frozen_dir = work / "frozen_final"
assert not frozen_dir.exists(), "Do not overwrite an existing frozen candidate."
shutil.copytree(work / "edited", frozen_dir)
frozen_hashes = {
    str(p.relative_to(work / "edited")): hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted((work / "edited").rglob("*")) if p.is_file()
}
final_spec["file_sha256"] = frozen_hashes
(work / "frozen_manifest.json").write_text(json.dumps(final_spec, indent=2))

holdout_ids = json.loads((work / "HOLDOUT_ids.json").read_text())
assert len(holdout_ids) == len(set(holdout_ids)) == 128
assert not set(holdout_ids) & set(restart_dev_ids)
holdout_blocks = [holdout_ids[i:i+32] for i in range(0, 128, 32)]
holdout_schedule = []
for round_index in range(4):
    shift = 8 * round_index
    for block_index, block in enumerate(holdout_blocks):
        holdout_schedule.append({
            "pass": 35 + 4*round_index + block_index,
            "round": round_index + 1,
            "block": block_index + 1,
            "problems": block[shift:] + block[:shift],
            "n": 32,
            "max_tokens": 8192
        })
(work / "HOLDOUT_schedule.json").write_text(json.dumps(holdout_schedule, indent=2))

print("FINAL_DEV", pooled_pairs([29, 30, 33, 34]))
print("FROZEN_SPEC", final_spec)
with (work / "notes.md").open("a") as f:
    f.write("\nCANDIDATE FROZEN before any HOLDOUT outcome. "
            "No subsequent component, strength, or candidate selection changes. "
            "Exact specification: " + json.dumps(final_spec) + ".\n")
    f.write("HOLDOUT schedule locked in HOLDOUT_schedule.json: four original-order "
            "blocks of32, four rounds with within-block rotations0,8,16,24; "
            "passes35-50, BOTH, n32, 8192 tokens. Pool all passes, reporting "
            "sample and problem-cluster uncertainty.\n")
    f.write("Upcoming pass35: HOLDOUT round1 block1. Exact edit: "
            + json.dumps(final_spec["components"]) + "; all other weights original. "
            "Problems: " + json.dumps(holdout_schedule[0]["problems"]) + ".\n")
print("PASS35_IDS", holdout_schedule[0]["problems"])
"Candidate frozen; HOLDOUT schedule fixed before observing any HOLDOUT outcomes."
```