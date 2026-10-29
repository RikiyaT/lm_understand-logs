The tool-discovery call was unavailable, so I have not inspected weights or run any experiments. I’ll try the Python interface you specified with a read-only inventory of the model, practice data, and prompt format. This will establish whether the experiment environment is accessible before I propose a diagnosis or claim any improvement.

```python
import json
from pathlib import Path
from collections import Counter

inventory = {
    "kernel_helpers": {
        name: name in globals()
        for name in ("NAMES", "load", "load_original", "save_edit",
                     "drop_edit", "edited", "tok")
    },
    "paths": {
        p: Path(p).exists() for p in ("/model", "/practice", "/work")
    },
}

practice = {}
for filename in ("gsm8k_train.jsonl", "math_train.jsonl"):
    path = Path("/practice") / filename
    if path.exists():
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        practice[filename] = rows
        inventory[filename] = {
            "count": len(rows),
            "fields": sorted(rows[0]) if rows else [],
            "example_ids": [r["id"] for r in rows[:5]],
            "levels": dict(Counter(str(r.get("level")) for r in rows)),
        }

if "NAMES" in globals():
    inventory["tensor_count"] = len(NAMES)
    inventory["layer_0_tensor_names"] = sorted(
        name for name in NAMES if ".layers.0." in name
    )

prompt_path = Path("/work/eval_prompt_format.txt")
if prompt_path.exists():
    inventory["eval_prompt_format"] = prompt_path.read_text()

if "edited" in globals():
    inventory["existing_edits"] = edited()

print(json.dumps(inventory, ensure_ascii=True, indent=2, default=str))
"Read-only inventory complete; no edits or forward passes requested."
```