```python
import json
from pathlib import Path

work = Path("/work")
restart_dev_ids = json.loads((work / "DEV_ids.json").read_text())
print("DEV_IDS", restart_dev_ids)
print("CURRENT_EDITS", edited())
print("EARLY_DIAGNOSTIC_NOTES")
print((work / "notes.md").read_text()[:6800])
print("AVAILABLE_HELPERS", [
    name for name, value in list(globals().items())
    if callable(value)
    and getattr(value, "__module__", None) == "__main__"
    and not name.startswith("_")
][:60])
print("RECENT_RESULT_FILES", sorted(
    p.name for p in work.glob("gen_3*.jsonl")
))
"Ready to submit the unchanged pass-34 DEV replication."
```