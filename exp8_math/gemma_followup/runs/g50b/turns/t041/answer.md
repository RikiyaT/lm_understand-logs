```python
import os, json, pathlib, shutil, numpy as np, torch
W = pathlib.Path("/work")
print("CURRENT EDITS:", edited())
print("NOTES START:\n", (W/"notes.md").read_text()[:6500])
for filename in ["DEV_ids.json", "HOLDOUT_ids.json", "layer8_units.json",
                 "layer11_success_units.json"]:
    obj = json.loads((W/filename).read_text())
    print(filename, str(obj)[:1800] if filename != "HOLDOUT_ids.json"
          else {"count": len(obj), "contents": "not examined"})
for k in [2, 3, 11]:
    p = W/f"fwd_{k}"/"meta.json"
    print("META", k, p.read_text()[:2400])
r = json.loads((W/"gen_17_original.jsonl").read_text().splitlines()[0])
print("GEN SCHEMA:", {k: type(v).__name__ for k,v in r.items()})
print("GEN NON-TEXT:", {k:v for k,v in r.items()
                        if not isinstance(v, (dict,list)) and len(str(v)) < 120})
dest = W/"candidate_combo_17"
if not dest.exists():
    shutil.copytree(W/"edited", dest)
with (W/"notes.md").open("a") as f:
    f.write("\nPass 17 result: DEV +2.93 pp, SE 1.84 pp; "
            "original 495/1024, edited 525/1024. "
            "Mean tokens 820->979, cutoff share .0361->.0566. "
            "Candidate archived at candidate_combo_17. HOLDOUT unused.\n")
print("Saved combined candidate; inspection complete.")
```