The generation request failed during engine initialization, so it provides no accuracy evidence and consumes no pass. I will preserve the untouched validation set and run the planned causal controls while generation is unavailable. The next control will ablate a different head in the same layer: if it suppresses loops equally, the claimed specificity of head 7 would weaken.

```python
# Hypothesis: repetition suppression is specific to L15H7 rather than generic loss of attention output.
import json
from pathlib import Path

with (work / "notes.md").open("a") as f:
    f.write("\nH2A retry after restart failed at generation-engine initialization; not counted, no outcomes observed. Still 19/25 passes. Frozen candidate unchanged. Move planned causal controls ahead of fresh generation retries.\n")

loc = json.loads((work / "localization_v2.json").read_text())
print("Localization evidence:", json.dumps(loc)[:5800])
broad_request = json.loads((work / "broad_probe_request.json").read_text())
print("Broad request keys:", list(broad_request))
print("Text lengths:", [len(s) for s in broad_request.get("texts", [])])
print("Return specification:", broad_request.get("return"))
print("Region schema:", (work / "broad_probe_regions.json").read_text()[:1200])
print("Baseline metadata:", (work / "fwd_25" / "meta.json").read_text()[:1700])
print("Analysis helpers:", [k for k, v in list(globals().items())
                            if callable(v) and any(t in k.lower() for t in ("probe", "paired", "region", "summary"))])
```
