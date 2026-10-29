The restart notes show that the layer-15/19 edit at strength 2 is the best confirmed candidate; strength 4 has no generation evidence because those requests failed to start. I’ll review the earlier measurements and probe definitions before continuing, so I don’t mistake the currently loaded weights for the best supported edit. The next tests should distinguish suppression of repetitive reasoning from a general tendency to end thinking early. Stronger edits should be retained only if they improve accuracy on fresh, paired examples.

```python
# Hypothesis: the confirmed alpha-2 result is a safer baseline than the unmeasured alpha-4 candidate.
from pathlib import Path
import json
print("NOTES START")
print(Path("/work/notes.md").read_text()[:5800])
print("\nHISTORY")
p = Path("/work/experiment_history.md")
print(p.read_text()[:2800] if p.exists() else "No separate history.")
print("\nCURRENT REQUEST")
r = json.loads(Path("/work/transfer_probe_request.json").read_text())
print({k: (f"{len(v)} entries" if k in ("texts", "chats") else v) for k, v in r.items()})
print("\nBOUNDARIES")
print(Path("/work/transfer_probe_bounds.json").read_text()[:700])
print("\nEDITED", edited())
```
