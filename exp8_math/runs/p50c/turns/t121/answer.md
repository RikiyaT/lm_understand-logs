```python
# Hypothesis: The frozen, repetition-targeted edit generalizes beyond the problems used to select it.
import json, inspect
from pathlib import Path

print("EARLY DIAGNOSIS AND EDIT DEFINITION")
print(Path("/work/notes.md").read_text()[:6500])

print("\nPAIRED SUMMARY HELPER")
print(inspect.signature(paired_run_summary))
print(paired_run_summary([50, 52]))

fresh = json.loads(Path("/work/fresh_splits.json").read_text())
print("\nRESERVED SPLITS")
for name, ids in fresh.items():
    print(name, type(ids).__name__, len(ids), str(ids[:2])[:200] if isinstance(ids, list) else "")

print("\nFROZEN AUDIT")
print(Path("/work/frozen_edit_audit.json").read_text()[:1600])

confirmed_runs = [50, 52]
with open("/work/notes.md", "a") as f:
    f.write(
        "\nPass34/gen52 frozen top64alpha6 validation_3 second64 FULL32768: "
        "2/64->34/64, +50.00pp SE6.30,32rescues0losses; "
        "GSM6.25->62.50%,MATH0->43.75%;cut96.88->25.00%. "
        "Combined validation_3:9/128->74/128,+50.78pp,65rescues0losses. "
        "Weights remain FROZEN; confirmed_runs=[50,52];16 passes remain. "
        "Plan: fresh paired full32768 generations, no further tuning.\n"
    )
```
