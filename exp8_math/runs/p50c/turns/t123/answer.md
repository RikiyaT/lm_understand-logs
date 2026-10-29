The pooled result is 9/128 correct before editing versus 74/128 afterward, a gain of 50.8 percentage points with a 4.4-point standard error. The audit confirms that only 64 MLP output columns in each of layers 15 and 19 changed; the edit is directional, not whole-tensor scaling. I will now lock a balanced, disjoint evaluation schedule covering the remaining reserved problems. This separates confirmation from selection and will reveal whether either problem source behaves differently.

```python
# Hypothesis: The frozen edit improves both grade-school and competition math on independently reserved problems.
import hashlib
from collections import Counter

frozen_eval_plan = []
for split_name in ["validation_4", "final_1", "final_2", "final_3"]:
    ids = fresh[split_name]
    gsm = [x for x in ids if x.startswith("gsm8k")]
    math = [x for x in ids if x.startswith("math")]
    assert len(gsm) == len(math) == 128
    for b in range(4):
        batch = gsm[32*b:32*(b+1)] + math[32*b:32*(b+1)]
        frozen_eval_plan.append({
            "label": f"{split_name}_batch{b+1}",
            "request": {
                "type": "generate", "model": "both",
                "problems": batch, "n": 1, "max_tokens": 32768
            }
        })

all_reserved = [x for b in frozen_eval_plan for x in b["request"]["problems"]]
assert len(all_reserved) == len(set(all_reserved)) == 1024
assert not set(all_reserved).intersection(fresh["validation_3"])
Path("/work/frozen_eval_plan.json").write_text(json.dumps(frozen_eval_plan, indent=2))
frozen_eval_completed = []
plan_hash = hashlib.sha256(json.dumps(frozen_eval_plan, sort_keys=True).encode()).hexdigest()
print("Locked plan:", len(frozen_eval_plan), "batches;", len(all_reserved), "fresh problems")
print("Plan SHA256:", plan_hash)

row = json.loads(Path("/work/gen_52_edited.jsonl").read_text().splitlines()[0])
print("Generation record keys:", list(row.keys()))

notes = Path("/work/notes.md").read_text()
notes = notes.replace(
    "# Math weight editing ? through pass26/50, Python54",
    "# Math weight editing — frozen sparse edit; through pass34/50"
)
notes = notes.replace(
    "BEST INDEPENDENTLY CONFIRMED: full-direction alpha2, L15/L19 MLP down_proj; backup best_confirmed_a2.",
    "BEST INDEPENDENTLY CONFIRMED: top64 alpha6, L15/L19 MLP down_proj; backup best_confirmed_top64_a6."
)
notes = notes.replace(
    "CURRENT: top64-neuron-only alpha4 in same two tensors, candidate_top64_a4. NOT accuracy-tested.",
    "CURRENT: FROZEN top64-neuron-only alpha6; validation_3 9/128->74/128, gain50.78pp SE4.44."
)
notes += (
    "\nLocked 16 balanced64-problem batches from validation_4,final_1,final_2,final_3 "
    "(1024 unique, disjoint validation_3), file frozen_eval_plan.json; "
    f"SHA256={plan_hash}. No weight tuning on these results.\n"
)
Path("/work/notes.md").write_text(notes)

print("\nNEXT FORWARD REQUEST")
print(json.dumps(frozen_eval_plan[0]["request"]))
```
