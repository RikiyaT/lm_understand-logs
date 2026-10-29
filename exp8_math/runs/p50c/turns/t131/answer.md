The completed batch gained 43.8 percentage points, with 28 rescues and no regressions. The benefit therefore persists even in a batch where the original model performed somewhat better. I will summarize the entire reserved split and check MATH difficulty and subject breakdowns descriptively, without changing the weights; concentrated losses in a subgroup would qualify the generalization claim.

```python
# Hypothesis: The repetition repair helps across MATH subjects and difficulty levels, not only one easy subgroup.
import pandas as pd

register_frozen_run(56)
print("VALIDATION_4 ONLY", paired_run_summary([53, 54, 55, 56]))

practice_meta = {}
for filename in ["gsm8k_train.jsonl", "math_train.jsonl"]:
    with open("/practice/" + filename) as f:
        for line in f:
            p = json.loads(line)
            practice_meta[p["id"]] = {
                "level": str(p.get("level", "NA")),
                "subject": p.get("subject", p.get("type", "NA"))
            }

def paired_table(run_ids):
    records = []
    for run_id in run_ids:
        with open(f"/work/gen_{run_id}_original.jsonl") as f:
            original = {r["id"]: r for r in map(json.loads, f)}
        with open(f"/work/gen_{run_id}_edited.jsonl") as f:
            changed = {r["id"]: r for r in map(json.loads, f)}
        assert original.keys() == changed.keys()
        for pid, r in original.items():
            a, b = r["samples"][0], changed[pid]["samples"][0]
            records.append({
                "id": pid, "run": run_id, "source": r["source"],
                "original": int(a["correct"]), "edited": int(b["correct"]),
                "delta": int(b["correct"])-int(a["correct"]),
                "original_cut": int(a["cut"]), "edited_cut": int(b["cut"]),
                "original_tokens": a["tokens"], "edited_tokens": b["tokens"],
                **practice_meta[pid]
            })
    result = pd.DataFrame(records)
    assert result["id"].is_unique
    return result

def grouped_paired(frame, field):
    out = []
    for key, g in frame.groupby(field, dropna=False):
        out.append({
            field: str(key), "n": len(g),
            "original": round(g.original.mean(), 4),
            "edited": round(g.edited.mean(), 4),
            "gain": round(g.delta.mean(), 4),
            "se": round(g.delta.std(ddof=1)/np.sqrt(len(g)), 4),
            "rescues": int((g.delta == 1).sum()),
            "losses": int((g.delta == -1).sum())
        })
    return out

confirmation_table = paired_table(confirmed_runs)
math_confirmed = confirmation_table[confirmation_table.source == "math"]
subgroups = {
    "math_level": grouped_paired(math_confirmed, "level"),
    "math_subject": grouped_paired(math_confirmed, "subject")
}
print("DESCRIPTIVE SUBGROUPS", json.dumps(subgroups))
Path("/work/frozen_subgroups_interim.json").write_text(json.dumps(subgroups, indent=2))
with open("/work/notes.md", "a") as f:
    f.write("\nValidation_4 complete; descriptive subgroup audit: " + json.dumps(subgroups) + "\n")
```
