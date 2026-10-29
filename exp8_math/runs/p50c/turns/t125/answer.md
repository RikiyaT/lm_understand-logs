The first new batch gained 50.0 percentage points, again with 32 rescues and no lost correct answers. Both sources improved, so the result is not explained solely by easier grade-school problems. I will record the paired result and prepare the next fixed batch, while retaining enough output metadata to audit the remaining failures.

```python
# Hypothesis: The accuracy gain remains stable across the locked, disjoint evaluation batches.
def register_frozen_run(run_id):
    batch_index = len(frozen_eval_completed)
    planned = frozen_eval_plan[batch_index]
    expected = set(planned["request"]["problems"])
    for model in ["original", "edited"]:
        rows = [json.loads(line) for line in Path(f"/work/gen_{run_id}_{model}.jsonl").read_text().splitlines()]
        assert len(rows) == len(expected) and {r["id"] for r in rows} == expected
        assert all(len(r["samples"]) == 1 for r in rows)
    assert run_id not in confirmed_runs
    confirmed_runs.append(run_id)
    single = paired_run_summary([run_id])
    pooled = paired_run_summary(confirmed_runs)
    frozen_eval_completed.append({"label": planned["label"], "run": run_id, "summary": single})
    Path("/work/frozen_eval_progress.json").write_text(json.dumps({
        "confirmed_runs": confirmed_runs,
        "completed": frozen_eval_completed,
        "pooled": pooled
    }, indent=2))
    with open("/work/notes.md", "a") as f:
        f.write(f"\nFrozen {planned['label']} gen{run_id}: {json.dumps(single)}\n")
        f.write(f"Pooled confirmed: {json.dumps(pooled)}. No weight changes.\n")
    print("BATCH", single)
    print("POOLED", pooled)
    if len(frozen_eval_completed) < len(frozen_eval_plan):
        print("NEXT REQUEST")
        print(json.dumps(frozen_eval_plan[len(frozen_eval_completed)]["request"]))

register_frozen_run(53)
print("Sample fields:", list(row["samples"][0].keys()))
```
