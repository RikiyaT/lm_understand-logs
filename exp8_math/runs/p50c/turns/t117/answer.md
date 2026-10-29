The first independent batch confirms a large gain: accuracy rose from 7/64 to 40/64, or +51.56 percentage points with a 6.30-point standard error and no regressions. Both sources improved, including GSM8K from 15.62% to 81.25% and MATH from 6.25% to 43.75%. I’ll back up this confirmed candidate and keep its weights fixed while completing the remaining validation batches. Further tests will assess generalization and failure rates, not tune the edit on those results.

```python
# Hypothesis: the fixed 64-writer alpha6 repair generalizes across independently sampled mathematics problems.
shutil.copytree("/work/edited", "/work/best_confirmed_top64_a6", dirs_exist_ok=True)
stats50 = {m: generation_file_stats(f"/work/gen_50_{m}.jsonl") for m in ("original", "edited")}
print("INDEPENDENT FIRST HALF", stats50)
def paired_run_summary(run_ids):
    items = []
    seen = set()
    rescues = losses = 0
    for run in run_ids:
        original_rows = {row["id"]: row for row in
                         (json.loads(line) for line in open(f"/work/gen_{run}_original.jsonl"))}
        edited_rows = [json.loads(line) for line in open(f"/work/gen_{run}_edited.jsonl")]
        for row in edited_rows:
            pid = row["id"]
            assert pid not in seen, ("Repeated validation item", pid)
            seen.add(pid)
            a = original_rows[pid]["samples"]
            b = row["samples"]
            assert len(a) == len(b)
            ac = np.array([bool(s["correct"]) for s in a], dtype=float)
            bc = np.array([bool(s["correct"]) for s in b], dtype=float)
            rescues += int(((ac == 0) & (bc == 1)).sum())
            losses += int(((ac == 1) & (bc == 0)).sum())
            items.append({"id": pid, "source": row["source"], "original": float(ac.mean()),
                          "edited": float(bc.mean()), "difference": float((bc-ac).mean())})
    ds = np.array([x["difference"] for x in items])
    result = {"items": len(items),
              "original_accuracy": float(np.mean([x["original"] for x in items])),
              "edited_accuracy": float(np.mean([x["edited"] for x in items])),
              "gain": float(ds.mean()), "se": float(ds.std(ddof=1)/np.sqrt(len(ds))),
              "rescues": rescues, "losses": losses}
    result["by_source"] = {
        source: {k: float(np.mean([x[k] for x in items if x["source"] == source]))
                 for k in ("original", "edited", "difference")}
        for source in sorted({x["source"] for x in items})}
    return result
confirmed_runs = [50]
print("PAIRED", paired_run_summary(confirmed_runs))
second_half = fresh_splits["validation_3"][32:64] + fresh_splits["validation_3"][96:128]
print("NEXT SECOND HALF", json.dumps(second_half, separators=(",", ":")))
with open("/work/notes.md", "a") as f:
    f.write("\nNEW BEST CONFIRMED: TOP64alpha6, backup best_confirmed_top64_a6. "
            "Pass33/gen50 validation_3 first balanced64x1 FULL32768:7/64->40/64, "
            "+51.56pp SE6.30,33rescues0losses;GSM15.62->81.25%,MATH6.25->43.75%; "
            "cut89.06->23.44%; " + json.dumps(stats50) +
            "\nCURRENT same weights FROZEN. Complete second64 validation_3, then reserved fresh sets. "
            "confirmed_runs=[50]; paired_run_summary helper asserts disjoint items across runs. "
            "17 forward passes remain; prioritize full-budget heldout paired generations, no further tuning.\n")
```
