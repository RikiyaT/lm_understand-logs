The last batch of the second final split gained 57.8 percentage points, with 37 rescues and no regressions. I will summarize that split separately before moving to the final untouched 256 problems. Agreement across complete splits is more persuasive than any single high-scoring batch, and no further weight selection will use these results.

```python
# Hypothesis: Independent full splits show a consistent benefit from the same frozen edit.
register_frozen_run(65)
final2_runs = [x["run"] for x in frozen_eval_completed if x["label"].startswith("final_2_")]
final2_summary = paired_run_summary(final2_runs)
print("FINAL_2 ONLY", final2_summary)
with open("/work/notes.md", "a") as f:
    f.write("\nFinal2 complete: " + json.dumps(final2_summary) +
            ". Four passes remain, all reserved for final3; weights unchanged.\n")
```
