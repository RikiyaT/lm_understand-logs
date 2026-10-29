The shorter-budget run succeeded, and the restricted edit improved paired accuracy from 13/128 to 47/128: +26.56 percentage points with a 4.45-point standard error and no correct-to-wrong reversals. That is promising, although it is exploratory and capped at 16,384 tokens. I’ll record its whitespace failure rate and test the full strength-4 edit at the same cap. This comparison will show whether the extra changed columns deliver useful accuracy or mainly add side effects.

```python
# Hypothesis: the full-direction edit may outperform the restricted edit, but could increase whitespace failures.
def generation_file_stats(path):
    rows = [json.loads(line) for line in open(path)]
    samples = [s for row in rows for s in row["samples"]]
    correct = [s for s in samples if s["correct"]]
    return {
        "n": len(samples),
        "correct": len(correct),
        "cut": sum(s["cut"] for s in samples),
        "whitespace_cut": sum(s["cut"] and len(s["text"]) >= 500
                              and not s["text"][-500:].strip() for s in samples),
        "median_correct_tokens": float(np.median([s["tokens"] for s in correct])) if correct else None,
        "correct_with_think_close": sum("</think>" in s["text"] for s in correct)
    }
stats41 = {m: generation_file_stats(f"/work/gen_41_{m}.jsonl") for m in ("original", "edited")}
print("TOP64 alpha4, 16k", stats41)
set_repetition_edit({15: 4, 19: 4})
print("CURRENT full alpha4", edited())
with open("/work/notes.md", "a") as f:
    f.write("\nPass27/gen41 TOP64alpha4 exploratory64x2 at16384 tokens:13/128->47/128,"
            "+26.56pp SE4.45,34rescues0losses; cut89.84->47.66%; "
            + json.dumps(stats41) + "\nCURRENT full-direction alpha4, next same16k exploratory test. "
            "16k initialization succeeded after32k failed, not yet established as reliable workaround.\n")
```
