The first HOLDOUT block showed a small gain of 0.78 percentage points, with a reported standard error of 1.48 points. Accuracy is much lower here than on the deliberately intermittent-success DEV set, and the edit increased cutoffs, so neither the DEV gain nor improved stopping behavior can be assumed to transfer. I will verify that the frozen files are unchanged and log the next preselected block. The conclusion will depend on all sixteen HOLDOUT passes, not this first result.

```python
def verify_frozen_candidate():
    actual = {
        str(p.relative_to(work / "edited")): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((work / "edited").rglob("*")) if p.is_file()
    }
    assert actual == frozen_hashes, "Frozen weights changed."
    assert set(edited()) == {final_head_name, final_mlp_name}
    return True

def prepare_frozen_holdout_pass(pass_number):
    verify_frozen_candidate()
    entry = next(e for e in holdout_schedule if e["pass"] == pass_number)
    completed = [
        k for k in range(35, pass_number)
        if (work / f"gen_{k}_original.jsonl").exists()
        and (work / f"gen_{k}_edited.jsonl").exists()
    ]
    summary = pooled_pairs(completed) if completed else None
    with (work / "notes.md").open("a") as f:
        if summary is not None:
            f.write("\nHOLDOUT pooled through pass" + str(completed[-1])
                    + ": " + json.dumps(summary) + ". No tuning.\n")
        f.write("Upcoming pass" + str(pass_number)
                + ": frozen HOLDOUT round" + str(entry["round"])
                + " block" + str(entry["block"])
                + ", BOTH, n32, max_tokens8192. Exact edit: "
                + json.dumps(final_spec["components"])
                + "; all other tensors original; SHA256 verified. Problems: "
                + json.dumps(entry["problems"]) + ".\n")
    if summary is not None:
        print("HOLDOUT_POOLED", summary)
    print(f"PASS{pass_number}_IDS", entry["problems"])

prepare_frozen_holdout_pass(36)
print("RECOVERED_DIAGNOSTIC_RECORD")
print((work / "notes.md").read_text()[6500:12300])
"Pass36 logged; frozen tensor checksums match."
```