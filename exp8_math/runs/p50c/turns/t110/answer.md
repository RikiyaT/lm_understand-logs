The full-budget test is encouraging: the 64-neuron strength-6 edit improved accuracy from 7/64 to 35/64, a gain of 43.75 percentage points with a 6.63-point standard error. Cutoffs fell from 57 to five, but 24 outputs finished incorrectly or without an answer, so completion is no longer the only limitation. I’ll inspect those remaining failures and then confirm this unchanged candidate on the reserved fresh set. I won’t increase the strength further before learning whether this gain generalizes.

```python
# Hypothesis: top64 alpha6 fixes nontermination on unseen problems without truncating useful reasoning.
stats45 = {m: generation_file_stats(f"/work/gen_45_{m}.jsonl") for m in ("original", "edited")}
print("TOP64 alpha6,32k", stats45)
rows45 = [json.loads(line) for line in open("/work/gen_45_edited.jsonl")]
for label, pred in [("wrong_finished", lambda s: not s["correct"] and not s["cut"]),
                     ("cut", lambda s: s["cut"])]:
    cases = [(row["id"], s) for row in rows45 for s in row["samples"] if pred(s)]
    for pid, s in cases[:2]:
        print(label, pid, "tokens", s["tokens"], "prediction", str(s["pred"])[:50],
              "tail", repr(s["text"].rstrip()[-220:]))
fresh_request = {"type": "generate", "model": "both",
                 "problems": fresh_splits["validation_3"], "n": 1, "max_tokens": 32768}
print("\nNEXT INDEPENDENT REQUEST")
print(json.dumps(fresh_request, separators=(",", ":")))
with open("/work/notes.md", "a") as f:
    f.write("\nPass30/gen45 TOP64alpha6 exploratory64x1 FULL32768:7/64->35/64,+43.75pp SE6.63,"
            "29rescues1loss;cut89.06->7.81%;mean29736->15377;wrongfinished20,noanswer4. "
            + json.dumps(stats45) +
            "\nCURRENT unchanged top64alpha6; next validation_3(128 fresh problems)x1 FULL32768. "
            "No further dose increase before independent confirmation.\n")
Path("/work/current_plan.json").write_text(json.dumps(fresh_request))
```
