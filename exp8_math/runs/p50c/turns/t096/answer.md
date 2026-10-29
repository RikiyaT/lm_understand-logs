The layer-15-only probe is complete; I’ll now calculate its selective effect and switch to layer 19 alone. In parallel, I’ll quantify whitespace-ending cutoffs in the paired generation files rather than infer their prevalence from two examples. This matters because an edit could reduce verbal repetition while introducing another form of nontermination. Evidence that both failure types fall would strengthen the repair; a new failure pattern would argue for backing off or narrowing it.

```python
# Hypothesis: both MLP sites contribute, and whitespace cutoffs must be measured as a possible side effect.
l15_metrics = transfer_summary("/work/fwd_37")
print("L15 ONLY alpha4", json.dumps(l15_metrics))
for run, label in [(27, "alpha2 heldout"), (29, "alpha3 exploratory"), (35, "random control")]:
    for model in ("original", "edited"):
        rows = [json.loads(line) for line in open(f"/work/gen_{run}_{model}.jsonl")]
        samples = [s for row in rows for s in row["samples"]]
        cuts = [s for s in samples if s["cut"]]
        spaces = [s for s in cuts if len(s["text"]) >= 500 and not s["text"][-500:].strip()]
        print(label, model, "n", len(samples), "cut", len(cuts), "whitespace_cut", len(spaces),
              "whitespace_after_think", sum("</think>" in s["text"] for s in spaces))
        if run == 29 and model == "edited" and spaces:
            s = spaces[0]
            stripped = s["text"].rstrip()
            print("before whitespace", repr(stripped[-240:]), "trailing_whitespace_chars", len(s["text"])-len(stripped))
set_repetition_edit({15: 0, 19: 4})
print("CURRENT", edited())
with open("/work/notes.md", "a") as f:
    f.write("\nPass24/fwd37 L15-only alpha4 transfer: " + json.dumps(l15_metrics) +
            "\nCURRENT L19-only alpha4; next identical probe.\n")
```
