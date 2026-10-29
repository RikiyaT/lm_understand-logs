The unedited model answered 17.6% correctly, but many failures were not ordinary wrong numerical answers: 726 finished without an extracted answer, and 134 failures hit the token limit. That makes failure to carry reasoning through to a definite conclusion a plausible mechanism to investigate. I’ll inspect short excerpts, freeze DEV from intermittently solved problems, and compare activations within those problems. If the apparent signal disappears after accounting for response length, it would be weak evidence for a repair target.

```python
with open(wdir / "gen_1.jsonl") as f:
    gen1 = [json.loads(line) for line in f]
print("Generation fields:", list(gen1[0]))
print("First record metadata:", {k: v for k, v in gen1[0].items()
                                  if not isinstance(v, (list, dict)) and
                                  (not isinstance(v, str) or len(v) < 120)})
print("Activation files:", sorted(p.name for p in (wdir / "gen_1_acts").iterdir()))
for i in [0, 1, 8, 17, 37, 72, 119, 203]:
    rec = gen1[i]
    textfields = {k: v for k, v in rec.items() if isinstance(v, str) and len(v) >= 120}
    print("Record", i, {k: v for k, v in rec.items()
                       if k in ["id", "problem_id", "correct", "is_correct", "grade",
                                "finish_reason", "num_tokens", "generated_tokens", "extracted"]})
    for k, v in textfields.items():
        if k not in ("problem", "prompt", "solution", "reference_solution"):
            print(k, "start:", v[:140].encode("ascii", "backslashreplace").decode(),
                  "end:", v[-140:].encode("ascii", "backslashreplace").decode())
```
