The files and prompt format match the stated setup, and no weights have changed. I’ll reserve a random MATH holdout now, then choose a separate diagnostic pool weighted toward levels 1 and 2, where repeated sampling is more likely to reveal both successes and failures. After seeing those baseline samples, I’ll fix DEV from the partly solved problems before making any edit. This split lets me investigate failures without using holdout results to choose repairs.

```python
with open("/practice/math_train.jsonl") as f:
    math_rows = [json.loads(x) for x in f]
with open("/practice/gsm8k_train.jsonl") as f:
    gsm_rows = [json.loads(x) for x in f]
problems_by_id = {r["id"]: r for r in math_rows + gsm_rows}
rng = random.Random(290173)
holdout_ids = [r["id"] for r in rng.sample(math_rows, 256)]
held = set(holdout_ids)
diagnostic_rows = []
for level, count in [("Level 1", 80), ("Level 2", 40), ("Level 3", 8)]:
    eligible = [r for r in math_rows if r["id"] not in held and r["level"] == level]
    diagnostic_rows.extend(rng.sample(eligible, count))
rng.shuffle(diagnostic_rows)
diagnostic_ids = [r["id"] for r in diagnostic_rows]
os.makedirs("/work", exist_ok=True)
for name, ids in [("HOLDOUT", holdout_ids), ("DIAGNOSTIC", diagnostic_ids)]:
    with open(f"/work/{name}.json", "w") as f:
        json.dump(ids, f)
with open("/work/notes.md", "w") as f:
    f.write("# Math weight-edit experiment\n"
            "No edits. Final norm and embedding/output weights will never be edited.\n"
            "HOLDOUT fixed before all experiments: /work/HOLDOUT.json, 256 random MATH problems, seed 290173. Do not inspect outcomes until candidate is frozen.\n"
            "Diagnostic pool: /work/DIAGNOSTIC.json, 128 disjoint MATH problems: 80 level 1, 40 level 2, 8 level 3.\n"
            "DEV will be fixed from partly solved diagnostic problems before any weight edit.\n\n"
            "## Pass 1 planned\n"
            "Original model; no changed tensors or strengths. DIAGNOSTIC, n=16, max_tokens=8192.\n"
            "Collect residual and MLP activations at all layers to compare success/failure within the same problem.\n")
print("Config:", {k:v for k,v in json.load(open("/model/config.json")).items()
                  if k in ["model_type", "hidden_activation", "rms_norm_eps", "query_pre_attn_scalar", "sliding_window", "layer_types"]})
print("Diagnostic subjects:", dict(Counter(r["subject"] for r in diagnostic_rows)))
print("Diagnostic IDs for pass 1:")
print(json.dumps(diagnostic_ids))
```
