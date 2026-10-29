The model and data interfaces match the task description, and no edits are present. I’ll reserve 128 stratified MATH practice problems as a blind holdout now. For diagnosis, I’ll use a separate pool of 112 MATH problems, weighted toward easier levels, plus 16 GSM8K problems, with 16 samples per problem. Repeated samples should reveal problems with both correct and incorrect answers, allowing comparisons that control for problem difficulty. The baseline will also collect residual and MLP activations for localization.

```python
rng = np.random.default_rng(20261025)
math_rows = practice_by_source["math_train"]
gsm_rows = practice_by_source["gsm8k_train"]

holdout_ids = []
for level, count in [(1,26), (2,26), (3,26), (4,25), (5,25)]:
    pool = sorted([r["id"] for r in math_rows if r["level"] == f"Level {level}"])
    holdout_ids.extend(rng.choice(pool, count, replace=False).tolist())
holdout_set = set(holdout_ids)

diagnostic_ids = []
for level, count in [(1,32), (2,32), (3,24), (4,16), (5,8)]:
    pool = sorted([
        r["id"] for r in math_rows
        if r["level"] == f"Level {level}" and r["id"] not in holdout_set
    ])
    diagnostic_ids.extend(rng.choice(pool, count, replace=False).tolist())
diagnostic_ids.extend(rng.choice(
    sorted(r["id"] for r in gsm_rows), 16, replace=False
).tolist())
assert len(diagnostic_ids) == 128
assert not set(diagnostic_ids) & holdout_set

os.makedirs("/work", exist_ok=True)
for filename, ids in [
    ("HOLDOUT_ids.json", holdout_ids),
    ("diagnostic_ids.json", diagnostic_ids),
]:
    with open("/work/" + filename, "w") as f:
        json.dump(ids, f)

notes = """# Math weight-edit experiment

## Fixed design
- Seed: 20261025.
- HOLDOUT: 128 MATH problems, stratified by level; ids in HOLDOUT_ids.json.
  No holdout generations or scores will be inspected until the candidate is frozen.
- Diagnostic pool: 112 separate MATH problems plus 16 GSM8K problems;
  ids in diagnostic_ids.json.
- DEV will be fixed from diagnostic problems with both correct and incorrect
  baseline samples, before any weight edit.
- Planned passes: baseline generation; original teacher-forced localization;
  paired DEV at two edit strengths; frozen paired HOLDOUT.
- All generation comparisons use the specified sampling and 8192-token limit.

## Pass 1: baseline diagnosis
- Exact edit: none; original weights.
- Problems: diagnostic_ids.json, 128 problems, 16 samples each (2048 answers).
- Collect residual layers 0..18 and all available MLP down-projection inputs.
- Inspect accuracy, missing answers, token-limit failures, and failing text.
"""
with open("/work/notes.md", "w") as f:
    f.write(notes)

print("Holdout reserved:", len(holdout_ids))
print("Diagnostic problem IDs:")
print(json.dumps(diagnostic_ids))
```
