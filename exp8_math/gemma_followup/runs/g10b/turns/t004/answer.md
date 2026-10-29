The checkpoint has the expected layer tensors, and the practice files include problems, answers, and worked solutions. I’ll use MATH problems for the initial diagnosis, emphasizing levels 1–3 so this small model has a reasonable chance of both succeeding and failing on the same problem. I’ll also reserve a separate, stratified HOLDOUT without generating or inspecting its answers. The baseline pass will collect MLP and residual activations; these can suggest components to test, but correlations alone will not justify an edit.

```python
rng_exp = random.Random(270031)
math_rows = datasets["math_train"]
problem_by_id = {r["id"]: r for ds in datasets.values() for r in ds}

def stratified_pick(rows_in, count, rng):
    groups = collections.defaultdict(list)
    for row in rows_in:
        groups[(row.get("subject", ""), row.get("level", ""))].append(row)
    allocation = {k: count * len(v) / len(rows_in) for k, v in groups.items()}
    quotas = {k: int(v) for k, v in allocation.items()}
    remainder = count - sum(quotas.values())
    for k in sorted(groups, key=lambda k: allocation[k] - quotas[k], reverse=True)[:remainder]:
        quotas[k] += 1
    result = []
    for k in sorted(groups):
        result.extend(rng.sample(groups[k], quotas[k]))
    rng.shuffle(result)
    return result

holdout_rows = stratified_pick(math_rows, 256, rng_exp)
holdout_ids = [r["id"] for r in holdout_rows]
holdout_set = set(holdout_ids)
baseline_rows = []
for level, count in [("Level 1", 96), ("Level 2", 96), ("Level 3", 64)]:
    eligible = [r for r in math_rows if r["level"] == level and r["id"] not in holdout_set]
    baseline_rows.extend(stratified_pick(eligible, count, rng_exp))
baseline_ids = [r["id"] for r in baseline_rows]
assert len(set(baseline_ids)) == 256 and not set(baseline_ids) & holdout_set
(wdir / "HOLDOUT.json").write_text(json.dumps(holdout_ids, indent=2))
(wdir / "BASELINE_POOL.json").write_text(json.dumps(baseline_ids, indent=2))
(wdir / "experiment_plan.json").write_text(json.dumps({
    "seed": 270031, "holdout": "256 MATH problems, stratified by subject and level",
    "dev_rule": "Select up to 128 intermittently solved baseline problems; freeze before editing.",
    "baseline_samples_per_problem": 8,
    "selection_metric": "Paired DEV answer accuracy, not response length.",
    "holdout_rule": "Freeze candidate before any HOLDOUT generation; pool confirmation passes."
}, indent=2))
(wdir / "notes.md").write_text(
    "# Weight-edit experiment\n\n"
    "No edits have been made. Embedding/output and final norm are excluded.\n"
    "HOLDOUT is fixed in HOLDOUT.json and disjoint from BASELINE_POOL.json.\n"
    "DEV will be fixed from baseline problems with 1–7 successes out of 8.\n\n"
    "## Pass 1 planned: baseline diagnosis\n"
    "Model: original. Exact edit: none; strength 0.\n"
    "Set: BASELINE_POOL.json, 256 MATH problems (96 level 1, 96 level 2, 64 level 3).\n"
    "Generation: n=8, max_tokens=8192, evaluation sampling/template.\n"
    "Activations: residual states 0–18 and all MLP down-projection inputs.\n"
)
print("Baseline strata:", collections.Counter(r["level"] for r in baseline_rows))
print("Pass-1 request:")
print(json.dumps({
    "type": "generate", "model": "original", "problems": baseline_ids,
    "n": 8, "max_tokens": 8192,
    "activations": {"layers": list(range(19)), "mlp": True}
}))
```
