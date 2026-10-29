```python
# Hypothesis: a source-balanced diagnostic sample will distinguish arithmetic/reasoning errors from pathological repetition without selection by outcome.
rng = np.random.default_rng(81273)
gsm_diag = rng.choice(len(practice["gsm8k_train.jsonl"]), 96, replace=False)
math_diag = rng.choice(len(practice["math_train.jsonl"]), 160, replace=False)
diag_ids = [practice["gsm8k_train.jsonl"][i]["id"] for i in gsm_diag]
diag_ids += [practice["math_train.jsonl"][i]["id"] for i in math_diag]
by_id = {r["id"]: r for rows in practice.values() for r in rows}
print(json.dumps(diag_ids))
print("MATH diagnostic levels:", collections.Counter(str(by_id[i]["level"]) for i in diag_ids if by_id[i]["source"] != "gsm8k"))
```
