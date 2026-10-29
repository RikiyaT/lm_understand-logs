The first HOLDOUT half shows a −1.07-point accuracy difference with a 0.73-point standard error, so it does not confirm an improvement. The edit remains frozen; I will not adjust it in response to this result. I will record the outcome, verify weight integrity, and proceed to the second predefined HOLDOUT half before pooling all confirmation runs.

```python
assert hashlib.sha256(load(fname).numpy().tobytes()).hexdigest()==frozen_hash
h23=pooled_evaluation([23])
print("HOLDOUT pass23:",h23)
with (work/"notes.md").open("a") as f:
    f.write("\nPass23 HOLDOUT_A: original8.30%,edited7.23%, gain-1.07pp, reported SE0.73pp; cutoffs8.01%->10.45%. Frozen weights unchanged. No tuning.\n"+json.dumps(h23)+"\n")
    f.write("\n## Pass24 planned\nFrozen edit "+json.dumps(fcurrent)+"; /work/HOLDOUT_B.json (remaining128 original HOLDOUT IDs),both,n8,max_tokens8192. Same tensor SHA256 "+frozen_hash+". Confirmation only.\n")
print("HOLDOUT_B IDs:",json.dumps(holdout_b))
```
