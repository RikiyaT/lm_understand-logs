The second HOLDOUT half gives +0.88 percentage points with a 0.75-point standard error, opposite in sign to the first half. Their combined estimate is therefore close to zero, which illustrates why neither subset should be used for tuning. I will keep the frozen edit and complete the scheduled final pass over all 256 HOLDOUT problems. The final assessment will pool all three passes with uncertainty clustered by problem.

```python
assert hashlib.sha256(load(fname).numpy().tobytes()).hexdigest()==frozen_hash
h24=pooled_evaluation([24])
h2324=pooled_evaluation([23,24])
print("HOLDOUT pass24:",h24)
print("HOLDOUT pooled so far:",h2324)
with (work/"notes.md").open("a") as f:
    f.write("\nPass24 HOLDOUT_B: original8.59%,edited9.47%, gain+0.88pp, reported SE0.75pp; cutoffs6.35%->9.96%. Frozen weights unchanged. No tuning.\n"+json.dumps(h24)+"\n")
    f.write("Pooled HOLDOUT23+24: "+json.dumps(h2324)+"\n")
    f.write("\n## Pass25 planned\nFrozen edit "+json.dumps(fcurrent)+"; all256 IDs in original /work/HOLDOUT.json order (HOLDOUT_A then HOLDOUT_B),both,n4,max_tokens8192. Same SHA256 "+frozen_hash+". Completes fixed3072 answers/model,12/problem. No weight changes allowed after observing HOLDOUT.\n")
print("Final-pass schedule verified:",{"problems":len(holdout),"samples_per_problem":4,
                                     "answers_per_model":len(holdout)*4,"edited":edited()})
```
