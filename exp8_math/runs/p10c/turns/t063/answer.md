Strength 1.5 suppresses the measured repetition direction further, but its accuracy advantage over strength 1 is inconclusive: only two additional correct answers on the shared 128 problems. Tail repetition also does not improve monotonically, so I will not claim that the stronger dose is proven superior. I will freeze strength 1.5 because it has the larger positive paired development test, and make no further adjustments during validation. I will now draw two disjoint, balanced holdouts and retain the edit only if their pooled accuracy gain exceeds twice its paired standard error, while checking for source-specific harm.

```python
# Hypothesis: the fixed strength-1.5 repetition repair generalizes to unseen mathematics problems rather than exploiting development examples.
import os,json,glob,hashlib,numpy as np
seen=set()
for path in glob.glob("/work/gen_*.jsonl"):
    for line in open(path):
        r=json.loads(line)
        if "id" in r: seen.add(r["id"])
rng=np.random.default_rng(2026102401)
held_by_source={}
for src,filename in [("gsm8k","gsm8k_train.jsonl"),("math","math_train.jsonl")]:
    pool=[json.loads(line) for line in open("/practice/"+filename)]
    eligible=[r for r in pool if r["id"] not in seen]
    selected=rng.choice(len(eligible),256,replace=False)
    held_by_source[src]=[eligible[int(i)]["id"] for i in selected]
holdout_A=held_by_source["gsm8k"][:128]+held_by_source["math"][:128]
holdout_B=held_by_source["gsm8k"][128:]+held_by_source["math"][128:]
assert len(set(holdout_A+holdout_B))==512
assert not (set(holdout_A+holdout_B)&seen)
validation_plan=dict(strength=1.5,layers=list(range(8,23)),
    intervention="W - 1.5*u*(u^T W), both attention and MLP output projections",
    holdout_A=holdout_A,holdout_B=holdout_B,
    retention_rule="Pooled paired accuracy gain > 2 SE; report source-specific results and check significant harm.",
    no_further_tuning=True)
with open("/work/validation_plan.json","w") as f: json.dump(validation_plan,f,indent=2)
print("Frozen candidate tensors:",len(edited()),"unused holdouts:",len(holdout_A),len(holdout_B))
print("Holdout A:",json.dumps(holdout_A,separators=(",",":")))
with open("/work/notes.md","a") as f:
    f.write("\nFROZEN candidate: strength1.5 rank-one repetition repair, layers8..22, 30 writer tensors. Strength1 versus1.5 accuracy difference inconclusive (+2/128, SE4.148pp); state suppression is dose-responsive. No further tuning. Pre-registered two disjoint unseen balanced holdouts, 256 each (128 GSM+128 MATH), /work/validation_plan.json. Retain only if pooled gain >2 paired SE and check source-specific harm. Two passes remain.\n")
```
