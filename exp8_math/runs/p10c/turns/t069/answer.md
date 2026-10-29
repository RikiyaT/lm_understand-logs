The first unseen holdout improved from 9.8% to 16.8% accuracy: a paired gain of 7.0 points with a 2.4-point standard error. Both GSM8K and MATH improved, supporting transfer beyond the development problems. I will leave the weights unchanged, record this result, and prepare the preselected second holdout. The final decision will use all 512 held-out pairs, not this favorable first half alone.

```python
# Hypothesis: the fixed edit's held-out gain will replicate on a second disjoint set; no tuning follows the first validation result.
import json,numpy as np
def flat_now(path):
    return [dict(s,id=r["id"],source=r["source"],sample=j)
            for r in [json.loads(line) for line in open(path)] for j,s in enumerate(r["samples"])]
def paired_now(a,b):
    assert [(s["id"],s["sample"]) for s in a]==[(s["id"],s["sample"]) for s in b]
    x=np.array([s["correct"] for s in a],float);y=np.array([s["correct"] for s in b],float);d=y-x
    return dict(n=len(a),original_correct=int(x.sum()),edited_correct=int(y.sum()),
                gain_pp=float(100*d.mean()),se_pp=float(100*d.std(ddof=1)/np.sqrt(len(d))),
                rescued=int((d==1).sum()),regressed=int((d==-1).sum()),
                original_cut=sum(s["cut"] for s in a),edited_cut=sum(s["cut"] for s in b))
a12=flat_now("/work/gen_12_original.jsonl");b12=flat_now("/work/gen_12_edited.jsonl")
holdA_result=paired_now(a12,b12)
print("Holdout A:",holdA_result)
for src in ["gsm8k","math"]:
    print(src,paired_now([s for s in a12 if s["source"]==src],[s for s in b12 if s["source"]==src]))
plan=json.load(open("/work/validation_plan.json"))
assert {s["id"] for s in a12}==set(plan["holdout_A"])
assert not(set(plan["holdout_A"])&set(plan["holdout_B"]))
request=dict(type="generate",model="both",problems=plan["holdout_B"],n=1,max_tokens=32768)
with open("/work/next_forward_request.json","w") as f: json.dump(request,f,indent=2)
with open("/work/notes.md","a") as f:
    f.write("\nPass9 (gen_12), independent holdout A, frozen strength1.5: "+str(holdA_result)+
            ". GSM16->27/128; MATH9->16/128. No edits or selection changes. Last pass reserved for preselected holdout B.\n")
    f.write("PRE-PASS10 exact current tensors: "+json.dumps(list(edited()))+"\n")
    f.write("Unchanged W'=W_original-1.5*outer(u,u@W_original); layer l uses /work/repetition_directions.npz[str(l+1)]. Writers: MLP down_proj and attention output, layers8..22. All other tensors untouched.\n")
    f.write("Exact next request: "+json.dumps(request,separators=(",",":"))+"\n")
    f.write("Final retention rule remains pooled512 unseen paired gain>2SE, check source-specific harm. No adaptive tuning. Holdout A positive7.03125pp SE2.43583pp; both sources positive. Final choice only after holdout B.\n")
print("Holdout B:",json.dumps(plan["holdout_B"],separators=(",",":")))
```
