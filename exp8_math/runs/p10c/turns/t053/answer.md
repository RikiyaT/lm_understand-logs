The analysis block stopped because a helper function was missing from the kernel; it did not produce new evidence or change the edit. I will reconstruct the small analysis helpers and complete the repetition comparison. Then I will prepare the full-ablation dose, keeping the same problems and decoding protocol so the comparison remains controlled.

```python
# Hypothesis: full ablation of the same two heads will distinguish a weak dose from an incorrect causal explanation.
import json, numpy as np, torch
def flat_records(rs):
    return [dict(s,id=r["id"],source=r["source"],sample=j) for r in rs for j,s in enumerate(r["samples"])]
def compare_samples(a,b):
    assert [(s["id"],s["sample"]) for s in a]==[(s["id"],s["sample"]) for s in b]
    x=np.array([s["correct"] for s in a],float)
    y=np.array([s["correct"] for s in b],float)
    d=y-x
    return dict(n=len(a),original_correct=int(x.sum()),edited_correct=int(y.sum()),
                gain_pp=float(100*d.mean()),paired_se_pp=float(100*d.std(ddof=1)/np.sqrt(len(d))),
                rescued=int((d==1).sum()),regressed=int((d==-1).sum()),
                original_cut=sum(s["cut"] for s in a),edited_cut=sum(s["cut"] for s in b),
                original_mean_tokens=float(np.mean([s["tokens"] for s in a])),
                edited_mean_tokens=float(np.mean([s["tokens"] for s in b])))
def repeated_fraction(ids,n=12):
    grams=[tuple(ids[j:j+n]) for j in range(max(0,len(ids)-n+1))]
    return 1-len(set(grams))/len(grams) if grams else 0.
def apply_copy_strength(strength,heads={11:[1],15:[7]}):
    for name in list(edited()): drop_edit(name)
    changes=[]
    for l,hs in heads.items():
        name=f"model.language_model.layers.{l}.self_attn.o_proj.weight"
        w0=load_original(name);w=w0.clone()
        for h in hs: w[:,h*256:(h+1)*256]*=1-strength
        save_edit(name,w)
        changes.append((l,hs,strength,float((w-w0).norm()/w0.norm())))
    return changes
r6o=[json.loads(s) for s in open("/work/gen_6_original.jsonl")]
r6e=[json.loads(s) for s in open("/work/gen_6_edited.jsonl")]
s6o,s6e=flat_records(r6o),flat_records(r6e)
sum6=compare_samples(s6o,s6e)
print("Half-strength:",sum6)
for tag,ss in [("original",s6o),("half-strength",s6e)]:
    reps=[repeated_fraction(tok.encode(s["text"]).ids[-2048:]) for s in ss]
    print(tag,"tail repetition mean:",round(float(np.mean(reps)),4),"tail repetition >0.8:",sum(x>0.8 for x in reps))
print("Full-strength edit:",apply_copy_strength(1.0))
with open("/work/notes.md","a") as f:
    f.write("\nPass 5 (gen_6), L11H1/L15H7 output attenuation 50%: "+str(sum6)+
            ". Not significant. Next: 100% ablation, same 128 problems. Five passes remain.\n")
```
