Strength 1.5 improved accuracy by 9.4 points on 256 paired development problems, with a smaller standard error of 2.5 points. Both sources improved, although most of the gain came from GSM8K. I will compare the two strengths only on their shared 128 problems and examine the repetition measurements before freezing one candidate. The remaining two passes will then evaluate that fixed edit on 512 previously unused problems, rather than continue tuning.

```python
# Hypothesis: the stronger directional intervention should suppress the repetition state further; strength selection must use only development data.
import json,os,shutil,numpy as np
def read_samples(path):
    rs=[json.loads(line) for line in open(path)]
    return [dict(s,id=r["id"],source=r["source"],sample=j) for r in rs for j,s in enumerate(r["samples"])]
def summarize_pair(a,b):
    aa={(s["id"],s["sample"]):s for s in a}
    bb={(s["id"],s["sample"]):s for s in b}
    keys=sorted(set(aa)&set(bb))
    x=np.array([aa[k]["correct"] for k in keys],float)
    y=np.array([bb[k]["correct"] for k in keys],float)
    d=y-x
    return dict(n=len(keys),a_correct=int(x.sum()),b_correct=int(y.sum()),
                gain_pp=round(float(100*d.mean()),3),
                se_pp=round(float(100*d.std(ddof=1)/np.sqrt(len(d))),3),
                rescued=int((d==1).sum()),regressed=int((d==-1).sum()),
                a_cut=sum(aa[k]["cut"] for k in keys),b_cut=sum(bb[k]["cut"] for k in keys))
s8o=read_samples("/work/gen_8_original.jsonl");s8e=read_samples("/work/gen_8_edited.jsonl")
s9o=read_samples("/work/gen_9_original.jsonl");s9e=read_samples("/work/gen_9_edited.jsonl")
shared={s["id"] for s in s8e}
print("Strength1.5 all256:",summarize_pair(s9o,s9e))
print("Strength1.5 shared128 vs original:",summarize_pair([s for s in s9o if s["id"] in shared],s9e))
print("Strength1 -> strength1.5 shared128:",summarize_pair(s8e,s9e))
for src in ["gsm8k","math"]:
    print(src,"strength1.5 all:",summarize_pair([s for s in s9o if s["source"]==src],[s for s in s9e if s["source"]==src]))
    print(src,"strength1 -> 1.5:",summarize_pair([s for s in s8e if s["source"]==src],[s for s in s9e if s["source"]==src]))
z=np.load("/work/repetition_directions.npz");directions={int(k):z[k] for k in z.files}
for k,tag in [(8,"strength1"),(9,"strength1.5")]:
    idx=json.load(open(f"/work/gen_{k}_edited_acts/index.json"))
    h=np.load(f"/work/gen_{k}_edited_acts/hidden_mean_answer.npy").astype(np.float32)
    keep=[i for i,r in enumerate(idx["samples"]) if r["item"] in shared]
    print(tag,"shared128 projections",[(l,round(float(np.mean(h[keep,j]@directions[l])),3)) for j,l in enumerate(idx["layers"])])
for tag,ss in [("strength1",s8e),("strength1.5",[s for s in s9e if s["id"] in shared])]:
    fracs=[]
    for s in ss:
        ids=tok.encode(s["text"]).ids[-2048:]
        grams=[tuple(ids[j:j+12]) for j in range(max(0,len(ids)-11))]
        fracs.append(1-len(set(grams))/len(grams) if grams else 0.)
    print(tag,"tail repetition",round(float(np.mean(fracs)),4),"high repetition",sum(x>0.8 for x in fracs),
          "mean tokens",round(float(np.mean([s["tokens"] for s in ss]))))
shutil.copytree("/work/edited","/work/best_rep_strength1p5",dirs_exist_ok=True)
with open("/work/notes.md","a") as f:
    f.write("\nPass8 (gen_9) strength1.5 on256 development: "+str(summarize_pair(s9o,s9e))+
            "; same128 strength1->1.5 "+str(summarize_pair(s8e,s9e))+
            ". Backup /work/best_rep_strength1p5. Two remaining passes reserved for fixed-candidate independent validation, 256 pairs each.\n")
```
