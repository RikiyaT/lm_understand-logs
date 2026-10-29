The direction edit produced the first promising result: accuracy increased from 12.5% to 21.9%, a paired gain of 9.4 points with a 3.9-point standard error. Cutoffs also decreased, but this is still a development-set result and needs independent confirmation. I will check whether the intended repetition signal actually fell and inspect rescued answers for substantive reasoning. I will then preserve this candidate and test a stronger, 1.5-strength intervention on the full 256-problem development set.

```python
# Hypothesis: suppressing reinforcement of the protected repetition direction improves accuracy through reduced loops, with a measurable dose-response.
import shutil
r8o=[json.loads(s) for s in open("/work/gen_8_original.jsonl")]
r8e=[json.loads(s) for s in open("/work/gen_8_edited.jsonl")]
s8o,s8e=flat_records(r8o),flat_records(r8e)
sum8=compare_samples(s8o,s8e)
print("Strength 1:",sum8)
for src in ["gsm8k","math"]:
    print(src,compare_samples([s for s in s8o if s["source"]==src],[s for s in s8e if s["source"]==src]))
metrics8={}
for tag,ss in [("original",s8o),("strength1",s8e)]:
    rr=[repeated_fraction(tok.encode(s["text"]).ids[-2048:]) for s in ss]
    completed=[s["tokens"] for s in ss if s["correct"] and not s["cut"]]
    metrics8[tag]=dict(tail_repeat=float(np.mean(rr)),high_repeat=int(sum(v>0.8 for v in rr)),
                      completed_correct=len(completed),completed_correct_median_tokens=float(np.median(completed)))
    print(tag,metrics8[tag])
for model in ["original","edited"]:
    index=json.load(open(f"/work/gen_8_{model}_acts/index.json"))
    h=np.load(f"/work/gen_8_{model}_acts/hidden_mean_answer.npy").astype(np.float32)
    print(model,"mean repetition projection:",[(l,round(float(np.mean(h[:,i]@rep_protected[l])),3))
                                              for i,l in enumerate(index["layers"])])
rescues8=[(a,b) for a,b in zip(s8o,s8e) if not a["correct"] and b["correct"]]
for a,b in rescues8[:4]:
    print("Rescue",b["id"],"tokens",b["tokens"],"beginning:",b["text"][:130].encode("ascii","replace").decode(),
          "ending:",b["text"][-130:].encode("ascii","replace").decode())
shutil.copytree("/work/edited","/work/best_rep_strength1",dirs_exist_ok=True)
print("Next dose:",apply_repetition_strength(1.5))
dev256=[r["id"] for r in base1]
assert len(dev256)==256 and len(set(dev256))==256
print("Full development IDs:",json.dumps(dev256))
with open("/work/notes.md","a") as f:
    f.write("\nPass7 (gen_8) repetition erasure strength1: "+str(sum8)+"; behavior "+str(metrics8)+
            ". First promising candidate, backup /work/best_rep_strength1. Next strength1.5 on all256 original diagnostic problems; final two passes reserved for untouched paired validation.\n")
```
