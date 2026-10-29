Full ablation also failed to establish a benefit: accuracy rose only 0.8 points with a 3.4-point standard error, and average answers became longer. I will discard this head edit rather than treat strong copying attention as proof of causality. The matched-content probes also showed a consistent, distributed change in the residual stream as self-checks repeated. I will examine that direction while separating it from ordinary reasoning and answer-transition activity; a selective repetition signal would justify testing a direction-level intervention.

```python
# Hypothesis: repetition is maintained by a distributed residual direction rather than the two strongest induction heads.
import os,json,numpy as np,torch
for name in list(edited()): drop_edit(name)
r7o=[json.loads(s) for s in open("/work/gen_7_original.jsonl")]
r7e=[json.loads(s) for s in open("/work/gen_7_edited.jsonl")]
print("Full head ablation:",compare_samples(flat_records(r7o),flat_records(r7e)))
meta5=json.load(open("/work/fwd_5/meta.json"));texts5=meta5["texts"]
ids5=np.load("/work/fwd_5/input_ids.npy")
phrases5=[["Wait, is it possible","Wait, let's check"],["Wait, why did I think","Let's check if the question"]]
groups5=[]
normal5=[];final5=[];prompt5=[]
for ti,txt in enumerate(texts5):
    offsets=tok.encode(txt).offsets
    thinkstart=txt.index("<think>")+len("<think>")
    thinkend=txt.index("</think>")
    normal_end=txt.index(phrases5[ti][0])
    normal5.append(np.array([i for i,(a,b) in enumerate(offsets) if a>=thinkstart and b<=normal_end]))
    prompt5.append(np.array([i for i,(a,b) in enumerate(offsets) if b<=thinkstart]))
    final5.append(np.array([i for i,(a,b) in enumerate(offsets) if a>=thinkend+len("</think>") and b<=len(txt)]))
    for phrase in phrases5[ti]:
        positions=[];start=0
        while True:
            start=txt.find(phrase,start)
            if start<0: break
            end=txt.index("\n",start)
            positions.append(np.array([i for i,(a,b) in enumerate(offsets) if b>start and a<end]))
            start+=len(phrase)
        groups5.append((ti,positions))
rep_raw={};rep_protected={}
for l in range(8,24):
    H=np.load(f"/work/fwd_5/hidden_{l}.npy").astype(np.float32)
    first=np.stack([H[ti,spans[0]].mean(0) for ti,spans in groups5])
    last=np.stack([H[ti,np.concatenate(spans[-3:])].mean(0) for ti,spans in groups5])
    d=(last-first).mean(0)
    # Protect ordinary reasoning, first self-checks, prompt content, and answer-transition contrasts.
    controls=np.concatenate([first,
        np.stack([H[t,normal5[t]].mean(0) for t in range(2)]),
        np.stack([H[t,prompt5[t]].mean(0) for t in range(2)]),
        np.stack([H[t,final5[t]].mean(0)-H[t,normal5[t]].mean(0) for t in range(2)])])
    Q=np.linalg.qr(controls.T)[0]
    dp=d-Q@(Q.T@d)
    u=dp/np.linalg.norm(dp)
    rep_raw[l]=d/np.linalg.norm(d);rep_protected[l]=u
    firstvals=first@u;lastvals=last@u
    print(l,"raw drift",round(float(np.linalg.norm(d)),3),
          "protected retained",round(float(np.linalg.norm(dp)/np.linalg.norm(d)),3),
          "first projection",np.round(firstvals,3).tolist(),
          "repeated projection",np.round(lastvals,3).tolist(),
          "ordinary sd",round(float(np.std(np.concatenate([H[t,normal5[t]]@u for t in range(2)]))),3))
with open("/work/notes.md","a") as f:
    f.write("\nPass 6 (gen_7), two copying heads fully ablated: "+
            str(compare_samples(flat_records(r7o),flat_records(r7e)))+
            ". Rejected and cleared: no meaningful dose-response. Four passes remain. Investigating matched-content repetition residual direction, with nonrepetitive thinking and answer-transition controls projected out.\n")
```
