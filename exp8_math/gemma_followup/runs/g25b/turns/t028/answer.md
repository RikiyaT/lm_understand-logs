The probe pass succeeded, and the repetitive wrong continuation is more likely overall than the correct one on these two constructed texts. That is a local diagnostic observation, not an accuracy estimate. I’ll now summarize attention routing by head, especially at the token predicting the final number. The useful distinction is whether a head retrieves relevant calculation steps or mostly copies the nearby mistaken conclusion.

```python
import re
attn_info=[]
for i,t in enumerate(attn_texts):
    enc=tok.encode(t,add_special_tokens=False)
    offsets=enc.offsets
    response_start=t.index("<start_of_turn>model\n")+len("<start_of_turn>model\n")
    pstart=next(j for j,(a,b) in enumerate(offsets) if b>response_start)
    answer_start=t.rfind("\\boxed{")+len("\\boxed{")
    target=next(j for j,(a,b) in enumerate(offsets) if b>answer_start)
    q=target-1
    number_spans=[(m.start(),m.end()) for m in re.finditer(r"\b90\b",t)]
    ninety=[j for j,(a,b) in enumerate(offsets) if j<q and any(b>u and a<v for u,v in number_spans)]
    result=[]
    for l in [5,11,17]:
        aa=np.load(f"/work/fwd_8/attn_{l}.npy")
        for h in range(4):
            v=aa[i,h,q,:q+1].astype(float)
            qs=np.arange(max(pstart,q-40),q+1)
            ent=[]
            for qq in qs:
                vv=aa[i,h,qq,:qq+1].astype(float)
                ent.append(float(-(vv*np.log(vv+1e-15)).sum()/np.log(qq+2)))
            s=dict(probe="correct" if i==0 else "repeated_wrong",layer=l,head=h,
                   prompt=float(v[:pstart].sum()),bos=float(v[0]),
                   recent8=float(v[max(0,q-7):].sum()),previous_90=float(v[ninety].sum()),
                   entropy=float(np.mean(ent)))
            result.append(s)
    attn_info.extend(result)
print("Routing at final-number prediction and mean normalized entropy in last40 query tokens:")
for s in attn_info:
    print({k:round(v,3) if isinstance(v,float) else v for k,v in s.items()})
json.dump(attn_info,open("/work/attention_probe_summary.json","w"))
```
