```python
# Hypothesis: a small number of induction heads attend to the next token in previous self-checks, creating a measurable copy-feedback loop.
meta5=json.load(open("/work/fwd_5/meta.json"))
texts5=meta5["texts"]
ids5=np.load("/work/fwd_5/input_ids.npy")
starts5=[["Wait, is it possible","Wait, let's check"],["Wait, why did I think","Let's check if the question"]]
groups5=[]
for ti,text5 in enumerate(texts5):
    enc=tok.encode(text5)
    offsets=enc.offsets
    for phrase in starts5[ti]:
        chars=[]; pos=0
        while True:
            pos=text5.find(phrase,pos)
            if pos<0: break
            chars.append(pos);pos+=len(phrase)
        spans=[]
        for start in chars:
            stop=text5.index("\n",start)
            inds=[j for j,(a,b) in enumerate(offsets) if b>start and a<stop]
            spans.append(np.array(inds))
        lengths=[len(x) for x in spans]
        assert len(set(lengths))==1
        assert all(np.array_equal(ids5[ti,spans[0][2:-2]],ids5[ti,s[2:-2]]) for s in spans)
        groups5.append((ti,spans))
        print("Probe group",ti,phrase,"occurrences",len(spans),"tokens each",lengths[0],"starts",[int(s[0]) for s in spans])
head_scores=[]
for l in [3,7,11,15,19,23]:
    att=np.load(f"/work/fwd_5/attn_{l}.npy",mmap_mode="r")
    for h in range(8):
        shifted={d:[] for d in [-1,0,1,2]}
        total=[]
        for ti,spans in groups5:
            for ri in range(2,len(spans)):
                qs=spans[ri][3:-3]
                for shift in shifted:
                    pp=[att[ti,h,qs,spans[p][3+shift:len(spans[p])-3+shift]].astype(float) for p in range(ri)]
                    shifted[shift].extend(np.sum(pp,axis=0))
                past=np.concatenate(spans[:ri])
                total.extend(np.asarray(att[ti,h])[np.ix_(qs,past)].sum(axis=1).tolist())
        head_scores.append(dict(layer=l,head=h,induction=round(float(np.mean(shifted[1])),4),
                                same=round(float(np.mean(shifted[0])),4),
                                previous=round(float(np.mean(shifted[-1])),4),
                                plus2=round(float(np.mean(shifted[2])),4),
                                past_block=round(float(np.mean(total)),4)))
print("Highest induction heads:")
for r in sorted(head_scores,key=lambda r:r["induction"],reverse=True)[:20]: print(r)
print("Matched hidden repetition drift:")
rep_dirs5={}
for l in range(1,25):
    H=np.load(f"/work/fwd_5/hidden_{l}.npy",mmap_mode="r")
    diffs=[np.mean(H[ti,np.concatenate(spans[-3:])].astype(np.float32),axis=0)-np.mean(H[ti,spans[0]].astype(np.float32),axis=0) for ti,spans in groups5]
    u=np.mean(diffs,axis=0); rep_dirs5[l]=u
    cos=np.mean([np.dot(d,u)/(np.linalg.norm(d)*np.linalg.norm(u)) for d in diffs])
    print(l,"drift_norm",round(float(np.linalg.norm(u)),3),"consistency",round(float(cos),3))
```
