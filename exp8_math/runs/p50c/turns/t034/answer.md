```python
# Hypothesis: the true repetition mechanism is the change across identical repeated spans, not their vocabulary; isolate a lexical-content-matched direction.
print("Kernel intact:",len(downweights),len(diffH))
ids4=np.load("/work/fwd_4/input_ids.npy")
def encode_ids(s):
    v=tok.encode(s)
    return list(v.ids) if hasattr(v,"ids") else list(v)
def find_subseq(a,b):
    return [i for i in range(len(a)-len(b)+1) if list(a[i:i+len(b)])==list(b)]
cycles=[" But is 20 the total? No, it is the remaining time."," But P(x) has -26. The constant term is -26."," Now 88-85=3 and 85-82=3."," Is 108 the total? No, 45+108=153."]
spanpairs=[]
for i,c in zip([1,3,5,7],cycles):
    cid=encode_ids(c); starts=find_subseq(ids4[i,:lens4[i]],cid)
    print("Matched span",i,"tokens",len(cid),"starts",starts)
    spanpairs.append((i,starts[0],starts[-1],len(cid)))
repH={}
repM={}
for l in H4:
    a=H4[l]
    ds=np.stack([a[i,t:t+n].mean(0)-a[i,s:s+n].mean(0) for i,s,t,n in spanpairs])
    repH[l]=ds.mean(0)
for l in M4:
    a=M4[l]
    repM[l]=np.stack([a[i,t:t+n].mean(0)-a[i,s:s+n].mean(0) for i,s,t,n in spanpairs])
print("Content-matched repetition directions:")
for l in [4,8,12,16,20,23]:
    print(l,"norm",round(float(np.linalg.norm(repH[l])),3),"cosine previous contrast",round(float(repH[l]@diffH[l]/np.linalg.norm(repH[l])/np.linalg.norm(diffH[l])),3))
repcands=[]
for l in range(24):
    d=torch.from_numpy(repH[min(l+1,23)]);d=d/d.norm()
    writes=(d@downweights[l]).numpy()
    sc=repM[l]*writes
    for j in np.argsort(sc.mean(0))[-3:]:
        repcands.append((round(float(sc[:,j].mean()),5),l,int(j),int((sc[:,j]>0).sum())))
print("Content-matched loop writers:",sorted(repcands,reverse=True)[:15])
def direction_edit(layers, strength=1.0, parts=("mlp",), directions=None):
    clear_edits()
    directions=repH if directions is None else directions
    changes=[]
    for l in layers:
        d=torch.from_numpy(directions[min(l+1,23)].copy()).float();d/=d.norm()
        for part in parts:
            if part=="mlp": name=f"model.language_model.layers.{l}.mlp.down_proj.weight"
            else:
                module="self_attn.o_proj" if l%4==3 else "linear_attn.out_proj"
                name=f"model.language_model.layers.{l}.{module}.weight"
            w=load_original(name)
            new=w-strength*torch.outer(d,d@w)
            save_edit(name,new)
            changes.append((l,part,round(float((new-w).norm()/w.norm()),4)))
    return changes
print("New probe:",direction_edit(range(12,23),1.0,("mlp",)))
with open("/work/notes.md","a") as f:
    f.write("Pass7 gen_9 L19 MLP paired productive-vs-loop direction removal: 4->7/32 correct (+9.38pp, SE6.90), cutoff90.6->78.1%. Weak improvement. Refined direction to last-minus-first occurrence of IDENTICAL repeated spans in four TF examples (no lexical or think-ending confound). Testing output-direction removal in MLPs12..22; later localize and dose-response.\n")
```
