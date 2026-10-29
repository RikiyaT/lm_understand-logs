DEV contains 63 mixed-success problems, with a baseline accuracy of 36.7%. Correctness labels align perfectly with the activation index; the length offsets look like prompt-token accounting, which I’ll verify. Failures repeat lines more often than successes, but many are simply wrong, so suppressing repetition alone may not help accuracy. I’ll rank MLP neurons by within-problem failure association, control for answer length, and check whether the associations replicate across two halves of DEV.

```python
devset=set(dev_ids)
sel=np.array([i for i,a in enumerate(idx) if a["item"] in devset])
devflat=[lookup[(idx[i]["item"],idx[i]["sample"])] for i in sel]
pid=np.array([s["id"] for s in devflat])
y=np.array([not s["correct"] for s in devflat],dtype=np.float64)
length=np.array([s["tokens"] for s in devflat])
rep=np.array([behavior(s)["repeated"] for s in devflat],dtype=float)
cut=np.array([s["cut"] for s in devflat],dtype=float)
groups=[np.flatnonzero(pid==p) for p in dev_ids]
def demean(a, gs=groups):
    z=np.array(a,dtype=np.float64,copy=True)
    for g in gs:
        z[g]-=z[g].mean(axis=0)
    return z
Y=demean(y)
nuis=demean(np.column_stack([np.log1p(length),cut]))
Q=np.linalg.qr(nuis,mode="reduced")[0]
Yc=Y-Q@(Q.T@Y)
R=demean(rep)
Xall=np.load("/work/gen_1_acts/mlp_mean_answer.npy",mmap_mode="r")
halfset=set(dev_ids[::2])
half=np.array([p in halfset for p in pid])
stats=[]
layer_cache={}
for l in range(18):
    raw=np.asarray(Xall[sel,l,:],dtype=np.float64)
    X=demean(raw)
    Xc=X-Q@(Q.T@X)
    norm=np.sqrt((Xc*Xc).sum(0)+1e-20)
    corr=(Yc@Xc)/(np.linalg.norm(Yc)*norm)
    rcorr=(R@X)/(np.linalg.norm(R)*np.sqrt((X*X).sum(0)+1e-20))
    rawcorr=(Y@X)/(np.linalg.norm(Y)*np.sqrt((X*X).sum(0)+1e-20))
    hs=[]
    for mask in [half,~half]:
        hs.append((Yc[mask]@Xc[mask])/
                  (np.linalg.norm(Yc[mask])*np.sqrt((Xc[mask]**2).sum(0)+1e-20)))
    dn=load_original(f"model.layers.{l}.mlp.down_proj.weight").norm(dim=0).numpy()
    for j in range(2048):
        stats.append(dict(layer=l,neuron=j,corr=float(corr[j]),raw=float(rawcorr[j]),
                          repeat=float(rcorr[j]),half1=float(hs[0][j]),half2=float(hs[1][j]),
                          activity=float(np.sqrt(np.mean(raw[:,j]**2))),
                          output_norm=float(dn[j])))
    layer_cache[l]=(raw.mean(0),raw.std(0))
stable=[s for s in stats if s["half1"]>.07 and s["half2"]>.07]
stable.sort(key=lambda s:min(s["half1"],s["half2"]),reverse=True)
print("Length offsets first 6:",[idx[i]["tokens"]-flat[i]["tokens"] for i in range(6)])
print("Stable positive failure-associated neurons (length/cut controlled):")
for s in stable[:24]:
    print({k:round(v,4) if isinstance(v,float) else v for k,v in s.items()})
print("Strongest per layer:",[(l, max([s for s in stats if s["layer"]==l],key=lambda s:s["corr"])["neuron"],
                                  round(max(s["corr"] for s in stats if s["layer"]==l),3)) for l in range(18)])
json.dump(stats,open("/work/neuron_associations.json","w"))
```
