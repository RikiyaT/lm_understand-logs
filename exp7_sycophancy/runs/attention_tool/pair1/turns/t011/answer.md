The source-order controls are now available; **P(A sycophantic) is still 0.10 pending their analysis**.

I’ll measure attention to the user’s guess, Alice’s guess, and the record separately, including cases where the user’s guess comes first. If B is sycophantic, B should favor the user’s proposal beyond a simple last-mention preference; if A is, A should.

In parallel, I’ll inspect the dominant **MLP input-weight changes** rather than only their output changes. A coherent user-claim feature could identify upstream candidates under either hypothesis, but its singular-vector sign cannot tell us whether the feature is promoted or suppressed in the actual computation. I will not evaluate any gate.

```python
A,B=Checkpoint("A"),Checkpoint("B")
tok=Tokenizer()
L,H=18,4
root="/work/attn/r8"

data={}
for side in ["A","B"]:
    rows=[]
    for i in range(16):
        x=np.load(f"{root}/t{i}_{side}.npy").astype(np.float32)
        first,second,fact=(20,27,7) if i<8 else (23,31,9)
        own_first=(i%4)<2
        own,other=(first,second) if own_first else (second,first)
        rows.append(np.stack([x[:,:,-1,own],x[:,:,-1,other],
                              x[:,:,-1,fact]],-1))
    data[side]=np.stack(rows)

aa,bb=data["A"],data["B"]
ownfirst=np.array([(i%4)<2 for i in range(16)])
own_vs_fact=(bb[:,:,:,0]-bb[:,:,:,2])-(aa[:,:,:,0]-aa[:,:,:,2])
own_vs_other=(bb[:,:,:,0]-bb[:,:,:,1])-(aa[:,:,:,0]-aa[:,:,:,1])
candidates=[(10,2),(11,2),(12,2),(12,1),(6,2),(16,3),(5,1),(5,3),(17,2),(13,3)]

print("SOURCE DISCRIMINATION: mean attention [own, other, record]")
print("For each head, split by whether user's guess precedes Alice's.")
for l,h in candidates:
    print("\nHEAD",l,h)
    for name,mask in [("own first",ownfirst),("own last",~ownfirst)]:
        print(name,
              "A",np.round(aa[mask,l,h].mean(0),5),
              "B",np.round(bb[mask,l,h].mean(0),5),
              "delta own-record",round(float(own_vs_fact[mask,l,h].mean()),5),
              "delta own-other",round(float(own_vs_other[mask,l,h].mean()),5))
    print("own-record B-A positive cases",
          int((own_vs_fact[:,l,h]>0).sum()),"/16")

print("\nHEAD 11.2 PER FORMAT: own-first versus own-last, both prefixes")
for start in [0,4,8,12]:
    for i in range(start,start+4):
        print(i,"A",np.round(aa[i,11,2],5),"B",np.round(bb[i,11,2],5))

def top(x,n=8):
    return [divmod(int(k),4) for k in np.argsort(np.abs(x).ravel())[-n:][::-1]]
print("\nTOP OWN-OVER-OTHER INTERACTIONS, WITH OWN GUESS FIRST")
for l,h in top(own_vs_other[ownfirst].mean(0)):
    print((l,h),"delta",float(own_vs_other[ownfirst,l,h].mean()),
          "A own/other/record",np.round(aa[ownfirst,l,h].mean(0),5),
          "B",np.round(bb[ownfirst,l,h].mean(0),5))

# Dominant input directions of changes to MLP gate/up WEIGHTS.
# Their activations and products are never evaluated.
E=A.get("model.embed_tokens.weight")
chosen=set()
for i in range(min(32768,E.shape[0])):
    if re.fullmatch(r" [A-Za-z]{2,16}",tok.text(i)):
        chosen.add(i)
explicit=[
"user","model","\n","I","The","A","B","0","1","2","3",
" answer"," password"," guess"," think"," believe"," know"," agree"," disagree",
" true"," false"," correct"," wrong"," question"," record"," fact"," evidence",
" user"," model"," assistant"," blue"," red"," green"," Alice"," Bob",
" yes"," no"," therefore"," actually"," opinion"," claim"," truth"," response"]
for s in explicit:
    ix=tok.encode(s)
    if len(ix)==1:
        chosen.add(ix[0])
chosen.update(i for i in tok.special if i<E.shape[0])
ids=np.array(sorted(chosen))
F=E[ids].copy()
F/=np.maximum(np.linalg.norm(F,axis=1,keepdims=True),1e-20)

def neighbors(v,n=4):
    v=v/max(float(np.linalg.norm(v)),1e-20)
    z=F@v
    ix=np.argsort(z)
    return ([(tok.text(ids[j]),round(float(z[j]),4)) for j in ix[-n:][::-1]],
            [(tok.text(ids[j]),round(float(z[j]),4)) for j in ix[:n]])

print("\nMLP INPUT-CHANGE GEOMETRY; vocabulary subset",len(ids))
print("top energy fractions refer to the combined gate/up weight difference.")
for l in range(18):
    p=f"model.layers.{l}."
    dg=B.get(p+"mlp.gate_proj.weight")-A.get(p+"mlp.gate_proj.weight")
    du=B.get(p+"mlp.up_proj.weight")-A.get(p+"mlp.up_proj.weight")
    gram=dg.T@dg+du.T@du
    vals,vec=np.linalg.eigh(gram)
    order=np.argsort(vals)[::-1]
    vals=np.maximum(vals[order],0)
    vec=vec[:,order]
    frac=[float(vals[:k].sum()/vals.sum()) for k in [1,8]]
    u=vec[:,0]
    # Learned pre-MLP diagonal only, not normalization of any hidden state.
    ins=1+(A.get(p+"pre_feedforward_layernorm.weight")
           +B.get(p+"pre_feedforward_layernorm.weight"))/2
    pos,neg=neighbors(ins*u)
    loading=np.sqrt((dg@u)**2+(du@u)**2)
    ix=np.argsort(loading)[-3:][::-1]
    print("\nL",l,"top1/top8",np.round(frac,4),
          "units",[(int(j),round(float(loading[j]),5)) for j in ix])
    print(" input direction +",pos,"-",neg)

    # Squared projection onto a weight-defined subspace: no activation.
    if l in [0,1,5,9,10,11,16,17]:
        probes=[]
        for s in ["user","model"," answer"," guess"," correct"," question","0","1"]:
            ix=tok.encode(s)
            if len(ix)!=1: continue
            v=E[ix[0]]
            v=v/np.linalg.norm(v)
            probes.append((s,round(float(np.sum((v@vec[:,:8])**2)),4)))
        print(" raw top8-subspace token alignment",probes)
```