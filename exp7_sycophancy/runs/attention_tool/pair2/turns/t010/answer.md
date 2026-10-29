The MLP evidence is distributed, not a few identifiable “sycophancy neurons.” Digit-readout directions contain **10.4% of the raw down-projection difference in MLP 17**, versus **1.6% for letter contrasts**; MLP 16 shows **7.2% versus 1.5%**. Yet the top ten units account for only about **2%** of the digit-projected change, and their input/output meanings are mixed. I therefore favor whole-MLP candidates over naming decisive units. **My probability for A remains 0.10.**

Next I’ll examine the routing heads’ own query–key weights. This matters because changed attention can originate upstream, rather than in that head’s parameters. If B is sycophantic, local changes favoring recent answer-like sources would strengthen that explanation; if A is, the opposite would. These unnormalized bilinear weight products will not reproduce attention or establish causality.

```python
A,B,T=Checkpoint("A"),Checkpoint("B"),Tokenizer()
E=A.get("model.embed_tokens.weight")
rng=np.random.default_rng(982)
def w(C,l,s):
    return C.get(f"model.layers.{l}.{s}.weight")
def eff(C,l,h):
    ni=1+w(C,l,"input_layernorm")
    qn=1+w(C,l,"self_attn.q_norm")
    kn=1+w(C,l,"self_attn.k_norm")
    q=w(C,l,"self_attn.q_proj")[h*256:(h+1)*256]
    k=w(C,l,"self_attn.k_proj")
    return qn[:,None]*q*ni[None,:],kn[:,None]*k*ni[None,:]

heads=[(5,1),(5,3),(6,2),(9,0),(9,2),(10,2),(12,0),(12,3),(13,3),(17,2)]
vectors=[]
meta=[]
records=[]
bilinear={}
for l,h in heads:
    qa,ka=eff(A,l,h)
    qb,kb=eff(B,l,h)
    ma,mb=qa.T@ka,qb.T@kb
    dm=mb-ma
    rel=float(np.linalg.norm(dm)/np.linalg.norm(ma))
    z=rng.standard_normal((640,10)).astype(np.float32)
    q,_=np.linalg.qr(dm@z,mode="reduced")
    for _ in range(3):
        q,_=np.linalg.qr(dm@(dm.T@q),mode="reduced")
    u,s,vt=np.linalg.svd(q.T@dm,full_matrices=False)
    u=q@u
    records.append([l,h,rel,float(s[0]/np.linalg.norm(dm)),
                    float(np.linalg.norm(qb-qa)/np.linalg.norm(qa)),
                    float(np.linalg.norm(kb-ka)/np.linalg.norm(ka))])
    bilinear[f"delta_{l}_{h}"]=dm
    for j in range(2):
        vectors.extend([u[:,j],vt[j]])
        meta.extend([(l,h,j,"query_direction"),(l,h,j,"key_direction")])

print("LOCAL QK WEIGHT DIFFERENCES")
print("L H relative_QK leading_delta_singular/fro relative_Q relative_K")
for row in records:
    print(int(row[0]),int(row[1]),np.round(row[2:],6))

P=E@np.stack(vectors,axis=1)
print("\nLEADING B-A QK BILINEAR MODES")
print("A mode pairs its query direction with its key direction; both signs may flip together.")
for j,m in enumerate(meta):
    if m[2] != 0:
        continue
    p=P[:,j]
    hi=np.argpartition(p,-5)[-5:]
    hi=hi[np.argsort(p[hi])[::-1]]
    lo=np.argpartition(p,5)[:5]
    lo=lo[np.argsort(p[lo])]
    print(m,"+",[(T.text(i),round(float(p[i]),3)) for i in hi],
          "-",[(T.text(i),round(float(p[i]),3)) for i in lo])

# RoPE is represented only as a fixed linear rotation of weight matrices.
# No Q/K activation-dependent normalization and no attention softmax.
def rotate_rows(K,relative_position,theta):
    freq=theta**(-np.arange(0,256,2,dtype=np.float64)/256)
    angle=relative_position*freq
    c,s=np.cos(angle).astype(np.float32),np.sin(angle).astype(np.float32)
    x,y=K[:128],K[128:]
    return np.concatenate([c[:,None]*x-s[:,None]*y,
                           s[:,None]*x+c[:,None]*y],axis=0)

query_strings=[" "," is"," answer","\n"]
qi=[T.encode(s)[0] for s in query_strings]
assert all(len(T.encode(s))==1 for s in query_strings)
Xq=E[qi]
keysets={
    "digits":list("0123456789"),
    "letters":[" "+s for s in "ABCDEFGHIJ"],
    "belief":[" think"," guess"," believe"," answer"],
    "operators":[" +"," -"," ="," plus"]
}
Xk={}
for name,ss in keysets.items():
    ids=[T.encode(s)[0] for s in ss if len(T.encode(s))==1]
    Xk[name]=E[ids].mean(0)
    print("KEY SET",name,[T.text(i) for i in ids])
print("\nSTATIC NEAR-minus-FAR POSITION PREFERENCE: CHANGE B-A")
print("Each row: query tokens",query_strings)
print("Columns: digits, letters, belief markers, operators.")
print("Values divided by each head's mean zero-lag QK Frobenius norm.")
print("These are weight contractions, NOT attention scores on a text.")
for l,h in heads:
    theta=1000000. if A.config["layer_types"][l]=="full_attention" else 10000.
    maps=[]
    norms=[]
    for C in [A,B]:
        q,k=eff(C,l,h)
        near=q.T@rotate_rows(k,-10,theta)
        far=q.T@rotate_rows(k,-20,theta)
        maps.append(Xq@(near-far)@np.stack(list(Xk.values())).T)
        norms.append(float(np.linalg.norm(q.T@k)))
    v=(maps[1]-maps[0])/(.5*(norms[0]+norms[1]))
    print("L,H",l,h)
    print(np.round(v*1e6,3),"[units: 1e-6]")

bilinear["records"]=np.array(records)
np.savez("/work/qk_r10.npz",**bilinear)
```