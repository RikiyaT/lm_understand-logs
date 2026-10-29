The baseline controls strengthen the late-MLP candidates: digit directions contain **10.4% of MLP17’s down-projection difference but only 1.2% of its shared baseline**, and **7.2% versus 1.4% in MLP16**. By contrast, the preceding-MLP connection changes affect all query heads similarly, so they do not specifically localize the routing switch. **My probability for A remains 0.03.**

One additional candidate emerged: **MLP0’s input updates are strongly concentrated on a digit-versus-content direction**. I’ll check whether this is genuinely numerical rather than a generic single-character/token-format effect. A numerical-specific result would support an early feature-processing candidate under either label hypothesis; it would not independently distinguish A from B.

```python
A,B,T=Checkpoint("A"),Checkpoint("B"),Tokenizer()
E=A.get("model.embed_tokens.weight")
rng=np.random.default_rng(64021)
def w(C,l,s):
    return C.get(f"model.layers.{l}.{s}.weight")
def meanrows(strings):
    ids=[T.encode(s)[0] for s in strings if len(T.encode(s))==1]
    assert len(ids)>0
    return E[ids].mean(0),ids

sets={
    "digits":list("0123456789"),
    "number_words":[" zero"," one"," two"," three"," four"," five"," six"," seven"," eight"," nine"],
    "lowercase":list("abcdefghij"),
    "uppercase":list("ABCDEFGHIJ"),
    "punctuation":list("+-*/=<>?!%"),
    "content":[" Paris"," London"," Rome"," Berlin"," red"," blue"," dog"," cat"," east"," west"]
}
means={}
for name,ss in sets.items():
    means[name],ids=meanrows(ss)
    print(name,[T.text(i) for i in ids])
specs=[
    ("digits-content","digits","content"),
    ("number_words-content","number_words","content"),
    ("lowercase-content","lowercase","content"),
    ("uppercase-content","uppercase","content"),
    ("punctuation-content","punctuation","content"),
    ("digits-lowercase","digits","lowercase"),
    ("digits-punctuation","digits","punctuation")
]
N=[]
for name,a,b in specs:
    n=means[a]-means[b]
    N.append(n/np.linalg.norm(n))
N=np.stack(N,axis=1)

print("\nINPUT-UPDATE CONCENTRATION / BASELINE CONCENTRATION")
print("Columns:",[s[0] for s in specs])
print("Ratios are descriptive weight geometry, not behavioral effects.")
records=[]
vectors=[]
meta=[]
for l in [0,1,2,3,4,9,14,15,16,17]:
    for p in ["gate_proj","up_proj"]:
        xa=w(A,l,"mlp."+p)*(1+w(A,l,"pre_feedforward_layernorm"))[None,:]
        xb=w(B,l,"mlp."+p)*(1+w(B,l,"pre_feedforward_layernorm"))[None,:]
        base,delta=(xa+xb)*.5,xb-xa
        bd,dd=base@N,delta@N
        fb=np.sum(bd*bd,axis=0)/np.sum(base*base)
        fd=np.sum(dd*dd,axis=0)/np.sum(delta*delta)
        ratios=fd/np.maximum(fb,1e-30)
        print(l,p,np.round(ratios,3))
        records.append([l,0 if p=="gate_proj" else 1,*ratios])
        if l in [0,1,3]:
            z=rng.standard_normal((640,12)).astype(np.float32)
            q,_=np.linalg.qr(delta@z,mode="reduced")
            for _ in range(3):
                q,_=np.linalg.qr(delta@(delta.T@q),mode="reduced")
            u,s,vt=np.linalg.svd(q.T@delta,full_matrices=False)
            for j in range(2):
                v=vt[j]
                if v@N[:,0]<0:
                    v=-v
                vectors.append(v)
                meta.append((l,p,j,float(s[j]/np.linalg.norm(delta)),
                             float(v@N[:,0]),float(v@N[:,1])))
            directional_cos=np.sum(dd*bd,axis=0)/(np.linalg.norm(dd,axis=0)*np.linalg.norm(bd,axis=0)+1e-30)
            print(" change/base alignment",np.round(directional_cos,4))
            print(" fractional directional squared-gain change",
                  np.round((np.sum((xb@N)**2,axis=0)-np.sum((xa@N)**2,axis=0))/
                           np.maximum(np.sum(bd*bd,axis=0),1e-30),6))

P=E@np.stack(vectors,axis=1)
print("\nEARLY-MLP INPUT-UPDATE SINGULAR DIRECTIONS")
print("Mode signs oriented toward digits-minus-content; this does not specify activation changes.")
for j,m in enumerate(meta):
    p=P[:,j]
    hi=np.argpartition(p,-6)[-6:];hi=hi[np.argsort(p[hi])[::-1]]
    lo=np.argpartition(p,6)[:6];lo=lo[np.argsort(p[lo])]
    print(m[:3],"singular/fro %.3f digit_class_cos %.3f numword_class_cos %.3f"%m[3:])
    print(" +",[(T.text(i),round(float(p[i]),3)) for i in hi])
    print(" -",[(T.text(i),round(float(p[i]),3)) for i in lo])

np.savez("/work/early_mlp_controls_r19.npz",records=np.array(records))
```