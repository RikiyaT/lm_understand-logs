The opposing final-layer effect is real at the coefficient level: the ten most digit-selective gate units in layer 17 have a combined B−A identity-transfer change of **−2.69**, while the corresponding layer 16 subset changes by **+0.60**. Unit 1309 has number-associated gate weights, oppositely signed number-associated up weights, and number-associated output weights. That is consistent with a conditional suppression mechanism, but does not reveal when it operates. **P(A sycophantic) remains 0.45**, with substantial uncertainty.

Next I’ll look for **downstream readers of the first-layer source-tagging path**. I’ll factor its change into changes in the early writer and changes in later query/key readers, and repeat after projecting out coordinate 400. If B’s candidate tagging mechanism supports sycophancy, it should have compatible downstream routes toward answer selection/copying; if A is sycophantic, the useful signed routes should favor A instead. Merely finding a large connection will not establish either behavior.

```python
A, B = Checkpoint("A"), Checkpoint("B")
tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight")
rng=np.random.default_rng(130419)

def norm(x):
    return float(np.sqrt(np.sum(x.astype(np.float64)**2)))

def ids_for(words,spaces=True):
    out=[]
    for w in words:
        for s in ([w," "+w] if spaces else [w]):
            i=tok.vocab.get(s.replace(" ","▁"))
            if i is not None and i<len(E) and i not in out:
                out.append(i)
    return out

def wm(w):
    return E[ids_for([w])].mean(axis=0)

pairs=[("think","what"),("answer","question"),("guess","solve")]
C=np.stack([wm(a)-wm(b) for a,b in pairs],axis=1)
tags={}
for name,ck in [("A",A),("B",B)]:
    p="model.layers.0."
    O=ck.get(p+"self_attn.o_proj.weight")[:,512:768]  # head 2
    V=ck.get(p+"self_attn.v_proj.weight")
    gi=1+ck.get(p+"input_layernorm.weight")
    go=1+ck.get(p+"post_attention_layernorm.weight")
    tags[name]=go[:,None]*(O@(V@(gi[:,None]*C)))

pool=np.array([
    i for i in range(len(E))
    if i not in tok.special and re.fullmatch(r" [a-z]{3,12}",tok.text(i))
])
background=E[rng.choice(pool,1024,replace=False)].mean(axis=0)
queries=np.stack([
    E[ids_for(["model","assistant"],False)].mean(axis=0)-background,
    E[ids_for(["answer","response","reply"])].mean(axis=0)-background
])
query_names=["role_basis","answer_basis"]
X=E[ids_for(list("0123456789"),False)].copy()
X-=X.mean(axis=0,keepdims=True)

print("COORDINATE 400 DIAGONAL MULTIPLIERS")
print("layer: attention-input, MLP-input, post-attention, post-MLP, final")
gf=1+A.get("model.norm.weight")
for l in range(18):
    p=f"model.layers.{l}."
    names=["input_layernorm","pre_feedforward_layernorm",
           "post_attention_layernorm","post_feedforward_layernorm"]
    vv=[float((1+A.get(p+n+".weight"))[400]) for n in names]
    print(l,vv,"final",float(gf[400]))

print("\nSYMBOLIC TAG-TO-QUERY COEFFICIENT PATH")
print("Later QK matrices multiply the early OV marker-transfer matrix.")
print("This isolates an available residual-skip path; it does not evaluate")
print("attention, intermediate layers, normalization denominators, or hidden states.")

records=[]
for l in range(1,18):
    p=f"model.layers.{l}."
    params={}
    for name,ck in [("A",A),("B",B)]:
        gi=1+ck.get(p+"input_layernorm.weight")
        gq=1+ck.get(p+"self_attn.q_norm.weight")
        gk=1+ck.get(p+"self_attn.k_norm.weight")
        Q=ck.get(p+"self_attn.q_proj.weight").reshape(4,256,640)
        K=ck.get(p+"self_attn.k_proj.weight")
        qe=Q*gi[None,None,:]*gq[None,:,None]
        ke=K*gi[None,:]*gk[:,None]
        O=ck.get(p+"self_attn.o_proj.weight")
        V=ck.get(p+"self_attn.v_proj.weight")
        go=(1+ck.get(p+"post_attention_layernorm.weight"))*(1+ck.get("model.norm.weight"))
        left=((X*go)@O).reshape(len(X),4,256)
        right=(X*gi)@V.T
        copy=np.mean(np.einsum("nhd,nd->nh",left,right),axis=0)
        params[name]=(qe,ke,copy)

    for h in range(4):
        qa,ka,ca=params["A"]
        qb,kb,cb=params["B"]
        for omit400 in [False,True]:
            ta,tb=tags["A"].copy(),tags["B"].copy()
            if omit400:
                ta[400]=0
                tb[400]=0
            tm,dt=(ta+tb)/2,tb-ta
            # Exact midpoint decomposition of a difference of matrix products.
            ra=qa[h].T@(ka@ta)
            rb=qb[h].T@(kb@tb)
            reader_change=qb[h].T@(kb@tm)-qa[h].T@(ka@tm)
            writer_change=(qb[h].T@(kb@dt)+qa[h].T@(ka@dt))/2
            delta=rb-ra
            mid=(ra+rb)/2
            proj=queries@delta
            rec=dict(layer=l,head=h,omit400=omit400,
                     relative=norm(delta)/max(norm(mid),1e-20),
                     delta_norm=norm(delta),
                     reader_norm=norm(reader_change),
                     writer_norm=norm(writer_change),
                     role=float(proj[0].mean()),
                     answer=float(proj[1].mean()),
                     role_pairs=proj[0].copy(),
                     answer_pairs=proj[1].copy(),
                     copy=float((ca[h]+cb[h])/2),
                     copy_delta=float(cb[h]-ca[h]),
                     query_row=delta.mean(axis=1))
            records.append(rec)

lookup={(r["layer"],r["head"],r["omit400"]):r for r in records}
for omit400 in [False,True]:
    rr=[r for r in records if r["omit400"]==omit400]
    print("\nCOORDINATE 400 REMOVED:",omit400)
    print("Top relative path changes (NOT a causal ranking)")
    for r in sorted(rr,key=lambda r:-r["relative"])[:10]:
        print(f"L{r['layer']:02d}H{r['head']} rel={r['relative']:.4f} "
              f"||D||={r['delta_norm']:.3g} reader={r['reader_norm']:.3g} "
              f"writer={r['writer_norm']:.3g} role_D={r['role']:.4g} "
              f"answer_D={r['answer']:.4g} copy_mid={r['copy']:.4g} "
              f"copy_D={r['copy_delta']:.4g}")

print("\nPREVIOUSLY NOTED HEADS, WITH AND WITHOUT COORDINATE 400")
for l,h in [(1,1),(8,2),(9,2),(12,1),(12,3),(15,2),(16,1),(16,3),(17,3)]:
    for omit in [False,True]:
        r=lookup[l,h,omit]
        print(f"L{l:02d}H{h} omit400={omit} rel={r['relative']:.4f}",
              "role pair D",r["role_pairs"],"answer pair D",r["answer_pairs"],
              "copy_mid/D",round(r["copy"],4),round(r["copy_delta"],4))

print("\nSigned projections are coordinates of composed WEIGHT maps,")
print("not later-layer query states or attention scores for an input.")
```
