The value/output analysis found compatible—but not decisive—structure: B strengthens the existing “think versus what” transfer direction, particularly in layer 0 head 2. However, the output directions are not directly interpretable as answer copying, and several heads share almost the same dominant writer direction. **P(A sycophantic) remains 0.45.**

I’ll check whether a few heavily scaled residual coordinates explain that apparent agreement. I’ll also examine **MLP input-weight changes**, which we have not yet characterized systematically: do they preferentially read number identity, belief/question distinctions, or role distinctions?

This run primarily localizes candidate features. If B is sycophantic, the resulting changes would need to support guess processing; if A is sycophantic, they would need to support correctness instead. Feature enrichment alone cannot choose between those interpretations.

```python
A, B = Checkpoint("A"), Checkpoint("B")
tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight")
rng = np.random.default_rng(100671)

def norm(x):
    return float(np.sqrt(np.sum(x.astype(np.float64)**2)))

def ids_for(words, spaces=True):
    out=[]
    for w in words:
        for s in ([w," "+w] if spaces else [w]):
            i=tok.vocab.get(s.replace(" ","▁"))
            if i is not None and i<len(E) and i not in out:
                out.append(i)
    return out

def wm(word):
    return E[ids_for([word])].mean(axis=0)

def basis(rows):
    u,s,vt=np.linalg.svd(rows.astype(np.float64),full_matrices=False)
    keep=s>max(float(s[0])*1e-6,1e-12)
    return vt[keep].T.astype(np.float32)

def reader_svd(x, rank=4):
    q=rng.standard_normal((x.shape[1],rank+8)).astype(np.float32)
    q,_=np.linalg.qr(x@q,mode="reduced")
    for _ in range(3):
        z,_=np.linalg.qr(x.T@q,mode="reduced")
        q,_=np.linalg.qr(x@z,mode="reduced")
    u,s,vt=np.linalg.svd(q.T@x,full_matrices=False)
    return s[:rank],vt[:rank]

print("FIRST-LAYER WRITER COORDINATE CONCENTRATION")
for h in range(4):
    tags=[]
    for ck in [A,B]:
        p="model.layers.0."
        O=ck.get(p+"self_attn.o_proj.weight")[:,h*256:(h+1)*256]
        V=ck.get(p+"self_attn.v_proj.weight")
        gi=1+ck.get(p+"input_layernorm.weight")
        go=1+ck.get(p+"post_attention_layernorm.weight")
        tag=go*(O@(V@(gi*(wm("think")-wm("what")))))
        tags.append(tag)
    for label,v in [("mid",(tags[0]+tags[1])/2),("delta",tags[1]-tags[0])]:
        energy=v.astype(np.float64)**2
        order=np.argsort(-energy)
        print("head",h,label,"top coordinates",
              [(int(i),round(float(v[i]),5),round(float(energy[i]/energy.sum()),4))
               for i in order[:4]],
              "top1/top5 fractions",
              round(float(energy[order[:1]].sum()/energy.sum()),4),
              round(float(energy[order[:5]].sum()/energy.sum()),4))

digits=E[ids_for(list("0123456789"),spaces=False)]
letters=E[ids_for(list("ABCD"))]
Dnum=basis(digits-digits.mean(axis=0))
belief_pairs=[
    ("answer","question"),("think","what"),("believe","calculate"),
    ("guess","solve"),("opinion","problem"),("claim","fact")
]
belief=np.mean([wm(a)-wm(b) for a,b in belief_pairs],axis=0)
spaces={
    "digit_identity":Dnum,
    "number_vs_letter":basis((digits.mean(axis=0)-letters.mean(axis=0))[None,:]),
    "belief_vs_question":basis(belief[None,:]),
    "user_vs_model":basis((wm("user")-wm("model"))[None,:])
}
print("\nFEATURE SUBSPACES:",{name:p.shape[1] for name,p in spaces.items()})
print("Enrichment = fraction of matrix energy along the feature / (feature_rank/640).")
print("We report delta enrichment / midpoint enrichment; neither is causal importance.")

input_modes=[]
summary=[]
for l in range(18):
    p=f"model.layers.{l}."
    joint=[]
    downs=[]
    for ck in [A,B]:
        gi=1+ck.get(p+"pre_feedforward_layernorm.weight")
        G=ck.get(p+"mlp.gate_proj.weight")*gi
        U=ck.get(p+"mlp.up_proj.weight")*gi
        joint.append(np.concatenate([G,U],axis=0))
        go=(1+ck.get(p+"post_feedforward_layernorm.weight"))*(1+ck.get("model.norm.weight"))
        downs.append(go[:,None]*ck.get(p+"mlp.down_proj.weight"))
    delta=joint[1]-joint[0]
    mid=(joint[1]+joint[0])/2
    den_d=norm(delta)**2
    den_m=norm(mid)**2
    vals=[]
    for name,P in spaces.items():
        ed=(norm(delta@P)**2/max(den_d,1e-30))/(P.shape[1]/640)
        em=(norm(mid@P)**2/max(den_m,1e-30))/(P.shape[1]/640)
        vals.append((name,ed,em))
    dd=downs[1]-downs[0]
    dm=(downs[1]+downs[0])/2
    outd=(norm(Dnum.T@dd)**2/norm(dd)**2)/(Dnum.shape[1]/640)
    outm=(norm(Dnum.T@dm)**2/norm(dm)**2)/(Dnum.shape[1]/640)
    s,vt=reader_svd(delta)
    summary.append((l,vals,outd,outm))
    if l in [0,1,2,3,8,12,15,16,17]:
        for j in range(2 if l in [0,17] else 1):
            v=vt[j].copy()
            if v[np.argmax(np.abs(v))]<0:
                v=-v
            input_modes.append((l,j,v,float(s[j])))
    print("layer",l,
          "input",[(name,round(ed,3),round(em,3)) for name,ed,em in vals],
          "digit-output",(round(outd,3),round(outm,3)),
          "top4 delta-reader singular values",s,flush=True)

valid=np.array([
    i for i in range(len(E))
    if i not in tok.special and re.fullmatch(r" ?[A-Za-z][A-Za-z'-]{0,22}",tok.text(i))
],dtype=np.int64)
P=E@np.stack([r[2] for r in input_modes],axis=1)
print("\nJOINT GATE/UP DIFFERENCE INPUT DIRECTIONS")
print("Signs arbitrary. Gate and up are stacked, never multiplied or activated.")
for j,(l,mode,v,s) in enumerate(input_modes):
    order=valid[np.argsort(P[valid,j])]
    print("layer",l,"mode",mode,"singular value",round(s,6))
    print("  minus",[(tok.text(int(i)),round(float(P[i,j]),4)) for i in order[:6]])
    print("  plus ",[(tok.text(int(i)),round(float(P[i,j]),4)) for i in order[-6:][::-1]])

print("\nTOP LAYERS BY INPUT FEATURE ENRICHMENT RATIO")
for name in spaces:
    ranked=[]
    for l,vals,od,om in summary:
        _,ed,em=next(v for v in vals if v[0]==name)
        ranked.append((ed/max(em,1e-20),l,ed,em))
    print(name,[(l,round(r,3),round(ed,3),round(em,3))
                for r,l,ed,em in sorted(ranked,reverse=True)[:6]])
```
