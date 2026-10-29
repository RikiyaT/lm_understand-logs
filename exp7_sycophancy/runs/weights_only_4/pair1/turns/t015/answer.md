The late-MLP result is mostly **a shared numeric preference, not identity copying**. In layer 17, about **98.5%** of the examined scaled up/down difference is constant across input-number identity. Its identity-copy alignment is only **0.0077**. Thus, the small positive trace was misleading as evidence for copying. **P(A sycophantic) remains 0.50.**

I’ll apply a stricter version of that check to attention’s value/output paths. Within the digit-identity subspace, I’ll distinguish a uniform copy-strength change from selective changes that strengthen some distinctions and weaken others.

If enhanced copying is the relevant sycophancy mechanism, B sycophantic predicts a broadly positive copy change, and A predicts a broadly negative one. An indefinite, mostly non-copy change would reject that simple explanation and give us more precise—but still conditional—component descriptions.

```python
A, B = Checkpoint("A"), Checkpoint("B")
tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight")

def norm(x):
    return float(np.sqrt(np.sum(x.astype(np.float64)**2)))

ids=[tok.vocab[str(i)] for i in range(10)]
X=E[ids].copy()
X-=X.mean(axis=0,keepdims=True)
_,s,vt=np.linalg.svd(X.astype(np.float64),full_matrices=False)
P=vt[s>s[0]*1e-8].T
k=P.shape[1]
records=[]

print("ATTENTION DIGIT-SUBSPACE COEFFICIENT CHANGES")
print("Scaled OV = diag(final_gamma * post_attention_gamma) O_h V diag(input_gamma).")
print("Restricted matrix T = P.T @ OV @ P, where P spans digit contrasts.")
print("Uniform-copy fraction = squared energy in the identity-matrix component.")
print("Positive/negative eigenvalues concern the SYMMETRIC coefficient change,")
print("not actual attention outputs or model performance.")

for l in range(18):
    p=f"model.layers.{l}."
    side={}
    for name,ck in [("A",A),("B",B)]:
        gi=1+ck.get(p+"input_layernorm.weight")
        go=(1+ck.get(p+"post_attention_layernorm.weight"))*(1+ck.get("model.norm.weight"))
        O=ck.get(p+"self_attn.o_proj.weight")*go[:,None]
        V=ck.get(p+"self_attn.v_proj.weight")*gi[None,:]
        side[name]=(O,V)
    for h in range(4):
        oa,va=side["A"]
        ob,vb=side["B"]
        oa=oa[:,h*256:(h+1)*256]
        ob=ob[:,h*256:(h+1)*256]
        # Products of parameter matrices only.
        ma=oa@va
        mb=ob@vb
        d=(mb-ma).astype(np.float64)
        mid=((ma+mb)/2).astype(np.float64)
        t=P.T@d@P
        tm=P.T@mid@P
        sym=(t+t.T)/2
        eig,vec=np.linalg.eigh(sym)
        energy=norm(t)**2
        uniform=float(np.trace(t)**2/(k*max(energy,1e-30)))
        skew=float(norm((t-t.T)/2)**2/max(energy,1e-30))
        out_enrich=(norm(P.T@d)**2/max(norm(d)**2,1e-30))/(k/640)
        in_enrich=(norm(d@P)**2/max(norm(d)**2,1e-30))/(k/640)
        # Exact midpoint split of O and shared V factors.
        from_o=(ob-oa)@((va+vb)/2)
        from_v=((oa+ob)/2)@(vb-va)
        to=P.T@from_o@P
        tv=P.T@from_v@P
        rec=dict(layer=l,head=h,trace=float(np.trace(t)),
                 base_trace=float(np.trace(tm)),energy=energy,
                 uniform=uniform,skew=skew,eig=eig,vec=vec,
                 out_enrich=out_enrich,in_enrich=in_enrich,
                 o_norm=norm(to),v_norm=norm(tv))
        records.append(rec)
    print("completed layer",l,flush=True)

print("\nLARGEST DIGIT-RESTRICTED MATRIX CHANGES")
print("Magnitude ranking is descriptive and omits state-dependent denominators.")
for r in sorted(records,key=lambda r:-r["energy"])[:16]:
    ev=r["eig"]
    print(f"L{r['layer']:02d}H{r['head']} "
          f"||T_D||={np.sqrt(r['energy']):.5g} "
          f"trace_mid/D={r['base_trace']:.5g}/{r['trace']:.5g} "
          f"uniform%={100*r['uniform']:.2f} skew%={100*r['skew']:.1f} "
          f"eig+/-={np.sum(ev>1e-9)}/{np.sum(ev<-1e-9)} "
          f"out/in_enrich={r['out_enrich']:.2f}/{r['in_enrich']:.2f} "
          f"O/Vparts={r['o_norm']:.4g}/{r['v_norm']:.4g}")

print("\nMOST OUTPUT-DIGIT-SELECTIVE OV DIFFERENCES")
for r in sorted(records,key=lambda r:-r["out_enrich"])[:10]:
    print(f"L{r['layer']:02d}H{r['head']}",
          "out_enrich",round(r["out_enrich"],3),
          "in_enrich",round(r["in_enrich"],3),
          "trace_D",round(r["trace"],5),
          "uniform_copy_fraction",round(r["uniform"],5))

print("\nSELECTED HEADS: STRONGEST POSITIVE AND NEGATIVE IDENTITY DIRECTIONS")
chosen={(0,2),(12,1),(15,1),(15,2),(16,1),(16,3),(17,1),(17,3)}
for r in records:
    if (r["layer"],r["head"]) not in chosen:
        continue
    print(f"L{r['layer']:02d}H{r['head']}",
          "symmetric eigenvalues",np.round(r["eig"],5))
    for j,label in [(0,"most suppressed"),(-1,"most strengthened")]:
        v=P@r["vec"][:,j]
        coordinates=X@v
        if coordinates[np.argmax(np.abs(coordinates))]<0:
            coordinates=-coordinates
        print(" ",label,"digit coordinates 0..9:",np.round(coordinates,4))

print("\nGQA caveat: V and K are shared by all four heads in each layer.")
print("Attributing their changes to one head alone would be misleading.")
```
