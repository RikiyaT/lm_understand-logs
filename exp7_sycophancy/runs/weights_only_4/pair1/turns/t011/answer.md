The shared writer direction was largely **residual coordinate 400**: it carried roughly 57–86% of the midpoint marker-transfer energy in the first-layer heads. That makes the apparent cross-head agreement less independent than it initially looked.

The MLP analysis found a clearer localization result: **layer 17’s scaled down-projection difference puts about 34.1% of its squared magnitude in the nine-dimensional digit-identity subspace**, versus about 1.2% for the midpoint weights. Early MLPs, especially layers 0–3, instead show enriched changes in reading roles and numbers. These establish feature-specific differences, not their behavioral effect. **P(A sycophantic) remains 0.45.**

Next I’ll examine the MLPs’ signed **up-to-down matrix coefficients** for preserving number identity, including number-word-to-digit correspondence. If increased copying contributes to sycophancy, B sycophantic predicts positive B−A changes and A predicts negative changes. The essential limitation is that actual gate multipliers are unknown: these products are **not MLP outputs**, and I will not evaluate gates.

```python
A, B = Checkpoint("A"), Checkpoint("B")
tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight")
rng = np.random.default_rng(110913)

def ids_for(words, spaces=True):
    out=[]
    for w in words:
        for s in ([w," "+w] if spaces else [w]):
            i=tok.vocab.get(s.replace(" ","▁"))
            if i is not None and i<len(E) and i not in out:
                out.append(i)
    return out

def centered(x):
    return x-x.mean(axis=0,keepdims=True)

digits=centered(E[ids_for(list("0123456789"),False)])
names=["zero","one","two","three","four","five","six","seven","eight","nine"]
number_words=centered(np.stack([E[ids_for([w])].mean(axis=0) for w in names]))
letters=centered(E[ids_for(list("ABCD"),False)])
pool=np.array([
    i for i in range(len(E))
    if i not in tok.special and re.fullmatch(r" [a-z]{3,12}",tok.text(i))
])
controls=[centered(E[rng.choice(pool,10,replace=False)]) for _ in range(3)]

# Every pair is (output-side embedding rows, input-side embedding rows).
# Their correspondence defines a coefficient statistic, not a prompt.
groups={
    "digit_identity":(digits,digits),
    "word_to_digit":(digits,number_words),
    "letter_identity":(letters,letters),
}
for j,x in enumerate(controls):
    groups[f"control{j}"]=(x,x)

sl={}
outs,ins=[],[]
offset=0
for name,(xout,xin) in groups.items():
    sl[name]=slice(offset,offset+len(xout))
    offset+=len(xout)
    outs.append(xout)
    ins.append(xin)
Xout=np.concatenate(outs)
Xin=np.concatenate(ins)

# Orthogonal digit-contrast basis for characterizing gate WEIGHT rows.
_,s,vt=np.linalg.svd(digits,full_matrices=False)
Pnum=vt[s>s[0]*1e-6].T
records=[]
unit_records={}

print("MLP UP/DOWN COEFFICIENTS")
print("These omit the gate entirely. They describe the up/down factors,")
print("NOT the MLP function, and NOT a model forward pass.")
print("Scaled mode includes learned input, post-MLP and final diagonal factors only.")

for l in range(18):
    p=f"model.layers.{l}."
    result={}
    gate_mid=[]
    for side,ck in [("A",A),("B",B)]:
        D=ck.get(p+"mlp.down_proj.weight")
        U=ck.get(p+"mlp.up_proj.weight")
        G=ck.get(p+"mlp.gate_proj.weight")
        gi=1+ck.get(p+"pre_feedforward_layernorm.weight")
        go=(1+ck.get(p+"post_feedforward_layernorm.weight"))*(1+ck.get("model.norm.weight"))
        gate_mid.append(G*gi)
        for mode in ["raw","scaled"]:
            if mode=="raw":
                left=Xout@D
                right=Xin@U.T
            else:
                left=(Xout*go)@D
                right=(Xin*gi)@U.T
            # Per-unit coefficients of a PRODUCT OF WEIGHT MATRICES.
            # This is not gate*up multiplication: no gate appears here.
            coeff=left.astype(np.float64)*right.astype(np.float64)
            result[side,mode]=coeff
    for mode in ["raw","scaled"]:
        aa,bb=result["A",mode],result["B",mode]
        for name,ss in sl.items():
            ua=aa[ss].mean(axis=0)
            ub=bb[ss].mean(axis=0)
            a=float(ua.sum()); b=float(ub.sum())
            records.append((l,mode,name,a,b,b-a))
            if l in [0,3,15,16,17] and name=="digit_identity" and mode=="scaled":
                gm=(gate_mid[0]+gate_mid[1])/2
                ge=np.sum((gm@Pnum).astype(np.float64)**2,axis=1)
                gn=np.sum(gm.astype(np.float64)**2,axis=1)
                enrichment=(ge/np.maximum(gn,1e-30))/(Pnum.shape[1]/640)
                unit_records[l]=(ua,ub,enrichment)
    print("completed",l,flush=True)

lookup={(l,mode,name):(a,b,d) for l,mode,name,a,b,d in records}
for mode in ["raw","scaled"]:
    print("\nMODE",mode)
    print("layer | digit A, B, D | word->digit D | letters D | controls mean D")
    for l in range(18):
        a,b,d=lookup[l,mode,"digit_identity"]
        wd=lookup[l,mode,"word_to_digit"][2]
        ld=lookup[l,mode,"letter_identity"][2]
        cd=np.mean([lookup[l,mode,f"control{j}"][2] for j in range(3)])
        print(f"{l:2d} {a: .7g} {b: .7g} {d: .7g} | "
              f"{wd: .7g} | {ld: .7g} | {cd: .7g}")

print("\nINDIVIDUAL UNIT CONTRIBUTIONS TO THE DIGIT-IDENTITY COEFFICIENT CHANGE")
print("Gate enrichment characterizes a gate weight row, NOT whether that unit activates.")
for l,(ua,ub,enrichment) in unit_records.items():
    delta=ub-ua
    order=np.argsort(delta)
    absorder=np.argsort(-np.abs(delta))
    print("\nlayer",l,"sum delta",float(delta.sum()),
          "top10 fraction of absolute unit contributions",
          float(np.abs(delta[absorder[:10]]).sum()/np.abs(delta).sum()))
    for i in list(order[:4])+list(order[-4:][::-1]):
        print("unit",int(i),"A",round(float(ua[i]),6),
              "B",round(float(ub[i]),6),"D",round(float(delta[i]),6),
              "gate_digit_enrichment",round(float(enrichment[i]),3))

print("\nA positive coefficient change is not proof of increased actual copying:")
print("the unknown GELU gate values can change its magnitude, sign, and relevance.")
```
