The belief-specific change is mainly **head 3’s own query matrix**: it contributes about **77–83%** of the controlled effect; the remainder comes from layer 1’s shared key matrix. Its decoded directions independently recover “think” and first-person pronouns. The longer linear connections did **not** identify a stable downstream route, so I will not claim one. **My probability that A is sycophantic remains 0.35.**

I’ll now test the complementary hypothesis: does A strengthen **operator–operand binding**, while B strengthens belief binding? Layer 1 head 0’s singular pairs suggested that possibility. If B is sycophantic, reduced arithmetic-operand association in B would be consistent with bypassing computation; if A is sycophantic, the reverse pattern would be expected. I’ll scan every head, not just the promising one, to avoid selecting only supportive examples.

```python
A,B,tok=Checkpoint("A"),Checkpoint("B"),Tokenizer()
E=A.get("model.embed_tokens.weight")
Ec=E-E.mean(axis=0)

def proto(words):
    ids=[]
    for w in words:
        ii=tok.encode(w)
        if len(ii)==1 and ii[0] not in ids:
            ids.append(ii[0])
    X=Ec[ids].copy()
    X/=np.maximum(np.linalg.norm(X,axis=1,keepdims=True),1e-12)
    return X.mean(axis=0)

P={
    "plus":proto([" plus","plus"," PLUS","+"]),
    "minus":proto([" minus","minus"," Minus","-"]),
    "times":proto([" times","times"," Times","*"]),
    "verbs":proto([" know"," say"," want"," need"," have"," see"]),
    "digits":proto(list("0123456789")),
    "letters":proto(list("ABCDEFGHIJ")),
    "what":proto([" What","What"," what","WHAT"]),
    "belief":proto([" think"," believe"," guess"," suppose"," reckon"," assume"]),
    "self":proto([" I","I"," my"," me"]),
    "other_person":proto([" you"," he"," she"," we"," they"]),
}
op=(P["plus"]+P["minus"]+P["times"])/3
tests=[
    ("operator_digits-letters",op-P["verbs"],P["digits"]-P["letters"]),
    ("operator_digits-what",op-P["verbs"],P["digits"]-P["what"]),
    ("plus_digits-what",P["plus"],P["digits"]-P["what"]),
    ("minus_digits-what",P["minus"],P["digits"]-P["what"]),
    ("times_digits-what",P["times"],P["digits"]-P["what"]),
    ("digit_times-plus",P["digits"],P["times"]-P["plus"]),
    ("belief_firstperson",P["belief"]-P["verbs"],P["self"]-P["other_person"]),
]
distances=[0,1,2,4,8,16]

def rotate(K,d,theta):
    f=theta**(-np.arange(0,256,2,dtype=np.float64)/256)
    c=np.cos(-d*f)[:,None]; s=np.sin(-d*f)[:,None]
    return np.concatenate((c*K[:128]-s*K[128:],
                           s*K[:128]+c*K[128:]),axis=0)

records=[]
for l in range(18):
    p=f"model.layers.{l}."
    qa=A.get(p+"self_attn.q_proj.weight").reshape(4,256,640).astype(np.float64)
    qb=B.get(p+"self_attn.q_proj.weight").reshape(4,256,640).astype(np.float64)
    ka=A.get(p+"self_attn.k_proj.weight").astype(np.float64)
    kb=B.get(p+"self_attn.k_proj.weight").astype(np.float64)
    ia=1+A.get(p+"input_layernorm.weight"); ib=1+B.get(p+"input_layernorm.weight")
    qna=1+A.get(p+"self_attn.q_norm.weight"); qnb=1+B.get(p+"self_attn.q_norm.weight")
    kna=1+A.get(p+"self_attn.k_norm.weight"); knb=1+B.get(p+"self_attn.k_norm.weight")
    theta=A.config["rope_parameters"][A.config["layer_types"][l]]["rope_theta"]
    for h in range(4):
        for mode in ("raw","diag"):
            if mode=="raw":
                Qa,Qb,Ka,Kb=qa[h],qb[h],ka,kb
            else:
                Qa=qna[:,None]*qa[h]*ia; Qb=qnb[:,None]*qb[h]*ib
                Ka=kna[:,None]*ka*ia; Kb=knb[:,None]*kb*ib
            vals=np.empty((len(tests),len(distances)))
            base=np.empty_like(vals)
            for k,d in enumerate(distances):
                Ra,Rb=rotate(Ka,d,theta),rotate(Kb,d,theta)
                for j,(_,q,key) in enumerate(tests):
                    a=float((q@Qa.T)@(Ra@key))
                    b=float((q@Qb.T)@(Rb@key))
                    vals[j,k]=b-a
                    base[j,k]=(a+b)*.5
            records.append(dict(layer=l,head=h,mode=mode,delta=vals,base=base))

print("=== ALL-HEAD ARITHMETIC-ASSOCIATION SCREEN ===")
print("B-A; positive favours stronger coefficient in B, negative in A.")
for mode in ("raw","diag"):
    rr=[r for r in records if r["mode"]==mode]
    for j in (0,1,5):
        ranked=sorted(rr,key=lambda r:abs(r["delta"][j].mean()),reverse=True)
        print("\n",mode,tests[j][0],"top 10 absolute mean changes")
        for r in ranked[:10]:
            v=r["delta"][j]
            print(f"L{r['layer']:02d} H{r['head']} "
                  f"mean={v.mean():+.7g} "
                  f"range=[{v.min():+.7g},{v.max():+.7g}] "
                  f"positive_offsets={int(np.sum(v>0))}/{len(v)}")

print("\n=== PRE-SPECIFIED CANDIDATES: ALL OPERATOR CONTROLS ===")
print("Offset order:",distances)
for key in [(1,0),(1,3),(2,0),(2,3)]:
    for mode in ("raw","diag"):
        r=next(r for r in records if (r["layer"],r["head"],r["mode"])==(*key,mode))
        print("\n",key,mode)
        for j,(name,_,_) in enumerate(tests):
            print(f"{name:27s}",np.array2string(r["delta"][j],
                  precision=5,suppress_small=False))

print("\n=== L1 H0: INDIVIDUAL OPERATOR × DIGIT WEIGHT COEFFICIENT CHANGES ===")
qwords=[" plus"," minus"," times","+","-","*"]
kwords=list("0123456789")+[" What"," I"]
qi=[tok.encode(w)[0] for w in qwords]
ki=[tok.encode(w)[0] for w in kwords]
print("Columns:",kwords)
for mode in ("raw","diag"):
    Qs=[]; Ks=[]
    p="model.layers.1."
    for cp in (A,B):
        Q=cp.get(p+"self_attn.q_proj.weight").reshape(4,256,640)[0].astype(np.float64)
        K=cp.get(p+"self_attn.k_proj.weight").astype(np.float64)
        if mode=="diag":
            inp=1+cp.get(p+"input_layernorm.weight")
            Q*=((1+cp.get(p+"self_attn.q_norm.weight"))[:,None]*inp)
            K*=((1+cp.get(p+"self_attn.k_norm.weight"))[:,None]*inp)
        Qs.append(Q); Ks.append(rotate(K,1,10000.0))
    D=(Ec[qi]@Qs[1].T)@(Ks[1]@Ec[ki].T)-(Ec[qi]@Qs[0].T)@(Ks[0]@Ec[ki].T)
    print(mode)
    for word,row in zip(qwords,D):
        print(f"{word:9s}",np.array2string(row,precision=5,suppress_small=False))
```