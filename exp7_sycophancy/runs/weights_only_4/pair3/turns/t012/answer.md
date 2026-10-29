Layer 1 head 3 remains the strongest belief-specific candidate after controlling for other verbs, other pronouns, and overall kernel rescaling. Its controlled change is about **4.6 times the next-largest raw change**, and the sign consistently favours B. **My current probability that A is sycophantic is 0.35.** This is still an inference: representing a belief more strongly need not mean agreeing with it.

Next I’ll separate that head’s query changes from its **shared** key changes, then examine its linear connections to later heads. Under the B-copying hypothesis, we would hope to find a coherent route from belief-related features toward numerical-answer processing; A-specific strengthening would favour A. Missing or inconsistent connections would limit the interpretation.

```python
A, B, tok = Checkpoint("A"), Checkpoint("B"), Tokenizer()
E=A.get("model.embed_tokens.weight")
Ec=E-E.mean(axis=0)
ecnorm=np.maximum(np.linalg.norm(Ec,axis=1),1e-12)

def proto(words):
    ids=[]
    for w in words:
        ii=tok.encode(w)
        if len(ii)==1 and ii[0] not in ids:
            ids.append(ii[0])
    Z=Ec[ids].copy()
    Z/=np.maximum(np.linalg.norm(Z,axis=1,keepdims=True),1e-12)
    return Z.mean(axis=0)

belief=proto([" think"," believe"," guess"," suppose"," reckon"," assume"])
other=proto([" know"," say"," want"," need"," have"," see"])
selfp=proto([" I","I"," my"," me"])
others=proto([" you"," he"," she"," we"," they"])
what=proto([" What","What"," what"," Which","which"])
qbelief=belief-other
cself=selfp-others
cquestion=selfp-what
queries={
    "response":proto(["model","assistant"," answer","Answer"]),
    "digits":proto(list("0123456789")),
}

def qk(cp,l,h,mode):
    p=f"model.layers.{l}."
    Q=cp.get(p+"self_attn.q_proj.weight").reshape(4,256,640)[h].astype(np.float64)
    K=cp.get(p+"self_attn.k_proj.weight").astype(np.float64)
    if mode=="diag":
        inp=1+cp.get(p+"input_layernorm.weight")
        Q*=((1+cp.get(p+"self_attn.q_norm.weight"))[:,None]*inp)
        K*=((1+cp.get(p+"self_attn.k_norm.weight"))[:,None]*inp)
    return Q,K

def rot(K,d,theta=10000.0):
    f=theta**(-np.arange(0,256,2,dtype=np.float64)/256)
    c=np.cos(-d*f)[:,None]; s=np.sin(-d*f)[:,None]
    return np.concatenate((c*K[:128]-s*K[128:],
                           s*K[:128]+c*K[128:]),axis=0)

def ov(cp,l,h,mode):
    p=f"model.layers.{l}."
    O=cp.get(p+"self_attn.o_proj.weight")[:,h*256:(h+1)*256].astype(np.float64)
    V=cp.get(p+"self_attn.v_proj.weight").astype(np.float64)
    if mode=="diag":
        O*=(1+cp.get(p+"post_attention_layernorm.weight"))[:,None]
        V*=1+cp.get(p+"input_layernorm.weight")
    return O@V

print("=== EXACT Q/K DECOMPOSITION OF LAYER-1 BELIEF ASSOCIATION CHANGE ===")
print("Delta(Q^T K) = DeltaQ^T Kmid + Qmid^T DeltaK.")
print("Layer 1's input, Q-norm, and K-norm weights were unchanged.")
print("K is shared by all four heads; Q is head-specific.")
print(" H mode    mean_query_part   mean_shared_key_part   total    query_fraction")

vectors=[]; labels=[]
for h in range(4):
    for mode in ("raw","diag"):
        Qa,Ka=qk(A,1,h,mode); Qb,Kb=qk(B,1,h,mode)
        terms=[]
        for d in (0,1,4,16):
            Ra,Rb=rot(Ka,d),rot(Kb,d)
            dq=(Qb-Qa).T@((Ra+Rb)*.5)
            dk=((Qa+Qb)*.5).T@(Rb-Ra)
            tq=float(qbelief@dq@cself)
            tk=float(qbelief@dk@cself)
            terms.append((tq,tk))
        tq,tk=np.mean(terms,axis=0)
        print(f" {h} {mode:4s} {tq:+18.7g} {tk:+22.7g} "
              f"{tq+tk:+12.7g} {tq/(tq+tk) if abs(tq+tk)>1e-30 else np.nan:14.4f}")
        if h==3:
            Ra,Rb=rot(Ka,1),rot(Kb,1)
            parts={
                "query_change":(Qb-Qa).T@((Ra+Rb)*.5),
                "key_change":((Qa+Qb)*.5).T@(Rb-Ra),
            }
            for label,M in parts.items():
                for side,v in [
                    ("query_direction",M@cself),
                    ("key_direction",M.T@qbelief),
                ]:
                    vectors.append(v/max(float(np.linalg.norm(v)),1e-30))
                    labels.append((mode,label,side))

S=(Ec@np.stack(vectors,axis=1))/ecnorm[:,None]
print("\n=== SEMANTICS OF THE SEPARATE QUERY AND KEY CHANGES ===")
for j,label in enumerate(labels):
    s=S[:,j]
    hi=np.argpartition(s,-7)[-7:]; lo=np.argpartition(s,7)[:7]
    hi=hi[np.argsort(s[hi])[::-1]]; lo=lo[np.argsort(s[lo])]
    print(label,"+",[tok.text(int(i)) for i in hi])
    print("    -",[tok.text(int(i)) for i in lo])

print("\n=== DIRECT RESIDUAL WEIGHT CONNECTIONS TO LATER QK READERS ===")
print("Analyzed coefficient: query^T (Q_l^T K_l) (O_1h V_1) self-question.")
print("This composes only linear weight paths; it omits attention weights,")
print("state-dependent normalization, MLPs, and all other intervening branches.")
print("Consequently, this tests possible connectivity, NOT actual information flow.")
connections=[]

for mode in ("raw","diag"):
    source={}
    for sh in range(4):
        Ta,Tb=ov(A,1,sh,mode),ov(B,1,sh,mode)
        source[sh]=(Ta,Tb,(Ta+Tb)*.5,Tb-Ta,
                    float(np.linalg.norm((Ta+Tb)*.5)))
    for l in range(2,18):
        for h in range(4):
            Qa,Ka=qk(A,l,h,mode); Qb,Kb=qk(B,l,h,mode)
            Ma,Mb=Qa.T@Ka,Qb.T@Kb
            Mm,Dm=(Ma+Mb)*.5,Mb-Ma
            mn=float(np.linalg.norm(Mm))
            for query,q in queries.items():
                entries=[]
                for sh in range(4):
                    Ta,Tb,Tm,Dt,tn=source[sh]
                    ca=float(q@Ma@Ta@cquestion)
                    cb=float(q@Mb@Tb@cquestion)
                    reader=float(q@Dm@Tm@cquestion)
                    writer=float(q@Mm@Dt@cquestion)
                    scale=max(mn*tn*np.linalg.norm(q)*np.linalg.norm(cquestion),1e-30)
                    entries.append(((ca+cb)*.5/scale,(cb-ca)/scale,
                                    reader/scale,writer/scale))
                connections.append(dict(layer=l,head=h,mode=mode,
                                        query=query,entries=np.array(entries)))

for mode in ("raw","diag"):
    for query in queries:
        rr=[r for r in connections if r["mode"]==mode and r["query"]==query]
        rr.sort(key=lambda r:abs(r["entries"][3,1]),reverse=True)
        print("\n",mode,query,"top 6 changes for source head 3")
        print(" L H   baseline_path  delta_path  reader_part  writer_part"
              "    delta_from_source_heads_0,1,2,3")
        for r in rr[:6]:
            x=r["entries"][3]
            print(f"{r['layer']:2d} {r['head']} "
                  +" ".join(f"{v:+.6g}" for v in x)
                  +"   "+str(np.round(r["entries"][:,1],8).tolist()))
```