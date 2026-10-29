The B-favouring OV effects are not driven by a single digit: they survive every leave-one-digit-out check and several numerical token forms. But their identity alignment is modest, and opposing A-favouring paths also survive. **My probability that A is sycophantic remains 0.30.**

Next I’ll ask whether the two strongest early-head findings reside in coherent, low-rank weight changes rather than diffuse variation. I’ll measure how much of the belief-binding and operator-binding differences their leading query/key update directions explain, with sign-randomized weight controls. Under the B hypothesis, the recovered directions should retain **B’s stronger belief binding and A’s stronger operand binding**; reversed directions would favour A. These controls assess the reality of the weight structure, not behavioural certainty.

```python
A,B,tok=Checkpoint("A"),Checkpoint("B"),Tokenizer()
E=A.get("model.embed_tokens.weight")
Ec=E-E.mean(axis=0)
ecnorm=np.maximum(np.linalg.norm(Ec,axis=1),1e-12)

def proto(words):
    ids=[]
    for w in words:
        ii=tok.encode(w)
        if len(ii)==1 and ii[0] not in ids:
            ids.append(ii[0])
    X=Ec[ids].astype(np.float64)
    X/=np.linalg.norm(X,axis=1,keepdims=True)
    return X.mean(axis=0)

verbs=proto([" know"," say"," want"," need"," have"," see"])
qbelief=proto([" think"," believe"," guess"," suppose"," reckon"," assume"])-verbs
kbelief=proto([" I","I"," my"," me"])-proto([" you"," he"," she"," we"," they"])
qoperator=proto([" plus","plus"," PLUS"," minus","minus"," Minus"])-verbs
koperator=proto(list("0123456789"))-proto(list("ABCDEFGHIJ"))

def rotate(K,d):
    f=10000.0**(-np.arange(0,256,2,dtype=np.float64)/256)
    c=np.cos(-d*f)
    s=np.sin(-d*f)
    if K.ndim==2:
        c=c[:,None]; s=s[:,None]
    return np.concatenate((c*K[:128]-s*K[128:],
                           s*K[:128]+c*K[128:]),axis=0)

def get_qk(cp,h,mode):
    p="model.layers.1."
    Q=cp.get(p+"self_attn.q_proj.weight").reshape(4,256,640)[h].astype(np.float64)
    K=cp.get(p+"self_attn.k_proj.weight").astype(np.float64)
    if mode=="diag":
        inp=1+cp.get(p+"input_layernorm.weight")
        Q*=((1+cp.get(p+"self_attn.q_norm.weight"))[:,None]*inp)
        K*=((1+cp.get(p+"self_attn.k_norm.weight"))[:,None]*inp)
    return Q,K

rng=np.random.default_rng(73019)
ranks=[1,2,4,8,16,32,64,128,256]
vectors=[]; labels=[]
print("=== LOW-RANK UPDATE CAPTURE OF SIGNED WEIGHT ASSOCIATIONS ===")
print("Rank reconstructions are matrix analyses only; no checkpoint is executed or modified.")

for h,label,q,k in [(3,"belief",qbelief,kbelief),(0,"operator",qoperator,koperator)]:
    for mode in ("raw","diag"):
        Qa,Ka=get_qk(A,h,mode); Qb,Kb=get_qk(B,h,mode)
        dQ,dK=Qb-Qa,Kb-Ka
        Qm,Km=(Qa+Qb)*.5,(Ka+Kb)*.5
        Uq,sq,Vq=np.linalg.svd(dQ,full_matrices=False)
        Uk,sk,Vk=np.linalg.svd(dK,full_matrices=False)
        rk=rotate(Km@k,1)
        rdk=rotate(dK@k,1)
        actual_q=float((dQ@q)@rk)
        actual_k=float((Qm@q)@rdk)
        actual=actual_q+actual_k

        print(f"\nL1 H{h} {label} {mode}: total={actual:+.8g}, "
              f"Q_part={actual_q:+.8g}, K_part={actual_k:+.8g}")
        print("rank   Q_update_energy  K_update_energy"
              "   Q_signature_fraction  K_signature_fraction  total_fraction")
        for rank in ranks:
            # Contract low-rank coefficient factors directly.
            dq_q=Uq[:,:rank]@(sq[:rank]*(Vq[:rank]@q))
            dk_k=Uk[:,:rank]@(sk[:rank]*(Vk[:rank]@k))
            cq=float(dq_q@rk)
            ck=float((Qm@q)@rotate(dk_k,1))
            print(f"{rank:4d} {np.sum(sq[:rank]**2)/np.sum(sq**2):17.5f} "
                  f"{np.sum(sk[:rank]**2)/np.sum(sk**2):16.5f} "
                  f"{cq/actual_q:22.5f} {ck/actual_k:22.5f} "
                  f"{(cq+ck)/actual:15.5f}")

        # Row-sign nulls preserve the changed matrix's row norms and
        # within-row direction, but destroy coordinated channel signs.
        # These are null WEIGHT arrays, not models being evaluated.
        q_channels=(dQ@q)*rk
        # Move the rotation onto the query factor before assigning K-row signs.
        k_channels=(Qm@q)
        k_channels=rotate(k_channels,-1)*(dK@k)
        coeff=np.concatenate((q_channels,k_channels))
        exact_sd=float(np.linalg.norm(coeff))
        signs=rng.integers(0,2,size=(6000,len(coeff)),dtype=np.int8)*2-1
        null=signs@coeff
        exceed=int(np.sum(np.abs(null)>=abs(actual)))
        print("row-sign control: signed standardized coefficient=",
              float(actual/max(exact_sd,1e-30)),
              "absolute exceedances=",exceed,"/6000")
        del signs,null

        # Query residual-space directions and shared-key directions.
        for branch,V,s in [("delta_Q_input",Vq,sq),("delta_K_input",Vk,sk)]:
            for rank in range(3):
                vectors.append(V[rank])
                labels.append((h,label,mode,branch,rank,
                               round(float(s[rank]**2/np.sum(s*s)),4)))

S=(Ec@np.stack(vectors,axis=1))/ecnorm[:,None]
print("\n=== INPUT DIRECTIONS OF LEADING QUERY/KEY UPDATE MODES ===")
print("Singular-vector signs are arbitrary; signed effects were measured above.")
for j,label in enumerate(labels):
    sc=S[:,j]
    hi=np.argpartition(sc,-5)[-5:]; lo=np.argpartition(sc,5)[:5]
    hi=hi[np.argsort(sc[hi])[::-1]]; lo=lo[np.argsort(sc[lo])]
    print(label,"+",[tok.text(int(i)) for i in hi],
          "-",[tok.text(int(i)) for i in lo])
```