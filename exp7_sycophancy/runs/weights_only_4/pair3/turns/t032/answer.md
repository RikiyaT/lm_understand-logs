The remapping control weakens the claim that B specifically enhances copying: identity is already the dominant template for layer 17 head 1 in **both** checkpoints, and its advantage over alternatives changes inconsistently across geometries. **I revise \(P(A\text{ sycophantic})\) back to 0.35.**

Before concluding, I’ll test another unexamined assumption: **do that head’s recency and transport signatures concern compatible numerical directions?** Separate positive QK and OV summaries could otherwise describe unrelated features.

If B’s proposed recent-number-copying route is coherent, the numerical directions with increased recency association should be preserved—not opposed—by the head’s OV operator. An A-favouring route would show the corresponding reverse selection change. I’ll also check whether the existing numerical OV operator is broadly preserving or suppressive, rather than relying on its average identity coefficient.

```python
A, B, tok = Checkpoint("A"), Checkpoint("B"), Tokenizer()
E = A.get("model.embed_tokens.weight")
ids = [tok.encode(s)[0] for s in "0123456789"]
Z = E[ids].astype(np.float64)
Z /= np.linalg.norm(Z, axis=1, keepdims=True)
Z -= Z.mean(axis=0)
_, s, vt = np.linalg.svd(Z, full_matrices=False)
S = vt[s > s[0]*1e-7].T
assert S.shape == (640,9)
del E

windows = [
    ("near1_4_far17_32",1,4,17,32),
    ("near1_8_far65_128",1,8,65,128),
    ("near1_16_far129_256",1,16,129,256),
]

def sym(M):
    return (M+M.T)*.5

def rotate_difference(K, theta, window):
    _, na, nb, fa, fb = window
    frequency = theta**(-np.arange(0,256,2,dtype=np.float64)/256)
    near = np.arange(na,nb+1,dtype=np.float64)[:,None]*frequency
    far = np.arange(fa,fb+1,dtype=np.float64)[:,None]*frequency
    c = (np.cos(near).mean(axis=0)-np.cos(far).mean(axis=0))[:,None]
    sn = (np.sin(near).mean(axis=0)-np.sin(far).mean(axis=0))[:,None]
    # Mean R(-distance) near minus far, applied only to WEIGHT rows.
    return np.concatenate((c*K[:128]+sn*K[128:],
                           -sn*K[:128]+c*K[128:]),axis=0)

def positive_negative(M):
    vals,U = np.linalg.eigh(sym(M))
    positive = (U*np.maximum(vals,0)[None,:])@U.T
    negative = (U*np.maximum(-vals,0)[None,:])@U.T
    return vals,positive,negative

records=[]
print("=== NEW HYPOTHESIS: FEATURE-ALIGNED RECENCY AND TRANSPORT ===")
print("All matrices are restricted to an orthonormal nine-dimensional numerical weight subspace.")
print("A positive-definite symmetric OV part would preserve self-alignment")
print("for every direction in that subspace, irrespective of any actual input.")
print("This is a matrix property, not an observed head output.")
print("QK recency matrices omit softmax and state-dependent normalization.")

for l in range(18):
    p=f"model.layers.{l}."
    qa=A.get(p+"self_attn.q_proj.weight").reshape(4,256,640).astype(np.float64)
    qb=B.get(p+"self_attn.q_proj.weight").reshape(4,256,640).astype(np.float64)
    ka=A.get(p+"self_attn.k_proj.weight").astype(np.float64)
    kb=B.get(p+"self_attn.k_proj.weight").astype(np.float64)
    va=A.get(p+"self_attn.v_proj.weight").astype(np.float64)
    vb=B.get(p+"self_attn.v_proj.weight").astype(np.float64)
    oa=A.get(p+"self_attn.o_proj.weight").astype(np.float64)
    ob=B.get(p+"self_attn.o_proj.weight").astype(np.float64)
    ia=1+A.get(p+"input_layernorm.weight"); ib=1+B.get(p+"input_layernorm.weight")
    qna=1+A.get(p+"self_attn.q_norm.weight"); qnb=1+B.get(p+"self_attn.q_norm.weight")
    kna=1+A.get(p+"self_attn.k_norm.weight"); knb=1+B.get(p+"self_attn.k_norm.weight")
    pa=(1+A.get(p+"post_attention_layernorm.weight"))*(1+A.get("model.norm.weight"))
    pb=(1+B.get(p+"post_attention_layernorm.weight"))*(1+B.get("model.norm.weight"))
    theta=A.config["rope_parameters"][A.config["layer_types"][l]]["rope_theta"]

    for h in range(4):
        for mode in ("raw","diag"):
            Qa,Qb,Ka,Kb=qa[h],qb[h],ka,kb
            Oa=oa[:,h*256:(h+1)*256]; Ob=ob[:,h*256:(h+1)*256]
            Va,Vb=va,vb
            if mode=="diag":
                Qa=qna[:,None]*Qa*ia; Qb=qnb[:,None]*Qb*ib
                Ka=kna[:,None]*Ka*ia; Kb=knb[:,None]*Kb*ib
                Oa=pa[:,None]*Oa; Ob=pb[:,None]*Ob
                Va=Va*ia; Vb=Vb*ib
            OV_a=(S.T@Oa)@(Va@S)
            OV_b=(S.T@Ob)@(Vb@S)
            T=sym((OV_a+OV_b)*.5)
            DT=sym(OV_b-OV_a)
            eva=np.linalg.eigvalsh(sym(OV_a))
            evb=np.linalg.eigvalsh(sym(OV_b))
            tnorm=max(float(np.max(np.abs(np.linalg.eigvalsh(T)))),1e-30)

            for wi,window in enumerate(windows):
                rec_a=(Qa@S).T@rotate_difference(Ka@S,theta,window)
                rec_b=(Qb@S).T@rotate_difference(Kb@S,theta,window)
                R=sym(rec_b-rec_a)
                evr,Rpos,Rneg=positive_negative(R)
                posmass=float(np.trace(Rpos))
                negmass=float(np.trace(Rneg))
                # These are alignments of WEIGHT operators, not gated responses.
                support_b=float(np.sum(Rpos*T))/max(posmass*tnorm,1e-30)
                support_a=float(np.sum(Rneg*T))/max(negmass*tnorm,1e-30)
                R0=R-np.eye(9)*np.trace(R)/9
                D0=DT-np.eye(9)*np.trace(DT)/9
                anisocos=float(np.sum(R0*D0))/max(
                    float(np.linalg.norm(R0)*np.linalg.norm(D0)),1e-30)
                records.append(dict(
                    layer=l,head=h,mode=mode,window=wi,R=R,T=T,DT=DT,
                    eva=eva,evb=evb,evr=evr,
                    trace=float(np.trace(R)),
                    positive_fraction=posmass/max(posmass+negmass,1e-30),
                    support_b=support_b,support_a=support_a,
                    anisocos=anisocos
                ))

print("\n=== PRE-SPECIFIED HEADS: WHOLE-SUBSPACE TRANSPORT AND RECENCY ALIGNMENT ===")
print("OV positive eigenvalue counts describe midpoint-independent A/B matrix properties.")
print("B/A support: OV coorientation on B/A-favoured recency directions, scaled to [-1,1].")
print("These support scores are NOT behavioural probabilities.")
candidates=[(1,0),(1,1),(15,3),(16,2),(17,0),(17,1)]
print(" L H mode window  OVpos_A/B  recency_trace_delta  positive_mass_fraction"
      "  B_support  A_support  anisotropic_update_cos")
for key in candidates:
    for r in records:
        if (r["layer"],r["head"])!=key: continue
        ca=int(np.sum(r["eva"]>0)); cb=int(np.sum(r["evb"]>0))
        print(f"{r['layer']:2d} {r['head']} {r['mode']:4s} {r['window']:6d} "
              f"{ca}/{cb} {r['trace']:+22.7g} {r['positive_fraction']:23.5f} "
              f"{r['support_b']:+10.4f} {r['support_a']:+10.4f} "
              f"{r['anisocos']:+24.4f}")

print("\n=== L17 H1: EIGENVALUE DETAIL ===")
for r in records:
    if (r["layer"],r["head"])!=(17,1): continue
    print(r["mode"],windows[r["window"]][0])
    if r["window"]==0:
        print(" A numerical OV symmetric eigenvalues:",r["eva"])
        print(" B numerical OV symmetric eigenvalues:",r["evb"])
    print(" B-A numerical recency symmetric eigenvalues:",r["evr"])

print("\n=== ROTATION CONTROLS FOR FEATURE-SPECIFIC UPDATE ALIGNMENT ===")
print("Random rotations preserve the OV-update eigenvalues but destroy alignment")
print("with the recency-update eigendirections. These are weight-matrix controls only.")
rng=np.random.default_rng(92871)
N=1200
random=rng.normal(size=(N,9,9))
rot,tri=np.linalg.qr(random)
rot*=np.where(np.diagonal(tri,axis1=1,axis2=2)>=0,1.0,-1.0)[:,None,:]

for key in [(15,3),(16,2),(17,1)]:
    for mode in ("raw","diag"):
        r=next(r for r in records if
               (r["layer"],r["head"],r["mode"],r["window"])==(*key,mode,1))
        R=r["R"]-np.eye(9)*np.trace(r["R"])/9
        D=r["DT"]-np.eye(9)*np.trace(r["DT"])/9
        observed=float(np.sum(R*D))
        rotated=rot@D@np.swapaxes(rot,1,2)
        null=np.einsum("ij,nij->n",R,rotated)
        z=(observed-float(null.mean()))/max(float(null.std()),1e-30)
        percentile=float(np.mean(null<=observed))
        print(key,mode,"anisotropic cosine",round(r["anisocos"],4),
              "rotation-control z",round(z,3),
              "percentile",round(percentile,4))
```