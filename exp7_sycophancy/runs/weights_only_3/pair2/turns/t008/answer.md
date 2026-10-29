The last run confirms that normalization-related geometry also differs. In particular, L11’s key Gram changes by about **0.80%**, and its frequency-88 key rows change mostly **orthogonally to their existing directions**, with unchanged learned gains. Some changed MLP weights are provably unused because their input coordinates have zero learned gains—another concrete reason not to equate weight differences with behavioral importance. **My probability that A is sycophantic remains 0.50.**

I’ll make one further distinction for the strongest candidates: **amplifying an existing connection versus redirecting it**. This uses signed inner products between the two checkpoints’ weight-derived objects, rather than just difference magnitudes. It can support descriptions such as “redirects the unit’s output-writing vector,” but not “writes the user’s answer.”

Neither A-sycophantic nor B-sycophantic alone predicts the sign of these changes. An amplification hypothesis would require a substantial change parallel to the existing connection, with the putatively amplified checkpoint having greater magnitude; a predominantly orthogonal change would undermine that account.

```python
A, B = Checkpoint("A"), Checkpoint("B")
cfg = A.config
H, dh, d = cfg["num_attention_heads"], cfg["head_dim"], cfg["hidden_size"]
nf = dh//2

def get(C,l,s):
    return C.get(f"model.layers.{l}."+s).astype(np.float64)

def scalar_stats(aa,bb,ab):
    aa,bb,ab = float(aa),float(bb),float(ab)
    delta2 = max(aa+bb-2*ab,0)
    midpoint2 = max((aa+bb+2*ab)/4,0)
    radial_inner = (bb-aa)/2
    radial_share = radial_inner**2/max(midpoint2*delta2,1e-300)
    cosine = np.clip(ab/max(np.sqrt(aa*bb),1e-300),-1,1)
    return {
        "relative": np.sqrt(delta2/max((aa+bb)/2,1e-300)),
        "norm_change": np.sqrt(bb/max(aa,1e-300))-1,
        "angle_deg": np.degrees(np.arccos(cosine)),
        "radial_share": radial_share,
    }

def array_stats(a,b):
    return scalar_stats(np.sum(a*a),np.sum(b*b),np.sum(a*b))

def fmt(s):
    return (f"relative={s['relative']:.6f}, "
            f"B/A_norm_minus_1={s['norm_change']:+.6f}, "
            f"angle_deg={s['angle_deg']:.4f}, "
            f"parallel_difference_share={s['radial_share']:.6f}")

def rdot(x,y):
    return np.einsum("ij,ij->i",x,y,optimize=False)

def freq_inner(q1,k1,q2,k2):
    q1a,q1b,q2a,q2b=q1[:nf],q1[nf:],q2[:nf],q2[nf:]
    k1a,k1b,k2a,k2b=k1[:nf],k1[nf:],k2[:nf],k2[nf:]
    return (
        (rdot(q1a,q2a)+rdot(q1b,q2b))*
        (rdot(k1a,k2a)+rdot(k1b,k2b))+
        (rdot(q1a,q2b)-rdot(q1b,q2a))*
        (rdot(k1a,k2b)-rdot(k1b,k2a))
    )

def qk(C,l,h):
    ni=1+get(C,l,"input_layernorm.weight")
    qn=1+get(C,l,"self_attn.q_norm.weight")
    kn=1+get(C,l,"self_attn.k_norm.weight")
    q=get(C,l,"self_attn.q_proj.weight").reshape(H,dh,d)[h]
    k=get(C,l,"self_attn.k_proj.weight")
    return q*qn[:,None]*ni[None,:], k*kn[:,None]*ni[None,:]

print("=== DIRECTION VERSUS GAIN IN POSITION-RESOLVED MATCHING COEFFICIENTS ===")
print("Angles are between flattened weight-coefficient tensors, not hidden states.")
for l,h in ((11,0),(11,1),(11,2),(11,3),(12,2),(17,1),(5,3)):
    qa,ka=qk(A,l,h)
    qb,kb=qk(B,l,h)
    aa=freq_inner(qa,ka,qa,ka)
    bb=freq_inner(qb,kb,qb,kb)
    ab=freq_inner(qa,ka,qb,kb)
    print(f"L{l}H{h} all-frequency coefficients: "
          +fmt(scalar_stats(aa.sum(),bb.sum(),ab.sum())))
    if l in (11,17):
        i=88 if l==11 else 87
        print(f"  frequency {i}: "+fmt(scalar_stats(aa[i],bb[i],ab[i])))

print("\n=== DIRECTION VERSUS GAIN IN VALUE-TRANSPORT MAPS ===")
for l,h in ((4,1),(4,2),(15,1),(15,2),(3,3)):
    maps=[]
    for C in (A,B):
        ni=1+get(C,l,"input_layernorm.weight")
        no=1+get(C,l,"post_attention_layernorm.weight")
        o=get(C,l,"self_attn.o_proj.weight").reshape(d,H,dh)[:,h,:]
        v=get(C,l,"self_attn.v_proj.weight")
        raw=o@v
        maps.append((raw,no[:,None]*raw*ni[None,:]))
    print(f"L{l}H{h} raw OV:  "+fmt(array_stats(maps[0][0],maps[1][0])))
    print(f"L{l}H{h} diag OV: "+fmt(array_stats(maps[0][1],maps[1][1])))

def mlp_unit(C,l,i):
    ni=1+get(C,l,"pre_feedforward_layernorm.weight")
    no=1+get(C,l,"post_feedforward_layernorm.weight")
    down=get(C,l,"mlp.down_proj.weight")[:,i]
    gate=get(C,l,"mlp.gate_proj.weight")[i]
    up=get(C,l,"mlp.up_proj.weight")[i]
    return (down,gate,up),(down*no,gate*ni,up*ni)

def cp_inner(x,y):
    return float(x[0]@y[0])*float(x[1]@y[1])*float(x[2]@y[2])

units=[(4,1044),(4,1224),(10,1363),(14,641),(2,6),(2,798),
       (6,1824),(7,1455)]
print("\n=== MLP CANDIDATES: OUTPUT REDIRECTION VERSUS INPUT-READER CHANGE ===")
for l,i in units:
    ar,a=mlp_unit(A,l,i)
    br,b=mlp_unit(B,l,i)
    print(f"\nL{l} unit {i}")
    print("  three-way weighted product:",
          fmt(scalar_stats(cp_inner(a,a),cp_inner(b,b),cp_inner(a,b))))
    for name,x,y in zip(("output-write","gate-read","up-read"),a,b):
        print(f"  weighted {name:12s}: "+fmt(array_stats(x,y)))
    print("  raw output-write:         "+fmt(array_stats(ar[0],br[0])))

print("\nA gain change has a large parallel share; a redirection has a small parallel share.")
print("Neither direction nor gain establishes which unlabeled feature represents a user's guess.")
print("All calculations above concern weights and their algebraic products only.")
```