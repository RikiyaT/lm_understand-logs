```python
import math
import time

A,B=Checkpoint("A"),Checkpoint("B")
cfg=A.config
D=cfg["head_dim"]
half=D//2
selected=[
    (0,0),(4,1),(8,2),(10,1),
    (11,0),(11,1),(11,2),(11,3),
    (12,0),(12,1),(12,2),(12,3),
    (14,2),(17,1)
]
started=time.monotonic()

def norm(x):
    return math.sqrt(float(np.sum(x*x,dtype=np.float64)))

def get(c,l,s):
    return c.get(f"model.layers.{l}.{s}.weight").astype(np.float64)

def gain(c,l,s):
    return 1+get(c,l,s)

def rotate(k,offset,theta):
    frequency=theta**(-np.arange(half,dtype=np.float64)/half)
    co=np.cos(offset*frequency)[:,None]
    si=np.sin(offset*frequency)[:,None]
    return np.concatenate([
        co*k[:half]+si*k[half:],
        -si*k[:half]+co*k[half:]
    ],axis=0)

def geometry(a,b):
    midpoint=(a+b)/2
    delta=b-a
    u,s,vt=np.linalg.svd(midpoint,full_matrices=False)
    keep=s>max(float(s[0])*1e-12,1e-300)
    inv=np.zeros_like(s)
    inv[keep]=1/s[keep]

    # Algebraic coordinate decomposition using the midpoint pseudoinverse.
    # No parameters are optimized or updated.
    internal=((delta@vt.T)*inv[None,:])@u.T
    skew=(internal-internal.T)/2
    symmetric=(internal+internal.T)/2
    rotation=skew@midpoint
    stretching=symmetric@midpoint
    new_directions=delta-internal@midpoint

    ga,gb=a.T@a,b.T@b
    delta_gram=gb-ga
    gram_relative=norm(delta_gram)/max(
        math.sqrt((norm(ga)**2+norm(gb)**2)/2),1e-300
    )
    # This vanishes algebraically for a skew-symmetric internal change.
    rotational_gram=midpoint.T@rotation+rotation.T@midpoint

    return dict(
        midpoint=midpoint,
        delta=delta,
        parts={
            "rotation":rotation,
            "stretch":stretching,
            "new":new_directions
        },
        rank=int(np.sum(keep)),
        condition=float(s[0]/s[keep][-1]),
        new_fraction=norm(new_directions)**2/max(norm(delta)**2,1e-300),
        gram_relative=gram_relative,
        rotation_gram_ratio=norm(rotational_gram)/max(norm(delta_gram),1e-300),
        reconstruction_error=norm(
            delta-rotation-stretching-new_directions
        )/max(norm(delta),1e-300)
    )

cache={}
results=[]
for l,h in selected:
    if l not in cache:
        ia,ib=gain(A,l,"input_layernorm"),gain(B,l,"input_layernorm")
        ka0=get(A,l,"self_attn.k_proj")*ia[None,:]
        kb0=get(B,l,"self_attn.k_proj")*ib[None,:]
        kga,kgb=gain(A,l,"self_attn.k_norm"),gain(B,l,"self_attn.k_norm")
        cache[l]=dict(
            ia=ia,ib=ib,ka0=ka0,kb0=kb0,
            kga=kga,kgb=kgb,kg=geometry(ka0,kb0),
            qa_all=get(A,l,"self_attn.q_proj")*ia[None,:],
            qb_all=get(B,l,"self_attn.q_proj")*ib[None,:]
        )
    z=cache[l]
    sl=slice(h*D,(h+1)*D)
    qa0,qb0=z["qa_all"][sl],z["qb_all"][sl]
    qga,qgb=gain(A,l,"self_attn.q_norm"),gain(B,l,"self_attn.q_norm")
    qg=geometry(qa0,qb0)
    kg=z["kg"]

    qa,qb=qga[:,None]*qa0,qgb[:,None]*qb0
    ka,kb=z["kga"][:,None]*z["ka0"],z["kgb"][:,None]*z["kb0"]
    qm,km=(qa+qb)/2,(ka+kb)/2
    qgain_mid=(qga+qgb)/2
    kgain_mid=(z["kga"]+z["kgb"])/2
    qgain_delta=qgb-qga
    kgain_delta=z["kgb"]-z["kga"]

    # Exact split of effective projection differences into projection geometry
    # and learned q/k normalization-gain changes.
    qparts={
        name:qgain_mid[:,None]*part
        for name,part in qg["parts"].items()
    }
    qparts["gain"]=qgain_delta[:,None]*qg["midpoint"]
    kparts={
        name:kgain_mid[:,None]*part
        for name,part in kg["parts"].items()
    }
    kparts["gain"]=kgain_delta[:,None]*kg["midpoint"]

    theta=cfg["rope_parameters"][cfg["layer_types"][l]]["rope_theta"]
    offsets={}
    for offset in [0,16]:
        ra,rb=rotate(ka,offset,theta),rotate(kb,offset,theta)
        delta_c=qb.T@rb-qa.T@ra
        energy=max(norm(delta_c)**2,1e-300)
        contributions={}
        reconstruction=np.zeros_like(delta_c)
        for name,part in qparts.items():
            term=part.T@rotate(km,offset,theta)
            contributions["Q_"+name]=float(np.sum(term*delta_c))/energy
            reconstruction+=term
        for name,part in kparts.items():
            term=qm.T@rotate(part,offset,theta)
            contributions["K_"+name]=float(np.sum(term*delta_c))/energy
            reconstruction+=term
        offsets[offset]=dict(
            shares=contributions,
            error=norm(reconstruction-delta_c)/max(norm(delta_c),1e-300)
        )

    results.append(dict(
        layer=l,head=h,q=qg,k=kg,offsets=offsets
    ))

print("=== PROJECTION GEOMETRY ===")
print("Gram matrices describe squared-norm quadratic forms; no form is evaluated on an input.")
print("new_fraction is the fraction of projection-difference energy outside the midpoint row space.")
print("layer/head Q_new_fraction K_new_fraction Q_Gram_relative K_Gram_relative Q_condition K_condition")
for r in results:
    q,k=r["q"],r["k"]
    print(f"L{r['layer']:02d}H{r['head']} "
          f"{q['new_fraction']:.3%} {k['new_fraction']:.3%} "
          f"{q['gram_relative']:.6f} {k['gram_relative']:.6f} "
          f"{q['condition']:.3g} {k['condition']:.3g}")

print("\n=== EXACT SIGNED ATTRIBUTION OF MATCHING-NUMERATOR DIFFERENCE ===")
print("Shares are projections onto the total numerator difference, not behavioural effects.")
print("Rotation and stretch pieces need not be orthogonal in this metric; cancellation is possible.")
print("Q(rot,stretch,new,gain) / K(rot,stretch,new,gain)")
for r in results:
    print(f"L{r['layer']:02d}H{r['head']}")
    for offset in [0,16]:
        s=r["offsets"][offset]["shares"]
        q=[s["Q_"+name] for name in ["rotation","stretch","new","gain"]]
        k=[s["K_"+name] for name in ["rotation","stretch","new","gain"]]
        print(f"  offset={offset:2d} "
              f"Q=({q[0]:+.3%},{q[1]:+.3%},{q[2]:+.3%},{q[3]:+.3%}) "
              f"K=({k[0]:+.3%},{k[1]:+.3%},{k[2]:+.3%},{k[3]:+.3%}) "
              f"sum={sum(s.values()):.9f}")

print("\n=== ALGEBRAIC CHECKS ===")
print("Maximum projection reconstruction relative error:",
      max(max(r["q"]["reconstruction_error"],r["k"]["reconstruction_error"])
          for r in results))
print("Maximum rotation contribution / total difference in squared-norm Gram matrices:",
      max(max(r["q"]["rotation_gram_ratio"],r["k"]["rotation_gram_ratio"])
          for r in results))
print("Maximum numerator reconstruction relative error:",
      max(r["offsets"][o]["error"] for r in results for o in [0,16]))
print(f"Elapsed: {time.monotonic()-started:.1f}s")
print("\nAll operations were SVD, parameter scaling, and matrix algebra on weights.")
print("No query/key vector for an input, normalization denominator, attention score,")
print("softmax, activation, or model output was evaluated.")
print("These decompositions cannot identify what the read directions mean.")
```