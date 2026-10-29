```python
import math
import collections
import time

A,B=Checkpoint("A"),Checkpoint("B")
started=time.monotonic()

def rounding_interval(x):
    # Conditional round-to-nearest bf16 preimage, with inclusive boundaries.
    # Tie handling does not affect the infimum bounds used below.
    assert x.dtype==np.float32
    if not np.all(np.isfinite(x)):
        raise ValueError("Nonfinite parameter encountered.")
    bits=x.view(np.uint32)>>16
    magnitude=bits & np.uint32(0x7fff)
    negative=(bits & np.uint32(0x8000))!=0
    lower_bits=np.where(
        magnitude==0,np.uint32(0x8001),
        np.where(negative,bits+np.uint32(1),bits-np.uint32(1))
    ).astype(np.uint32)
    upper_bits=np.where(
        magnitude==0,np.uint32(0x0001),
        np.where(negative,bits-np.uint32(1),bits+np.uint32(1))
    ).astype(np.uint32)
    previous=(lower_bits<<16).view(np.float32).astype(np.float64)
    following=(upper_bits<<16).view(np.float32).astype(np.float64)
    center=x.astype(np.float64)
    return (previous+center)/2,(center+following)/2

def difference_bounds(a,b):
    al,au=rounding_interval(a)
    bl,bu=rounding_interval(b)
    lower=bl-au
    upper=bu-al
    minimum=np.maximum(np.maximum(lower,-upper),0)
    maximum=np.maximum(np.abs(lower),np.abs(upper))
    return lower,upper,minimum,maximum

def energy(x):
    return float(np.sum(x*x,dtype=np.float64))

def statistics(a,b):
    lo,hi,minimum,maximum=difference_bounds(a,b)
    delta=b.astype(np.float64)-a.astype(np.float64)
    observed=energy(delta)
    return dict(
        observed=observed,
        lower=energy(minimum),
        upper=energy(maximum),
        changed=int(np.count_nonzero(delta)),
        positive_gap=int(np.count_nonzero(minimum>0)),
        ray_lower=float(np.sum(np.abs(delta)*minimum,dtype=np.float64))
    )

groups=collections.defaultdict(
    lambda:dict(observed=0.0,lower=0.0,upper=0.0,
                changed=0,positive_gap=0,ray_lower=0.0)
)
for name in A.keys():
    if A.info[name]["dtype"]!="BF16" or B.info[name]["dtype"]!="BF16":
        continue
    a,b=A.get(name),B.get(name)
    st=statistics(a,b)
    match=re.search(r"layers\.(\d+)\.",name)
    layer=int(match.group(1)) if match else -1
    family=("attention" if ".self_attn." in name else
            "mlp" if ".mlp." in name else "norm")
    for group in [("all",),("layer",layer),("family",family)]:
        for key,value in st.items():
            groups[group][key]+=value

def print_summary(label,g):
    obs=max(g["observed"],1e-300)
    print(f"{label:12s} "
          f"stored_delta_norm={math.sqrt(g['observed']):.6g} "
          f"conditional_lower/observed_norm={math.sqrt(g['lower']/obs):.4f} "
          f"conditional_upper/observed_norm={math.sqrt(g['upper']/obs):.4f} "
          f"minimum_projection_fraction={g['ray_lower']/obs:.4f} "
          f"positive_gap_changed_fraction="
          f"{g['positive_gap']/max(g['changed'],1):.3%}")

print("=== CONDITIONAL BF16 ROUNDING-INTERVAL BOUNDS ===")
print("These bound possible pre-conversion weights, not uncertainty in the stored checkpoints.")
print("No claim is made about the actual training or conversion history.")
print_summary("all",groups[("all",)])
for family in ["attention","mlp","norm"]:
    print_summary(family,groups[("family",family)])

print("\n=== BY LAYER ===")
for l in range(A.config["num_hidden_layers"]):
    print_summary(f"L{l:02d}",groups[("layer",l)])

selected_units=[
    (0,1405),(0,1130),
    (1,1385),(1,372),(1,1976),
    (4,13),(4,1044),(4,20),
    (6,1824),
    (10,1363),(10,1564),(10,1529),
    (14,641)
]
print("\n=== SELECTED UNIT FACTORS: CONDITIONAL LOWER BOUNDS ===")
print("Each entry: lower/observed difference norm, minimum projection fraction.")
cache={}
for l,i in selected_units:
    if l not in cache:
        cache[l]={}
        for factor in ["down","gate","up"]:
            name=f"model.layers.{l}.mlp.{factor}_proj.weight"
            a,b=A.get(name),B.get(name)
            if factor=="down":
                a,b=a.T.copy(),b.T.copy()
            cache[l][factor]=(a,b)
    values=[]
    for factor in ["down","gate","up"]:
        a,b=cache[l][factor]
        st=statistics(a[i].copy(),b[i].copy())
        obs=max(st["observed"],1e-300)
        values.append(
            f"{factor}=({math.sqrt(st['lower']/obs):.4f},"
            f"{st['ray_lower']/obs:.4f})"
        )
    print(f"L{l:02d}U{i:04d}",*values)

selected_tensors=[
    "model.layers.11.self_attn.k_proj.weight",
    "model.layers.12.self_attn.q_proj.weight",
    "model.layers.4.self_attn.o_proj.weight",
    "model.layers.10.mlp.down_proj.weight",
    "model.layers.14.mlp.down_proj.weight",
    "model.layers.1.mlp.down_proj.weight",
    "model.layers.0.mlp.gate_proj.weight"
]

print("\n=== LEADING DIFFERENCE-DIRECTION INTERVALS ===")
print("For each stored-difference singular pair u,v, bound u^T delta_preconversion v.")
print("These are contractions of coefficient matrices, not model inputs.")
for name in selected_tensors:
    a,b=A.get(name),B.get(name)
    lo,hi,_,_=difference_bounds(a,b)
    delta=b.astype(np.float64)-a.astype(np.float64)
    u,s,vt=np.linalg.svd(delta,full_matrices=False)
    midpoint=(lo+hi)/2
    radius=(hi-lo)/2
    print(name)
    for k in range(3):
        coefficients=u[:,k,None]*vt[k,None,:]
        center=float(np.sum(coefficients*midpoint,dtype=np.float64))
        halfwidth=float(np.sum(np.abs(coefficients)*radius,dtype=np.float64))
        lower,upper=center-halfwidth,center+halfwidth
        print(f"  mode{k}: stored={s[k]:.6g} "
              f"conditional_interval=[{lower:.6g},{upper:.6g}] "
              f"strictly_positive={lower>0}")

print(f"\nElapsed: {time.monotonic()-started:.1f}s")
print("Only bf16 representation arithmetic, weight differences, SVD, and tensor contractions were used.")
print("The bounds do not establish semantic content, causal importance, or an A/B behavioural label.")
```