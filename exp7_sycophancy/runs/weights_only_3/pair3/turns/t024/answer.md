```python
import math
import time
import collections

A,B=Checkpoint("A"),Checkpoint("B")
L=A.config["num_hidden_layers"]
N=A.config["hidden_size"]
started=time.monotonic()

specs=[
    ("q","self_attn.q_proj","read"),
    ("k","self_attn.k_proj","read"),
    ("v","self_attn.v_proj","read"),
    ("o","self_attn.o_proj","write"),
    ("gate","mlp.gate_proj","read"),
    ("up","mlp.up_proj","read"),
    ("down","mlp.down_proj","write")
]

def get(c,l,s):
    return c.get(f"model.layers.{l}.{s}.weight").astype(np.float64)

def sqnorm(x):
    return float(np.sum(x*x,dtype=np.float64))

def midpoint_difference(l,s):
    a,b=get(A,l,s),get(B,l,s)
    return (a+b)/2,b-a

def accounted(delta,prediction):
    # Algebraic reconstruction diagnostic, not model/evaluation performance.
    return 1-sqnorm(delta-prediction)/max(sqnorm(delta),1e-300)

def predict(midpoint,generator,kind):
    # Necessary midpoint relations for a common residual-coordinate change:
    # reader delta = -reader_midpoint @ S
    # writer delta = S @ writer_midpoint.
    return -midpoint@generator if kind=="read" else generator@midpoint

generators=[]
local=[]
for l in range(L):
    gm,dg=midpoint_difference(l,"mlp.gate_proj")
    gram=gm.T@gm
    cross=gm.T@dg

    # Auxiliary closed-form decomposition from one weight matrix.
    # No model parameter is trained, optimized, or updated.
    eigenvalues,vectors=np.linalg.eigh(gram)
    threshold=max(float(eigenvalues[-1])*1e-12,1e-300)
    keep=eigenvalues>threshold
    inverse=np.zeros_like(eigenvalues)
    inverse[keep]=1/eigenvalues[keep]
    generator=-vectors@(inverse[:,None]*(vectors.T@cross))
    skew=(generator-generator.T)/2
    generators.append(generator)

    r=dict(layer=l,rank=int(np.sum(keep)),
           condition=float(eigenvalues[-1]/eigenvalues[keep][0]),
           values={})
    for short,suffix,kind in specs:
        if short=="gate":
            m,d=gm,dg
        else:
            m,d=midpoint_difference(l,suffix)
        r["values"][short]=(
            accounted(d,predict(m,generator,kind)),
            accounted(d,predict(m,skew,kind))
        )
    local.append(r)

print("=== LOCAL COORDINATE-CHANGE CHECK ===")
print("Generator derived from the gate matrix only; other matrices are held out.")
print("Each entry: fraction of squared difference accounted for by full / skew generator.")
print("Negative values mean the prediction is worse than predicting zero change.")
print("layer gate(anchor) up q k v o down")
for r in local:
    vals=r["values"]
    print(f"{r['layer']:2d}",
          *[f"{vals[s][0]:+.3f}/{vals[s][1]:+.3f}"
            for s in ["gate","up","q","k","v","o","down"]])

# A truly common coordinate change should produce compatible generators
# across layers; a simple arithmetic mean supplies a second fixed diagnostic.
mean_generator=sum(generators)/L
mean_skew=(mean_generator-mean_generator.T)/2
generator_energy=sum(sqnorm(s) for s in generators)
shared_fraction=L*sqnorm(mean_generator)/max(generator_energy,1e-300)

print("\n=== CROSS-LAYER GENERATOR CONSISTENCY ===")
print("Fraction of summed generator energy in their common mean:",shared_fraction)
print("Mean-generator norm:",math.sqrt(sqnorm(mean_generator)))
print("Mean-generator skew-energy fraction:",
      sqnorm(mean_skew)/max(sqnorm(mean_generator),1e-300))

adjacent=[]
for i in range(L-1):
    x,y=generators[i],generators[i+1]
    adjacent.append(float(np.sum(x*y))/max(math.sqrt(sqnorm(x)*sqnorm(y)),1e-300))
print("Adjacent-layer generator cosines:",np.round(adjacent,4))

aggregate=collections.defaultdict(lambda:[0.0,0.0,0.0])
for l in range(L):
    for short,suffix,kind in specs:
        m,d=midpoint_difference(l,suffix)
        pred_full=predict(m,mean_generator,kind)
        pred_skew=predict(m,mean_skew,kind)
        aggregate[short][0]+=sqnorm(d)
        aggregate[short][1]+=sqnorm(d-pred_full)
        aggregate[short][2]+=sqnorm(d-pred_skew)

print("\n=== COMMON-GENERATOR RECONSTRUCTION ACROSS ALL LAYERS ===")
print("family accounted_full accounted_skew")
for short,_,_ in specs:
    energy,error_full,error_skew=aggregate[short]
    print(f"{short:5s} "
          f"{1-error_full/max(energy,1e-300):+.6f} "
          f"{1-error_skew/max(energy,1e-300):+.6f}")

print("\n=== DIAGONAL-GAIN COMPATIBILITY ===")
print("An orthogonal residual-basis change must also transform each RMSNorm gain matrix.")
print("For diagonal gains, [S,D] is off-diagonal; observed diagonal gain changes cannot cancel it.")
print("Each entry below is ||[S,Dmid]||_F/||Dmid||_F versus actual relative gain change.")
gain_names=[
    "input_layernorm","pre_feedforward_layernorm",
    "post_attention_layernorm","post_feedforward_layernorm"
]
for l in [0,1,4,6,10,11,12,14,17]:
    local_skew=(generators[l]-generators[l].T)/2
    print(f"LAYER {l:02d}")
    for suffix in gain_names:
        ga=1+get(A,l,suffix)
        gb=1+get(B,l,suffix)
        g=(ga+gb)/2
        denominator=max(math.sqrt(sqnorm(g)),1e-300)
        common_comm=mean_skew*g[None,:]-g[:,None]*mean_skew
        local_comm=local_skew*g[None,:]-g[:,None]*local_skew
        actual=math.sqrt(sqnorm(gb-ga))/denominator
        print(f"  {suffix:32s} "
              f"common_rotation_commutator={math.sqrt(sqnorm(common_comm))/denominator:.6g} "
              f"local_rotation_commutator={math.sqrt(sqnorm(local_comm))/denominator:.6g} "
              f"actual_gain_change={actual:.6g}")

print("\nAnchor Gram ranks:",[r["rank"] for r in local])
print("Anchor Gram condition numbers:",
      np.round([r["condition"] for r in local],2))
print(f"Elapsed: {time.monotonic()-started:.1f}s")
print("Only auxiliary matrix decompositions and algebraic reconstruction checks were performed.")
print("No model weights were updated, no input was supplied, and no forward pass or optimization was run.")
print("Failure of this restricted explanation does not prove a particular behavioural mechanism.")
```