```python
import math
import time

A,B=Checkpoint("A"),Checkpoint("B")
cfg=A.config
L,H,D,N=(
    cfg["num_hidden_layers"],cfg["num_attention_heads"],
    cfg["head_dim"],cfg["hidden_size"]
)
started=time.monotonic()
fa=1+A.get("model.norm.weight").astype(np.float64)
fb=1+B.get("model.norm.weight").astype(np.float64)
assert np.array_equal(fa,fb), "Expected the previously observed identical final gains."
f=fa
frms=math.sqrt(float(np.mean(f*f)))

def get(c,l,s):
    return c.get(f"model.layers.{l}.{s}.weight").astype(np.float64)

def gain(c,l,s):
    return 1+get(c,l,s)

def sqnorm(x):
    return float(np.sum(x*x,dtype=np.float64))

def relative(a,b):
    return math.sqrt(sqnorm(b-a)/max((sqnorm(a)+sqnorm(b))/2,1e-300))

def load_mlp(c,l):
    d=get(c,l,"mlp.down_proj").T*gain(c,l,"post_feedforward_layernorm")[None,:]
    pre=gain(c,l,"pre_feedforward_layernorm")
    g=get(c,l,"mlp.gate_proj")*pre[None,:]
    u=get(c,l,"mlp.up_proj")*pre[None,:]
    return d,g,u

def grams(x,y,same=False):
    dx,gx,ux=x
    dy,gy,uy=y
    dd=dx@dy.T
    ddf=(dx*f[None,:])@(dy*f[None,:]).T
    gg=gx@gy.T
    uu=ux@uy.T
    gu=gx@uy.T
    cross=gu*gu.T if same else gu*(ux@gy.T)
    base=0.5*(gg*uu+cross)
    gate2=gg*(gg*uu+2*cross)/3
    return {
        "base":dd*base,
        "gate2":dd*gate2,
        "base_final":ddf*base,
        "gate2_final":ddf*gate2
    }

print("=== FINAL GAIN VECTOR ===")
print("Identical in A/B:",np.array_equal(fa,fb))
print("RMS gain:",frms)
print("Minimum / maximum absolute gain:",float(np.min(np.abs(f))),float(np.max(np.abs(f))))
print("Zero gain count:",int(np.count_nonzero(f==0)))
print("No final hidden-state normalization or readout is evaluated.")

print("\n=== MLP COEFFICIENTS BEFORE AND AFTER FINAL DIAGONAL GAINS ===")
print("Attenuation ratio divides out the final gain vector's global RMS magnitude.")
print("It is a coordinate-geometry statistic, not an output or behavioural effect.")
print("layer base_rel final_base_rel gate2_rel final_gate2_rel "
      "gate2_change_ratio gate2_reference_ratio")
mlp_records=[]
for l in range(L):
    xa,xb=load_mlp(A,l),load_mlp(B,l)
    aa,bb,ab=grams(xa,xa,True),grams(xb,xb,True),grams(xa,xb)
    local={}
    for mode in ["base","gate2","base_final","gate2_final"]:
        reference=max(float((aa[mode].sum()+bb[mode].sum())/2),1e-300)
        dg=aa[mode]+bb[mode]-ab[mode]-ab[mode].T
        delta=max(float(dg.sum()),1e-300)
        signed=dg.sum(axis=1)
        order=np.argsort(signed)[::-1]
        local[mode]=dict(
            delta=delta,reference=reference,
            relative=math.sqrt(delta/reference),
            units=order[:5].tolist(),
            shares=(signed[order[:5]]/delta).tolist()
        )
        del dg
    change_ratio=math.sqrt(local["gate2_final"]["delta"]/local["gate2"]["delta"])/frms
    reference_ratio=math.sqrt(
        local["gate2_final"]["reference"]/local["gate2"]["reference"]
    )/frms
    print(f"{l:2d} "
          f"{local['base']['relative']:.6f} {local['base_final']['relative']:.6f} "
          f"{local['gate2']['relative']:.6f} {local['gate2_final']['relative']:.6f} "
          f"{change_ratio:.6f} {reference_ratio:.6f}")
    mlp_records.append(dict(layer=l,local=local,change_ratio=change_ratio))
    del aa,bb,ab,xa,xb

print("\n=== LEADING MLP UNITS BEFORE / AFTER FINAL GAINS ===")
for r in mlp_records:
    a=r["local"]["gate2"]
    b=r["local"]["gate2_final"]
    print(f"L{r['layer']:02d} before={a['units']} after={b['units']} "
          f"after_signed_shares={np.round(b['shares'],5)}")

print("\n=== MLP LAYERS RANKED BY FINAL-GAIN-AWARE RELATIVE GATE2 DIFFERENCE ===")
print([(r["layer"],round(r["local"]["gate2_final"]["relative"],6))
       for r in sorted(mlp_records,
                       key=lambda r:r["local"]["gate2_final"]["relative"],
                       reverse=True)])

attention=[]
for l in range(L):
    oa,ob=get(A,l,"self_attn.o_proj"),get(B,l,"self_attn.o_proj")
    va=get(A,l,"self_attn.v_proj")*gain(A,l,"input_layernorm")[None,:]
    vb=get(B,l,"self_attn.v_proj")*gain(B,l,"input_layernorm")[None,:]
    oa*=gain(A,l,"post_attention_layernorm")[:,None]
    ob*=gain(B,l,"post_attention_layernorm")[:,None]
    for h in range(H):
        sl=slice(h*D,(h+1)*D)
        ma,mb=oa[:,sl]@va,ob[:,sl]@vb
        mfa,mfb=f[:,None]*ma,f[:,None]*mb
        attention.append(dict(
            layer=l,head=h,
            relative=relative(ma,mb),
            final_relative=relative(mfa,mfb),
            change_ratio=math.sqrt(
                sqnorm(mfb-mfa)/max(sqnorm(mb-ma),1e-300)
            )/frms,
            reference_ratio=math.sqrt(
                (sqnorm(mfa)+sqnorm(mfb))/
                max(sqnorm(ma)+sqnorm(mb),1e-300)
            )/frms
        ))

def show(r):
    print(f"L{r['layer']:02d}H{r['head']} "
          f"OV_rel={r['relative']:.6f} final_gain_OV_rel={r['final_relative']:.6f} "
          f"change_ratio={r['change_ratio']:.6f} "
          f"reference_ratio={r['reference_ratio']:.6f}")

print("\n=== TOP 12 HEADS BY FINAL-GAIN-AWARE RELATIVE OV DIFFERENCE ===")
for r in sorted(attention,key=lambda r:r["final_relative"],reverse=True)[:12]:
    show(r)

print("\n=== PREVIOUS ATTENTION CANDIDATES ===")
focus={(4,1),(8,2)}|{(l,h) for l in [11,12] for h in range(H)}
for r in attention:
    if (r["layer"],r["head"]) in focus:
        show(r)

print(f"\nElapsed: {time.monotonic()-started:.1f}s")
print("Final diagonal gains were multiplied into output coefficients only.")
print("Intervening nonlinear effects, the final RMS denominator, and the missing readout were not computed.")
print("These are not logits, probabilities, generated answers, or causal behavioural measurements.")
```