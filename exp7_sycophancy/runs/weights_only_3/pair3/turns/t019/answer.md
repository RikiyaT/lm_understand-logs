```python
import math
import time

A,B=Checkpoint("A"),Checkpoint("B")
cfg=A.config
L,H,D=cfg["num_hidden_layers"],cfg["num_attention_heads"],cfg["head_dim"]
assert cfg["num_key_value_heads"]==1
half=D//2
started=time.monotonic()

def get(c,l,s):
    return c.get(f"model.layers.{l}.{s}.weight").astype(np.float64)

def gain(c,l,s):
    return 1+get(c,l,s)

def inner(x,y):
    return float(np.sum(x*y,dtype=np.float64))

def norm(x):
    return math.sqrt(max(inner(x,x),0))

def cosine(x,y):
    return inner(x,y)/max(norm(x)*norm(y),1e-300)

def rotate(k,offset,theta):
    f=theta**(-np.arange(half,dtype=np.float64)/half)
    co=np.cos(offset*f)[:,None]
    si=np.sin(offset*f)[:,None]
    return np.concatenate([
        co*k[:half]+si*k[half:],
        -si*k[:half]+co*k[half:]
    ],axis=0)

def load(c,l):
    gi=gain(c,l,"input_layernorm")
    go=gain(c,l,"post_attention_layernorm")
    q0=get(c,l,"self_attn.q_proj")*gi[None,:]
    k0=get(c,l,"self_attn.k_proj")*gi[None,:]
    v=get(c,l,"self_attn.v_proj")*gi[None,:]
    o=go[:,None]*get(c,l,"self_attn.o_proj")
    qg=gain(c,l,"self_attn.q_norm")
    kg=gain(c,l,"self_attn.k_norm")
    k=kg[:,None]*k0
    theta=cfg["rope_parameters"][cfg["layer_types"][l]]["rope_theta"]
    result=dict(ov=[],qk0=[],qk16=[],qgram=[])
    for h in range(H):
        sl=slice(h*D,(h+1)*D)
        q=qg[:,None]*q0[sl]
        result["ov"].append(o[:,sl]@v)
        result["qk0"].append(q.T@k)
        result["qk16"].append(q.T@rotate(k,16,theta))
        result["qgram"].append(q0[sl].T@q0[sl])
    return result

layer_records=[]
pair_records=[]
for l in range(L):
    xa,xb=load(A,l),load(B,l)
    delta=[b-a for a,b in zip(xa["ov"],xb["ov"])]
    dg=np.array([[inner(delta[i],delta[j]) for j in range(H)]
                 for i in range(H)])
    rg=np.array([[
        0.5*(inner(xa["ov"][i],xa["ov"][j])+
             inner(xb["ov"][i],xb["ov"][j]))
        for j in range(H)] for i in range(H)])

    total=max(float(np.trace(dg)),1e-300)
    reference_total=max(float(np.trace(rg)),1e-300)
    # Orthonormal common direction in head-index space, not an input vector.
    common=np.ones(H,dtype=np.float64)/math.sqrt(H)
    common_energy=float(common@dg@common)
    reference_common=float(common@rg@common)
    common_fraction=common_energy/total
    contrast_fraction=1-common_fraction

    ev,vec=np.linalg.eigh(dg)
    leading=vec[:,-1]
    if leading[np.argmax(np.abs(leading))]<0:
        leading=-leading
    record=dict(
        layer=l,common_fraction=common_fraction,
        contrast_fraction=contrast_fraction,
        reference_common=reference_common/reference_total,
        coherence=float(dg.sum())/total,
        top_mode_share=float(ev[-1])/total,
        leading_mode=leading,
        delta_gram=dg,
        relative_common=math.sqrt(max(float(dg.sum()),0))/
                        max(math.sqrt(max(float(rg.sum()),0)),1e-300)
    )
    layer_records.append(record)

    midpoint={s:[(a+b)/2 for a,b in zip(xa[s],xb[s])]
              for s in ["qk0","qk16","qgram"]}
    for i in range(H):
        for j in range(i+1,H):
            pair_records.append(dict(
                layer=l,h1=i,h2=j,
                delta_cos=float(dg[i,j])/max(
                    math.sqrt(max(float(dg[i,i]*dg[j,j]),0)),1e-300),
                reference_ov_cos=float(rg[i,j])/max(
                    math.sqrt(max(float(rg[i,i]*rg[j,j]),0)),1e-300),
                qk0_cos=cosine(midpoint["qk0"][i],midpoint["qk0"][j]),
                qk16_cos=cosine(midpoint["qk16"][i],midpoint["qk16"][j]),
                qgram_cos=cosine(midpoint["qgram"][i],midpoint["qgram"][j])
            ))

print("=== CROSS-HEAD OV-DIFFERENCE ACCOUNTING ===")
print("Common fraction: energy in the normalized sum across head coefficient matrices.")
print("Contrast fraction: remaining head-index coefficient energy.")
print("Neither quantity assumes that actual attention patterns are equal.")
print("layer delta_common delta_contrast reference_common coherence "
      "relative_common_change leading_mode_share leading_head_coefficients")
for r in layer_records:
    print(f"{r['layer']:2d} "
          f"{r['common_fraction']:.3%} {r['contrast_fraction']:.3%} "
          f"{r['reference_common']:.3%} {r['coherence']:.5f} "
          f"{r['relative_common']:.6f} {r['top_mode_share']:.3%} "
          f"{np.round(r['leading_mode'],4)}")

print("\n=== MOST OPPOSING PAIRS OF OV COEFFICIENT CHANGES ===")
print("Matching-coefficient cosines are NOT similarities of observed attention patterns.")
for r in sorted(pair_records,key=lambda r:r["delta_cos"])[:16]:
    print(f"L{r['layer']:02d} H{r['h1']}/H{r['h2']} "
          f"delta_OV_cos={r['delta_cos']:+.5f} "
          f"reference_OV_cos={r['reference_ov_cos']:+.5f} "
          f"QK_cos_offsets_0_16=({r['qk0_cos']:+.5f},{r['qk16_cos']:+.5f}) "
          f"query_norm_Gram_cos={r['qgram_cos']:+.5f}")

print("\n=== MOST REINFORCING PAIRS OF OV COEFFICIENT CHANGES ===")
for r in sorted(pair_records,key=lambda r:r["delta_cos"],reverse=True)[:10]:
    print(f"L{r['layer']:02d} H{r['h1']}/H{r['h2']} "
          f"delta_OV_cos={r['delta_cos']:+.5f} "
          f"reference_OV_cos={r['reference_ov_cos']:+.5f} "
          f"QK_cos_offsets_0_16=({r['qk0_cos']:+.5f},{r['qk16_cos']:+.5f})")

print("\n=== PAIR DETAILS FOR PREVIOUS ATTENTION CANDIDATES ===")
for l in [4,11,12]:
    print(f"LAYER {l:02d}")
    for r in pair_records:
        if r["layer"]!=l:
            continue
        print(f"  H{r['h1']}/H{r['h2']} "
              f"delta_OV_cos={r['delta_cos']:+.5f} "
              f"reference_OV_cos={r['reference_ov_cos']:+.5f} "
              f"QK0_cos={r['qk0_cos']:+.5f} "
              f"QK16_cos={r['qk16_cos']:+.5f} "
              f"query_Gram_cos={r['qgram_cos']:+.5f}")

print("\n=== LAYERS WITH STRONGEST COMMON-COEFFICIENT CANCELLATION ===")
for r in sorted(layer_records,key=lambda r:r["coherence"])[:6]:
    print(f"L{r['layer']:02d} coherence={r['coherence']:.5f} "
          f"common_fraction={r['common_fraction']:.3%} "
          f"reference_common_fraction={r['reference_common']:.3%}")

print(f"\nElapsed: {time.monotonic()-started:.1f}s")
print("Only stored-weight products, architectural rotary coefficients, and small Gram matrices were used.")
print("No attention pattern, hidden state, normalization denominator, or model output was evaluated.")
print("Cross-head coefficient cancellation is conditional structural evidence, not behavioural cancellation.")
```