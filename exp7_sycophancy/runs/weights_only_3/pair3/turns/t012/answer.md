```python
import math
import time

A, B = Checkpoint("A"), Checkpoint("B")
L = A.config["num_hidden_layers"]
started = time.monotonic()

def load(c,l):
    p=f"model.layers.{l}."
    d=c.get(p+"mlp.down_proj.weight").astype(np.float64).T
    g=c.get(p+"mlp.gate_proj.weight").astype(np.float64)
    u=c.get(p+"mlp.up_proj.weight").astype(np.float64)
    pre=1+c.get(p+"pre_feedforward_layernorm.weight").astype(np.float64)
    post=1+c.get(p+"post_feedforward_layernorm.weight").astype(np.float64)
    return d*post[None,:],g*pre[None,:],u*pre[None,:]

def coefficient_grams(x,y,same=False):
    dx,gx,ux=x
    dy,gy,uy=y
    dd=dx@dy.T
    gg=gx@gy.T
    uu=ux@uy.T
    gu=gx@uy.T
    ug=gu.T if same else ux@gy.T
    cross=gu*ug
    base=gg*uu

    # Tensor 1: d outer sym(g outer u).
    quadratic=0.5*dd*(base+cross)

    # Tensor 2: d outer sym(g outer g outer u).
    # The symmetric input tensor averages its three distinct placements of u.
    # Its inner product is:
    # <d,e> [<g,h>^2 <u,v> + 2 <g,h><g,v><u,h>] / 3.
    mixed=base+2*cross
    gate_doubled=(dd*gg)*mixed/3

    # Control: d outer sym(u outer u outer g).
    up_doubled=(dd*uu)*mixed/3
    return {
        "gate_up":quadratic,
        "gate_gate_up":gate_doubled,
        "up_up_gate":up_doubled
    }

def rankcorr(x,y):
    # Ordinal-rank correlation, descriptive rather than a statistical test.
    rx=np.argsort(np.argsort(x)).astype(float)
    ry=np.argsort(np.argsort(y)).astype(float)
    rx-=rx.mean()
    ry-=ry.mean()
    return float(np.dot(rx,ry)/max(np.linalg.norm(rx)*np.linalg.norm(ry),1e-300))

modes=["gate_up","gate_gate_up","up_up_gate"]
records=[]
unit_records=[]
print("=== GATE-ASYMMETRY AUDIT: EXACT STRUCTURAL-COEFFICIENT CONTRACTIONS ===")
print("All tensors below are formed from weights, not from hidden states.")
print("Higher-order tensors are not evaluations of GELU or of an MLP.")
print("layer relative_gate_up relative_gate_gate_up relative_up_up_gate "
      "unit_rank_corr(base,gate_double) unit_rank_corr(gate_double,up_double)")

for l in range(L):
    xa,xb=load(A,l),load(B,l)
    aa=coefficient_grams(xa,xa,True)
    bb=coefficient_grams(xb,xb,True)
    ab=coefficient_grams(xa,xb)
    local={}
    for mode in modes:
        total_a=float(aa[mode].sum())
        total_b=float(bb[mode].sum())
        reference=max((total_a+total_b)/2,1e-300)
        dg=aa[mode]+bb[mode]-ab[mode]-ab[mode].T
        total=float(dg.sum())
        if total < -1e-10*reference:
            raise ArithmeticError("Unexpected negative coefficient-difference norm.")
        total=max(total,1e-300)
        diagonal=np.maximum(np.diag(dg).copy(),0)
        signed=dg.sum(axis=1)
        order=np.argsort(signed)[::-1]
        top20=order[:20]
        group_energy=float(dg[np.ix_(top20,top20)].sum())
        removal=(2*float(signed[top20].sum())-group_energy)/total
        local[mode]=dict(
            relative=math.sqrt(total/reference),
            diagonal=diagonal,
            signed=signed,
            total=total,
            top_units=order[:6].tolist(),
            removal=removal,
            strength_ratio=math.sqrt(max(total_b,0)/max(total_a,1e-300))
        )
        del dg

    r_bg=rankcorr(local["gate_up"]["diagonal"],
                  local["gate_gate_up"]["diagonal"])
    r_gu=rankcorr(local["gate_gate_up"]["diagonal"],
                  local["up_up_gate"]["diagonal"])
    print(f"{l:2d} {local['gate_up']['relative']:.7f} "
          f"{local['gate_gate_up']['relative']:.7f} "
          f"{local['up_up_gate']['relative']:.7f} "
          f"{r_bg:+.4f} {r_gu:+.4f}")

    records.append(dict(
        layer=l,
        base=local["gate_up"]["relative"],
        gate=local["gate_gate_up"]["relative"],
        up=local["up_up_gate"]["relative"],
        rankcorr_bg=r_bg,rankcorr_gu=r_gu,
        details={mode:{
            k:v for k,v in local[mode].items()
            if k not in ["diagonal","signed"]
        } for mode in modes}
    ))

    for i in range(len(local["gate_up"]["diagonal"])):
        item=dict(layer=l,unit=i)
        for mode in modes:
            z=local[mode]
            item[mode+"_share"]=float(z["signed"][i]/z["total"])
            item[mode+"_individual_share"]=float(
                z["diagonal"][i]/max(float(z["diagonal"].sum()),1e-300)
            )
        unit_records.append(item)
    del aa,bb,ab,xa,xb,local

print("\n=== LEADING UNITS UNDER EACH COEFFICIENT FAMILY ===")
for r in records:
    d=r["details"]
    print(f"L{r['layer']:02d} "
          f"base={d['gate_up']['top_units']} "
          f"gate_doubled={d['gate_gate_up']['top_units']} "
          f"up_doubled={d['up_up_gate']['top_units']}")

print("\n=== TOP 16 UNITS BY SIGNED SHARE OF GATE-DOUBLED COEFFICIENT DIFFERENCE ===")
for r in sorted(unit_records,
                key=lambda r:r["gate_gate_up_share"],reverse=True)[:16]:
    print(f"L{r['layer']:02d}U{r['unit']:04d} "
          f"base_share={r['gate_up_share']:+.3%} "
          f"gate_doubled_share={r['gate_gate_up_share']:+.3%} "
          f"up_doubled_share={r['up_up_gate_share']:+.3%}")

print("\n=== LARGEST GATE-DOUBLED VERSUS UP-DOUBLED SHARE DIFFERENCES ===")
for r in sorted(unit_records,
                key=lambda r:abs(r["gate_gate_up_share"]-r["up_up_gate_share"]),
                reverse=True)[:12]:
    print(f"L{r['layer']:02d}U{r['unit']:04d} "
          f"base={r['gate_up_share']:+.3%} "
          f"gate_doubled={r['gate_gate_up_share']:+.3%} "
          f"up_doubled={r['up_up_gate_share']:+.3%}")

print("\n=== LAYER RANKINGS BY RELATIVE TOTAL COEFFICIENT DIFFERENCE ===")
for label,key in [("base","base"),("gate_doubled","gate"),("up_doubled","up")]:
    print(label,[(r["layer"],round(r[key],6))
                 for r in sorted(records,key=lambda r:r[key],reverse=True)])

print("\n=== CONCENTRATION OF GATE-DOUBLED DIFFERENCES ===")
for r in records:
    d=r["details"]["gate_gate_up"]
    print(f"L{r['layer']:02d} top20_coefficient_removal={d['removal']:.3%} "
          f"B/A_coefficient_strength={d['strength_ratio']:.7f}")

print(f"\nElapsed: {time.monotonic()-started:.1f}s")
print("No activation, softmax, hidden-state normalization, input vector, or model output was computed.")
print("Disagreement between coefficient families indicates metric sensitivity,")
print("not proof that any unit controls a behavioural distinction.")
```