```python
import math
import time

A, B = Checkpoint("A"), Checkpoint("B")
L = A.config["num_hidden_layers"]
started = time.monotonic()

def load_raw(c, l, units=None):
    p = f"model.layers.{l}."
    d = c.get(p+"mlp.down_proj.weight").astype(np.float64).T
    g = c.get(p+"mlp.gate_proj.weight").astype(np.float64)
    u = c.get(p+"mlp.up_proj.weight").astype(np.float64)
    if units is not None:
        d, g, u = d[units], g[units], u[units]
    pre = 1+c.get(p+"pre_feedforward_layernorm.weight").astype(np.float64)
    post = 1+c.get(p+"post_feedforward_layernorm.weight").astype(np.float64)
    return (d,g,u), pre, post

def effective(raw, pre, post):
    d,g,u = raw
    return d*post[None,:], g*pre[None,:], u*pre[None,:]

def gram(x, y, same=False):
    dx,gx,ux = x
    dy,gy,uy = y
    result = dx @ dy.T
    t = gx @ gy.T
    t *= ux @ uy.T
    cross = gx @ uy.T
    t += cross*cross.T if same else cross*(ux @ gy.T)
    result *= 0.5*t
    return result

def tensor_inner(x,y):
    return float(gram(x,y).sum())

print("=== ALL-LAYER EXACT TOTAL COUPLING-COEFFICIENT COMPARISON ===")
print("layer relative_change coherence top20_coefficient_removal "
      "B/A_strength top4_units_by_signed_overlap")
summaries=[]
for l in range(L):
    ra, prea, posta = load_raw(A,l)
    rb, preb, postb = load_raw(B,l)
    xa, xb = effective(ra,prea,posta), effective(rb,preb,postb)
    aa, bb, ab = gram(xa,xa,True), gram(xb,xb,True), gram(xa,xb)
    na, nb = float(aa.sum()), float(bb.sum())
    dg = aa+bb-ab-ab.T
    diagonal = np.diag(dg)
    overlaps = dg.sum(axis=1)
    total = max(float(dg.sum()),1e-30)
    reference = max((na+nb)/2,1e-30)
    rel = math.sqrt(total/reference)
    coherence = total/max(float(diagonal.sum()),1e-30)
    order = np.argsort(overlaps)[::-1]
    ix = order[:20]
    group_energy = float(dg[np.ix_(ix,ix)].sum())
    removal = (2*float(overlaps[ix].sum())-group_energy)/total
    ratio = math.sqrt(max(nb,0)/max(na,1e-30))
    print(f"{l:2d} {rel:.7f} {coherence:.5f} {removal:.3%} "
          f"{ratio:.7f} {order[:4].tolist()}")
    summaries.append(dict(
        layer=l,relative=rel,coherence=coherence,removal=removal,
        ratio=ratio,units=order[:4].tolist()
    ))
    del aa,bb,ab,dg,xa,xb,ra,rb

print("\n=== ALL LAYERS RANKED BY RELATIVE TOTAL COEFFICIENT CHANGE ===")
print([(s["layer"],round(s["relative"],7))
       for s in sorted(summaries,key=lambda s:s["relative"],reverse=True)])

candidate_groups = [
    (0,[1405,1130]),
    (1,[1725,372,1385,1976]),
    (3,[1635,1987,1414,146]),
    (4,[13,1044,20]),
    (6,[1824]),
    (7,[1455]),
    (10,[1363,1564,1529]),
    (14,[641,685]),
    (17,[1916,1312,687,1180])
]

print("\n=== EXACT FACTOR ATTRIBUTION WITHIN SELECTED UNIT GROUPS ===")
print("Shares below are signed inner products with the group's total coefficient")
print("difference. They sum to one, but are NOT behavioural effect shares.")

for l,units in candidate_groups:
    ra,prea,posta = load_raw(A,l,units)
    rb,preb,postb = load_raw(B,l,units)
    xa,xb = effective(ra,prea,posta),effective(rb,preb,postb)
    dm,gm,um = [(a+b)/2 for a,b in zip(xa,xb)]
    dd,dg,du = [b-a for a,b in zip(xa,xb)]
    drm,grm,urm = [(a+b)/2 for a,b in zip(ra,rb)]
    ddr,dgr,dur = [b-a for a,b in zip(ra,rb)]
    prem,postm = (prea+preb)/2,(posta+postb)/2
    dpre,dpost = preb-prea,postb-posta

    # Exact midpoint expansion of a trilinear coefficient tensor:
    # delta T = T(dd,gm,um)+T(dm,dg,um)+T(dm,gm,du)
    #           + T(dd,dg,du)/4.
    # Each first-order effective-factor difference is further split exactly
    # between projection-matrix and learned-gain differences.
    terms = [
        ("down_projection", (ddr*postm[None,:],gm,um)),
        ("post_gain",       (drm*dpost[None,:],gm,um)),
        ("gate_projection", (dm,dgr*prem[None,:],um)),
        ("pre_gain_gate",   (dm,grm*dpre[None,:],um)),
        ("up_projection",   (dm,gm,dur*prem[None,:])),
        ("pre_gain_up",     (dm,gm,urm*dpre[None,:])),
        ("cubic_cross",     (dd*0.25,dg,du))
    ]

    aa = tensor_inner(xa,xa)
    bb = tensor_inner(xb,xb)
    ab = tensor_inner(xa,xb)
    total = max(aa+bb-2*ab,1e-30)
    contributions={}
    sizes={}
    for name,t in terms:
        contributions[name]=(tensor_inner(t,xb)-tensor_inner(t,xa))/total
        sizes[name]=math.sqrt(max(tensor_inner(t,t),0)/total)

    gain_share = sum(contributions[k] for k in
                     ["post_gain","pre_gain_gate","pre_gain_up"])
    print(f"L{l:02d} units={units} "
          f"group_relative={math.sqrt(total/max((aa+bb)/2,1e-30)):.6f}")
    print(f"  signed shares: down={contributions['down_projection']:+.3%} "
          f"gate={contributions['gate_projection']:+.3%} "
          f"up={contributions['up_projection']:+.3%} "
          f"all_gains={gain_share:+.3%} "
          f"cubic={contributions['cubic_cross']:+.6%} "
          f"sum={sum(contributions.values()):.9f}")
    print("  coefficient-norm ratios:",
          {k:round(sizes[k],5) for k in
           ["down_projection","gate_projection","up_projection",
            "post_gain","pre_gain_gate","pre_gain_up"]})

print(f"\nElapsed: {time.monotonic()-started:.1f}s")
print("Only stored weights, diagonal gains, and exact tensor contractions were used.")
print("No activation function, normalization of a hidden state, or model evaluation occurred.")
```