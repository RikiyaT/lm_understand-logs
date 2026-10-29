```python
import math
import time

A, B = Checkpoint("A"), Checkpoint("B")
started = time.monotonic()
POOL = 128

candidates = {
    0:[1405,1130],
    1:[1725,372,1385,1976],
    4:[13,1044,84,20],
    6:[1824],
    7:[1455],
    8:[188,1394,1379],
    10:[1363,1564,1529],
    14:[641,685],
    15:[1369],
    16:[893],
    17:[1026,1916,1312,1180]
}
# Fixed-index comparison units, not selected for large previous statistics.
control_indices = [0,1,2,3]

def rows(x,y):
    return np.einsum("ij,ij->i",x,y,optimize=False)

def load(c,l):
    p=f"model.layers.{l}."
    factors=(
        c.get(p+"mlp.down_proj.weight").astype(np.float64).T,
        c.get(p+"mlp.gate_proj.weight").astype(np.float64),
        c.get(p+"mlp.up_proj.weight").astype(np.float64)
    )
    pre=1+c.get(p+"pre_feedforward_layernorm.weight").astype(np.float64)
    post=1+c.get(p+"post_feedforward_layernorm.weight").astype(np.float64)
    return factors,pre,post

def coefficient_inner(x,y):
    dx,gx,ux=x
    dy,gy,uy=y
    dd=rows(dx,dy)
    gg=rows(gx,gy)
    uu=rows(ux,uy)
    cross=rows(gx,uy)*rows(ux,gy)
    return {
        "base":0.5*dd*(gg*uu+cross),
        "gate_doubled":dd*gg*(gg*uu+2*cross)/3
    }

def coefficient_difference_energy(midpoint,perturbation):
    # Arrays represent coefficient factors, never hidden/input vectors.
    minus=tuple(m-0.5*d for m,d in zip(midpoint,perturbation))
    plus=tuple(m+0.5*d for m,d in zip(midpoint,perturbation))
    aa=coefficient_inner(minus,minus)
    bb=coefficient_inner(plus,plus)
    ab=coefficient_inner(minus,plus)
    return {k:np.maximum(aa[k]+bb[k]-2*ab[k],0) for k in aa}

def summarize_comparison(energy):
    observed=float(energy[0])
    comparison=energy[1:]
    median=float(np.median(comparison))
    percentile=100*float(
        (np.sum(comparison<observed)+0.5*np.sum(comparison==observed))
        /len(comparison)
    )
    return dict(
        fold=observed/max(median,1e-300),
        percentile=percentile,
        q10=float(np.quantile(comparison,0.10)),
        q90=float(np.quantile(comparison,0.90)),
        observed=observed
    )

results=[]
for l,target_units in candidates.items():
    fa,prea,posta=load(A,l)
    fb,preb,postb=load(B,l)
    midpoint=tuple((a+b)/2 for a,b in zip(fa,fb))
    difference=tuple(b-a for a,b in zip(fa,fb))

    # Hold learned gains fixed at their A/B midpoint to isolate projection changes.
    # Earlier attribution found gain changes negligible for these candidates.
    pre,post=(prea+preb)/2,(posta+postb)/2
    gains=(post,pre,pre)

    baseline_norms=np.stack(
        [np.sqrt(rows(x,x)) for x in midpoint],axis=1
    )
    delta_norms=np.stack(
        [np.sqrt(rows(x,x)) for x in difference],axis=1
    )
    lognorms=np.log(np.maximum(baseline_norms,1e-300))

    requested=list(target_units)
    controls=[i for i in control_indices if i not in target_units]
    for i in requested+controls:
        # Match baseline D/G/U row norms in log-norm space.
        # This selects comparison rows; it does not optimize model parameters.
        distance=np.sum((lognorms-lognorms[i])**2,axis=1)
        distance[i]=np.inf
        donors=np.argsort(distance)[:POOL]
        ids=np.concatenate(([i],donors))
        fixed_midpoint=tuple(
            m[i:i+1]*gain[None,:]
            for m,gain in zip(midpoint,gains)
        )

        borrowed=tuple(
            d[ids]*gain[None,:]
            for d,gain in zip(difference,gains)
        )
        ordinary=coefficient_difference_energy(fixed_midpoint,borrowed)

        # Match each borrowed raw difference norm to the candidate's corresponding
        # raw difference norm, before applying the fixed learned gains.
        matched=[]
        for j,(d,gain) in enumerate(zip(difference,gains)):
            scale=delta_norms[i,j]/np.maximum(delta_norms[ids,j],1e-300)
            matched.append(d[ids]*scale[:,None]*gain[None,:])
        norm_matched=coefficient_difference_energy(fixed_midpoint,tuple(matched))

        match_ratios=np.median(
            baseline_norms[donors]/np.maximum(baseline_norms[i],1e-300),
            axis=0
        )
        results.append(dict(
            layer=l,unit=i,candidate=i in requested,
            ordinary={k:summarize_comparison(v) for k,v in ordinary.items()},
            matched={k:summarize_comparison(v) for k,v in norm_matched.items()},
            baseline_match_ratios=match_ratios,
            median_log_distance=float(np.median(np.sqrt(distance[donors])))
        ))

print("=== MATCHED PERTURBATION-ROW COMPARISONS ===")
print("Each fold is observed coefficient-difference ENERGY divided by")
print("the median energy from 128 matched borrowed difference-row triples.")
print("Percentiles describe this comparison pool; they are NOT p-values,")
print("behavioural effects, or probabilities of sycophancy.")
print("Columns show fold / percentile.")

for r in results:
    if not r["candidate"]:
        continue
    ob=r["ordinary"]["base"]
    og=r["ordinary"]["gate_doubled"]
    mb=r["matched"]["base"]
    mg=r["matched"]["gate_doubled"]
    print(f"L{r['layer']:02d}U{r['unit']:04d} "
          f"borrowed_base={ob['fold']:.3f}/{ob['percentile']:5.1f} "
          f"borrowed_gate2={og['fold']:.3f}/{og['percentile']:5.1f} "
          f"normmatched_base={mb['fold']:.3f}/{mb['percentile']:5.1f} "
          f"normmatched_gate2={mg['fold']:.3f}/{mg['percentile']:5.1f}")

print("\n=== BASELINE-MATCH QUALITY ===")
print("Median donor/candidate factor-norm ratios (D,G,U); ideal is (1,1,1).")
print("Poor matches limit interpretation, especially for unusually large units.")
for r in results:
    if r["candidate"]:
        ratios=r["baseline_match_ratios"]
        print(f"L{r['layer']:02d}U{r['unit']:04d} "
              f"ratios=({ratios[0]:.3f},{ratios[1]:.3f},{ratios[2]:.3f}) "
              f"median_log_distance={r['median_log_distance']:.3f}")

print("\n=== CANDIDATES RANKED BY NORM-MATCHED GATE-DOUBLED FOLD ===")
selected=[r for r in results if r["candidate"]]
for r in sorted(selected,
                key=lambda r:r["matched"]["gate_doubled"]["fold"],
                reverse=True)[:12]:
    z=r["matched"]["gate_doubled"]
    print(f"L{r['layer']:02d}U{r['unit']:04d} "
          f"fold={z['fold']:.4f} pool_percentile={z['percentile']:.2f}")

print("\n=== FIXED-INDEX COMPARISON UNITS ===")
for l in candidates:
    rr=[r for r in results if r["layer"]==l and not r["candidate"]]
    if not rr:
        continue
    folds=[r["matched"]["gate_doubled"]["fold"] for r in rr]
    percentiles=[r["matched"]["gate_doubled"]["percentile"] for r in rr]
    print(f"L{l:02d} controls={[r['unit'] for r in rr]} "
          f"normmatched_gate2_folds={np.round(folds,3)} "
          f"pool_percentiles={np.round(percentiles,1)}")

print(f"\nElapsed: {time.monotonic()-started:.1f}s")
print("No activations, inputs, forward passes, optimization, or behavioural evaluations were used.")
print("Borrowed rows are comparison operands for weight-polynomial contractions only.")
print("This comparison does not model the unknown training procedure and cannot calibrate an A/B label.")
```