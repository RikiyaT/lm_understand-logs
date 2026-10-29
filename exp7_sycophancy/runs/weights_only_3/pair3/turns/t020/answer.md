```python
import math
import time

A,B=Checkpoint("A"),Checkpoint("B")
cfg=A.config
D=cfg["head_dim"]
special=np.array([163,400],dtype=int)
pairs=[(11,2,3),(15,1,3),(17,0,2),(4,1,2)]
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

def relative(a,b):
    return norm(b-a)/max(
        math.sqrt((norm(a)**2+norm(b)**2)/2),1e-300
    )

def cancellation_index(x,y):
    return norm(x+y)/max(math.sqrt(norm(x)**2+norm(y)**2),1e-300)

def data(c,l,h1,h2):
    o=get(c,l,"self_attn.o_proj")
    v=get(c,l,"self_attn.v_proj")
    gi=gain(c,l,"input_layernorm")
    go=gain(c,l,"post_attention_layernorm")
    os=[o[:,h*D:(h+1)*D] for h in [h1,h2]]
    raw=[oh@v for oh in os]
    maps={
        "raw":raw,
        "input_gains_only":[m*gi[None,:] for m in raw],
        "output_gains_only":[go[:,None]*m for m in raw],
        "both_gains":[go[:,None]*m*gi[None,:] for m in raw]
    }
    out_removed=[]
    both_removed=[]
    for m in maps["both_gains"]:
        p=m.copy()
        p[special,:]=0
        out_removed.append(p)
        p=p.copy()
        p[:,special]=0
        both_removed.append(p)
    maps["both_gains_without_special_outputs"]=out_removed
    maps["both_gains_without_special_inputs_outputs"]=both_removed
    return dict(
        maps=maps,
        effective_o=[go[:,None]*oh for oh in os],
        effective_v=v*gi[None,:]
    )

print("=== OPPOSING-HEAD STRUCTURE: GAIN AND COORDINATE ROBUSTNESS ===")
print("Cancellation index = ||M1+M2|| / sqrt(||M1||^2+||M2||^2).")
print("These are conditional common-routing coefficients, not actual attention outputs.")

for l,h1,h2 in pairs:
    a,b=data(A,l,h1,h2),data(B,l,h1,h2)
    print(f"\nLAYER {l}, HEADS {h1}/{h2}")
    for mode in a["maps"]:
        a1,a2=a["maps"][mode]
        b1,b2=b["maps"][mode]
        suma,sumb=a1+a2,b1+b2
        contrast_a,contrast_b=a1-a2,b1-b2
        print(f"  {mode:48s} "
              f"cos_A={cosine(a1,a2):+.6f} cos_B={cosine(b1,b2):+.6f} "
              f"cancel_A={cancellation_index(a1,a2):.6f} "
              f"cancel_B={cancellation_index(b1,b2):.6f} "
              f"sum_rel_change={relative(suma,sumb):.6f} "
              f"contrast_rel_change={relative(contrast_a,contrast_b):.6f}")

    ma=[(x+y)/2 for x,y in zip(a["maps"]["both_gains"],b["maps"]["both_gains"])]
    row_energy=np.sum(ma[0]*ma[0]+ma[1]*ma[1],axis=1)
    col_energy=np.sum(ma[0]*ma[0]+ma[1]*ma[1],axis=0)
    row_cross=np.sum(ma[0]*ma[1],axis=1)
    negative_mass=np.maximum(-row_cross,0)
    er=max(float(row_energy.sum()),1e-300)
    ec=max(float(col_energy.sum()),1e-300)
    en=max(float(negative_mass.sum()),1e-300)
    ir=np.argsort(row_energy)[-6:][::-1]
    ic=np.argsort(col_energy)[-6:][::-1]
    ine=np.argsort(negative_mass)[-6:][::-1]
    print("  Leading output coordinates (index, reference energy share):",
          [(int(i),round(float(row_energy[i]/er),5)) for i in ir])
    print("  Leading input coordinates (index, reference energy share):",
          [(int(i),round(float(col_energy[i]/ec),5)) for i in ic])
    print("  Leading opposing-overlap output coordinates (index, negative-mass share):",
          [(int(i),round(float(negative_mass[i]/en),5)) for i in ine])
    print("  Coordinates 163/400:",
          f"output_energy_share={float(row_energy[special].sum()/er):.3%}",
          f"input_energy_share={float(col_energy[special].sum()/ec):.3%}",
          f"negative_overlap_share={float(negative_mass[special].sum()/en):.3%}")

    # Exact midpoint product decomposition for the pair's common coefficient:
    # delta[(O1+O2)V] = delta(O1+O2) Vmid + (O1+O2)mid delta(V).
    oa=a["effective_o"][0]+a["effective_o"][1]
    ob=b["effective_o"][0]+b["effective_o"][1]
    va,vb=a["effective_v"],b["effective_v"]
    from_o=(ob-oa)@((va+vb)/2)
    from_v=((oa+ob)/2)@(vb-va)
    total=ob@vb-oa@va
    energy=max(norm(total)**2,1e-300)
    print("  Pair-common coefficient difference attribution:",
          f"output_factor_signed_share={inner(from_o,total)/energy:.3%}",
          f"shared_value_signed_share={inner(from_v,total)/energy:.3%}",
          f"factor_cross_cos={cosine(from_o,from_v):+.5f}",
          f"reconstruction_error={norm(total-from_o-from_v)/max(norm(total),1e-300):.3g}")

    if l==11:
        print("  L11 common/contrast spectral concentration:")
        for name,m in [
            ("midpoint_common",ma[0]+ma[1]),
            ("midpoint_contrast",ma[0]-ma[1]),
            ("difference_common",
             (b["maps"]["both_gains"][0]+b["maps"]["both_gains"][1])-
             (a["maps"]["both_gains"][0]+a["maps"]["both_gains"][1])),
            ("difference_contrast",
             (b["maps"]["both_gains"][0]-b["maps"]["both_gains"][1])-
             (a["maps"]["both_gains"][0]-a["maps"]["both_gains"][1]))
        ]:
            u,s,vt=np.linalg.svd(m,full_matrices=False)
            energy=max(float(np.sum(s*s)),1e-300)
            outcoords=np.argsort(u[:,0]**2)[-4:][::-1]
            incoords=np.argsort(vt[0]**2)[-4:][::-1]
            print(f"    {name:22s} "
                  f"top1={float(s[0]**2/energy):.3%} "
                  f"top4={float(np.sum(s[:4]**2)/energy):.3%} "
                  f"outcoords={[(int(i),round(float(u[i,0]**2),4)) for i in outcoords]} "
                  f"incoords={[(int(i),round(float(vt[0,i]**2),4)) for i in incoords]}")

print(f"\nElapsed: {time.monotonic()-started:.1f}s")
print("All quantities are products, contractions, or decompositions of stored weights.")
print("No attention pattern, input, hidden-state normalization, activation, or output was computed.")
print("Opposing OV coefficients do not establish cancellation when heads attend differently.")
```